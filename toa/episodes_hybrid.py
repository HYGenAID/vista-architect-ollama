"""
Hybrid LLM + code episode derivation for TOA.

Combines:
1. Code-generated imaging-centered hints (deterministic, fast)
2. LLM episode cutting with strict validation (flexible, clinically coherent)
3. Automatic fallback to deterministic if LLM fails

For thoracic tumor board: episodes are built around MAJOR imaging studies.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Callable
import json
import re
import hashlib
from pathlib import Path

# Allowed episode kinds
KINDS = {
    # New treatment-line-based kinds (primary)
    "baseline",              # Background info predating oncological disease
    "diagnosis",             # Initial diagnostic workup
    "treatment_line",        # Line of therapy (preferred going forward)
    "post_oncological",      # Hospice/palliative care only

    # Legacy kinds (backward compatibility)
    "systemic_tx_line",      # Old name for treatment_line
    "imaging_based_state",
    "rt_course",
    "surgery_course",
    "major_ae",
    "hospitalization",
    "progression",
    "surveillance"
}

def _eid(seed: str) -> str:
    """Generate deterministic episode ID from seed."""
    return hashlib.blake2b(seed.encode("utf-8"), digest_size=12).hexdigest()


def categorize_imaging(event: Dict[str, Any]) -> str:
    """
    Categorize imaging by clinical significance for episode logic.

    Returns: "baseline" | "response" | "progression" | "surveillance" | "unknown"
    """
    desc = (event.get("description") or "").lower()

    # Progression markers
    if any(k in desc for k in [
        "progression", "progressed", " pd", "worsening",
        "increased", "larger", "new lesion", "new met",
        "growing", "enlarging"
    ]):
        return "progression"

    # Response markers
    if any(k in desc for k in [
        "partial response", " pr ", "complete response", " cr ",
        "decreased", "smaller", "reduced", "shrinking",
        "improving", "resolved", "resolution"
    ]):
        return "response"

    # Stable disease
    if any(k in desc for k in [
        "stable disease", " sd ", "stable", "no change",
        "unchanged", "no progression"
    ]):
        return "surveillance"

    # Baseline/staging
    if any(k in desc for k in [
        "staging", "baseline", "initial", "diagnosis",
        "first", "pre-treatment"
    ]):
        return "baseline"

    return "unknown"


def build_imaging_centered_hints(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build imaging-centered change points for thoracic tumor board.

    Imaging studies define disease states. Everything else is context.

    Returns:
        hints dict with imaging_anchors, context_events, and rules
    """
    imaging_anchors = []
    context_events = []
    treatment_starts = []
    surgery_events = []
    diagnosis_events = []

    for idx, e in enumerate(events):
        event_type = (e.get("type") or "").lower()
        desc = (e.get("description") or "").lower()
        modality = (e.get("modality") or "").lower()
        priority = (e.get("priority") or "").upper()

        # MAJOR imaging = episode anchor
        if event_type == "imaging" and priority == "MAJOR":
            category = categorize_imaging(e)

            imaging_anchors.append({
                "idx": idx,
                "event_id": e.get("event_id"),
                "date": e.get("valid_time_start"),
                "modality": modality or "unknown",
                "site": e.get("site") or "unknown",
                "category": category,
                "reason": f"{modality or 'imaging'} {e.get('site', 'study')}: {category}",
                "description": e.get("description", "")[:80],
                "suggested_group": None  # LLM will assign semantic groups
            })

        # Diagnosis events
        elif event_type == "diagnostic" and any(k in desc for k in [
            "biopsy", "pathology", "malignant", "cancer", "carcinoma", "mesothelioma"
        ]):
            diagnosis_events.append({
                "idx": idx,
                "event_id": e.get("event_id"),
                "date": e.get("valid_time_start"),
                "reason": "diagnosis/biopsy/pathology"
            })

        # Surgery events
        elif event_type == "surgery":
            surgery_events.append({
                "idx": idx,
                "event_id": e.get("event_id"),
                "date": e.get("valid_time_start"),
                "reason": "surgical intervention"
            })

        # Treatment starts (track for tx line episodes)
        elif event_type == "treatment" and any(k in desc for k in [
            "start", "initiate", "began", "commenced", "first cycle", "cycle 1"
        ]):
            treatment_starts.append({
                "idx": idx,
                "event_id": e.get("event_id"),
                "date": e.get("valid_time_start"),
                "reason": "treatment start/switch"
            })

        # Context events (attach to nearest imaging)
        elif event_type in ("treatment", "diagnostic", "symptom", "adverse_effect", "lab", "examination"):
            context_events.append({
                "idx": idx,
                "event_id": e.get("event_id"),
                "date": e.get("valid_time_start"),
                "type": event_type,
                "attach_to": "nearest_imaging_before",
                "reason": f"context for imaging-based decision"
            })

    return {
        "imaging_anchors": imaging_anchors,
        "context_events": context_events,
        "treatment_starts": treatment_starts,
        "surgery_events": surgery_events,
        "diagnosis_events": diagnosis_events,
        "episode_count_estimate": len(imaging_anchors) + len(surgery_events) + len(diagnosis_events),
        "rules": {
            "thoracic_specific": True,
            "anchor_type": "imaging_studies",
            "group_context_window_days": 90,
            "prefer_imaging_anchors": True
        }
    }


def _events_lite(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert events to lightweight format for LLM input."""
    out = []
    for i, e in enumerate(events):
        out.append({
            "idx": i,
            "event_id": e.get("event_id"),
            "date": e.get("valid_time_start"),
            "type": e.get("type"),
            "description": (e.get("description") or "")[:160],
            "priority": e.get("priority", "MINOR"),
            "modality": e.get("modality"),
            "site": e.get("site"),
        })
    return out


def llm_cut_episodes(
    events: List[Dict[str, Any]],
    chat_client: Callable[[str, str], str],
    system_prompt: str,
    profile_name: str = "thoracic_tumor_board",
    source_text: Optional[str] = None,
    deterministic_context: Optional[str] = None,
    ct_date_vector: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Call LLM once with events_lite + hints to get episodes JSON.

    When source_text is provided, the LLM also produces patient_state
    on the last episode (combined episode + patient info extraction).

    Args:
        events: Full timeline events (already deduped/rolled)
        chat_client: LLM function (system_prompt, user_prompt) -> str
        system_prompt: System prompt for LLM
        profile_name: Profile name for overlay rules
        source_text: Final chunk of clinical text for patient_state extraction
        deterministic_context: Graph search hits for LLM attention guidance
        ct_date_vector: Confirmed CT/PET-CT study dates from structured records

    Returns:
        List of episode dicts (last episode may contain patient_state)

    Raises:
        ValueError: If LLM returns invalid JSON or violates constraints
    """
    # Load prompt template
    prompt_path = Path(__file__).resolve().parent / "prompts" / "episodes_from_events.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Episodes prompt not found: {prompt_path}")

    base_prompt = prompt_path.read_text(encoding="utf-8")

    # Build hints
    hints = build_imaging_centered_hints(events)

    # Build payload
    payload = {
        "events_lite": _events_lite(events),
        "hints": hints,
        "profile_rules": f"Profile: {profile_name} (thoracic tumor board specific)"
    }

    # Add optional enrichment inputs for patient_state extraction
    if source_text:
        # Truncate source_text to ~50k chars to stay within token budget
        max_source_chars = 50_000
        if len(source_text) > max_source_chars:
            payload["source_text"] = source_text[:max_source_chars] + "\n... [truncated]"
        else:
            payload["source_text"] = source_text
    if deterministic_context:
        payload["deterministic_context"] = deterministic_context
    if ct_date_vector:
        payload["ct_date_vector"] = ct_date_vector

    # Construct user prompt
    user_prompt = base_prompt + "\n\nDATA:\n" + json.dumps(payload, ensure_ascii=False, indent=2)

    # Call LLM
    has_enrichment = " + patient_state" if source_text else ""
    print(f"Calling LLM for episode cutting{has_enrichment} ({len(events)} events, {len(hints['imaging_anchors'])} imaging anchors)...")
    raw = chat_client(system_prompt, user_prompt)

    # Extract JSON from response
    # Handle ```json ... ``` markers
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", raw)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Extract first complete JSON object (handles double-JSON from concurrent LLM calls)
        from .llm_first_pass import _extract_first_json_object
        try:
            json_str = _extract_first_json_object(raw)
        except ValueError:
            raise ValueError("LLM episodes: no JSON found in response")

    # Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try repair before giving up
        from .llm_first_pass import _repair_json
        try:
            data = json.loads(_repair_json(json_str))
        except json.JSONDecodeError:
            raise ValueError(f"LLM episodes: invalid JSON - {e}")

    episodes = data.get("episodes", [])
    notes = data.get("notes", [])

    print(f"LLM returned {len(episodes)} episodes")
    if notes:
        print(f"LLM notes: {'; '.join(notes[:3])}")

    return episodes


def auto_populate_event_ids(
    episodes: List[Dict[str, Any]],
    events: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Automatically populate event_ids for each episode based on date ranges.

    Rules:
    - baseline: Include all baseline_information type events (regardless of date)
    - diagnosis/treatment_line/post_oncological: Include all events where start_date <= event.date <= end_date
    - If episode already has event_ids, leave it unchanged (backward compatibility)

    Args:
        episodes: List of episode dicts (may or may not have event_ids)
        events: List of all event dicts

    Returns:
        Episodes with event_ids populated
    """
    from datetime import datetime

    for ep in episodes:
        # Skip if already has event_ids (backward compatibility)
        if "event_ids" in ep and ep["event_ids"]:
            continue

        kind = ep.get("kind")
        start_date = ep.get("start_date")
        end_date = ep.get("end_date")

        event_ids = []

        if kind == "baseline":
            # Baseline: include all baseline_information events regardless of date
            # Backward compatibility: also include critical_information if no baseline_information events found
            baseline_events = [
                e["event_id"] for e in events
                if e.get("type") == "baseline_information"
            ]
            if baseline_events:
                event_ids = baseline_events
            else:
                # Fallback for old data that only has critical_information
                event_ids = [
                    e["event_id"] for e in events
                    if e.get("type") == "critical_information"
                ]
        else:
            # Other episodes: include all events in date range
            if start_date:
                start_dt = datetime.fromisoformat(start_date.split('T')[0])
                end_dt = datetime.fromisoformat(end_date.split('T')[0]) if end_date else datetime(9999, 12, 31)

                for e in events:
                    event_date_str = e.get("valid_time_start", "")
                    if event_date_str:
                        event_dt = datetime.fromisoformat(event_date_str.split('T')[0])
                        if start_dt <= event_dt <= end_dt:
                            event_ids.append(e["event_id"])

        ep["event_ids"] = event_ids

    return episodes


def validate_episodes(
    episodes: List[Dict[str, Any]],
    events: List[Dict[str, Any]]
) -> tuple[bool, List[str]]:
    """
    Validate episodes against strict invariants.

    Returns:
        (is_valid, error_messages)
    """
    errs = []

    if not isinstance(episodes, list):
        return False, ["episodes not a list"]

    if len(episodes) == 0:
        return False, ["no episodes generated"]

    # Build event lookup
    event_ids = {e.get("event_id") for e in events}
    dates = [e.get("valid_time_start") for e in events if e.get("valid_time_start")]
    min_date, max_date = (min(dates), max(dates)) if dates else ("", "")

    # Build imaging event IDs for imaging_based_state validation
    imaging_event_ids = {
        e.get("event_id") for e in events
        if e.get("type") == "imaging"
    }

    last_start = ""
    seen_lines = set()

    for i, ep in enumerate(episodes):
        kind = ep.get("kind")
        start_date = ep.get("start_date") or ""
        end_date = ep.get("end_date") or start_date
        anchor_id = ep.get("anchor_event_id")
        event_ids_list = ep.get("event_ids", [])

        # Check kind
        if kind not in KINDS:
            errs.append(f"Episode {i}: invalid kind '{kind}' (allowed: {KINDS})")

        # Check dates
        if not start_date:
            errs.append(f"Episode {i}: missing start_date")

        if start_date and end_date and start_date > end_date:
            errs.append(f"Episode {i}: start_date > end_date ({start_date} > {end_date})")

        # Check anchor exists (except baseline which can have synthetic anchor)
        if anchor_id and anchor_id not in event_ids:
            # Allow synthetic baseline anchors
            if not (kind == "baseline" and "baseline" in anchor_id.lower()):
                errs.append(f"Episode {i}: anchor_event_id '{anchor_id}' not in events")

        # Check date range (except baseline which can be 1 day before first event)
        if min_date and start_date and start_date < min_date:
            # Allow baseline to be 1 day before earliest event
            if not (kind == "baseline" and start_date >= min_date.split('T')[0]):  # Compare just dates
                # More lenient: allow baseline to start up to 1 day before
                from datetime import datetime, timedelta
                min_dt = datetime.fromisoformat(min_date.split('T')[0])
                start_dt = datetime.fromisoformat(start_date.split('T')[0])
                if not (kind == "baseline" and (min_dt - start_dt).days <= 1):
                    errs.append(f"Episode {i}: start_date {start_date} before earliest event {min_date}")
        if max_date and end_date and end_date > max_date:
            errs.append(f"Episode {i}: end_date {end_date} after latest event {max_date}")

        # Check monotonic start dates
        if start_date and last_start and start_date < last_start:
            errs.append(f"Episode {i}: episodes not sorted by start_date ({start_date} < {last_start})")
        last_start = start_date or last_start

        # Kind-specific validation
        if kind in ("systemic_tx_line", "treatment_line"):
            line_num = ep.get("line_number")
            if not isinstance(line_num, int) or line_num < 1:
                errs.append(f"Episode {i}: {kind} must have line_number ≥ 1")
            elif line_num in seen_lines:
                errs.append(f"Episode {i}: duplicate line_number {line_num}")
            seen_lines.add(line_num)

        elif kind == "imaging_based_state":
            # Anchor must be imaging event
            if anchor_id and anchor_id not in imaging_event_ids:
                errs.append(f"Episode {i}: imaging_based_state anchor must be imaging event")

            disease_state = ep.get("disease_state")
            if disease_state not in ["baseline", "response", "stable", "progression", None]:
                errs.append(f"Episode {i}: invalid disease_state '{disease_state}'")

        # Check event_ids reference valid events
        for eid in event_ids_list:
            if eid not in event_ids:
                errs.append(f"Episode {i}: event_id '{eid}' in event_ids not found in events")

    # Cap check
    if len(episodes) > 25:
        errs.append(f"Too many episodes ({len(episodes)} > 25)")

    # Light validation of patient_state on last episode (warn only, don't fail)
    if episodes:
        last_ep = episodes[-1]
        ps = last_ep.get("patient_state")
        if ps:
            for field in ["name", "diagnosis", "tnm_staging", "driver_mutations"]:
                if not ps.get(field):
                    print(f"  ⚠️  patient_state missing recommended field: {field}")

    return (len(errs) == 0), errs


def assign_ids(episodes: List[Dict[str, Any]], pid: str) -> List[Dict[str, Any]]:
    """Assign deterministic episode_id to each episode."""
    out = []
    for ep in episodes:
        seed = f"{pid}|{ep.get('kind')}|{ep.get('start_date')}|{ep.get('anchor_event_id', '')}"
        e = dict(ep)
        if e.get("episode_id") == "temp" or not e.get("episode_id"):
            e["episode_id"] = _eid(seed)
        out.append(e)
    return out


def derive_episodes_llm(
    pid: str,
    events: List[Dict[str, Any]],
    chat_client: Callable[[str, str], str],
    system_prompt: str = "",
    profile_name: str = "thoracic_tumor_board",
    max_retries: int = 1,
    source_text: Optional[str] = None,
    deterministic_context: Optional[str] = None,
    ct_date_vector: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Derive episodes using LLM with automatic validation and retry.

    Falls back to deterministic episodes if LLM fails validation after retries.

    Args:
        pid: Patient ID
        events: Timeline events (already deduped/sorted)
        chat_client: LLM function
        system_prompt: System prompt
        profile_name: Profile name
        max_retries: Number of retries with validator feedback
        source_text: Final chunk of clinical text for patient_state extraction
        deterministic_context: Graph search hits for LLM attention guidance
        ct_date_vector: Confirmed CT/PET-CT study dates from structured records

    Returns:
        List of validated episodes (last may contain patient_state)
    """
    for attempt in range(max_retries + 1):
        try:
            # Call LLM
            llm_episodes = llm_cut_episodes(
                events, chat_client, system_prompt, profile_name,
                source_text=source_text,
                deterministic_context=deterministic_context,
                ct_date_vector=ct_date_vector,
            )

            # Auto-populate event_ids based on date ranges
            llm_episodes = auto_populate_event_ids(llm_episodes, events)

            # Validate
            is_valid, errors = validate_episodes(llm_episodes, events)

            if is_valid:
                print(f"✅ LLM episodes validated successfully")
                assigned = assign_ids(llm_episodes, pid)
                merged = merge_initial_workup_episodes(assigned)
                return merged

            # Validation failed
            print(f"⚠️  LLM episodes validation failed (attempt {attempt + 1}/{max_retries + 1}):")
            for err in errors[:5]:  # Show first 5 errors
                print(f"   - {err}")

            if attempt < max_retries:
                # Retry with validator feedback
                print(f"Retrying with validator feedback...")
                feedback = "\n\nVALIDATION ERRORS FROM PREVIOUS ATTEMPT:\n"
                feedback += "\n".join([f"- {err}" for err in errors])
                feedback += "\n\nPlease fix these errors and return ONLY valid JSON."

                system_prompt_retry = system_prompt + feedback
                llm_episodes = llm_cut_episodes(
                    events, chat_client, system_prompt_retry, profile_name,
                    source_text=source_text,
                    deterministic_context=deterministic_context,
                    ct_date_vector=ct_date_vector,
                )

                # Auto-populate event_ids for retry
                llm_episodes = auto_populate_event_ids(llm_episodes, events)

                is_valid_retry, errors_retry = validate_episodes(llm_episodes, events)
                if is_valid_retry:
                    print(f"✅ LLM episodes validated after retry")
                    assigned = assign_ids(llm_episodes, pid)
                    merged = merge_initial_workup_episodes(assigned)
                    return merged

                print(f"⚠️  Retry still has validation errors")
                for err in errors_retry[:5]:
                    print(f"   - {err}")

        except Exception as e:
            print(f"❌ LLM episode cutting failed: {e}")
            if attempt >= max_retries:
                break

    # Fall back to deterministic
    print("🔄 Falling back to deterministic episode derivation")
    from .episodes import derive_episodes
    episodes = derive_episodes(pid, events)
    merged = merge_initial_workup_episodes(episodes)
    return merged


def merge_initial_workup_episodes(episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge consecutive 'initial_diagnostics' imaging_group episodes into single initial workup.

    Auto-tags the first imaging_based_state episode as 'initial_diagnostics' if not already tagged,
    because by definition the story starts with initial workup.

    Only applies to initial_diagnostics - other episode types remain separate.

    Args:
        episodes: List of episode dictionaries

    Returns:
        List with consecutive initial_diagnostics episodes merged
    """
    if not episodes:
        return episodes

    # Sort by start_date for chronological processing
    sorted_episodes = sorted(episodes, key=lambda e: e.get('start_date', '1970-01-01'))

    # Auto-tag first imaging_based_state episode as initial_diagnostics if not already tagged
    # (by definition, the first imaging is part of initial workup)
    for ep in sorted_episodes:
        if ep.get('kind') == 'imaging_based_state':
            if not ep.get('imaging_group'):
                ep['imaging_group'] = 'initial_diagnostics'
            break  # Only tag the first imaging_based_state episode

    merged = []
    i = 0

    while i < len(sorted_episodes):
        current = sorted_episodes[i].copy()

        # Only merge initial_diagnostics imaging_group
        if (current.get('kind') == 'imaging_based_state' and
            current.get('imaging_group') == 'initial_diagnostics'):

            # Find all consecutive initial_diagnostics episodes
            j = i + 1
            contexts_to_merge = [current.get('clinical_context', '')]

            while j < len(sorted_episodes):
                next_ep = sorted_episodes[j]

                # Stop if not initial_diagnostics
                if (next_ep.get('kind') != 'imaging_based_state' or
                    next_ep.get('imaging_group') != 'initial_diagnostics'):
                    break

                # Merge into current
                current['end_date'] = next_ep.get('end_date', current['end_date'])

                # Merge event_ids (unique)
                current_event_ids = set(current.get('event_ids', []))
                current_event_ids.update(next_ep.get('event_ids', []))
                current['event_ids'] = list(current_event_ids)

                # Collect clinical context
                if next_ep.get('clinical_context', '').strip():
                    contexts_to_merge.append(next_ep['clinical_context'])

                j += 1

            # Number contexts if multiple were merged
            if len(contexts_to_merge) > 1:
                non_empty = [c for c in contexts_to_merge if c.strip()]
                if len(non_empty) > 1:
                    current['clinical_context'] = " ".join([f"{idx+1}. {ctx}" for idx, ctx in enumerate(non_empty)])
                elif non_empty:
                    current['clinical_context'] = non_empty[0]

            merged.append(current)
            i = j
        else:
            # Not initial_diagnostics - keep as is
            merged.append(current)
            i += 1

    return merged
