
from __future__ import annotations
from typing import List, Dict, Any, Callable, Optional, Tuple, Union
from pathlib import Path
import json
import os
from .io.ehr_xml import iter_chunks
from .llm_first_pass import extract_chunk_events
from .dedupe_rollup import rollup_same_day, sort_events, stable_event_id
from .episodes import derive_episodes
from .episodes_hybrid import derive_episodes_llm
from .profiles import Profile, load_profile

def default_chat_client(system_prompt: str, user_prompt: str) -> str:
    raise RuntimeError("No chat client provided. Pass chat_client=... to extract().")


def correct_episode_dates(
    episodes: List[Dict[str, Any]],
    events: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Correct episode start_date and end_date based on actual timeline events.

    For each episode, sets:
    - start_date: date of the FIRST event in event_ids list
    - end_date: date of the LAST event in event_ids list

    This ensures episode dates match the actual events contained in the episode,
    rather than being based on when episodes start/end relative to each other.

    Args:
        episodes: List of episode dictionaries with event_ids
        events: List of timeline events

    Returns:
        List of episodes with corrected dates
    """
    # Build event ID to date lookup
    event_dates = {e.get("event_id"): e.get("valid_time_start") for e in events}

    corrected = []
    for ep in episodes:
        ep_copy = dict(ep)
        event_ids = ep.get("event_ids", [])

        if event_ids:
            # Get dates for all events in this episode
            dates = [event_dates.get(eid) for eid in event_ids]
            dates = [d for d in dates if d]  # Filter out None values

            if dates:
                # Sort dates to get first and last
                dates_sorted = sorted(dates)
                ep_copy["start_date"] = dates_sorted[0]
                ep_copy["end_date"] = dates_sorted[-1]

        corrected.append(ep_copy)

    return corrected

def extract(
    pid: str,
    xml_text: str,
    chat_client: Optional[Callable[[str,str], str]] = None,
    system_prompt: Optional[str] = None,
    profile: Optional[Union[str, Profile]] = None,
    tumor_type: Optional[str] = None,
    episodes_mode: str = "llm",  # "llm" | "code"
    output_dir: Optional[str] = None,  # For intermediate chunk saving
    chunk_chat_client: Optional[Callable[[str,str], str]] = None,  # Separate model for chunk extraction
    episode_chat_client: Optional[Callable[[str,str], str]] = None,  # Separate model for episode splitting
    chunking_strategy: Optional[str] = None,  # Override profile's chunking strategy ("full" | "streamlined")
    lumia_context: Optional[str] = None,  # Lumia graph context for provenance tracking
    deterministic_context: Optional[str] = None,  # Graph search hits for LLM attention guidance
    text_chunks: Optional[List[str]] = None,  # Pre-serialized graph chunks (bypass iter_chunks)
    ct_date_vector: Optional[List[str]] = None,  # Confirmed CT/PET-CT dates from structured data
    imaging_date_vector: Optional[List[dict]] = None,  # Full imaging vector (all modalities: date, modality, site)
) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
    """
    Orchestrates: chunk → first-pass events → normalize/dedupe/rollup → episodes
    Returns (events, episodes). Events are deduped & rolled per-day.

    Args:
        pid: Patient ID
        xml_text: Full EHR XML text
        chat_client: LLM chat function with signature (system_prompt, user_prompt) -> str
                    (used as default if chunk_chat_client or episode_chat_client not specified)
        system_prompt: Optional system prompt (overrides profile prompt if both provided)
        profile: Optional profile name (str) or Profile object for tumor-specific extraction
        tumor_type: Optional tumor type to override profile's default
        episodes_mode: "llm" (imaging-centered hybrid) or "code" (deterministic)
        output_dir: Optional directory to save intermediate chunk results (enables resume)
        chunk_chat_client: Optional separate LLM for chunk extraction (defaults to chat_client)
        episode_chat_client: Optional separate LLM for episode splitting (defaults to chat_client)
        chunking_strategy: Optional override for profile's chunking strategy ("full" | "streamlined")

    Returns:
        Tuple of (events, episodes) where events are deduplicated and rolled up per day
    """
    chat_client = chat_client or default_chat_client

    # Use stage-specific chat clients if provided, otherwise fall back to main chat_client
    chunk_client = chunk_chat_client or chat_client
    episode_client = episode_chat_client or chat_client

    # Load profile if provided
    profile_obj = None
    if profile is not None:
        if isinstance(profile, str):
            profile_obj = load_profile(profile)
        else:
            profile_obj = profile

    # Determine final system prompt
    final_system_prompt = system_prompt
    if final_system_prompt is None and profile_obj is not None:
        # Use profile's prompt template (will be assembled per chunk)
        final_system_prompt = "PROFILE"  # Marker to use profile assembly

    # 1) first pass over chunks
    raw_events: List[Dict[str, Any]] = []
    current_timeline = ""
    current_summary = ""

    # Get chunking config from profile
    max_chunk_size = 120_000  # default
    min_final_chunk_ratio = 0.8  # default
    strategy_from_profile = "streamlined"  # default
    if profile_obj:
        max_chunk_size = profile_obj.chunking_config.get("max_chunk_size", 120_000)
        min_final_chunk_ratio = profile_obj.chunking_config.get("min_final_chunk_ratio", 0.8)
        strategy_from_profile = profile_obj.chunking_config.get("strategy", "streamlined")

    # Allow CLI/caller to override strategy
    final_chunking_strategy = chunking_strategy if chunking_strategy is not None else strategy_from_profile

    # Get chunks: use pre-serialized if provided, otherwise chunk from XML
    if text_chunks is not None:
        chunks = text_chunks
        num_chunks = len(chunks)
        print(f"  Using {num_chunks} pre-serialized chunk(s) from graph store...")
    else:
        chunks = list(iter_chunks(xml_text, max_chars=max_chunk_size, min_final_chunk_ratio=min_final_chunk_ratio, strategy=final_chunking_strategy))
        num_chunks = len(chunks)
        print(f"  Processing {num_chunks} chunk(s) using '{final_chunking_strategy}' strategy...")
        if num_chunks > 1 and final_chunking_strategy == "streamlined":
            print(f"  Note: Early chunks pre-filtered to notes/procedures/conditions; final chunk is complete XML")

    # Setup intermediate save directory if provided
    chunk_dir = None
    if output_dir:
        chunk_dir = Path(output_dir) / ".chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

    for idx, chunk in enumerate(chunks):
        is_final_chunk = (idx == num_chunks - 1)
        chunk_file = chunk_dir / f"chunk_{idx}_events.json" if chunk_dir else None

        # Resume capability: skip if chunk already processed
        if chunk_file and chunk_file.exists():
            print(f"  Chunk {idx+1}/{num_chunks}: Loading from cache (already processed)")
            with open(chunk_file, 'r') as f:
                evs = json.load(f)
            raw_events.extend(evs)
            current_timeline = json.dumps(evs, indent=2)
            continue

        # Guardrail: verify previous chunk was saved (except for first chunk)
        if idx > 0 and chunk_dir:
            prev_chunk_file = chunk_dir / f"chunk_{idx-1}_events.json"
            if not prev_chunk_file.exists():
                raise RuntimeError(f"Guardrail failed: Previous chunk {idx-1} file not found: {prev_chunk_file}")

        # Determine if this chunk is streamlined (filtered)
        # In streamlined strategy: early chunks are filtered; final chunk is complete XML
        # In full strategy: all chunks are complete XML, never streamlined
        is_streamlined = (final_chunking_strategy == "streamlined") and (num_chunks > 1) and not is_final_chunk

        # Process chunk
        print(f"  Chunk {idx+1}/{num_chunks}: Extracting events...")

        # Assemble chunk-specific prompt if using profile
        chunk_prompt = final_system_prompt
        if final_system_prompt == "PROFILE" and profile_obj is not None:
            chunk_prompt = profile_obj.assemble_prompt(
                current_timeline=current_timeline,
                current_summary=current_summary,
                xml_chunk=chunk,
                is_final_chunk=is_final_chunk,
                is_streamlined=is_streamlined,
                tumor_type=tumor_type
            )

        # Filter imaging date vector to this chunk's time window
        chunk_imaging_vector = None
        if imaging_date_vector:
            from toa.graph_serializer import filter_imaging_vector_for_chunk
            chunk_imaging_vector = filter_imaging_vector_for_chunk(imaging_date_vector, chunk)
            if chunk_imaging_vector:
                modalities = {}
                for entry in chunk_imaging_vector:
                    modalities[entry['modality']] = modalities.get(entry['modality'], 0) + 1
                mod_str = ", ".join(f"{m}:{n}" for m, n in sorted(modalities.items()))
                print(f"    Imaging vector: {len(chunk_imaging_vector)} studies for this chunk ({mod_str})")

        evs = extract_chunk_events(
            chunk,
            system_prompt=chunk_prompt,
            chat_client=chunk_client,
            lumia_context=lumia_context,
            deterministic_context=deterministic_context,
            imaging_date_vector=chunk_imaging_vector,
            is_final_chunk=is_final_chunk
        )
        raw_events.extend(evs)

        # Save intermediate results
        if chunk_file:
            with open(chunk_file, 'w') as f:
                json.dump(evs, f, indent=2)
            # Guardrail: verify save succeeded
            if not chunk_file.exists():
                raise RuntimeError(f"Guardrail failed: Chunk {idx} save verification failed")
            print(f"  Chunk {idx+1}/{num_chunks}: Saved {len(evs)} events to {chunk_file.name}")

        # Update context for next chunk
        current_timeline = json.dumps(evs, indent=2)
        # Note: minimal_summary would need to be extracted from LLM response if needed

        # Rate limit delays removed (2026-02-10) — retry logic in prepare_patient.py handles 429s

    # 2) group by valid_time_start (YYYY-MM-DD) and roll up
    by_day: Dict[str, List[Dict[str, Any]]] = {}
    for e in raw_events:
        d = e.get("valid_time_start") or e.get("date") or ""
        if "date" in e and "valid_time_start" not in e:
            e["valid_time_start"] = e["date"]
        by_day.setdefault(d, []).append(e)

    events: List[Dict[str, Any]] = []
    for d, day_events in by_day.items():
        rolled = rollup_same_day(pid, day_events)
        events.extend(rolled)

    events = sort_events(events)

    # ensure event_id present
    for e in events:
        e.setdefault("event_id", stable_event_id(pid, e))
        e.setdefault("patient_id", pid)

    # 3) episodes
    # Rate limit delays removed (2026-02-10) — retry logic in prepare_patient.py handles 429s

    # NOTE: Episode enrichment (source_text, deterministic_context, ct_date_vector)
    # is preserved in episodes_hybrid.py but bypassed here. Patient info extraction
    # now happens in a separate gpt-5 call in prepare_patient.py Step 2.
    # To re-enable enrichment, uncomment the params below.

    if episodes_mode == "llm":
        # Hybrid: LLM with imaging-centered hints + validation + fallback
        profile_name = profile if isinstance(profile, str) else "thoracic_tumor_board"
        episodes = derive_episodes_llm(
            pid=pid,
            events=events,
            chat_client=episode_client,  # Use episode-specific client
            system_prompt=system_prompt or "",
            profile_name=profile_name,
            max_retries=1,
            # Enrichment bypassed — patient_info extracted separately by gpt-5
            # source_text=final_chunk,
            # deterministic_context=deterministic_context,
            # ct_date_vector=ct_date_vector,
        )
    else:
        # Deterministic code-based episode derivation
        episodes = derive_episodes(pid, events)

    # Correct episode dates based on actual timeline events
    # This ensures start_date = first event date, end_date = last event date
    episodes = correct_episode_dates(episodes, events)

    return events, episodes

def project_major_events(events: List[Dict[str,Any]], episodes: List[Dict[str,Any]]) -> Dict[str, Any]:
    """
    Back-compat projection for legacy timeline.json
    - include MAJOR events + episode anchors as MAJOR
    """
    major = []
    episode_anchor_ids = {ep.get("anchor_event_id") for ep in episodes if ep.get("anchor_event_id")}
    for e in events:
        if (e.get("priority")=="MAJOR") or (e.get("event_id") in episode_anchor_ids):
            major.append({
                "date": e.get("valid_time_start"),
                "type": e.get("type"),
                "description": e.get("description"),
                "priority": "MAJOR",
            })
    major = sorted(major, key=lambda x: (x.get("date") or "", x.get("type") or ""))
    return {"events": major}
