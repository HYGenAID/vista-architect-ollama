"""
Graph Serializer — Convert Lumia graph into compact clinical text + E-reference map.

Replaces raw XML with a structured, date-grouped clinical text format that:
- Groups measurements by exact timestamp (not same-day)
- Keeps ALL measurement values (not just abnormals)
- Embeds E-references inline for provenance tracking
- Splits at date boundaries for chunking

Output format:
    === 2020-01-15 ===
    [E1] note Attending Note: Patient presents with 3-week history of cough...
    [E2] procedure CT CHEST W IV CONTRAST
    [E3] condition C34.1 Malignant neoplasm of upper lobe

    --- Labs 2020-01-15 08:42 ---
    [E4] Hemoglobin A1c: 5.30 % of total Hgb
    [E5] Glucose mean value: 111 mg/dL

Usage:
    >>> from toa.graph_store import CohortGraphStore
    >>> store = CohortGraphStore("graph_store")
    >>> graph = store.load_patient_graph("136020661")
    >>> text, ref_map = serialize_graph(graph, "136020661")
    >>> chunks = chunk_serialized_text(text)
"""

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from toa.graph import TOAGraph


# Regex to match date headers: === YYYY-MM-DD === or === DEMOGRAPHICS ===
_DATE_HEADER_RE = re.compile(r"^=== (\d{4}-\d{2}-\d{2}) ===$", re.MULTILINE)


def extract_chunk_date_range(chunk_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract the earliest and latest date from a chunk's date headers.

    Args:
        chunk_text: Serialized chunk text containing === YYYY-MM-DD === headers

    Returns:
        (earliest_date, latest_date) as YYYY-MM-DD strings, or (None, None) if no dates found.
    """
    dates = _DATE_HEADER_RE.findall(chunk_text)
    if not dates:
        return None, None
    dates_sorted = sorted(set(dates))
    return dates_sorted[0], dates_sorted[-1]


def filter_vector_for_chunk(ct_date_vector: List[str],
                            chunk_text: str,
                            buffer_days: int = 30) -> List[str]:
    """Filter CT date vector to dates relevant to a chunk's time window.

    Returns vector dates that fall within the chunk's date range, plus a buffer
    on each side to catch jittered dates near boundaries.

    Args:
        ct_date_vector: Full CT date vector (YYYY-MM-DD, descending)
        chunk_text: Serialized chunk text
        buffer_days: Extra days on each side of chunk range (default 30)

    Returns:
        Filtered CT dates within the chunk window, descending order.
    """
    earliest, latest = extract_chunk_date_range(chunk_text)
    if not earliest or not latest:
        return ct_date_vector  # Can't determine range, return full vector

    from datetime import datetime, timedelta
    try:
        range_start = datetime.strptime(earliest, "%Y-%m-%d") - timedelta(days=buffer_days)
        range_end = datetime.strptime(latest, "%Y-%m-%d") + timedelta(days=buffer_days)
    except ValueError:
        return ct_date_vector

    filtered = []
    for d in ct_date_vector:
        try:
            dt = datetime.strptime(d[:10], "%Y-%m-%d")
            if range_start <= dt <= range_end:
                filtered.append(d)
        except ValueError:
            continue

    return filtered


def filter_imaging_vector_for_chunk(imaging_vector: List[dict],
                                     chunk_text: str,
                                     buffer_days: int = 30) -> List[dict]:
    """Filter full imaging vector (all modalities) to a chunk's time window.

    Args:
        imaging_vector: List of {date, modality, site} dicts (descending)
        chunk_text: Serialized chunk text
        buffer_days: Extra days on each side of chunk range (default 30)

    Returns:
        Filtered imaging entries within the chunk window, descending order.
    """
    earliest, latest = extract_chunk_date_range(chunk_text)
    if not earliest or not latest:
        return imaging_vector

    from datetime import datetime, timedelta
    try:
        range_start = datetime.strptime(earliest, "%Y-%m-%d") - timedelta(days=buffer_days)
        range_end = datetime.strptime(latest, "%Y-%m-%d") + timedelta(days=buffer_days)
    except ValueError:
        return imaging_vector

    filtered = []
    for entry in imaging_vector:
        try:
            dt = datetime.strptime(entry['date'][:10], "%Y-%m-%d")
            if range_start <= dt <= range_end:
                filtered.append(entry)
        except (ValueError, KeyError):
            continue

    return filtered


def serialize_graph(
    graph: TOAGraph,
    patient_id: str,
    max_note_chars: Optional[int] = None,
) -> Tuple[str, Dict[str, str]]:
    """
    Convert a Lumia graph into compact clinical text with E-references.

    Args:
        graph: Lumia hierarchical graph (TOAGraph)
        patient_id: Patient ID (for logging)
        max_note_chars: Optional max characters per note text (None = no limit)

    Returns:
        (serialized_text, reference_map)
        - serialized_text: Compact clinical text with [E1], [E2] tags
        - reference_map: Dict mapping 'E1' -> node_id (e.g., 'pid_note_12345')
    """
    # Extract Person node for demographics (DOB, sex, etc.)
    person_nodes = [
        data for _, data in graph.G.nodes(data=True)
        if data.get('node_type') == 'Person'
    ]

    # Extract all Event nodes from graph
    event_nodes = [
        (node_id, data)
        for node_id, data in graph.G.nodes(data=True)
        if data.get('node_type') == 'Event'
    ]

    # Sort by date, then by node_id for stability
    event_nodes.sort(key=lambda x: (x[1].get('date', '9999-99-99'), x[0]))

    # Build reference map: E1 -> node_id
    reference_map = {}
    for i, (node_id, _data) in enumerate(event_nodes, start=1):
        reference_map[f"E{i}"] = node_id

    # Separate measurement events from non-measurement events
    # Measurements have a timestamp with time component for grouping
    measurement_types = {'measurement', 'lab', 'vital'}

    # Group events by date
    events_by_date = defaultdict(list)
    for i, (node_id, data) in enumerate(event_nodes, start=1):
        date = data.get('date', 'UNKNOWN')
        events_by_date[date].append((i, node_id, data))

    # Build serialized text
    lines = []

    # Demographics header (from Person node)
    if person_nodes:
        person = person_nodes[0]
        lines.append("=== DEMOGRAPHICS ===")
        demo_fields = [
            ('birthdate', 'Date of Birth'),
            ('gender', 'Sex'),
            ('age_years', 'Age'),
            ('race', 'Race'),
            ('ethnicity', 'Ethnicity'),
        ]
        for key, label in demo_fields:
            val = person.get(key)
            if val:
                lines.append(f"  {label}: {val}")
        lines.append("")

    for date in sorted(events_by_date.keys()):
        date_events = events_by_date[date]

        lines.append(f"=== {date} ===")

        # Split into measurements and non-measurements
        measurements = []
        non_measurements = []
        for ref_num, node_id, data in date_events:
            event_type = _get_event_type(node_id)
            if event_type in measurement_types:
                measurements.append((ref_num, node_id, data))
            else:
                non_measurements.append((ref_num, node_id, data))

        # Emit non-measurement events first
        for ref_num, node_id, data in non_measurements:
            ref_id = f"E{ref_num}"
            event_type = data.get('type', _get_event_type(node_id))
            name = data.get('name', data.get('label', data.get('description', '')))
            value = data.get('value', '')
            code = data.get('code', '')
            text = _get_full_text(graph, node_id, max_note_chars)

            # Detect imaging procedures and relabel them
            if event_type in ('procedure', 'proc'):
                modality = _detect_imaging_modality(name, code)
                if modality:
                    event_type = f"imaging_procedure [{modality}]"

            # Build the most informative line possible
            if text:
                lines.append(f"[{ref_id}] {event_type} {name}: {text}")
            elif value and name:
                lines.append(f"[{ref_id}] {event_type} {name}: {value}")
            elif name:
                lines.append(f"[{ref_id}] {event_type} {name}")
            else:
                lines.append(f"[{ref_id}] {event_type}")

        # Group measurements by exact timestamp
        if measurements:
            meas_by_ts = defaultdict(list)
            for ref_num, node_id, data in measurements:
                ts = data.get('timestamp', data.get('date', date))
                meas_by_ts[ts].append((ref_num, node_id, data))

            for ts in sorted(meas_by_ts.keys()):
                ts_events = meas_by_ts[ts]
                # Format timestamp: show time if available
                ts_display = ts if ts != date else date
                lines.append(f"--- Labs {ts_display} ---")

                for ref_num, node_id, data in ts_events:
                    ref_id = f"E{ref_num}"
                    name = data.get('name', data.get('label', data.get('description', '')))
                    value = data.get('value', '')
                    unit = data.get('unit', '')

                    if value and name:
                        val_str = f"{value} {unit}".strip() if unit else str(value)
                        lines.append(f"[{ref_id}] {name}: {val_str}")
                    elif name:
                        text = _get_full_text(graph, node_id, max_note_chars)
                        if text:
                            lines.append(f"[{ref_id}] {name}: {text}")
                        else:
                            lines.append(f"[{ref_id}] {name}")

        lines.append("")  # Blank line between dates

    return "\n".join(lines), reference_map


def chunk_serialized_text(
    serialized_text: str,
    max_chunk_chars: int = 120_000,
    max_final_chunk_chars: Optional[int] = None,
    strategy: str = "streamlined",
) -> List[str]:
    """
    Split serialized text into LLM-sized chunks at date-group boundaries.

    Strategies:
        "streamlined" (default): Early chunks are filtered to notes/procedures/conditions
            only. Final chunk is preserved in full (all event types). Matches the XML
            pipeline's streamlined chunking strategy.
        "full": All chunks contain all event types.

    Args:
        serialized_text: Output from serialize_graph()
        max_chunk_chars: Maximum characters per early chunk (default: 120K)
        max_final_chunk_chars: Maximum characters for the final chunk (default: same as max_chunk_chars).
            Keep this smaller for best accuracy — the final chunk feeds Step 2 JSON generation.
        strategy: "streamlined" or "full"

    Returns:
        List of text chunks, each <= max_chunk_chars (or max_final_chunk_chars for last)
    """
    if max_final_chunk_chars is None:
        max_final_chunk_chars = max_chunk_chars

    if len(serialized_text) <= max_chunk_chars:
        return [serialized_text]

    # Split into date blocks at "=== " markers
    blocks = []
    current_block_lines = []

    for line in serialized_text.split("\n"):
        if line.startswith("=== ") and current_block_lines:
            blocks.append("\n".join(current_block_lines))
            current_block_lines = [line]
        else:
            current_block_lines.append(line)

    if current_block_lines:
        blocks.append("\n".join(current_block_lines))

    # Reserve final chunk: take ~max_final_chunk_chars worth of blocks from the end
    final_blocks = []
    final_size = 0
    split_idx = len(blocks)
    for i in range(len(blocks) - 1, -1, -1):
        block_size = len(blocks[i]) + 1
        if final_size + block_size > max_final_chunk_chars and final_blocks:
            break
        final_blocks.insert(0, blocks[i])
        final_size += block_size
        split_idx = i

    # If everything fits in the final chunk, return as single chunk
    if split_idx == 0:
        return [serialized_text]

    beginning_blocks = blocks[:split_idx]
    final_chunk = "\n".join(final_blocks)

    # Filter early blocks if streamlined strategy
    if strategy == "streamlined":
        beginning_blocks = [_filter_block_streamlined(b) for b in beginning_blocks]
        beginning_blocks = [b for b in beginning_blocks if b.strip()]

    # Assemble early blocks into max_chunk_chars chunks
    chunks = []
    current_chunk_parts = []
    current_size = 0

    for block in beginning_blocks:
        block_size = len(block) + 1
        if current_size + block_size > max_chunk_chars and current_chunk_parts:
            chunks.append("\n".join(current_chunk_parts))
            current_chunk_parts = [block]
            current_size = block_size
        else:
            current_chunk_parts.append(block)
            current_size += block_size

    if current_chunk_parts:
        text = "\n".join(current_chunk_parts)
        if text.strip():
            chunks.append(text)

    # Final chunk: always complete (unfiltered)
    chunks.append(final_chunk)

    return chunks


# Event types to keep in streamlined early chunks (matches XML pipeline)
_STREAMLINED_KEEP_TYPES = {
    'note', 'procedure', 'proc', 'condition', 'cond', 'imaging', 'img',
    'imaging_procedure',  # Annotated imaging procedures
}


def _filter_block_streamlined(block: str) -> str:
    """Filter a date block to keep only notes, procedures, conditions, imaging."""
    filtered_lines = []
    for line in block.split("\n"):
        # Always keep date headers and blank lines
        if line.startswith("=== ") or not line.strip():
            filtered_lines.append(line)
            continue
        # Skip lab panels
        if line.startswith("--- Labs"):
            continue
        # Check event type: lines like "[E42] note Progress note: ..."
        if line.startswith("[E"):
            # Extract event type after "] "
            bracket_end = line.find("] ")
            if bracket_end >= 0:
                rest = line[bracket_end + 2:]
                event_type = rest.split(" ", 1)[0] if rest else ""
                if event_type in _STREAMLINED_KEEP_TYPES or event_type == 'imaging_procedure':
                    filtered_lines.append(line)
                # Skip other event types (measurement, drug_exposure, visit, observation, etc.)
                continue
        # Keep anything else (shouldn't happen, but safe)
        filtered_lines.append(line)
    return "\n".join(filtered_lines)


def _detect_imaging_modality(name: str, code: str = "") -> str:
    """Detect if a procedure is an imaging study, return modality or empty string.

    Matches CPT codes and procedure names against known imaging modalities.
    Uses the same keyword logic as deterministic_retrieval.get_imaging_date_vector().

    Returns:
        Modality string (e.g., "CT", "PET-CT", "MRI", "XR") or "" if not imaging.
    """
    name_upper = name.upper()
    code_upper = code.upper()

    # PET-CT / PET/CT (must check before plain CT)
    pet_ct_kw = ['PET/CT', 'PET-CT', 'PET WITH CONCURRENTLY ACQUIRED C',
                 'PET WITH CT', '78815', '78816']
    for kw in pet_ct_kw:
        if kw in name_upper or kw in code_upper:
            return "PET-CT"

    # PET only
    if 'PET' in name_upper.split() or '78811' in code_upper or '78812' in code_upper:
        return "PET"

    # MRI
    mri_kw = ['MRI', 'MAGNETIC RESONANCE', '70551', '70553', '71550', '74181']
    for kw in mri_kw:
        if kw in name_upper or kw in code_upper:
            return "MRI"

    # CT (check after PET-CT to avoid false matches)
    # Avoid bare "CT " which matches "IMPACT ", "CORRECT ", etc.
    ct_kw = ['COMPUTED TOMOGRAPHY', 'CT SCAN', 'CAT SCAN',
             'CT CHEST', 'CT ABDOMEN', 'CT PELVIS', 'CT HEAD', 'CT BRAIN',
             'CT THORAX', 'CT NECK', 'CT SPINE', 'CT SINUS',
             '71250', '71260', '71270', '74177', '74178', '70450', '70460']
    for kw in ct_kw:
        if kw in name_upper or kw in code_upper:
            return "CT"
    # Match "CT" at start of name or after known prefixes
    if name_upper.startswith('CT ') or name_upper.endswith(' CT') or name_upper.endswith('/CT'):
        return "CT"
    # Match Stanford proc format: "STANFORD_PROC/CT ..."
    if '/CT ' in code_upper:
        return "CT"

    # X-ray
    xr_kw = ['XR ', 'X-RAY', 'XRAY', 'RADIOGRAPH', 'CHEST 2 VIEW',
             'CHEST 1 VIEW', '71045', '71046', '71047', '71048']
    for kw in xr_kw:
        if kw in name_upper or kw in code_upper:
            return "XR"

    # Ultrasound (avoid bare "US " which matches "VENOUS ", "FOCUS ", etc.)
    us_kw = ['ULTRASOUND', 'ULTRASON', 'SONOGRAPH', '76604', '76700', '76641',
             'US ABDOMEN', 'US PELVIS', 'US CHEST', 'US LIVER', 'US THYROID',
             'US RENAL', 'US KIDNEY', 'US BREAST', 'US GUIDED']
    for kw in us_kw:
        if kw in name_upper or kw in code_upper:
            return "US"

    # Nuclear medicine (bone scan etc.)
    nuc_kw = ['BONE SCAN', 'SCINTIG', '78300', '78305', '78306']
    for kw in nuc_kw:
        if kw in name_upper or kw in code_upper:
            return "NM"

    return ""


def _get_event_type(node_id: str) -> str:
    """Extract event type from node ID (e.g., 'pid_note_12345' -> 'note')."""
    parts = node_id.split('_')
    # Namespaced IDs: pid_type_id or non-namespaced: type_id
    # Try to find the type part
    for part in parts:
        if part in ('note', 'lab', 'procedure', 'proc', 'condition', 'cond',
                     'drug', 'drug_exposure', 'imaging', 'img', 'vital',
                     'measurement', 'visit', 'observation', 'obs',
                     'allergy', 'device'):
            return part
    # Fallback: second part if namespaced, first part if not
    if len(parts) >= 3:
        return parts[1]  # namespaced: pid_type_id
    elif len(parts) >= 2:
        return parts[0]  # non-namespaced: type_id
    return "event"


def _get_full_text(
    graph: TOAGraph,
    event_node_id: str,
    max_chars: Optional[int] = None,
) -> str:
    """
    Get full text from XMLFragment children of event node.

    Unlike format_lumia_context.py which uses 150-char snippets,
    this returns full text since it replaces the XML source.

    Args:
        graph: Lumia graph
        event_node_id: Event node ID
        max_chars: Optional max characters (None = no limit)

    Returns:
        Full text content from XMLFragment children
    """
    children = list(graph.G.successors(event_node_id))

    texts = []
    for child_id in children:
        child_data = graph.G.nodes[child_id]
        if child_data.get('node_type') == 'XMLFragment':
            text = child_data.get('text', child_data.get('xml_text', ''))
            if text:
                texts.append(text.strip())

    combined = "\n".join(texts) if texts else ""

    if max_chars and len(combined) > max_chars:
        combined = combined[:max_chars] + "..."

    return combined
