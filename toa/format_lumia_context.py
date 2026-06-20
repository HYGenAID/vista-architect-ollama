"""
Format Lumia graph for LLM extraction with full provenance tracking.

This module creates a reference map for ALL Lumia events (not just search hits),
allowing the LLM to cite specific source events when extracting timeline objects.

Example:
    >>> lumia = TOAGraph.load('graphs/patient_lumia.graphml')
    >>> context, ref_map = format_lumia_for_extraction(lumia)
    >>> # ref_map: {'E1': 'note_12345', 'E2': 'note_67890', ..., 'E98': 'lab_99999'}
    >>> # context: Shows LLM all events with [E1], [E2] tags for citation
"""

from typing import Dict, Tuple, List
from toa.graph import TOAGraph
import networkx as nx


def format_lumia_for_extraction(
    lumia_graph: TOAGraph,
    max_snippet_chars: int = 150,
    max_events: int = None
) -> Tuple[str, Dict[str, str]]:
    """
    Format ALL Lumia events with short references for LLM extraction.

    Args:
        lumia_graph: Lumia hierarchical graph
        max_snippet_chars: Max characters per text snippet (default: 150)
        max_events: Optional limit on number of events to include

    Returns:
        (formatted_context, reference_map)
        - formatted_context: Context string showing all events with [E1], [E2] tags
        - reference_map: Dict mapping 'E1' → 'note_12345' (Lumia event ID)
    """
    # Extract all Event nodes from Lumia graph
    event_nodes = [
        (node_id, data)
        for node_id, data in lumia_graph.G.nodes(data=True)
        if data.get('node_type') == 'Event'
    ]

    # Sort by date
    event_nodes.sort(key=lambda x: x[1].get('date', '9999-99-99'))

    # Limit if specified
    if max_events:
        event_nodes = event_nodes[:max_events]

    # Create reference map
    reference_map = {}
    for i, (node_id, data) in enumerate(event_nodes, start=1):
        ref_id = f"E{i}"
        reference_map[ref_id] = node_id

    # Format context
    lines = []
    lines.append("=" * 80)
    lines.append("SOURCE EVENTS FROM EHR (cite these when extracting timeline)")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Total source events available: {len(event_nodes)}")
    lines.append("Use [E1], [E2], etc. to cite sources in 'source_event_refs' field.")
    lines.append("")

    # Group by date for readability
    from collections import defaultdict
    events_by_date = defaultdict(list)
    for i, (node_id, data) in enumerate(event_nodes, start=1):
        date = data.get('date', 'UNKNOWN')
        events_by_date[date].append((i, node_id, data))

    # Format each date group
    for date in sorted(events_by_date.keys()):
        lines.append(f"📅 {date}")

        for ref_num, node_id, data in events_by_date[date]:
            ref_id = f"E{ref_num}"

            # Get event type
            event_type = node_id.split('_')[0]  # e.g., 'note', 'lab', 'proc', 'img'
            note_id = data.get('note_id', data.get('proc_id', data.get('lab_id', '')))

            # Get text snippet from XMLFragment children
            text_snippet = _get_text_snippet(lumia_graph, node_id, max_snippet_chars)

            # Format line
            if text_snippet:
                lines.append(f"  [{ref_id}] {event_type} {note_id}: {text_snippet}")
            else:
                lines.append(f"  [{ref_id}] {event_type} {note_id}")

        lines.append("")

    return "\n".join(lines), reference_map


def _get_text_snippet(lumia_graph: TOAGraph, event_node_id: str, max_chars: int) -> str:
    """
    Get text snippet from XMLFragment children of event node.

    Args:
        lumia_graph: Lumia graph
        event_node_id: Event node ID
        max_chars: Max characters to return

    Returns:
        Text snippet (truncated to max_chars)
    """
    # Find XMLFragment children
    children = list(lumia_graph.G.successors(event_node_id))

    for child_id in children:
        child_data = lumia_graph.G.nodes[child_id]
        if child_data.get('node_type') == 'XMLFragment':
            # Get text from fragment
            text = child_data.get('text', child_data.get('xml_text', ''))
            if text:
                # Clean and truncate
                text = text.strip()
                if len(text) > max_chars:
                    text = text[:max_chars] + "..."
                return text

    return ""


def create_full_reference_map(lumia_graph: TOAGraph) -> Dict[str, str]:
    """
    Create reference map for ALL Lumia events (minimal version, no context formatting).

    Args:
        lumia_graph: Lumia hierarchical graph

    Returns:
        Dict mapping 'E1' → 'note_12345' for all Event nodes
    """
    event_nodes = [
        (node_id, data)
        for node_id, data in lumia_graph.G.nodes(data=True)
        if data.get('node_type') == 'Event'
    ]

    # Sort by date
    event_nodes.sort(key=lambda x: x[1].get('date', '9999-99-99'))

    # Create map
    reference_map = {}
    for i, (node_id, data) in enumerate(event_nodes, start=1):
        ref_id = f"E{i}"
        reference_map[ref_id] = node_id

    return reference_map
