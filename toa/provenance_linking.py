"""
Provenance Linking: Connect TOA Timeline Events to HGraph Base Events

This module handles the critical linking step where timeline events (TOA layer)
are connected back to their source events in the base hierarchical graph (HGraph).

Architecture:
    XML → HGraph (Patient → Visit → Event) → TOA (TimelineEvent + Episode) → UI
             ↑                                    ↓
             └────────── SOURCED_FROM edges ──────┘

Key Innovation:
    Instead of fuzzy date matching, LLMs reference HGraph events using short IDs
    (E1, E2, E3...) that are deterministically mapped to base event nodes.
    This reduces hallucination and creates direct graph edges for provenance.

Example:
    >>> from toa.graph import TOAGraph
    >>> from toa.format_deterministic_context import format_context_for_llm
    >>> from toa.provenance_linking import add_timeline_events_with_provenance
    >>>
    >>> # Build HGraph from XML
    >>> hgraph = build_hierarchical_graph(patient_id)
    >>>
    >>> # Generate context with short references
    >>> context_snippet, ref_map = format_context_for_llm(hgraph)
    >>>
    >>> # LLM extraction (context-guided)
    >>> timeline_objects = extract_timeline_with_llm(patient_id, context_snippet)
    >>>
    >>> # Add timeline events with provenance edges
    >>> toa_graph = add_timeline_events_with_provenance(hgraph, timeline_objects, ref_map)
    >>> toa_graph.save(f"graphs/{patient_id}.graphml")
"""

from typing import List, Dict, Any, Tuple
from toa.graph import TOAGraph
import logging

logger = logging.getLogger(__name__)


def add_timeline_events_with_provenance(
    hgraph: TOAGraph,
    timeline_objects: List[Dict[str, Any]],
    reference_map: Dict[str, str],
    validate: bool = True
) -> TOAGraph:
    """
    Add timeline event nodes to HGraph with direct provenance edges.

    This is the critical linking step that connects the TOA layer back to the
    HGraph base layer, enabling complete provenance traceability.

    Args:
        hgraph: Base hierarchical graph (Patient → Visit → Event nodes)
        timeline_objects: List of extracted timeline events from LLM
        reference_map: Mapping from short refs (E1, E2...) to base event_ids
        validate: Whether to validate provenance coverage (default: True)

    Returns:
        Enhanced graph with:
        - TimelineEvent nodes (enriched clinical events)
        - SOURCED_FROM edges: TimelineEvent → HGraph Event nodes
        - PRECEDES edges: TimelineEvent → TimelineEvent (temporal)

    Raises:
        ValueError: If validation fails and critical errors are found

    Example:
        >>> hgraph = build_hierarchical_graph("136108176")
        >>> context, ref_map = format_context_for_llm(hgraph)
        >>> timeline_events = extract_timeline_with_llm("136108176", context)
        >>> toa_graph = add_timeline_events_with_provenance(
        ...     hgraph, timeline_events, ref_map
        ... )
        >>> # toa_graph now has both HGraph and TOA layers
    """
    logger.info(f"Adding {len(timeline_objects)} timeline events to HGraph")
    logger.info(f"Reference map has {len(reference_map)} entries")

    # Track provenance statistics
    stats = {
        'total_events': len(timeline_objects),
        'events_with_refs': 0,
        'events_without_refs': 0,
        'total_provenance_edges': 0,
        'invalid_refs': 0,
        'missing_base_events': 0
    }

    # Add timeline event nodes
    for tl_event in timeline_objects:
        event_id = tl_event['event_id']

        # Create timeline event node
        hgraph.G.add_node(
            event_id,
            node_type='TimelineEvent',
            date=tl_event['date'],
            type=tl_event['type'],
            subtype=tl_event.get('subtype'),
            description=tl_event['short_description'],
            clinical_context=tl_event.get('clinical_context', ''),
            site=tl_event.get('site'),
            laterality=tl_event.get('laterality'),
            modality=tl_event.get('modality'),
            value=tl_event.get('value'),
            units=tl_event.get('units'),
            priority=tl_event.get('priority', 'medium'),
            notes=tl_event.get('notes', '')
        )

        # Add direct provenance edges to base HGraph events
        source_refs = tl_event.get('source_event_refs', [])

        if source_refs:
            stats['events_with_refs'] += 1

            for short_ref in source_refs:
                if short_ref not in reference_map:
                    logger.warning(
                        f"Timeline event {event_id}: Invalid reference '{short_ref}' "
                        f"not in reference map"
                    )
                    stats['invalid_refs'] += 1
                    continue

                base_event_id = reference_map[short_ref]

                # Verify base event exists in HGraph
                if not hgraph.G.has_node(base_event_id):
                    logger.warning(
                        f"Timeline event {event_id}: Referenced base event "
                        f"{base_event_id} not found in HGraph"
                    )
                    stats['missing_base_events'] += 1
                    continue

                # Add provenance edge: TimelineEvent → SOURCED_FROM → HGraph Event
                hgraph.G.add_edge(
                    event_id,
                    base_event_id,
                    edge_type='SOURCED_FROM',
                    reference=short_ref  # Store short ref for debugging
                )
                stats['total_provenance_edges'] += 1

                logger.debug(
                    f"Linked {event_id} → SOURCED_FROM → {base_event_id} "
                    f"(via {short_ref})"
                )
        else:
            stats['events_without_refs'] += 1
            logger.warning(
                f"Timeline event {event_id} has no source_event_refs field"
            )

    # Add temporal edges between timeline events
    sorted_events = sorted(
        [(e['event_id'], e['date']) for e in timeline_objects],
        key=lambda x: x[1]
    )

    for i in range(len(sorted_events) - 1):
        hgraph.G.add_edge(
            sorted_events[i][0],
            sorted_events[i+1][0],
            edge_type='PRECEDES'
        )

    # Log statistics
    logger.info("=" * 60)
    logger.info("PROVENANCE LINKING STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total timeline events: {stats['total_events']}")
    logger.info(f"Events with source refs: {stats['events_with_refs']} "
                f"({100*stats['events_with_refs']/stats['total_events']:.1f}%)")
    logger.info(f"Events without refs: {stats['events_without_refs']} "
                f"({100*stats['events_without_refs']/stats['total_events']:.1f}%)")
    logger.info(f"Total SOURCED_FROM edges added: {stats['total_provenance_edges']}")
    logger.info(f"Invalid references: {stats['invalid_refs']}")
    logger.info(f"Missing base events: {stats['missing_base_events']}")
    logger.info("=" * 60)

    # Validation
    if validate:
        _validate_provenance_coverage(hgraph, stats)

    return hgraph


def _validate_provenance_coverage(
    graph: TOAGraph,
    stats: Dict[str, int]
) -> None:
    """
    Validate provenance coverage and raise warnings/errors.

    Args:
        graph: Graph with timeline events and provenance edges
        stats: Statistics from provenance linking

    Raises:
        ValueError: If critical validation checks fail
    """
    total = stats['total_events']
    with_refs = stats['events_with_refs']

    # Check: At least 80% of events should have provenance
    coverage_rate = with_refs / total if total > 0 else 0

    if coverage_rate < 0.5:
        raise ValueError(
            f"CRITICAL: Only {coverage_rate*100:.1f}% of timeline events have "
            f"source references. Expected at least 50%."
        )
    elif coverage_rate < 0.8:
        logger.warning(
            f"LOW COVERAGE: Only {coverage_rate*100:.1f}% of timeline events "
            f"have source references. Target is 80%+."
        )
    else:
        logger.info(
            f"✓ Good provenance coverage: {coverage_rate*100:.1f}%"
        )

    # Check: Invalid references should be minimal
    invalid_rate = stats['invalid_refs'] / total if total > 0 else 0
    if invalid_rate > 0.05:
        logger.warning(
            f"HIGH INVALID RATE: {invalid_rate*100:.1f}% of references are "
            f"invalid (not in reference map)"
        )


def get_provenance_chain(
    graph: TOAGraph,
    timeline_event_id: str
) -> Dict[str, Any]:
    """
    Get complete provenance chain for a timeline event.

    Traverses SOURCED_FROM edges to retrieve all base HGraph events that
    contributed to this timeline event, along with their visit context.

    Args:
        graph: Complete graph with HGraph and TOA layers
        timeline_event_id: ID of timeline event to trace

    Returns:
        Dictionary containing:
        {
            'timeline_event': {...},  # TimelineEvent node attributes
            'source_events': [        # List of base HGraph events
                {
                    'event_id': 'evt_note_2024-01-18...',
                    'type': 'note',
                    'date': '2024-01-18',
                    'name': 'Progress Note',
                    'full_text': '...',
                    'visit': {
                        'name': 'Thoracic Medical Oncology Clinic',
                        'date': '2024-01-18'
                    }
                },
                ...
            ]
        }

    Example:
        >>> provenance = get_provenance_chain(graph, "tl_evt_001")
        >>> print(f"Timeline event: {provenance['timeline_event']['description']}")
        >>> print(f"Based on {len(provenance['source_events'])} source documents:")
        >>> for src in provenance['source_events']:
        ...     print(f"  - {src['type']} from {src['date']}: {src['name']}")
    """
    # Get timeline event node
    if not graph.G.has_node(timeline_event_id):
        raise ValueError(f"Timeline event {timeline_event_id} not found in graph")

    tl_event = dict(graph.G.nodes[timeline_event_id])
    tl_event['event_id'] = timeline_event_id

    # Traverse SOURCED_FROM edges to get base HGraph events
    source_events = []
    for u, v, edge_data in graph.G.out_edges(timeline_event_id, data=True):
        if edge_data.get('edge_type') == 'SOURCED_FROM':
            base_event = dict(graph.G.nodes[v])
            base_event['event_id'] = v

            # Get parent visit for context
            visit = _get_parent_visit(graph, v)

            source_events.append({
                'event_id': base_event['event_id'],
                'type': base_event.get('type', ''),
                'date': base_event.get('date', ''),
                'timestamp': base_event.get('timestamp', ''),
                'name': base_event.get('name', ''),
                'full_text': base_event.get('full_text', ''),
                'reference': edge_data.get('reference'),  # Short ref (E1, E2, etc.)
                'visit': {
                    'name': visit.get('name', ''),
                    'date': visit.get('start_date', visit.get('date', ''))
                } if visit else None
            })

    # Sort by date
    source_events.sort(key=lambda x: (x['date'], x.get('timestamp', '')))

    return {
        'timeline_event': tl_event,
        'source_events': source_events
    }


def _get_parent_visit(graph: TOAGraph, event_id: str) -> Dict[str, Any]:
    """Get parent Visit node for an Event node via VISIT_CONTAINS edges."""
    for u, v, edge_data in graph.G.in_edges(event_id, data=True):
        if edge_data.get('edge_type') == 'VISIT_CONTAINS':
            visit_data = dict(graph.G.nodes[u])
            visit_data['visit_id'] = u
            return visit_data
    return None


def audit_provenance_accuracy(
    graph: TOAGraph,
    timeline_event_id: str,
    verbose: bool = True
) -> None:
    """
    Audit provenance accuracy for manual review.

    Displays timeline event alongside source base events to enable
    human verification that the timeline event accurately represents
    the source material.

    Args:
        graph: Complete graph with HGraph and TOA layers
        timeline_event_id: ID of timeline event to audit
        verbose: Whether to print full text of source events

    Example:
        >>> audit_provenance_accuracy(graph, "tl_evt_001")

        Timeline Event: tl_evt_001
        Date: 2024-01-18
        Description: CT chest showing progression with new hepatic metastases

        Source Events (2):
          [E3] note (2024-01-18 14:30:00)
            Visit: Thoracic Medical Oncology Clinic
            Name: Progress Note
            Text: [First 200 chars of note...]

          [E4] image (2024-01-10)
            Visit: Radiology
            Name: CT Chest
            Text: [First 200 chars of report...]
    """
    provenance = get_provenance_chain(graph, timeline_event_id)

    print("=" * 70)
    print(f"PROVENANCE AUDIT: {timeline_event_id}")
    print("=" * 70)
    print()
    print("TIMELINE EVENT (TOA Layer):")
    print(f"  Date: {provenance['timeline_event']['date']}")
    print(f"  Type: {provenance['timeline_event']['type']}")
    print(f"  Description: {provenance['timeline_event']['description']}")
    if provenance['timeline_event'].get('clinical_context'):
        print(f"  Context: {provenance['timeline_event']['clinical_context']}")
    print()
    print(f"SOURCE EVENTS ({len(provenance['source_events'])} from HGraph):")
    print()

    for i, src in enumerate(provenance['source_events'], 1):
        print(f"  [{src['reference']}] {src['type']} ({src['date']} {src.get('timestamp', '')})")
        if src['visit']:
            print(f"    Visit: {src['visit']['name']}")
        print(f"    Name: {src['name']}")

        if verbose and src['full_text']:
            # Show first 300 chars
            text_preview = src['full_text'][:300]
            if len(src['full_text']) > 300:
                text_preview += "..."
            print(f"    Text: {text_preview}")
        elif src['full_text']:
            print(f"    Text: [{len(src['full_text'])} characters]")

        print()

    print("=" * 70)
    print()
    print("REVIEW QUESTION:")
    print("Does the timeline event description accurately reflect the source events?")
    print("=" * 70)


def validate_cohort_provenance(
    patient_ids: List[str],
    graphs_dir: str = "graphs"
) -> Dict[str, Any]:
    """
    Validate provenance coverage across a cohort.

    Args:
        patient_ids: List of patient IDs to validate
        graphs_dir: Directory containing graph files

    Returns:
        Summary statistics:
        {
            'total_patients': int,
            'total_timeline_events': int,
            'events_with_provenance': int,
            'coverage_rate': float,
            'by_patient': {patient_id: {...}, ...}
        }
    """
    from pathlib import Path

    cohort_stats = {
        'total_patients': len(patient_ids),
        'total_timeline_events': 0,
        'events_with_provenance': 0,
        'by_patient': {}
    }

    for patient_id in patient_ids:
        graph_path = Path(graphs_dir) / f"{patient_id}.graphml"

        if not graph_path.exists():
            logger.warning(f"Graph not found for patient {patient_id}")
            continue

        graph = TOAGraph.load(str(graph_path))

        # Count timeline events
        tl_events = [
            node_id for node_id, data in graph.G.nodes(data=True)
            if data.get('node_type') == 'TimelineEvent'
        ]

        # Count events with provenance
        events_with_prov = 0
        for tl_event_id in tl_events:
            # Check if has SOURCED_FROM edges
            has_provenance = any(
                edge_data.get('edge_type') == 'SOURCED_FROM'
                for u, v, edge_data in graph.G.out_edges(tl_event_id, data=True)
            )
            if has_provenance:
                events_with_prov += 1

        patient_stats = {
            'timeline_events': len(tl_events),
            'events_with_provenance': events_with_prov,
            'coverage_rate': events_with_prov / len(tl_events) if tl_events else 0
        }

        cohort_stats['by_patient'][patient_id] = patient_stats
        cohort_stats['total_timeline_events'] += len(tl_events)
        cohort_stats['events_with_provenance'] += events_with_prov

    cohort_stats['coverage_rate'] = (
        cohort_stats['events_with_provenance'] / cohort_stats['total_timeline_events']
        if cohort_stats['total_timeline_events'] > 0 else 0
    )

    return cohort_stats
