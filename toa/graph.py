"""
Timeline Object Architecture (TOA) as a Graph Database

This module implements TOA as an in-memory graph structure using NetworkX,
enabling powerful graph queries, temporal reasoning, and provenance tracking.

Graph Structure:
    Nodes:
        - Event: Timeline objects (imaging, treatment, diagnosis, etc.)
        - Episode: Clinical episodes (treatment lines, baseline, diagnosis)
        - Patient: Patient root node (optional, for multi-patient graphs)

    Edges:
        - PRECEDES: Temporal ordering between events
        - CONTAINS: Episode contains event
        - ANCHORED_BY: Episode anchored by imaging event
        - FOLLOWED_BY: Episode sequence
        - SOURCED_FROM: Event sourced from XML location (provenance)
        - SAME_DAY: Events on same date (co-occurrence)
        - Future: CAUSED, RESPONDED_WITH, DOSE_OF (causal relationships)

Example Usage:
    >>> graph = TOAGraph()
    >>> graph.load_patient("cohort1_136040534")
    >>>
    >>> # Query events in date range
    >>> events = graph.query_events_between_dates("2021-01-01", "2021-12-31")
    >>>
    >>> # Find temporal path
    >>> path = graph.find_temporal_path("evt_001", "evt_050")
    >>>
    >>> # Get episode context
    >>> events = graph.get_episode_events("ep_001")
    >>>
    >>> # Persist to file
    >>> graph.save("temp_jsons/cohort1_136040534/graph.graphml")
"""

from __future__ import annotations
import networkx as nx
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import hashlib


class TOAGraph:
    """
    Timeline Object Architecture as a directed multigraph.

    Represents clinical timelines as a graph where:
    - Nodes = Events + Episodes + Measurements + XMLFragments
    - Edges = Temporal relationships, episode membership, measurements, provenance

    Node Types:
    - Event: High-level clinical events (imaging, treatment, lab panel)
    - Episode: Clinical episodes (treatment lines)
    - Measurement: Individual lab values (WBC, hemoglobin, glucose, etc.)
    - XMLFragment: Source document provenance

    Attributes:
        G (nx.MultiDiGraph): NetworkX directed multigraph
        patient_id (str): Current patient ID (for single-patient graphs)
        provenance_mode (str): Provenance granularity ('date', 'note_id', 'timestamp', 'source_refs')
        include_measurements (bool): Whether to create individual measurement nodes
    """

    def __init__(self, provenance_mode: str = 'date', include_measurements: bool = True):
        """
        Initialize empty multigraph.

        Args:
            provenance_mode: Provenance granularity mode (default: 'date')
                - 'date': Day-level provenance (recommended, avoids hallucination)
                - 'note_id': Note ID level (future)
                - 'timestamp': Timestamp level (future)
                - 'source_refs': Legacy byte offset mode
            include_measurements: Whether to extract individual measurement nodes from lab events
                - True: Creates Measurement nodes (WBC, Hgb, etc.) linked to lab events
                - False: Only creates lab event nodes with aggregated description
        """
        self.G = nx.MultiDiGraph()
        self.patient_id = None
        self.provenance_mode = provenance_mode
        self.include_measurements = include_measurements

    # ========== Loading Methods ==========

    def load_patient(self, patient_id: str, base_dir: str = "temp_jsons") -> TOAGraph:
        """
        Load a patient's TOA artifacts into the graph.

        Reads timeline_objects.jsonl and episodes.json, creates nodes and edges.

        Args:
            patient_id: Patient identifier
            base_dir: Base directory containing patient subdirectories

        Returns:
            Self (for method chaining)

        Raises:
            FileNotFoundError: If required JSON files don't exist
        """
        self.patient_id = patient_id
        patient_dir = Path(base_dir) / patient_id

        # Load timeline objects
        timeline_path = patient_dir / "timeline_objects.jsonl"
        if not timeline_path.exists():
            raise FileNotFoundError(f"Timeline not found: {timeline_path}")

        timeline = self._load_jsonl(timeline_path)

        # Load episodes
        episodes_path = patient_dir / "episodes.json"
        if episodes_path.exists():
            episodes = self._load_json(episodes_path)
        else:
            episodes = []

        # Build graph
        self._build_graph(patient_id, timeline, episodes)

        return self

    def _build_graph(self, patient_id: str, timeline: List[Dict], episodes: List[Dict]):
        """
        Construct graph from timeline objects and episodes.

        Creates:
        1. Event nodes (from timeline objects)
        2. Episode nodes (from episodes)
        2.5. Measurement nodes (from lab event values) [if include_measurements=True]
        3. Temporal PRECEDES edges (between events)
        4. Episode CONTAINS edges (episode → events)
        5. Episode ANCHORED_BY edges (episode → imaging anchor)
        6. Episode FOLLOWED_BY edges (episode sequence)
        7. SAME_DAY edges (co-occurring events)
        8. SOURCED_FROM edges (event → XML provenance)
        """
        # 1. Add Event nodes
        for event in timeline:
            event_id = event.get('event_id')
            # Ensure patient_id is set
            event_data = dict(event)
            event_data['patient_id'] = patient_id
            event_data['node_type'] = 'Event'
            self.G.add_node(event_id, **event_data)

        # 2. Add Episode nodes
        for episode in episodes:
            episode_id = episode.get('episode_id')
            # Ensure patient_id is set
            episode_data = dict(episode)
            episode_data['patient_id'] = patient_id
            episode_data['node_type'] = 'Episode'
            self.G.add_node(episode_id, **episode_data)

        # 2.5. Add Measurement nodes (from lab events)
        if self.include_measurements:
            self._add_measurement_nodes(patient_id, timeline)

        # 3. Add temporal PRECEDES edges (ordered by date)
        self._add_temporal_edges(timeline)

        # 4. Add episode CONTAINS edges
        self._add_episode_contains_edges(episodes)

        # 5. Add episode ANCHORED_BY edges
        self._add_episode_anchor_edges(episodes)

        # 6. Add episode FOLLOWED_BY edges
        self._add_episode_sequence_edges(episodes)

        # 7. Add SAME_DAY edges (co-occurring events)
        self._add_same_day_edges(timeline)

        # 8. Add SOURCED_FROM edges (provenance)
        self._add_provenance_edges(timeline, mode=self.provenance_mode)

    def _add_temporal_edges(self, timeline: List[Dict]):
        """Create PRECEDES edges between temporally ordered events."""
        sorted_events = sorted(timeline, key=lambda e: e.get('valid_time_start', ''))

        for i in range(len(sorted_events) - 1):
            e1_id = sorted_events[i].get('event_id')
            e2_id = sorted_events[i+1].get('event_id')

            # Calculate days between events
            date1 = sorted_events[i].get('valid_time_start')
            date2 = sorted_events[i+1].get('valid_time_start')
            days = self._days_between(date1, date2)

            self.G.add_edge(
                e1_id,
                e2_id,
                edge_type='PRECEDES',
                days_between=days
            )

    def _add_episode_contains_edges(self, episodes: List[Dict]):
        """Create CONTAINS edges from episodes to their events."""
        for episode in episodes:
            ep_id = episode.get('episode_id')
            event_ids = episode.get('event_ids', [])

            for event_id in event_ids:
                if self.G.has_node(event_id):
                    self.G.add_edge(
                        ep_id,
                        event_id,
                        edge_type='CONTAINS'
                    )

    def _add_episode_anchor_edges(self, episodes: List[Dict]):
        """Create ANCHORED_BY edges from episodes to their imaging anchors."""
        for episode in episodes:
            ep_id = episode.get('episode_id')

            # Handle both single anchor and multiple anchors
            anchor_ids = episode.get('anchor_event_ids', [])
            if not anchor_ids:
                # Fallback to single anchor_event_id
                anchor_id = episode.get('anchor_event_id')
                if anchor_id:
                    anchor_ids = [anchor_id]

            for anchor_id in anchor_ids:
                if self.G.has_node(anchor_id):
                    self.G.add_edge(
                        ep_id,
                        anchor_id,
                        edge_type='ANCHORED_BY'
                    )

    def _add_episode_sequence_edges(self, episodes: List[Dict]):
        """Create FOLLOWED_BY edges between sequential episodes."""
        sorted_episodes = sorted(episodes, key=lambda ep: ep.get('start_date', ''))

        for i in range(len(sorted_episodes) - 1):
            ep1_id = sorted_episodes[i].get('episode_id')
            ep2_id = sorted_episodes[i+1].get('episode_id')

            self.G.add_edge(
                ep1_id,
                ep2_id,
                edge_type='FOLLOWED_BY'
            )

    def _add_same_day_edges(self, timeline: List[Dict]):
        """Create SAME_DAY edges between events on the same date."""
        # Group events by date
        by_date: Dict[str, List[str]] = {}
        for event in timeline:
            date = event.get('valid_time_start', '')
            event_id = event.get('event_id')
            if date:
                by_date.setdefault(date, []).append(event_id)

        # Create edges between same-day events
        for date, event_ids in by_date.items():
            if len(event_ids) > 1:
                # Create edges between all pairs (fully connected subgraph)
                for i in range(len(event_ids)):
                    for j in range(i + 1, len(event_ids)):
                        self.G.add_edge(
                            event_ids[i],
                            event_ids[j],
                            edge_type='SAME_DAY',
                            date=date
                        )
                        # Add reverse edge (undirected relationship)
                        self.G.add_edge(
                            event_ids[j],
                            event_ids[i],
                            edge_type='SAME_DAY',
                            date=date
                        )

    def _add_provenance_edges(self, timeline: List[Dict], mode: str = 'date'):
        """
        Create SOURCED_FROM edges with configurable granularity.

        Modes:
        - 'date': Day-level provenance using evidence_date (default, avoids hallucination)
        - 'note_id': Note ID level provenance (future)
        - 'timestamp': Precise timestamp level provenance (future)
        - 'source_refs': Legacy mode using source_refs field (for backward compatibility)

        Args:
            timeline: List of timeline events
            mode: Provenance granularity mode
        """
        if mode == 'date':
            self._add_date_level_provenance(timeline)
        elif mode == 'note_id':
            self._add_note_id_provenance(timeline)
        elif mode == 'timestamp':
            self._add_timestamp_provenance(timeline)
        elif mode == 'source_refs':
            self._add_source_refs_provenance(timeline)
        else:
            raise ValueError(f"Unknown provenance mode: {mode}")

    def _add_date_level_provenance(self, timeline: List[Dict]):
        """
        Day-level provenance: Link events to all XML entries from evidence_date.

        Advantages:
        - Simple, reliable (no LLM hallucination)
        - Clinically meaningful (matches chart review workflow)
        - Easy to query: "show me all XML from 2022-06-24"
        """
        for event in timeline:
            event_id = event.get('event_id')
            evidence_date = event.get('evidence_date')

            if evidence_date:
                # Create day-level XML fragment ID
                fragment_id = f"xml_date_{evidence_date}"

                # Add XML fragment node if not exists
                if not self.G.has_node(fragment_id):
                    self.G.add_node(
                        fragment_id,
                        node_type='XMLFragment',
                        evidence_date=evidence_date,
                        provenance_type='date_level'
                    )

                # Add SOURCED_FROM edge
                self.G.add_edge(
                    event_id,
                    fragment_id,
                    edge_type='SOURCED_FROM',
                    evidence_date=evidence_date
                )

    def _add_note_id_provenance(self, timeline: List[Dict]):
        """
        Note ID level provenance: Link events to specific note IDs.

        Future implementation: Extract note_id from event metadata.
        """
        for event in timeline:
            event_id = event.get('event_id')
            note_id = event.get('note_id')  # Future field

            if note_id:
                fragment_id = f"xml_note_{note_id}"

                if not self.G.has_node(fragment_id):
                    self.G.add_node(
                        fragment_id,
                        node_type='XMLFragment',
                        note_id=note_id,
                        provenance_type='note_id_level'
                    )

                self.G.add_edge(
                    event_id,
                    fragment_id,
                    edge_type='SOURCED_FROM',
                    note_id=note_id
                )

    def _add_timestamp_provenance(self, timeline: List[Dict]):
        """
        Timestamp level provenance: Link events to precise timestamps.

        Future implementation: Use detailed timestamp from event metadata.
        """
        for event in timeline:
            event_id = event.get('event_id')
            timestamp = event.get('timestamp')  # Future field (ISO8601 with time)

            if timestamp:
                fragment_id = f"xml_timestamp_{timestamp}"

                if not self.G.has_node(fragment_id):
                    self.G.add_node(
                        fragment_id,
                        node_type='XMLFragment',
                        timestamp=timestamp,
                        provenance_type='timestamp_level'
                    )

                self.G.add_edge(
                    event_id,
                    fragment_id,
                    edge_type='SOURCED_FROM',
                    timestamp=timestamp
                )

    def _add_source_refs_provenance(self, timeline: List[Dict]):
        """
        Legacy mode: Use source_refs field with byte offsets.

        For backward compatibility with existing source_refs format.
        """
        for event in timeline:
            event_id = event.get('event_id')
            source_refs = event.get('source_refs', [])

            for ref in source_refs:
                doc_id = ref.get('doc_id', 'unknown')
                offset = ref.get('offset', [0, 0])

                fragment_id = f"xml_{doc_id}_{offset[0]}_{offset[1]}"

                if not self.G.has_node(fragment_id):
                    self.G.add_node(
                        fragment_id,
                        node_type='XMLFragment',
                        doc_id=doc_id,
                        offset_start=offset[0],
                        offset_end=offset[1],
                        provenance_type='byte_offset'
                    )

                self.G.add_edge(
                    event_id,
                    fragment_id,
                    edge_type='SOURCED_FROM'
                )

    def _add_measurement_nodes(self, patient_id: str, timeline: List[Dict]):
        """
        Extract individual measurement nodes from lab events.

        For each lab event with a 'values' field, creates individual Measurement nodes
        for each lab value (e.g., WBC, hemoglobin, glucose) and links them via
        HAS_MEASUREMENT edges.

        This enables queries like:
        - "Get all WBC measurements in this episode"
        - "What was the lowest hemoglobin during carboplatin?"

        Args:
            patient_id: Patient identifier
            timeline: List of timeline events
        """
        import hashlib

        for event in timeline:
            if event.get('type') != 'lab':
                continue

            event_id = event.get('event_id')
            values = event.get('values', {})
            date = event.get('valid_time_start')
            evidence_date = event.get('evidence_date')

            if not values:
                continue

            # Create individual measurement nodes
            for lab_name, lab_value in values.items():
                # Parse value (e.g., "6.6↑" → value=6.6, direction="↑")
                value_parsed, direction = self._parse_lab_value(lab_value)

                # Create unique measurement ID
                measurement_id = hashlib.md5(
                    f"{event_id}_{lab_name}".encode()
                ).hexdigest()[:16]
                measurement_id = f"measure_{measurement_id}"

                # Add Measurement node
                self.G.add_node(
                    measurement_id,
                    node_type='Measurement',
                    patient_id=patient_id,
                    lab_name=lab_name,
                    value=value_parsed,
                    value_string=lab_value,
                    direction=direction,
                    date=date,
                    evidence_date=evidence_date,
                    parent_event_id=event_id
                )

                # Add HAS_MEASUREMENT edge
                self.G.add_edge(
                    event_id,
                    measurement_id,
                    edge_type='HAS_MEASUREMENT',
                    lab_name=lab_name
                )

    def _parse_lab_value(self, value_str: str) -> tuple:
        """
        Parse lab value string into numeric value and direction.

        Examples:
            "6.6↑" → (6.6, "high")
            "38↓" → (38.0, "low")
            "194" → (194.0, "normal")

        Returns:
            (value, direction) where direction is "high", "low", or "normal"
        """
        import re

        if not isinstance(value_str, str):
            return (None, "normal")

        # Remove arrows and extract number
        clean_value = value_str.replace('↑', '').replace('↓', '').strip()

        # Try to parse numeric value
        try:
            value = float(clean_value)
        except ValueError:
            value = None

        # Determine direction
        if '↑' in value_str:
            direction = "high"
        elif '↓' in value_str:
            direction = "low"
        else:
            direction = "normal"

        return (value, direction)

    # ========== Query Methods ==========

    def query_events_between_dates(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Get all events within a date range.

        Args:
            start_date: Start date (ISO format YYYY-MM-DD)
            end_date: End date (ISO format YYYY-MM-DD)

        Returns:
            List of event dictionaries

        Example:
            >>> events = graph.query_events_between_dates("2021-01-01", "2021-06-30")
        """
        events = []
        for node, data in self.G.nodes(data=True):
            if data.get('node_type') == 'Event':
                event_date = data.get('valid_time_start', '')
                if start_date <= event_date <= end_date:
                    events.append(data)

        return sorted(events, key=lambda e: e.get('valid_time_start', ''))

    def get_episode_events(self, episode_id: str) -> List[Dict[str, Any]]:
        """
        Get all events contained in an episode.

        Args:
            episode_id: Episode identifier

        Returns:
            List of event dictionaries
        """
        events = []
        for successor in self.G.successors(episode_id):
            edge_data = self.G.get_edge_data(episode_id, successor)
            if any(e.get('edge_type') == 'CONTAINS' for e in edge_data.values()):
                event_data = self.G.nodes[successor]
                if event_data.get('node_type') == 'Event':
                    events.append(event_data)

        return sorted(events, key=lambda e: e.get('valid_time_start', ''))

    def find_temporal_path(self, event1_id: str, event2_id: str) -> Optional[List[str]]:
        """
        Find shortest temporal path between two events.

        Args:
            event1_id: Source event ID
            event2_id: Target event ID

        Returns:
            List of event IDs forming the path, or None if no path exists

        Example:
            >>> path = graph.find_temporal_path("evt_001", "evt_050")
            >>> # Returns: ["evt_001", "evt_015", "evt_032", "evt_050"]
        """
        try:
            path = nx.shortest_path(self.G, event1_id, event2_id)
            return path
        except nx.NetworkXNoPath:
            return None

    def find_events_near_anchor(self, anchor_id: str, days: int = 90) -> List[Dict[str, Any]]:
        """
        Find events within N days of an anchor event (imaging-centered context).

        Args:
            anchor_id: Anchor event ID (typically imaging)
            days: Number of days before/after anchor (default: 90)

        Returns:
            List of event dictionaries within temporal window
        """
        if not self.G.has_node(anchor_id):
            return []

        anchor_date = self.G.nodes[anchor_id].get('valid_time_start', '')
        if not anchor_date:
            return []

        anchor_dt = datetime.fromisoformat(anchor_date)
        nearby_events = []

        for node, data in self.G.nodes(data=True):
            if data.get('node_type') == 'Event':
                event_date = data.get('valid_time_start', '')
                if event_date:
                    event_dt = datetime.fromisoformat(event_date)
                    if abs((event_dt - anchor_dt).days) <= days:
                        nearby_events.append(data)

        return sorted(nearby_events, key=lambda e: e.get('valid_time_start', ''))

    def get_same_day_events(self, event_id: str) -> List[Dict[str, Any]]:
        """
        Get all events that occurred on the same day as the given event.

        Args:
            event_id: Event identifier

        Returns:
            List of co-occurring event dictionaries
        """
        same_day = []
        for successor in self.G.successors(event_id):
            edge_data = self.G.get_edge_data(event_id, successor)
            if any(e.get('edge_type') == 'SAME_DAY' for e in edge_data.values()):
                event_data = self.G.nodes[successor]
                same_day.append(event_data)

        return same_day

    def get_event_provenance(self, event_id: str) -> List[Dict[str, Any]]:
        """
        Get XML source locations (provenance) for an event.

        Enables fast fact-checking by tracing back to original XML.
        Supports multiple provenance granularities (date, note_id, timestamp, byte-offset).

        Args:
            event_id: Event identifier

        Returns:
            List of XML fragment metadata with fields depending on provenance mode:
            - date mode: {evidence_date, provenance_type}
            - note_id mode: {note_id, provenance_type}
            - timestamp mode: {timestamp, provenance_type}
            - source_refs mode: {doc_id, offset_start, offset_end, provenance_type}

        Example:
            >>> # Day-level provenance
            >>> provenance = graph.get_event_provenance("evt_001")
            >>> # [{"evidence_date": "2022-06-24", "provenance_type": "date_level"}]

            >>> # Byte-offset provenance (legacy)
            >>> # [{"doc_id": "rad_report_123", "offset_start": 450, "offset_end": 892}]
        """
        fragments = []
        for successor in self.G.successors(event_id):
            edge_data = self.G.get_edge_data(event_id, successor)
            if any(e.get('edge_type') == 'SOURCED_FROM' for e in edge_data.values()):
                fragment_data = dict(self.G.nodes[successor])
                if fragment_data.get('node_type') == 'XMLFragment':
                    # Remove node_type from output (internal field)
                    fragment_data.pop('node_type', None)
                    fragments.append(fragment_data)

        return fragments

    def get_episode_sequence(self) -> List[Dict[str, Any]]:
        """
        Get all episodes in temporal sequence.

        Returns:
            List of episode dictionaries ordered by start_date
        """
        episodes = []
        for node, data in self.G.nodes(data=True):
            if data.get('node_type') == 'Episode':
                episodes.append(data)

        return sorted(episodes, key=lambda ep: ep.get('start_date', ''))

    def get_imaging_anchors(self) -> List[Dict[str, Any]]:
        """
        Get all imaging events that serve as episode anchors.

        Returns:
            List of imaging event dictionaries
        """
        anchors = []
        for node, data in self.G.nodes(data=True):
            if data.get('node_type') == 'Event' and data.get('type') == 'imaging':
                # Check if this event is an anchor for any episode
                for predecessor in self.G.predecessors(node):
                    edge_data = self.G.get_edge_data(predecessor, node)
                    if any(e.get('edge_type') == 'ANCHORED_BY' for e in edge_data.values()):
                        anchors.append(data)
                        break

        return sorted(anchors, key=lambda e: e.get('valid_time_start', ''))

    # ========== Graph Statistics ==========

    # ========== Measurement Queries ==========

    def get_measurements_in_episode(self, episode_id: str, lab_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all measurements (or specific lab) from events in an episode.

        This enables queries like:
        - "What was the lowest WBC during carboplatin treatment?"
        - "Show me all hemoglobin values in first line therapy"

        Args:
            episode_id: Episode identifier
            lab_name: Optional lab name filter (e.g., "WBC", "Hgb")

        Returns:
            List of measurement dictionaries with values sorted by date

        Example:
            >>> # Get all WBC measurements in episode 1
            >>> wbc_measurements = graph.get_measurements_in_episode("ep_001", "WBC")
            >>> for m in wbc_measurements:
            ...     print(f"{m['date']}: WBC {m['value']} {m['direction']}")
        """
        # Get all events in episode
        events = self.get_episode_events(episode_id)
        event_ids = [e.get('event_id') for e in events]

        measurements = []
        for event_id in event_ids:
            # Get measurements from this event
            for successor in self.G.successors(event_id):
                edge_data = self.G.get_edge_data(event_id, successor)
                if any(e.get('edge_type') == 'HAS_MEASUREMENT' for e in edge_data.values()):
                    measurement_data = dict(self.G.nodes[successor])
                    if measurement_data.get('node_type') == 'Measurement':
                        # Filter by lab name if specified
                        if lab_name is None or measurement_data.get('lab_name') == lab_name:
                            measurements.append(measurement_data)

        # Sort by date
        return sorted(measurements, key=lambda m: m.get('date', ''))

    def get_measurement_summary(self, lab_name: str, episode_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistical summary for a specific lab measurement.

        Args:
            lab_name: Lab name (e.g., "WBC", "Hgb", "Platelets")
            episode_id: Optional episode filter

        Returns:
            Dictionary with min, max, mean, count, dates of extremes

        Example:
            >>> # What was the lowest WBC during carboplatin?
            >>> summary = graph.get_measurement_summary("WBC", "ep_carboplatin")
            >>> print(f"Lowest: {summary['min']} on {summary['min_date']}")
            >>> print(f"Highest: {summary['max']} on {summary['max_date']}")
        """
        if episode_id:
            measurements = self.get_measurements_in_episode(episode_id, lab_name)
        else:
            # Get all measurements for this lab across timeline
            measurements = []
            for node, data in self.G.nodes(data=True):
                if data.get('node_type') == 'Measurement' and data.get('lab_name') == lab_name:
                    measurements.append(data)

        if not measurements:
            return {'count': 0, 'lab_name': lab_name}

        # Extract numeric values
        values = [m.get('value') for m in measurements if m.get('value') is not None]

        if not values:
            return {'count': len(measurements), 'lab_name': lab_name, 'error': 'No numeric values'}

        # Find extremes
        min_val = min(values)
        max_val = max(values)
        mean_val = sum(values) / len(values)

        # Find dates of extremes
        min_measurement = [m for m in measurements if m.get('value') == min_val][0]
        max_measurement = [m for m in measurements if m.get('value') == max_val][0]

        return {
            'lab_name': lab_name,
            'count': len(measurements),
            'min': min_val,
            'max': max_val,
            'mean': round(mean_val, 2),
            'min_date': min_measurement.get('date'),
            'max_date': max_measurement.get('date'),
            'values': values  # All values for plotting
        }

    def get_all_measurements(self, event_id: str) -> List[Dict[str, Any]]:
        """
        Get all individual measurements from a lab event.

        Args:
            event_id: Lab event identifier

        Returns:
            List of measurement dictionaries

        Example:
            >>> measurements = graph.get_all_measurements("lab_evt_001")
            >>> # [{"lab_name": "WBC", "value": 4.5, ...}, {"lab_name": "Hgb", ...}]
        """
        measurements = []
        for successor in self.G.successors(event_id):
            edge_data = self.G.get_edge_data(event_id, successor)
            if any(e.get('edge_type') == 'HAS_MEASUREMENT' for e in edge_data.values()):
                measurement_data = dict(self.G.nodes[successor])
                if measurement_data.get('node_type') == 'Measurement':
                    measurements.append(measurement_data)
        return measurements

    def stats(self) -> Dict[str, Any]:
        """
        Get graph statistics.

        Returns:
            Dictionary with node counts, edge counts, and graph metrics
        """
        event_count = sum(1 for _, d in self.G.nodes(data=True) if d.get('node_type') == 'Event')
        episode_count = sum(1 for _, d in self.G.nodes(data=True) if d.get('node_type') == 'Episode')
        xml_fragment_count = sum(1 for _, d in self.G.nodes(data=True) if d.get('node_type') == 'XMLFragment')
        measurement_count = sum(1 for _, d in self.G.nodes(data=True) if d.get('node_type') == 'Measurement')

        edge_types = {}
        for _, _, data in self.G.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        return {
            'patient_id': self.patient_id,
            'total_nodes': self.G.number_of_nodes(),
            'total_edges': self.G.number_of_edges(),
            'event_nodes': event_count,
            'episode_nodes': episode_count,
            'xml_fragment_nodes': xml_fragment_count,
            'measurement_nodes': measurement_count,
            'edge_types': edge_types,
            'graph_density': nx.density(self.G),
        }

    # ========== Persistence ==========

    def save(self, path: str):
        """
        Save graph to GraphML format.

        GraphML only supports primitive types (str, int, float, bool).
        Complex types (dict, list) are serialized to JSON strings.

        Args:
            path: Output file path (e.g., "temp_jsons/patient_id/graph.graphml")
        """
        # GraphML doesn't support None/dict/list values - clean them
        G_clean = self.G.copy()

        def clean_value(v):
            """Convert value to GraphML-compatible type."""
            if v is None:
                return None  # Will be filtered out
            elif isinstance(v, (str, int, float, bool)):
                return v
            elif isinstance(v, (dict, list)):
                # Serialize to JSON string
                return json.dumps(v)
            else:
                return str(v)

        # Clean node attributes
        for node, data in G_clean.nodes(data=True):
            cleaned_data = {k: clean_value(v) for k, v in data.items() if v is not None}
            G_clean.nodes[node].clear()
            G_clean.nodes[node].update(cleaned_data)

        # Clean edge attributes
        for u, v, key, data in G_clean.edges(data=True, keys=True):
            cleaned_data = {k: clean_value(v) for k, v in data.items() if v is not None}
            G_clean.edges[u, v, key].clear()
            G_clean.edges[u, v, key].update(cleaned_data)

        nx.write_graphml(G_clean, path)

    @classmethod
    def load(cls, path: str) -> TOAGraph:
        """
        Load graph from GraphML format.

        Args:
            path: Input file path

        Returns:
            TOAGraph instance
        """
        instance = cls()
        instance.G = nx.read_graphml(path)
        return instance

    # ========== Visualization (Optional) ==========

    def export_for_visualization(self, path: str):
        """
        Export graph to JSON format for visualization (e.g., D3.js, Cytoscape).

        Args:
            path: Output JSON file path
        """
        # Convert to node-link format
        data = nx.node_link_data(self.G)

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def export_timeline_context(self) -> Dict[str, Any]:
        """
        Export graph as standard timeline context for LLM.

        Returns the same structure as timeline.json + episodes.json,
        but sourced from graph instead of JSON files.

        This enables using the graph as the source of truth while maintaining
        the same LLM context structure that achieves 95% accuracy.

        Returns:
            Dictionary with:
            - events: List of timeline events
            - episodes: List of clinical episodes
            - summary: Basic statistics

        Example:
            >>> graph = load_patient_graph("patient_001")
            >>> context = graph.export_timeline_context()
            >>> # Feed to LLM (same as loading timeline.json + episodes.json)
            >>> llm.ask(query, context=json.dumps(context))
        """
        # Get all events (sorted by date)
        all_events = self.query_events_between_dates("1900-01-01", "2100-12-31")

        # Get all episodes
        episodes = self.get_episode_sequence()

        # Export in standard TOA format
        return {
            "events": [
                {
                    "date": e.get('date'),
                    "type": e.get('type'),
                    "subtype": e.get('subtype'),
                    "priority": e.get('priority'),
                    "description": e.get('description'),
                    "modality": e.get('modality'),
                    "site": e.get('site'),
                    "event_id": e.get('event_id'),
                    # Include values field for labs (if exists)
                    **({'values': e.get('values')} if e.get('values') else {})
                }
                for e in all_events
            ],
            "episodes": [
                {
                    "episode_id": ep.get('episode_id'),
                    "label": ep.get('label'),
                    "line_number": ep.get('line_number'),
                    "start_date": ep.get('start_date'),
                    "end_date": ep.get('end_date'),
                    "treatment_summary": ep.get('treatment_summary'),
                    "imaging_anchor_description": ep.get('imaging_anchor_description'),
                    "event_count": len(self.get_episode_events(ep['episode_id']))
                }
                for ep in episodes
            ],
            "summary": {
                "total_events": len(all_events),
                "total_episodes": len(episodes),
                "date_range": {
                    "start": all_events[0].get('date') if all_events else None,
                    "end": all_events[-1].get('date') if all_events else None
                },
                "event_types": self._count_by_field(all_events, 'type')
            }
        }

    def _count_by_field(self, items: List[Dict], field: str) -> Dict[str, int]:
        """Count items by field value."""
        counts = {}
        for item in items:
            value = item.get(field, 'unknown')
            counts[value] = counts.get(value, 0) + 1
        return counts

    # ========== Helper Methods ==========

    def _load_jsonl(self, path: Path) -> List[Dict]:
        """Load JSONL file (one JSON object per line)."""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def _load_json(self, path: Path) -> Any:
        """Load JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _days_between(self, date1: str, date2: str) -> Optional[int]:
        """Calculate days between two ISO dates."""
        if not date1 or not date2:
            return None
        try:
            d1 = datetime.fromisoformat(date1)
            d2 = datetime.fromisoformat(date2)
            return (d2 - d1).days
        except ValueError:
            return None


# ========== Convenience Functions ==========

def load_patient_graph(
    patient_id: str,
    base_dir: str = "temp_jsons",
    provenance_mode: str = 'date',
    include_measurements: bool = True
) -> TOAGraph:
    """
    Convenience function to load a patient's TOA graph.

    Args:
        patient_id: Patient identifier
        base_dir: Base directory containing patient subdirectories
        provenance_mode: Provenance granularity ('date', 'note_id', 'timestamp', 'source_refs')
        include_measurements: Whether to create individual measurement nodes from lab events

    Returns:
        TOAGraph instance

    Example:
        >>> from toa.graph import load_patient_graph
        >>> graph = load_patient_graph("cohort1_136040534")
        >>> stats = graph.stats()

        >>> # Use different provenance mode
        >>> graph = load_patient_graph("cohort1_136040534", provenance_mode='note_id')

        >>> # Disable measurement nodes
        >>> graph = load_patient_graph("cohort1_136040534", include_measurements=False)
    """
    graph = TOAGraph(provenance_mode=provenance_mode, include_measurements=include_measurements)
    graph.load_patient(patient_id, base_dir)
    return graph
