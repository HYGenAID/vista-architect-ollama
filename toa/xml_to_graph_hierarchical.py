"""
Hierarchical XML-to-Graph Parser

Preserves full XML event hierarchy with structured provenance:
- Encounter-level demographics, caresites, providers
- Visit-level grouping via visit_id
- Structured event types: note, procedure, condition, measurement, observation, drug_exposure, image
- Image ↔ Report linkage (radiology coupling)
- Granular XMLFragment nodes (note_id, procedure_id, image_id)

Key Features:
- VISIT_GROUPED edges: Link events by shared visit_id
- IMAGE_REPORT_FOR edges: Link radiology reports to images
- PROCEDURE_FOR edges: Link images to procedures
- Enhanced XMLFragment provenance (note_id, procedure_id, image_id level)

Example:
    >>> from toa.xml_to_graph_hierarchical import build_hierarchical_graph
    >>> graph = build_hierarchical_graph("test_cohort_2_136035634")
    >>>
    >>> # Query radiology report for an image
    >>> image_node = graph.get_node("img_12345")
    >>> report = graph.get_linked_radiology_report(image_node)
    >>>
    >>> # Query all events in a visit
    >>> visit_events = graph.get_events_by_visit_id("visit_26284334")
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import hashlib

from toa.graph import TOAGraph


def build_hierarchical_graph(
    patient_id: str,
    xml_source: Optional[str] = None,
    base_dir: str = "patient_records/thoracic_xmls",
    namespace_ids: bool = False
) -> TOAGraph:
    """
    Build hierarchical graph from patient XML preserving full structure.

    Args:
        patient_id: Patient identifier
        xml_source: Either path to XML file OR raw XML text content
                   (defaults to base_dir/patient_id.xml)
        base_dir: Base directory for XML files
        namespace_ids: If True, prefix all node IDs with {patient_id}_ to prevent
                      collisions when merging multiple patient graphs.

    Returns:
        TOAGraph with hierarchical structure:
        - Event nodes (typed: note, procedure, condition, measurement, observation, drug_exposure, image, visit)
        - XMLFragment nodes (note_id, procedure_id, image_id level)
        - VISIT_GROUPED edges (events with same visit_id)
        - IMAGE_REPORT_FOR edges (radiology reports → images)
        - PROCEDURE_FOR edges (images → procedures)
        - PRECEDES edges (temporal ordering)
        - SOURCED_FROM edges (granular provenance)
    """
    # Determine if xml_source is a file path or XML content
    if xml_source is None:
        # Default: load from file
        xml_path = Path(base_dir) / f"{patient_id}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"XML not found: {xml_path}")
        tree = ET.parse(xml_path)
        root = tree.getroot()
    elif xml_source.strip().startswith('<'):
        # Raw XML content
        root = ET.fromstring(xml_source)
    else:
        # File path
        xml_path = Path(xml_source)
        if not xml_path.exists():
            raise FileNotFoundError(f"XML not found: {xml_path}")
        tree = ET.parse(xml_path)
        root = tree.getroot()

    # Extract structured events
    events = parse_hierarchical_events(root, patient_id)

    # Namespace node IDs if requested (for multi-patient graph merging)
    if namespace_ids:
        _namespace_event_ids(events, patient_id)

    # Build graph with full structure
    graph = TOAGraph()
    graph.patient_id = patient_id
    _build_hierarchical_graph(graph, events, patient_id)

    return graph


def _namespace_event_ids(data: Dict[str, Any], patient_id: str):
    """Prefix all event IDs and cross-references with patient_id_ for merge safety."""
    prefix = f"{patient_id}_"

    # Namespace event IDs (events in data['events'] are the SAME objects as in
    # data['notes']/data['images']/etc., so we only need to modify them once)
    for event in data['events']:
        event['event_id'] = prefix + event['event_id']

    # Namespace visit index (event_ids here are strings, not references to event dicts,
    # so they need separate updating)
    new_visits = {}
    for visit_id, event_ids in data['visits'].items():
        # The event_ids in the visits dict are plain strings captured at parse time,
        # NOT references to the event dicts. Since we already modified the event dicts
        # above, we need to reconstruct these from the now-updated events.
        new_visits[visit_id] = [prefix + eid for eid in event_ids]
    data['visits'] = new_visits


def parse_hierarchical_events(root: ET.Element, patient_id: str) -> Dict[str, Any]:
    """
    Parse XML with full hierarchical structure preservation.

    Returns:
        Dictionary containing:
        - events: List of all events with structured metadata
        - visits: Dictionary mapping visit_id → list of event_ids
        - images: Dictionary mapping image_id → image event metadata
        - procedures: Dictionary mapping procedure_id → procedure event metadata
        - notes: Dictionary mapping note_id → note event metadata
        - demographics: Patient demographics from first encounter
    """
    events = []
    visits = defaultdict(list)  # visit_id → [event_ids]
    images = {}  # image_id → event
    procedures = {}  # procedure_id → event
    notes = {}  # note_id → event
    demographics = None

    for encounter in root.findall('.//encounter'):
        # Extract demographics (first encounter)
        if demographics is None:
            demographics = _extract_demographics(encounter)

        # Extract caresites and providers
        caresites = _extract_caresites(encounter)
        providers = _extract_providers(encounter)

        # Parse events in this encounter
        for entry in encounter.findall('.//entry'):
            timestamp = entry.get('timestamp')

            for event_elem in entry.findall('event'):
                event = _parse_hierarchical_event(
                    event_elem,
                    timestamp,
                    patient_id,
                    caresites,
                    providers
                )

                if event:
                    events.append(event)
                    event_id = event['event_id']

                    # Index by visit_id
                    visit_id = event.get('visit_id')
                    if visit_id:
                        visits[visit_id].append(event_id)

                    # Index by type-specific IDs
                    if event['type'] == 'image':
                        image_id = event.get('image_id')
                        if image_id:
                            images[image_id] = event

                    elif event['type'] == 'procedure':
                        procedure_id = event.get('procedure_id')
                        if procedure_id:
                            procedures[procedure_id] = event

                    elif event['type'] == 'note':
                        note_id = event.get('note_id')
                        if note_id:
                            notes[note_id] = event

    return {
        'events': events,
        'visits': visits,
        'images': images,
        'procedures': procedures,
        'notes': notes,
        'demographics': demographics
    }


def _extract_demographics(encounter: ET.Element) -> Dict[str, Any]:
    """Extract patient demographics from encounter."""
    person = encounter.find('person')
    if person is None:
        return {}

    demographics = {}

    # Birthdate
    birthdate = person.find('birthdate')
    if birthdate is not None:
        demographics['birthdate'] = birthdate.text

    # Age
    age_elem = person.find('.//age/years')
    if age_elem is not None:
        demographics['age_years'] = age_elem.text

    # Demographics block
    demo_elem = person.find('demographics')
    if demo_elem is not None:
        ethnicity = demo_elem.find('ethnicity')
        if ethnicity is not None:
            demographics['ethnicity'] = ethnicity.text

        gender = demo_elem.find('gender')
        if gender is not None:
            demographics['gender'] = gender.text

        race = demo_elem.find('race')
        if race is not None:
            demographics['race'] = race.text

    return demographics


def _extract_caresites(encounter: ET.Element) -> List[Dict[str, str]]:
    """Extract caresites from encounter."""
    caresites = []
    caresites_elem = encounter.find('caresites')

    if caresites_elem is not None:
        for caresite in caresites_elem.findall('caresite'):
            caresites.append({
                'care_site_id': caresite.get('care_site_id', ''),
                'care_site_name': caresite.get('care_site_name', '')
            })

    return caresites


def _extract_providers(encounter: ET.Element) -> List[Dict[str, str]]:
    """Extract providers from encounter."""
    providers = []
    providers_elem = encounter.find('providers')

    if providers_elem is not None:
        for provider in providers_elem.findall('provider'):
            providers.append({
                'provider_id': provider.get('provider_id', ''),
                'gender': provider.get('gender', ''),
                'speciality': provider.get('speciality', ''),
                'year_of_birth': provider.get('year_of_birth', '')
            })

    return providers


def _parse_hierarchical_event(
    event_elem: ET.Element,
    timestamp: str,
    patient_id: str,
    caresites: List[Dict],
    providers: List[Dict]
) -> Optional[Dict]:
    """
    Parse single event with full structured metadata.

    Returns event dictionary with type-specific fields:
    - note: note_id, name, code
    - procedure: procedure_id, name, code
    - condition: code (ICD10), name
    - measurement: value, unit, code (LOINC)
    - observation: value, code
    - drug_exposure: code (RxNorm), name
    - image: image_id, anatomic_site_source_value, procedure_id
    - visit: visit_detail
    """
    event_type = event_elem.get('type')
    if not event_type:
        return None

    # Parse timestamp
    try:
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M")
        date = dt.strftime("%Y-%m-%d")
        time = dt.strftime("%H:%M")
    except:
        date = timestamp[:10] if len(timestamp) >= 10 else timestamp
        time = timestamp[11:16] if len(timestamp) >= 16 else ""

    # Base event structure
    event = {
        'patient_id': patient_id,
        'timestamp': timestamp,
        'date': date,
        'time': time,
        'type': event_type,
        'node_type': 'Event',
        # Common fields
        'visit_id': event_elem.get('visit_id', ''),
        'provider_id': event_elem.get('provider_id', ''),
        'care_site_id': event_elem.get('care_site_id', ''),
        'code': event_elem.get('code', ''),
        'name': event_elem.get('name', ''),
        'value': (event_elem.text or '').strip(),
    }

    # Type-specific fields
    if event_type == 'note':
        event['note_id'] = event_elem.get('note_id', '')
        event['event_id'] = f"note_{event['note_id']}"

    elif event_type == 'procedure':
        event['procedure_id'] = event_elem.get('procedure_id', '')
        event['event_id'] = f"proc_{event['procedure_id']}"

    elif event_type == 'image':
        event['image_id'] = event_elem.get('image_id', '')
        event['anatomic_site'] = event_elem.get('anatomic_site_source_value', '')
        # Images may be linked to procedures
        event['procedure_id'] = event_elem.get('procedure_id', '')
        event['event_id'] = f"img_{event['image_id']}"

    elif event_type == 'condition':
        # ICD10 codes
        event['event_id'] = _generate_event_id(timestamp, event_type, event['code'])

    elif event_type == 'measurement':
        event['unit'] = event_elem.get('unit', '')
        event['event_id'] = _generate_event_id(timestamp, event_type, event['name'])

    elif event_type == 'observation':
        event['event_id'] = _generate_event_id(timestamp, event_type, event['name'])

    elif event_type == 'drug_exposure':
        # RxNorm codes
        event['event_id'] = _generate_event_id(timestamp, event_type, event['code'])

    elif event_type == 'visit' or event_type == 'visit_detail':
        event['event_id'] = f"visit_{event['visit_id']}"

    else:
        # Generic fallback
        event['event_id'] = _generate_event_id(timestamp, event_type, event.get('name', ''))

    return event


def _generate_event_id(timestamp: str, event_type: str, identifier: str) -> str:
    """Generate unique event ID from timestamp + type + identifier."""
    key = f"{timestamp}_{event_type}_{identifier}"
    return hashlib.md5(key.encode()).hexdigest()[:24]


def _add_visit_nodes(graph: TOAGraph, visits: Dict[str, List[str]], events: List[Dict]):
    """
    Add Visit nodes as parent containers for events (tree structure).

    Creates:
    - Visit nodes (one per unique visit_id)
    - VISIT_CONTAINS edges (Visit → Event, parent-child)

    This is O(n) instead of O(n²) for fully-connected visit groups.

    Args:
        graph: TOAGraph
        visits: Dictionary mapping visit_id → [event_ids]
        events: List of all events
    """
    # Get visit metadata from events
    visit_metadata = {}
    for event in events:
        visit_id = event.get('visit_id')
        if visit_id and visit_id not in visit_metadata:
            # Extract visit-level metadata from first event (exclude visit_id to avoid duplicate)
            visit_metadata[visit_id] = {
                'date': event.get('date'),
                'timestamp': event.get('timestamp'),
                'care_site_id': event.get('care_site_id'),
                'provider_id': event.get('provider_id'),
            }

    # Add Visit nodes
    for visit_id, event_ids in visits.items():
        if not event_ids:
            continue

        visit_node_id = f"visit_{visit_id}"

        # Add Visit node
        metadata = visit_metadata.get(visit_id, {})
        graph.G.add_node(
            visit_node_id,
            node_type='Visit',
            visit_id=visit_id,
            event_count=len(event_ids),
            **metadata
        )

        # Add VISIT_CONTAINS edges (parent → children)
        for event_id in event_ids:
            graph.G.add_edge(
                visit_node_id,
                event_id,
                edge_type='VISIT_CONTAINS',
                visit_id=visit_id
            )


def _build_hierarchical_graph(graph: TOAGraph, data: Dict[str, Any], patient_id: str):
    """
    Build graph from hierarchically structured events.

    Creates:
    1. Event nodes (with structured type-specific metadata)
    2. Visit nodes (parent containers for events)
    3. XMLFragment nodes (note_id, procedure_id, image_id level)
    4. PRECEDES edges (temporal ordering)
    5. VISIT_CONTAINS edges (Visit → Event, parent-child tree)
    6. IMAGE_REPORT_FOR edges (radiology report → image)
    7. PROCEDURE_FOR edges (image → procedure)
    8. SOURCED_FROM edges (event → XMLFragment)
    """
    events = data['events']
    visits = data['visits']
    images = data['images']
    procedures = data['procedures']
    notes = data['notes']
    demographics = data.get('demographics')

    # Add Person node with demographics (DOB, sex, race, ethnicity)
    if demographics:
        person_id = f"{patient_id}_person" if any('_' in e['event_id'] for e in events[:1]) else "person"
        graph.G.add_node(person_id, node_type='Person', **demographics)

    # Sort events by timestamp
    events_sorted = sorted(events, key=lambda e: e['timestamp'])

    # 1. Add Event nodes
    for event in events_sorted:
        event_id = event['event_id']
        graph.G.add_node(event_id, **event)

    # 2. Add Visit nodes (parent containers)
    _add_visit_nodes(graph, visits, events)

    # 3. Add XMLFragment nodes (granular provenance)
    _add_granular_xml_fragments(graph, events)

    # 4. Add PRECEDES edges (temporal ordering)
    for i in range(len(events_sorted) - 1):
        e1_id = events_sorted[i]['event_id']
        e2_id = events_sorted[i + 1]['event_id']

        graph.G.add_edge(
            e1_id,
            e2_id,
            edge_type='PRECEDES',
            relation='temporal'
        )

    # 5. Link radiology reports to images
    _link_radiology_reports(graph, notes, images, visits)

    # 6. Link images to procedures
    _link_images_to_procedures(graph, images, procedures)


def _add_granular_xml_fragments(graph: TOAGraph, events: List[Dict]):
    """
    Add XMLFragment nodes at granular level (note_id, procedure_id, image_id).

    Each event type gets appropriate fragment:
    - note events → XMLFragment(note_id)
    - procedure events → XMLFragment(procedure_id)
    - image events → XMLFragment(image_id)
    - others → XMLFragment(date)
    """
    for event in events:
        event_id = event['event_id']
        event_type = event['type']

        fragment_id = None
        fragment_data = {
            'node_type': 'XMLFragment',
            'provenance_type': None
        }

        # Determine fragment granularity
        if event_type == 'note':
            note_id = event.get('note_id')
            if note_id:
                fragment_id = f"xml_note_{note_id}"
                fragment_data['note_id'] = note_id
                fragment_data['provenance_type'] = 'note_id'

        elif event_type == 'procedure':
            procedure_id = event.get('procedure_id')
            if procedure_id:
                fragment_id = f"xml_proc_{procedure_id}"
                fragment_data['procedure_id'] = procedure_id
                fragment_data['provenance_type'] = 'procedure_id'

        elif event_type == 'image':
            image_id = event.get('image_id')
            if image_id:
                fragment_id = f"xml_img_{image_id}"
                fragment_data['image_id'] = image_id
                fragment_data['provenance_type'] = 'image_id'

        # Fallback to date-level
        if fragment_id is None:
            date = event.get('date')
            if date:
                fragment_id = f"xml_date_{date}"
                fragment_data['evidence_date'] = date
                fragment_data['provenance_type'] = 'date'

        # Add fragment node (if not exists)
        if fragment_id and not graph.G.has_node(fragment_id):
            graph.G.add_node(fragment_id, **fragment_data)

        # Add SOURCED_FROM edge
        if fragment_id:
            graph.G.add_edge(
                event_id,
                fragment_id,
                edge_type='SOURCED_FROM'
            )


def _link_radiology_reports(
    graph: TOAGraph,
    notes: Dict[str, Dict],
    images: Dict[str, Dict],
    visits: Dict[str, List[str]]
):
    """
    Link radiology report notes to their corresponding images.

    Strategy (from Gemini analysis):
    1. Match by visit_id (primary)
    2. Match by date + procedure name (secondary)
    """
    for note_id, note_event in notes.items():
        note_event_id = note_event['event_id']
        note_visit_id = note_event.get('visit_id')
        note_date = note_event.get('date')
        note_name = note_event.get('name', '').lower()
        note_value = note_event.get('value', '').lower()

        # Check if this is a radiology report (contains imaging keywords)
        if not any(kw in note_name or kw in note_value
                   for kw in ['ct', 'mri', 'pet', 'xr', 'x-ray', 'ultrasound', 'imaging', 'radiology']):
            continue

        # Strategy 1: Match by visit_id
        if note_visit_id:
            for image_id, image_event in images.items():
                if image_event.get('visit_id') == note_visit_id:
                    # Found match!
                    graph.G.add_edge(
                        note_event_id,
                        image_event['event_id'],
                        edge_type='IMAGE_REPORT_FOR',
                        matching_method='visit_id',
                        visit_id=note_visit_id
                    )

        # Strategy 2: Match by date + procedure name
        else:
            for image_id, image_event in images.items():
                image_date = image_event.get('date')
                image_procedure_id = image_event.get('procedure_id')

                # Same date?
                if image_date != note_date:
                    continue

                # Try to match procedure name from note content
                # (e.g., "CT CHEST" in note matches image with procedure "CT CHEST WO IV CONTRAST")
                if image_procedure_id:
                    # Find procedure name
                    # This is simplified - in production, would query procedure dict
                    graph.G.add_edge(
                        note_event_id,
                        image_event['event_id'],
                        edge_type='IMAGE_REPORT_FOR',
                        matching_method='date_and_content',
                        date=note_date
                    )


def _link_images_to_procedures(
    graph: TOAGraph,
    images: Dict[str, Dict],
    procedures: Dict[str, Dict]
):
    """
    Link image events to their ordering procedures.

    Images have procedure_id attribute that links to procedure event.
    """
    for image_id, image_event in images.items():
        procedure_id = image_event.get('procedure_id')

        if procedure_id and procedure_id in procedures:
            procedure_event = procedures[procedure_id]

            graph.G.add_edge(
                image_event['event_id'],
                procedure_event['event_id'],
                edge_type='PROCEDURE_FOR',
                procedure_id=procedure_id
            )


# ==========================================================================
# Query Helper Functions
# ==========================================================================

def get_events_by_visit(graph: TOAGraph, visit_id: str) -> List[Dict[str, Any]]:
    """
    Get all events in a visit (using tree structure).

    Args:
        graph: TOAGraph
        visit_id: Visit identifier

    Returns:
        List of events in this visit
    """
    visit_node_id = f"visit_{visit_id}"

    if not graph.G.has_node(visit_node_id):
        return []

    events = []

    # Traverse children from Visit node
    for child in graph.G.successors(visit_node_id):
        edge_data = graph.G.get_edge_data(visit_node_id, child)
        if any(e.get('edge_type') == 'VISIT_CONTAINS' for e in edge_data.values()):
            events.append(dict(graph.G.nodes[child]))

    return sorted(events, key=lambda e: e.get('timestamp', ''))


def get_radiology_report_for_image(graph: TOAGraph, image_event_id: str) -> Optional[Dict[str, Any]]:
    """
    Get radiology report note for an image.

    Args:
        graph: TOAGraph
        image_event_id: Image event identifier

    Returns:
        Note event dictionary or None
    """
    # Find incoming IMAGE_REPORT_FOR edges
    for predecessor in graph.G.predecessors(image_event_id):
        edge_data = graph.G.get_edge_data(predecessor, image_event_id)

        if any(e.get('edge_type') == 'IMAGE_REPORT_FOR' for e in edge_data.values()):
            return dict(graph.G.nodes[predecessor])

    return None


def get_images_for_procedure(graph: TOAGraph, procedure_event_id: str) -> List[Dict[str, Any]]:
    """
    Get all images associated with a procedure.

    Args:
        graph: TOAGraph
        procedure_event_id: Procedure event identifier

    Returns:
        List of image events
    """
    images = []

    for predecessor in graph.G.predecessors(procedure_event_id):
        edge_data = graph.G.get_edge_data(predecessor, procedure_event_id)

        if any(e.get('edge_type') == 'PROCEDURE_FOR' for e in edge_data.values()):
            images.append(dict(graph.G.nodes[predecessor]))

    return images


def get_all_radiology_studies(graph: TOAGraph) -> List[Dict[str, Any]]:
    """
    Get all radiology studies (procedure + images + report).

    Returns:
        List of study dictionaries with:
        - procedure: Procedure event
        - images: List of image events
        - report: Note event (if available)
    """
    studies = []

    # Find all imaging procedures
    for node_id, data in graph.G.nodes(data=True):
        if data.get('type') == 'procedure':
            name = data.get('name', '').lower()
            if any(kw in name for kw in ['ct', 'mri', 'pet', 'xr', 'x-ray', 'ultrasound']):
                # Get images
                images = get_images_for_procedure(graph, node_id)

                # Get report (check images for report links)
                report = None
                for image in images:
                    report = get_radiology_report_for_image(graph, image['event_id'])
                    if report:
                        break

                studies.append({
                    'procedure': data,
                    'images': images,
                    'report': report
                })

    return studies


# ==========================================================================
# Statistics
# ==========================================================================

def print_hierarchical_statistics(graph: TOAGraph):
    """Print statistics about hierarchical graph structure."""
    total_nodes = graph.G.number_of_nodes()
    total_edges = graph.G.number_of_edges()

    # Count by node type
    node_types = defaultdict(int)
    event_types = defaultdict(int)

    for node_id, data in graph.G.nodes(data=True):
        node_type = data.get('node_type', 'unknown')
        node_types[node_type] += 1

        if node_type == 'Event':
            event_type = data.get('type', 'unknown')
            event_types[event_type] += 1

    # Count by edge type
    edge_types = defaultdict(int)
    for u, v, data in graph.G.edges(data=True):
        edge_type = data.get('edge_type', 'unknown')
        edge_types[edge_type] += 1

    print("=" * 70)
    print("HIERARCHICAL GRAPH STATISTICS")
    print("=" * 70)
    print(f"Total nodes: {total_nodes:,}")
    print(f"Total edges: {total_edges:,}")

    print(f"\nNode types:")
    for node_type, count in sorted(node_types.items()):
        print(f"  {node_type}: {count:,}")

    print(f"\nEvent types:")
    for event_type, count in sorted(event_types.items(), key=lambda x: -x[1]):
        print(f"  {event_type}: {count:,}")

    print(f"\nEdge types:")
    for edge_type, count in sorted(edge_types.items(), key=lambda x: -x[1]):
        print(f"  {edge_type}: {count:,}")

    print("=" * 70)
