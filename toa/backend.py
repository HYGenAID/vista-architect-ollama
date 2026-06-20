"""
TOA Backend API - Clean interface for patient timeline extraction.

This module provides a simple, framework-agnostic API for extracting
clinical timelines from EHR XML data. Designed to be called from:
- Dash app (current)
- CLI scripts (batch processing)
- FastAPI service (future)
- Next.js backend (future)
- EHR hooks (production)

Key principle: Frontend never imports toa.engine directly.
"""

from __future__ import annotations
import os
import json
from typing import Dict, Any, Optional, Callable
from pathlib import Path

from .engine import extract, project_major_events
from .io import jsonio
from .graph import TOAGraph


class ExtractionResult:
    """Result of a patient timeline extraction."""

    def __init__(self, pid: str, output_dir: str, success: bool, error: Optional[str] = None):
        self.pid = pid
        self.output_dir = output_dir
        self.success = success
        self.error = error
        self.num_events = 0
        self.num_episodes = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        files = {
            "timeline_objects": f"{self.output_dir}/timeline_objects.jsonl",
            "episodes": f"{self.output_dir}/episodes.json",
            "timeline": f"{self.output_dir}/timeline.json"  # legacy format
        }

        # Add graph file if it exists
        graph_path = Path(self.output_dir) / "graph.graphml"
        if graph_path.exists():
            files["graph"] = str(graph_path)

        return {
            "patient_id": self.pid,
            "output_directory": self.output_dir,
            "success": self.success,
            "error": self.error,
            "num_events": self.num_events,
            "num_episodes": self.num_episodes,
            "files_created": files
        }


def extract_patient(
    pid: str,
    xml_text: str,
    chat_client: Callable[[str, str], str],
    profile: str = "thoracic_tumor_board",
    tumor_type: Optional[str] = None,
    output_dir: Optional[str] = None,
    episodes_mode: str = "llm",  # "llm" | "code"
    chunk_chat_client: Optional[Callable[[str, str], str]] = None,  # Separate model for chunk extraction
    episode_chat_client: Optional[Callable[[str, str], str]] = None,  # Separate model for episode splitting
    chunking_strategy: Optional[str] = None,  # Override profile's chunking strategy ("full" | "streamlined")
    export_graph: bool = True,  # Export NetworkX graph to .graphml file
    enable_provenance: bool = False  # Enable provenance linking (experimental)
) -> ExtractionResult:
    """
    Extract clinical timeline for a single patient.

    This is the ONLY public API for TOA extraction. All frontends (Dash, Next.js, CLI)
    should call this function.

    Args:
        pid: Patient ID
        xml_text: Full EHR XML content as string
        chat_client: LLM chat function with signature (system_prompt, user_prompt) -> str
                    (used as default if chunk_chat_client or episode_chat_client not specified)
        profile: Profile name (e.g., "thoracic_tumor_board")
        tumor_type: Override tumor type (defaults to profile's tumor_type)
        output_dir: Output directory (defaults to temp_jsons/{pid})
        episodes_mode: "llm" (LLM-based episodes) or "code" (deterministic)
        chunk_chat_client: Optional separate LLM for chunk extraction (defaults to chat_client)
        episode_chat_client: Optional separate LLM for episode splitting (defaults to chat_client)
        chunking_strategy: Override profile's chunking strategy ("full" | "streamlined")
        enable_provenance: Enable provenance linking (creates reference_map.json, saves Lumia graph)

    Returns:
        ExtractionResult with status and file paths

    Side Effects:
        Creates files in output_dir:
        - timeline_objects.jsonl: Full TOA events (one per line)
        - episodes.json: Clinical episodes
        - timeline.json: Legacy format for backward compatibility

    Example:
        >>> def my_chat(sys, usr):
        ...     return call_openai(sys, usr)
        >>> result = extract_patient("P001", xml, my_chat, "thoracic_tumor_board")
        >>> if result.success:
        ...     print(f"Created {result.num_events} events")
    """
    # Set default output directory
    if output_dir is None:
        output_dir = f"temp_jsons/{pid}"

    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate Lumia context for provenance (if enabled)
        lumia_context = None
        reference_map = None
        deterministic_context = None
        if enable_provenance:
            print(f"\n  Building Lumia graph for provenance tracking...")
            from .xml_to_graph_hierarchical import build_hierarchical_graph
            from .format_lumia_context import format_lumia_for_extraction

            try:
                lumia_graph = build_hierarchical_graph(pid, xml_text)
                lumia_context, reference_map = format_lumia_for_extraction(lumia_graph)
                print(f"  Lumia context generated: {len(reference_map)} source events available")

                # Generate deterministic search hits for LLM attention guidance
                try:
                    from .deterministic_retrieval import DeterministicRetriever
                    retriever = DeterministicRetriever(lumia_graph)
                    det_results = retriever.get_comprehensive_context()
                    deterministic_context = _format_deterministic_guidance(det_results)
                    ct_date_vector = retriever.get_ct_date_vector()
                    imaging_date_vector = retriever.get_imaging_date_vector()
                    print(f"  Deterministic search hits generated for LLM guidance")
                    if imaging_date_vector:
                        print(f"  Imaging date vector: {len(imaging_date_vector)} studies (CT: {len(ct_date_vector)})")
                except Exception as e:
                    print(f"  Deterministic retrieval failed (non-fatal): {e}")
                    ct_date_vector = None
                    imaging_date_vector = None

            except Exception as e:
                print(f"  Lumia graph build failed: {e}")
                print(f"     Continuing without provenance tracking...")
                enable_provenance = False  # Disable provenance if Lumia fails
                ct_date_vector = None
                imaging_date_vector = None
        else:
            ct_date_vector = None
            imaging_date_vector = None

        # Extract events and episodes using TOA engine
        events, episodes = extract(
            pid=pid,
            xml_text=xml_text,
            chat_client=chat_client,
            profile=profile,
            tumor_type=tumor_type,
            episodes_mode=episodes_mode,
            output_dir=output_dir,  # Enable intermediate chunk saving & resume
            chunk_chat_client=chunk_chat_client,  # Pass through stage-specific clients
            episode_chat_client=episode_chat_client,
            chunking_strategy=chunking_strategy,  # Pass through strategy override
            lumia_context=lumia_context,  # Pass Lumia context for provenance
            deterministic_context=deterministic_context,  # Pass search hits for LLM guidance
            ct_date_vector=ct_date_vector,  # Confirmed CT dates from structured data
            imaging_date_vector=imaging_date_vector,  # Full imaging vector (all modalities)
        )

        # NOTE: Episode enrichment bypassed — patient_info now extracted separately
        # by gpt-5 in prepare_patient.py Step 2. To re-enable, uncomment below:
        # patient_state = _extract_patient_state(episodes)
        # if patient_state:
        #     patient_info = _patient_state_to_patient_info(patient_state)
        #     jsonio.write_json(f"{output_dir}/patient_info.json", patient_info)

        # Save TOA outputs
        jsonio.write_jsonl(f"{output_dir}/timeline_objects.jsonl", events)
        jsonio.write_json(f"{output_dir}/episodes.json", episodes)

        # Project to legacy timeline.json format for backward compatibility
        timeline_legacy = project_major_events(events, episodes)
        jsonio.write_json(f"{output_dir}/timeline.json", timeline_legacy)

        # Optional: Add provenance post-processing
        if enable_provenance and reference_map:
            try:
                # Save reference map
                jsonio.write_json(f"{output_dir}/reference_map.json", reference_map)
                _add_provenance_postprocessing(pid, events, output_dir, reference_map)
            except Exception as e:
                print(f"  ⚠️  Provenance processing failed: {e}")
                # Don't fail extraction - events already saved!

        # Export graph database (optional, enabled by default)
        if export_graph:
            try:
                graph = TOAGraph()
                graph.load_patient(pid, base_dir=output_dir.rsplit('/', 1)[0])
                graph.save(f"{output_dir}/graph.graphml")
                print(f"  ✅ Graph exported: {output_dir}/graph.graphml")

                # Print graph stats
                stats = graph.stats()
                print(f"     Nodes: {stats['total_nodes']} ({stats['event_nodes']} events, "
                      f"{stats['episode_nodes']} episodes, {stats['xml_fragment_nodes']} XML fragments)")
                print(f"     Edges: {stats['total_edges']} ({', '.join(f'{k}: {v}' for k, v in stats['edge_types'].items())})")

            except Exception as e:
                print(f"  ⚠️  Graph export failed: {e}")
                # Don't fail the entire extraction if graph export fails

        # Create success result
        result = ExtractionResult(pid, output_dir, success=True)
        result.num_events = len(events)
        result.num_episodes = len(episodes)
        result.deterministic_context = deterministic_context

        return result

    except Exception as e:
        # Create failure result
        result = ExtractionResult(pid, output_dir, success=False, error=str(e))
        result.deterministic_context = None
        return result


def extract_patient_from_graph(
    pid: str,
    chat_client: Callable[[str, str], str],
    store_path: str = "graph_store",
    profile: str = "thoracic_tumor_board",
    tumor_type: Optional[str] = None,
    output_dir: Optional[str] = None,
    episodes_mode: str = "llm",
    chunk_chat_client: Optional[Callable[[str, str], str]] = None,
    episode_chat_client: Optional[Callable[[str, str], str]] = None,
    export_graph: bool = True,
    max_chunk_chars: int = 120_000,
    max_final_chunk_chars: Optional[int] = None,
) -> ExtractionResult:
    """
    Extract clinical timeline from pre-built graph store (no XML needed).

    Graph-first pipeline: loads Lumia graph from CohortGraphStore, serializes
    to compact clinical text, and feeds through the standard extraction engine.

    Args:
        pid: Patient ID (must exist in graph store)
        chat_client: LLM chat function (default for all stages)
        store_path: Path to CohortGraphStore directory
        profile: Profile name
        tumor_type: Override tumor type
        output_dir: Output directory (defaults to temp_jsons/{pid})
        episodes_mode: "llm" or "code"
        chunk_chat_client: Optional separate LLM for chunk extraction
        episode_chat_client: Optional separate LLM for episode splitting
        export_graph: Export NetworkX graph to .graphml file
        max_chunk_chars: Max characters per early chunk (default: 120K)
        max_final_chunk_chars: Max characters for final chunk (default: same as max_chunk_chars)

    Returns:
        ExtractionResult with status, file paths, and final_chunk attribute
    """
    from .graph_feeding import prepare_from_graph

    if output_dir is None:
        output_dir = f"temp_jsons/{pid}"

    try:
        os.makedirs(output_dir, exist_ok=True)

        # Prepare from graph store
        print(f"\n  Preparing from graph store ({store_path})...")
        prepared = prepare_from_graph(pid, store_path, max_chunk_chars=max_chunk_chars, max_final_chunk_chars=max_final_chunk_chars)

        # Extract imaging date vectors from graph
        ct_date_vector = None
        imaging_date_vector = None
        try:
            from .deterministic_retrieval import DeterministicRetriever
            retriever = DeterministicRetriever(prepared['graph'])
            ct_date_vector = retriever.get_ct_date_vector()
            imaging_date_vector = retriever.get_imaging_date_vector()
            if imaging_date_vector:
                print(f"  Imaging date vector: {len(imaging_date_vector)} studies (CT: {len(ct_date_vector)})")
        except Exception as e:
            print(f"  Imaging date vector extraction failed (non-fatal): {e}")

        # Extract events and episodes via engine (graph-first path)
        # E-references are inline in text_chunks, so no separate lumia_context
        events, episodes = extract(
            pid=pid,
            xml_text="",  # Not used when text_chunks provided
            chat_client=chat_client,
            profile=profile,
            tumor_type=tumor_type,
            episodes_mode=episodes_mode,
            output_dir=output_dir,
            chunk_chat_client=chunk_chat_client,
            episode_chat_client=episode_chat_client,
            text_chunks=prepared['text_chunks'],
            lumia_context=None,  # E-refs already inline in serialized text
            deterministic_context=prepared['deterministic_context'],
            ct_date_vector=ct_date_vector,
            imaging_date_vector=imaging_date_vector,
        )

        # NOTE: Episode enrichment bypassed — patient_info now extracted separately
        # by gpt-5 in prepare_patient.py Step 2. To re-enable, uncomment below:
        # patient_state = _extract_patient_state(episodes)
        # if patient_state:
        #     patient_info = _patient_state_to_patient_info(patient_state)
        #     jsonio.write_json(f"{output_dir}/patient_info.json", patient_info)

        # Save outputs
        jsonio.write_jsonl(f"{output_dir}/timeline_objects.jsonl", events)
        jsonio.write_json(f"{output_dir}/episodes.json", episodes)

        # Project to legacy timeline.json
        timeline_legacy = project_major_events(events, episodes)
        jsonio.write_json(f"{output_dir}/timeline.json", timeline_legacy)

        # Save reference map for provenance
        if prepared['reference_map']:
            jsonio.write_json(f"{output_dir}/reference_map.json", prepared['reference_map'])

        # Export TOA graph
        if export_graph:
            try:
                graph = TOAGraph()
                graph.load_patient(pid, base_dir=output_dir.rsplit('/', 1)[0])
                graph.save(f"{output_dir}/graph.graphml")
                print(f"  Graph exported: {output_dir}/graph.graphml")
            except Exception as e:
                print(f"  Graph export failed (non-fatal): {e}")

        # Create result
        result = ExtractionResult(pid, output_dir, success=True)
        result.num_events = len(events)
        result.num_episodes = len(episodes)
        # Attach final_chunk for Step 2 JSON generation
        result.final_chunk = prepared['final_chunk']
        result.deterministic_context = prepared.get('deterministic_context')

        return result

    except Exception as e:
        result = ExtractionResult(pid, output_dir, success=False, error=str(e))
        result.final_chunk = None
        result.deterministic_context = None
        return result


def extract_patient_from_file(
    pid: str,
    xml_path: str,
    chat_client: Callable[[str, str], str],
    profile: str = "thoracic_tumor_board",
    tumor_type: Optional[str] = None,
    output_dir: Optional[str] = None,
    episodes_mode: str = "llm",
    chunk_chat_client: Optional[Callable[[str, str], str]] = None,
    episode_chat_client: Optional[Callable[[str, str], str]] = None,
    chunking_strategy: Optional[str] = None,
    export_graph: bool = True,
    enable_provenance: bool = False
) -> ExtractionResult:
    """
    Extract clinical timeline for a single patient from XML file.

    Convenience wrapper around extract_patient() that reads XML from file.

    Args:
        pid: Patient ID
        xml_path: Path to EHR XML file
        chat_client: LLM chat function (default for all stages if stage-specific not provided)
        profile: Profile name
        tumor_type: Override tumor type
        output_dir: Output directory (defaults to temp_jsons/{pid})
        episodes_mode: "llm" or "code"
        chunk_chat_client: Optional separate LLM for chunk extraction
        episode_chat_client: Optional separate LLM for episode splitting
        chunking_strategy: Override profile's chunking strategy ("full" | "streamlined")

    Returns:
        ExtractionResult with status and file paths

    Raises:
        FileNotFoundError: If xml_path doesn't exist
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_text = f.read()

    return extract_patient(
        pid=pid,
        xml_text=xml_text,
        chat_client=chat_client,
        profile=profile,
        tumor_type=tumor_type,
        output_dir=output_dir,
        episodes_mode=episodes_mode,
        chunk_chat_client=chunk_chat_client,
        episode_chat_client=episode_chat_client,
        chunking_strategy=chunking_strategy,
        export_graph=export_graph,
        enable_provenance=enable_provenance
    )


def batch_extract_patients(
    patient_ids: list[str],
    xml_dir: str,
    chat_client: Callable[[str, str], str],
    profile: str = "thoracic_tumor_board",
    tumor_type: Optional[str] = None,
    output_base_dir: str = "temp_jsons",
    chunk_chat_client: Optional[Callable[[str, str], str]] = None,
    episode_chat_client: Optional[Callable[[str, str], str]] = None,
    chunking_strategy: Optional[str] = None,
    export_graph: bool = True
) -> Dict[str, ExtractionResult]:
    """
    Extract timelines for multiple patients (batch processing).

    Use this for pre-generating JSONs for a tumor board cohort before the meeting.

    Args:
        patient_ids: List of patient IDs
        xml_dir: Directory containing {pid}.xml files
        chat_client: LLM chat function (default for all stages)
        profile: Profile name
        tumor_type: Override tumor type
        output_base_dir: Base directory for outputs (creates {base}/{pid}/ for each)
        chunk_chat_client: Optional separate LLM for chunk extraction
        episode_chat_client: Optional separate LLM for episode splitting
        chunking_strategy: Override profile's chunking strategy ("full" | "streamlined")

    Returns:
        Dictionary mapping patient_id -> ExtractionResult

    Example:
        >>> patients = ["P001", "P002", "P003"]
        >>> results = batch_extract_patients(patients, "patient_records/thoracic_xmls/", my_chat)
        >>> successful = [pid for pid, r in results.items() if r.success]
        >>> print(f"{len(successful)}/{len(patients)} succeeded")
    """
    results = {}

    for pid in patient_ids:
        print(f"Processing patient {pid}...")

        xml_path = Path(xml_dir) / f"{pid}.xml"
        output_dir = f"{output_base_dir}/{pid}"

        try:
            result = extract_patient_from_file(
                pid=pid,
                xml_path=str(xml_path),
                chat_client=chat_client,
                profile=profile,
                tumor_type=tumor_type,
                output_dir=output_dir,
                chunk_chat_client=chunk_chat_client,
                episode_chat_client=episode_chat_client,
                chunking_strategy=chunking_strategy,
                export_graph=export_graph
            )
            results[pid] = result

            if result.success:
                print(f"  ✅ Success: {result.num_events} events, {result.num_episodes} episodes")
            else:
                print(f"  ❌ Failed: {result.error}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[pid] = ExtractionResult(pid, output_dir, success=False, error=str(e))

    return results


def check_patient_ready(pid: str, base_dir: str = "temp_jsons") -> bool:
    """
    Check if a patient's timeline JSONs are already generated and ready to use.

    Use this to determine if extraction is needed or if the dashboard can
    render immediately.

    Args:
        pid: Patient ID
        base_dir: Base directory containing patient subdirectories

    Returns:
        True if all required JSON files exist, False otherwise

    Example:
        >>> if check_patient_ready("P001"):
        ...     render_dashboard("P001")
        ... else:
        ...     extract_patient_from_file("P001", "data/P001.xml", chat_client)
    """
    patient_dir = Path(base_dir) / pid

    # Required files for dashboard rendering
    required_files = [
        "timeline.json",           # Legacy format (minimum requirement)
    ]

    # Optional TOA files (nice to have but not required for basic dashboard)
    optional_files = [
        "timeline_objects.jsonl",  # Full TOA events
        "episodes.json"            # Clinical episodes
    ]

    # Check if all required files exist
    for filename in required_files:
        file_path = patient_dir / filename
        if not file_path.exists():
            return False

    return True


def _patient_state_to_patient_info(patient_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reformat patient_state (from enriched episodes) into patient_info.json schema.

    Maps the flat patient_state dict to the 3-section format expected by the UI
    and downstream consumers (quick_eval.py, summary generation).

    Args:
        patient_state: Dict from last episode's patient_state field

    Returns:
        Dict with "PATIENT DEMOGRAPHICS", "TUMOR INFORMATION", "TREATMENTS" sections
    """
    ps = patient_state

    demographics = {
        "name": ps.get("name", "Unknown"),
        "date_of_birth": ps.get("date_of_birth", "Unknown"),
        "sex": ps.get("sex", "Unknown"),
        "height_cm": ps.get("height_cm"),
        "weight_kg": ps.get("weight_kg"),
        "ecog_performance_status": ps.get("ecog_performance_status", "Unknown"),
        "previous_conditions": ps.get("previous_conditions", ""),
        "therapy_toxicities": ps.get("therapy_toxicities", "None"),
        "allergies": ps.get("allergies", []),
        "smoking_history": ps.get("smoking_history", "Unknown"),
        "medications": ps.get("medications", []),
        "dnr": ps.get("dnr", "No"),
    }

    tumor_info = {
        "diagnosis": ps.get("diagnosis", "Unknown"),
        "tnm_staging": ps.get("tnm_staging", "Unknown"),
        "histology": ps.get("histology", "Unknown"),
        "metastasis_status": ps.get("metastasis_status", "Unknown"),
        "lymph_node_involvement": ps.get("lymph_node_involvement", "Unknown"),
        "driver_mutations": ps.get("driver_mutations", {}),
        "latest_updates": ps.get("latest_updates", []),
        "body_diagram_image": ps.get("body_diagram_image", ""),
    }

    treatments = {
        "current": ps.get("current_treatment", []),
        "previous": ps.get("previous_treatment", []),
        "radiation_therapy": ps.get("radiation_therapy", "No"),
        "alternatives": [],  # Not extracted from episodes (too speculative for LLM)
        "surgical_candidate": ps.get("surgical_candidate", {"eligible": False, "description": "Not assessed"}),
    }

    return {
        "PATIENT DEMOGRAPHICS": demographics,
        "TUMOR INFORMATION": tumor_info,
        "TREATMENTS": treatments,
    }


def _extract_patient_state(episodes: list) -> Optional[Dict[str, Any]]:
    """Extract and remove patient_state from the last episode, if present."""
    if not episodes:
        return None
    last_ep = episodes[-1]
    patient_state = last_ep.pop("patient_state", None)
    return patient_state


def _format_deterministic_guidance(det_results: dict) -> str:
    """
    Format deterministic retrieval results as plain-text LLM attention guidance.

    Unlike format_deterministic_context.py (which creates its own E-references),
    this produces plain text that complements the Lumia reference map without
    conflicting reference IDs.

    Args:
        det_results: Output from DeterministicRetriever.get_comprehensive_context()

    Returns:
        Plain-text guidance string for LLM prompt injection
    """
    lines = []
    lines.append("=" * 60)
    lines.append("GRAPH SEARCH HITS (key clinical variables found in EHR):")
    lines.append("=" * 60)

    def _snippet(text, max_len=250):
        return text[:max_len] + "..." if len(text) > max_len else text

    # Metastasis
    mets = det_results.get('metastasis', {})
    if mets.get('count', 0) > 0:
        lines.append(f"\nMETASTASIS ({mets['count']} mentions):")
        if mets.get('latest_note'):
            lines.append(f"  Latest note ({mets['latest_note']['date']}): \"{_snippet(mets['latest_note']['text'])}\"")
        if mets.get('latest_radiology'):
            lines.append(f"  Latest radiology ({mets['latest_radiology']['date']}): \"{_snippet(mets['latest_radiology']['text'])}\"")

    # Lymph nodes
    lymph = det_results.get('lymph_nodes', {})
    if lymph.get('count', 0) > 0:
        lines.append(f"\nLYMPH NODES ({lymph['count']} mentions):")
        if lymph.get('latest_note'):
            lines.append(f"  Latest note ({lymph['latest_note']['date']}): \"{_snippet(lymph['latest_note']['text'])}\"")
        if lymph.get('latest_radiology'):
            lines.append(f"  Latest radiology ({lymph['latest_radiology']['date']}): \"{_snippet(lymph['latest_radiology']['text'])}\"")

    # Driver mutations
    mutations = det_results.get('driver_mutations', {})
    if mutations:
        genes = [k for k in mutations if k != '_driver_mutation_keyword']
        if genes:
            lines.append(f"\nDRIVER MUTATIONS:")
            for gene in sorted(genes):
                info = mutations[gene]
                event = info['latest_event']
                lines.append(f"  {gene} ({info['count']} mentions, latest {event['date']}): \"{_snippet(event['text'])}\"")

    # Molecular testing (NGS panels, liquid biopsy, etc.)
    molecular = det_results.get('molecular_testing', {})
    if molecular.get('count', 0) > 0:
        lines.append(f"\nMOLECULAR/GENOMIC TESTING ({molecular['count']} mentions):")
        for m in molecular.get('mentions', []):
            lines.append(f"  ({m['date']}): \"{_snippet(m['text'], 400)}\"")

    # Smoking — use larger snippet to capture actual status value
    smoking = det_results.get('smoking', {})
    if smoking.get('count', 0) > 0:
        event = smoking['latest_mention']
        lines.append(f"\nSMOKING STATUS ({smoking['count']} mentions, latest {event['date']}):")
        lines.append(f"  \"{_snippet(event['text'], max_len=500)}\"")

    # ECOG — use larger snippet to capture the actual score (value often follows keyword)
    ecog = det_results.get('ecog', {})
    if ecog.get('count', 0) > 0:
        event = ecog['latest_mention']
        lines.append(f"\nECOG/PERFORMANCE STATUS ({ecog['count']} mentions, latest {event['date']}):")
        lines.append(f"  \"{_snippet(event['text'], max_len=500)}\"")

    # Radiation therapy
    radiation = det_results.get('radiation_therapy', {})
    if radiation.get('count', 0) > 0:
        lines.append(f"\nRADIATION THERAPY ({radiation['count']} mentions):")
        if radiation.get('latest_note'):
            lines.append(f"  Latest note ({radiation['latest_note']['date']}): \"{_snippet(radiation['latest_note']['text'])}\"")
        if radiation.get('latest_radiology'):
            lines.append(f"  Latest radiology ({radiation['latest_radiology']['date']}): \"{_snippet(radiation['latest_radiology']['text'])}\"")

    # Toxicity/Comorbidities
    tox = det_results.get('toxicity_comorbidities', {})
    if tox.get('count', 0) > 0:
        if tox.get('toxicities'):
            lines.append(f"\nTHERAPY TOXICITIES ({len(tox['toxicities'])} mentions):")
            for t in tox['toxicities'][:3]:
                lines.append(f"  ({t['date']}): \"{_snippet(t['text'], 200)}\"")
        if tox.get('comorbidities'):
            lines.append(f"\nCOMORBIDITIES ({len(tox['comorbidities'])} mentions):")
            for c in tox['comorbidities'][:3]:
                lines.append(f"  ({c['date']}): \"{_snippet(c['text'], 200)}\"")

    # Surgical history
    surgery = det_results.get('surgical_history', {})
    if surgery.get('count', 0) > 0:
        lines.append(f"\nSURGICAL HISTORY ({surgery['count']} mentions):")
        if surgery.get('latest_note'):
            lines.append(f"  Latest note ({surgery['latest_note']['date']}): \"{_snippet(surgery['latest_note']['text'], 400)}\"")
        if surgery.get('procedures'):
            for p in surgery['procedures']:
                lines.append(f"  Procedure ({p['date']}): \"{_snippet(p['text'], 300)}\"")

    # Code status / DNR (CRITICAL — display generator can't see middle of serialized text;
    # this surfaces the actual chart language regardless of position to prevent inversions)
    cs = det_results.get('code_status', {})
    if cs.get('count', 0) > 0:
        lines.append(f"\nCODE STATUS / DNR / POLST ({cs['count']} mentions, latest first):")
        for m in cs.get('mentions', [])[:5]:
            lines.append(f"  ({m['date']}): \"{_snippet(m['text'], 350)}\"")
    else:
        lines.append("\nCODE STATUS / DNR / POLST: no mentions found in record (default: No / Full Code).")

    # Drug exposures
    drugs = det_results.get('drug_exposures', {})
    if drugs.get('count', 0) > 0:
        lines.append(f"\nDRUG EXPOSURES ({drugs['count']} distinct):")
        for drug in drugs.get('drugs', []):
            date_str = drug['first_date'] if drug['first_date'] == drug['last_date'] else f"{drug['first_date']} to {drug['last_date']}"
            lines.append(f"  {drug['name']} ({date_str})")

    lines.append("\n" + "=" * 60)
    lines.append("Use these search hits to guide your attention when extracting")
    lines.append("clinical events from the XML below.")
    lines.append("=" * 60 + "\n")

    return '\n'.join(lines)


def _add_provenance_postprocessing(pid: str, events: list, output_dir: str, reference_map: dict) -> None:
    """
    Optional provenance post-processing: Add provenance edges using provided reference map.

    This function is called after timeline events are saved. It:
    1. Loads Lumia graph (if exists)
    2. Uses provided reference map (generated during extraction)
    3. Adds provenance edges (if events have source_event_refs)
    4. Saves enhanced graph

    Args:
        pid: Patient ID
        events: List of timeline event dictionaries
        output_dir: Output directory for JSONs
        reference_map: Dict mapping E1 → note_id (already generated during extraction)

    Side Effects:
        Creates files:
        - graphs/{pid}_toa_provenance.graphml (if refs exist)
    """
    print(f"\n  ℹ️  Provenance post-processing enabled")

    # Check if Lumia graph exists
    lumia_path = Path("graphs") / f"{pid}_lumia.graphml"
    if not lumia_path.exists():
        print(f"  ⚠️  No Lumia graph found at {lumia_path}")
        print(f"     Skipping provenance (run with Lumia graph build first)")
        return

    # Load Lumia graph
    print(f"  📊 Loading Lumia graph...")
    lumia_graph = TOAGraph.load(str(lumia_path))

    print(f"  ✅ Using reference map: {len(reference_map)} entries")

    # Check if events have source_event_refs
    events_with_refs = [e for e in events if e.get('source_event_refs')]
    if not events_with_refs:
        print(f"  ℹ️  No events with source_event_refs - skipping provenance linking")
        print(f"     (Prompts not yet configured for provenance)")
        return

    # Add provenance edges
    print(f"  🔗 Adding provenance edges ({len(events_with_refs)} events with refs)...")
    from .provenance_linking import add_timeline_events_with_provenance

    toa_graph = add_timeline_events_with_provenance(
        lumia_graph,
        events,
        reference_map,
        validate=False  # Skip validation to avoid errors
    )

    # Save enhanced graph
    toa_graph_path = Path("graphs") / f"{pid}_toa_provenance.graphml"
    toa_graph.save(str(toa_graph_path))

    stats = toa_graph.stats()
    print(f"  ✅ Provenance graph: {toa_graph_path}")
    print(f"     Nodes: {stats['total_nodes']}, Edges: {stats['total_edges']}")
