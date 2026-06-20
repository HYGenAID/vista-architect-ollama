"""
Graph Feeding Pipeline — Load graph from store, serialize, prepare contexts.

Orchestrates the graph-first extraction path:
    1. Load pre-built Lumia graph from CohortGraphStore
    2. Serialize to compact clinical text with E-references
    3. Chunk into LLM-sized pieces
    4. Generate deterministic search hints
    5. Return everything ready for engine.extract()

Usage:
    >>> from toa.graph_feeding import prepare_from_graph
    >>> prepared = prepare_from_graph("136020661")
    >>> # prepared['text_chunks'] -> List[str] for engine.extract()
    >>> # prepared['reference_map'] -> {'E1': 'pid_note_12345', ...}
    >>> # prepared['deterministic_context'] -> str for LLM attention guidance
    >>> # prepared['final_chunk'] -> last chunk for Step 2 JSON generation
"""

from typing import Any, Dict, Optional

from toa.graph_store import CohortGraphStore
from toa.graph_serializer import serialize_graph, chunk_serialized_text


def prepare_from_graph(
    patient_id: str,
    store_path: str = "graph_store",
    max_chunk_chars: int = 120_000,
    max_final_chunk_chars: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Load graph from store and prepare all inputs for extraction pipeline.

    Args:
        patient_id: Patient ID
        store_path: Path to CohortGraphStore directory
        max_chunk_chars: Max characters per early text chunk
        max_final_chunk_chars: Max characters for final chunk (default: same as max_chunk_chars)

    Returns:
        Dict with keys:
            text_chunks: List[str] - pre-serialized chunks for engine.extract()
            final_chunk: str - last chunk (for Step 2 JSON generation)
            deterministic_context: Optional[str] - search hits for LLM guidance
            reference_map: Dict[str, str] - E1 -> node_id mapping
            graph: TOAGraph - loaded Lumia graph
            metadata: Dict - patient metadata from store index
    """
    store = CohortGraphStore(store_path)

    # 1. Load pre-built graph
    print(f"  Loading graph from store: {store_path}/graphs/{patient_id}.graphml")
    graph = store.load_patient_graph(patient_id)
    metadata = store.get_patient_metadata(patient_id) or {}

    node_count = graph.G.number_of_nodes()
    print(f"  Graph loaded: {node_count} nodes")

    # 2. Serialize graph to compact clinical text
    print(f"  Serializing graph to clinical text...")
    serialized_text, reference_map = serialize_graph(graph, patient_id)
    print(f"  Serialized: {len(serialized_text):,} chars, {len(reference_map)} E-references")

    # 3. Chunk into LLM-sized pieces
    text_chunks = chunk_serialized_text(serialized_text, max_chunk_chars=max_chunk_chars, max_final_chunk_chars=max_final_chunk_chars)
    print(f"  Chunked into {len(text_chunks)} piece(s)")

    # 4. Generate deterministic search context
    deterministic_context = None
    try:
        from toa.deterministic_retrieval import DeterministicRetriever
        from toa.backend import _format_deterministic_guidance

        retriever = DeterministicRetriever(graph)
        det_results = retriever.get_comprehensive_context()
        deterministic_context = _format_deterministic_guidance(det_results)
        print(f"  Deterministic search hints generated")
    except Exception as e:
        print(f"  Deterministic retrieval failed (non-fatal): {e}")

    # 5. Assemble result — ensure demographics are always in final_chunk
    final_chunk = text_chunks[-1] if text_chunks else serialized_text

    # For multi-chunk patients, demographics end up in chunk 0 (first) but
    # Step 2 JSON generation only receives final_chunk (last). Prepend
    # the demographics block so DOB/Sex are always visible to Step 2.
    if len(text_chunks) > 1 and serialized_text.startswith("=== DEMOGRAPHICS ==="):
        next_block = serialized_text.find("\n=== ", len("=== DEMOGRAPHICS ==="))
        if next_block > 0:
            demo_block = serialized_text[:next_block].strip()
            if not final_chunk.startswith("=== DEMOGRAPHICS"):
                final_chunk = demo_block + "\n\n" + final_chunk
                print(f"  Prepended demographics to final_chunk ({len(demo_block)} chars)")

    return {
        "text_chunks": text_chunks,
        "final_chunk": final_chunk,
        "deterministic_context": deterministic_context,
        "reference_map": reference_map,
        "graph": graph,
        "metadata": metadata,
    }
