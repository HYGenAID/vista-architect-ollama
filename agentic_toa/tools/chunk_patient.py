#!/usr/bin/env python3
"""Plan chunks for an agentic-pipeline patient run.

Serializes the sandboxed Lumia graph (`agentic_toa/temp_graphs/{pid}.graphml`),
splits into character-bounded chunks (default 120K), and writes:
  - runs/{pid}/serialized_text.txt          (full serialized text, with [E1]..[EN] refs)
  - runs/{pid}/reference_map.json           (E1 -> node_id map for provenance)
  - runs/{pid}/chunks/chunk_<i>.txt         (one file per chunk)
  - runs/{pid}/chunk_plan.json              (chunk metadata)

The chunk_plan.json shape:
  {
    "pid": "...",
    "n_chunks": 3,
    "chunk_chars": 120000,
    "chunks": [
      {"index": 0, "path": "chunks/chunk_0.txt", "char_count": 115342, "first_eref": "E1", "last_eref": "E142"},
      ...
    ]
  }
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from toa.graph_store import CohortGraphStore
from toa.graph_serializer import serialize_graph, chunk_serialized_text


def first_last_eref(chunk_text: str) -> tuple[str | None, str | None]:
    refs = re.findall(r"\[E(\d+)\]", chunk_text)
    if not refs:
        return None, None
    return f"E{refs[0]}", f"E{refs[-1]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pid", required=True)
    ap.add_argument("--chunk-chars", type=int, default=120_000)
    ap.add_argument("--max-final-chunk-chars", type=int, default=None,
                    help="Final-chunk char cap (default: same as --chunk-chars)")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="Optional cap for development; truncates the planned chunks list")
    ap.add_argument("--graph-store-root", default=str(ROOT / "agentic_toa" / "temp_graphs"),
                    help="Where the sandboxed graphml lives")
    ap.add_argument("--runs-root", default=str(ROOT / "agentic_toa" / "runs"))
    args = ap.parse_args()

    pid = args.pid
    runs_dir = Path(args.runs_root) / pid
    chunks_dir = runs_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Load sandboxed graph via a CohortGraphStore-like pattern, but pointed at temp_graphs/.
    # CohortGraphStore expects a graphs/ subdir; the simplest reliable path is to use TOAGraph.load directly.
    from toa.graph import TOAGraph
    graphml_path = Path(args.graph_store_root) / f"{pid}.graphml"
    if not graphml_path.exists():
        print(f"ERROR: sandboxed graph not found: {graphml_path}", file=sys.stderr)
        print(f"Hint: run `python tools/copy_to_temp_graphs.py --pid {pid}` first.", file=sys.stderr)
        return 1
    graph = TOAGraph.load(str(graphml_path))

    serialized_text, ref_map = serialize_graph(graph, pid)
    (runs_dir / "serialized_text.txt").write_text(serialized_text)
    (runs_dir / "reference_map.json").write_text(json.dumps(ref_map, indent=2))

    chunks = chunk_serialized_text(
        serialized_text,
        max_chunk_chars=args.chunk_chars,
        max_final_chunk_chars=args.max_final_chunk_chars,
    )
    if args.max_chunks is not None:
        chunks = chunks[: args.max_chunks]

    chunk_records = []
    for i, chunk in enumerate(chunks):
        path = chunks_dir / f"chunk_{i}.txt"
        path.write_text(chunk)
        first_e, last_e = first_last_eref(chunk)
        chunk_records.append({
            "index": i,
            "path": str(path.relative_to(runs_dir)),
            "char_count": len(chunk),
            "first_eref": first_e,
            "last_eref": last_e,
        })

    plan = {
        "pid": pid,
        "n_chunks": len(chunks),
        "chunk_chars": args.chunk_chars,
        "max_final_chunk_chars": args.max_final_chunk_chars or args.chunk_chars,
        "serialized_text_chars": len(serialized_text),
        "reference_map_size": len(ref_map),
        "chunks": chunk_records,
    }
    (runs_dir / "chunk_plan.json").write_text(json.dumps(plan, indent=2))
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
