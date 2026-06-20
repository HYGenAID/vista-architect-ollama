#!/usr/bin/env python3
"""Run per-chunk event extraction for the agentic TOA pipeline.

Reads one chunk text file (produced by chunk_patient.py), injects the patient's
graph-search context (from toa.deterministic_retrieval), calls a Vertex LLM
(default: gemini-3.5-flash) with the chunk_events_v2 prompt, and emits a
validated JSON file:

    runs/{pid}/chunks/chunk_<i>.json

Schema (see prompts/chunk_events_v2.txt):
    {
      "events": [...],
      "before_chunk_mentions": [...],
      "background": {...}
    }

Exit non-zero on extraction or parse failure (so the orchestrator can retry).
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import gsgpt
from toa.graph import TOAGraph
from toa.deterministic_retrieval import DeterministicRetriever
from toa.backend import _format_deterministic_guidance

PROMPT_PATH = ROOT / "agentic_toa" / "prompts" / "chunk_events_v2.txt"


def extract_first_json(text: str) -> dict:
    """Pull the first balanced JSON object from a possibly noisy LLM response."""
    text = text.strip()
    # Strip code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Find first balanced { ... }
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object in response")
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("Unbalanced JSON braces in response")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pid", required=True)
    ap.add_argument("--chunk", type=int, required=True, help="Chunk index")
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--thinking", type=int, default=0,
                    help="thinking_budget for the LLM call (0 = no reasoning, Flash speed)")
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--runs-root", default=str(ROOT / "agentic_toa" / "runs"))
    ap.add_argument("--graph-store-root", default=str(ROOT / "agentic_toa" / "temp_graphs"))
    args = ap.parse_args()

    runs_dir = Path(args.runs_root) / args.pid
    chunks_dir = runs_dir / "chunks"
    chunk_text_path = chunks_dir / f"chunk_{args.chunk}.txt"
    if not chunk_text_path.exists():
        print(f"ERROR: {chunk_text_path} not found. Run chunk_patient.py first.", file=sys.stderr)
        return 1

    # Graph context (deterministic retrieval)
    graphml_path = Path(args.graph_store_root) / f"{args.pid}.graphml"
    if not graphml_path.exists():
        print(f"ERROR: {graphml_path} not found.", file=sys.stderr)
        return 1
    graph = TOAGraph.load(str(graphml_path))
    retriever = DeterministicRetriever(graph)
    det_context = _format_deterministic_guidance(retriever.get_comprehensive_context())

    # Compose prompt
    chunk_text = chunk_text_path.read_text()
    prompt_template = PROMPT_PATH.read_text()
    prompt = prompt_template.replace("{graph_context}", det_context).replace("{chunk_text}", chunk_text)

    # Call LLM
    t0 = time.time()
    response = gsgpt.chat(
        prompt,
        model=args.model,
        max_tokens=args.max_tokens,
        thinking_budget=args.thinking,
    )
    elapsed = time.time() - t0

    if not response:
        print(f"ERROR: empty response from {args.model} (elapsed {elapsed:.1f}s)", file=sys.stderr)
        return 2

    # Parse
    try:
        obj = extract_first_json(response)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"ERROR: failed to parse JSON: {e}", file=sys.stderr)
        print(f"--- raw response (first 500 chars) ---\n{response[:500]}", file=sys.stderr)
        return 3

    # Light schema sanity
    for key in ("events", "before_chunk_mentions", "background"):
        if key not in obj:
            print(f"WARN: missing key '{key}' in response — defaulting to empty.", file=sys.stderr)
            obj[key] = [] if key != "background" else {}

    # Tag the chunk
    obj["_chunk_meta"] = {
        "pid": args.pid,
        "chunk": args.chunk,
        "model": args.model,
        "thinking_budget": args.thinking,
        "elapsed_sec": round(elapsed, 2),
        "chunk_text_chars": len(chunk_text),
        "n_events": len(obj.get("events", [])),
        "n_before_chunk": len(obj.get("before_chunk_mentions", [])),
    }

    out_path = chunks_dir / f"chunk_{args.chunk}.json"
    out_path.write_text(json.dumps(obj, indent=2))
    print(f"OK: pid={args.pid} chunk={args.chunk} events={obj['_chunk_meta']['n_events']} "
          f"before_chunk={obj['_chunk_meta']['n_before_chunk']} elapsed={elapsed:.1f}s -> {out_path}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
