#!/usr/bin/env python3
"""Episode synthesis on the unified timeline.

Reads:
    runs/{pid}/timeline_objects.jsonl
    runs/{pid}/background.json

Writes:
    runs/{pid}/episodes.json
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

PROMPT_PATH = ROOT / "agentic_toa" / "prompts" / "episode_synthesis_v2.txt"


def extract_first_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
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
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--thinking", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--runs-root", default=str(ROOT / "agentic_toa" / "runs"))
    args = ap.parse_args()

    runs_dir = Path(args.runs_root) / args.pid
    timeline_path = runs_dir / "timeline_objects.jsonl"
    background_path = runs_dir / "background.json"
    if not timeline_path.exists():
        print(f"ERROR: {timeline_path} not found. Run unify_timeline.py first.", file=sys.stderr)
        return 1

    events = []
    for i, line in enumerate(timeline_path.read_text().splitlines()):
        line = line.strip()
        if line:
            ev = json.loads(line)
            ev["event_id"] = i
            events.append(ev)

    background = json.loads(background_path.read_text()) if background_path.exists() else {}

    # Compose prompt
    prompt_template = PROMPT_PATH.read_text()
    prompt = (
        prompt_template
        .replace("{background_block}", json.dumps(background, indent=2))
        .replace("{timeline_events}", json.dumps(events, indent=2))
    )

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

    try:
        obj = extract_first_json(response)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"ERROR: failed to parse JSON: {e}", file=sys.stderr)
        print(f"--- raw response (first 800 chars) ---\n{response[:800]}", file=sys.stderr)
        return 3

    if "episodes" not in obj:
        print("ERROR: response missing 'episodes' key", file=sys.stderr)
        return 4

    # Auto-populate event_ids by date range (consistent with existing pipeline)
    for ep in obj["episodes"]:
        sd = ep.get("start_date")
        ed = ep.get("end_date")
        kind = ep.get("kind")
        if kind == "baseline":
            # baseline holds all baseline_information events regardless of date
            ep["event_ids"] = [e["event_id"] for e in events
                               if e.get("type") == "baseline_information"]
        else:
            ep["event_ids"] = [
                e["event_id"] for e in events
                if sd and (ed is None or e.get("date", "9999") <= ed) and e.get("date", "0000") >= sd
            ]

    obj["_meta"] = {
        "pid": args.pid,
        "model": args.model,
        "thinking_budget": args.thinking,
        "elapsed_sec": round(elapsed, 2),
        "n_episodes": len(obj["episodes"]),
        "n_timeline_events": len(events),
    }
    out_path = runs_dir / "episodes.json"
    out_path.write_text(json.dumps(obj, indent=2))
    print(f"OK: pid={args.pid} episodes={len(obj['episodes'])} elapsed={elapsed:.1f}s -> {out_path}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
