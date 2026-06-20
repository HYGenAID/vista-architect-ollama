#!/usr/bin/env python3
"""Parallel-by-section display generator (v3).

The single-call display generator has to fill ~25 fields + a 4-line summary in
ONE LLM call. Empirical observation: a judge that scores ONE variable at a time
catches mistakes the same-model extractor makes when juggling all 25 at once.
This is the focus asymmetry. v3 narrows the extractor's focus to match.

Architecture:
  - 3 PARALLEL LLM calls, one per JSON section (PATIENT DEMOGRAPHICS, TUMOR
    INFORMATION, TREATMENTS). Each gets the full evidence packet but is asked
    to produce ONLY its section's fields.
  - 1 SEQUENTIAL summary call: takes the merged patient_info and writes the
    4-line tumor board note.

Outputs are identical in shape to v2 (patient_info.json + summary.json), so
downstream evaluators see no change.

Usage:
    python agentic_toa/tools/generate_displays_v3.py --pid 136093825 \
        --model claude-opus-4-6
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import gsgpt
from toa.graph_store import CohortGraphStore
from toa.deterministic_retrieval import DeterministicRetriever
from toa.backend import _format_deterministic_guidance
from toa.graph import TOAGraph

V2_PROMPT_PATH = ROOT / "agentic_toa" / "prompts" / "display_items_v2.txt"
V3_SECTION_PROMPT_PATH = ROOT / "agentic_toa" / "prompts" / "display_section_v3.txt"
V3_SUMMARY_PROMPT_PATH = ROOT / "agentic_toa" / "prompts" / "display_summary_v3.txt"

SECTIONS = ["PATIENT DEMOGRAPHICS", "TUMOR INFORMATION", "TREATMENTS"]


def parse_json_response(text: str) -> dict:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif text.startswith("```"):
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in response (first 200 chars: {text[:200]!r})")
    return json.loads(m.group(0))


def extract_section_schema(full_prompt: str, section: str) -> str:
    """Pull out the section's schema definition from the v2 prompt.

    The v2 prompt has sections delimited by:
      PATIENT DEMOGRAPHICS:
        ... fields ...
      TUMOR INFORMATION:
        ... fields ...
      TREATMENTS:
        ... fields ...
    """
    other = [s for s in SECTIONS if s != section]
    # Find start of this section
    start_re = re.compile(rf"^{re.escape(section)}:$", re.MULTILINE)
    m = start_re.search(full_prompt)
    if not m:
        return f"(section {section} schema not found)"
    start = m.start()
    # Find end: start of next section or end of schema
    end_idx = len(full_prompt)
    for s in other:
        m2 = re.search(rf"^{re.escape(s)}:$", full_prompt[start+1:], re.MULTILINE)
        if m2:
            cand = start + 1 + m2.start()
            if cand < end_idx:
                end_idx = cand
    # Also stop at "=== TUMOR BOARD NOTE FORMAT ===" or any other === block
    end_marker = re.search(r"^=== ", full_prompt[start+1:], re.MULTILINE)
    if end_marker:
        cand = start + 1 + end_marker.start()
        if cand < end_idx:
            end_idx = cand
    return full_prompt[start:end_idx].rstrip()


def build_evidence(args, runs_dir, graph):
    retriever = DeterministicRetriever(graph)
    det_context = _format_deterministic_guidance(retriever.get_comprehensive_context())
    ct_dates = retriever.get_ct_date_vector()

    demographics = {}
    demo_path = runs_dir / "demographics.json"
    if demo_path.exists():
        demographics = json.loads(demo_path.read_text())

    timeline_path = runs_dir / "timeline_objects.jsonl"
    events = []
    for i, line in enumerate(timeline_path.read_text().splitlines()):
        line = line.strip()
        if line:
            ev = json.loads(line); ev["event_id"] = i
            events.append(ev)

    background = json.loads((runs_dir / "background.json").read_text())
    episodes   = json.loads((runs_dir / "episodes.json").read_text())

    source_text = ""
    if args.include_source_text:
        st_path = runs_dir / "serialized_text.txt"
        if st_path.exists():
            full = st_path.read_text()
            source_text = full[-args.source_text_tail_chars:] if len(full) > args.source_text_tail_chars else full
    if not source_text:
        source_text = "[source_text omitted]"

    return {
        "demographics": demographics, "events": events, "background": background,
        "episodes": episodes, "det_context": det_context, "ct_dates": ct_dates,
        "source_text": source_text,
    }


SECTION_PROMPT_TEMPLATE = """You are a highest-expert-level oncology data assistant working at Stanford Tumor Board meetings (subtype: {tumor_type}).

You are producing ONLY one section of the patient_info JSON: **{focus_section}**.

A different parallel LLM call is filling in the other sections. Stay focused — do not waste tokens producing fields outside your section. Your output MUST be a single JSON object with exactly ONE top-level key: "{focus_section}".

=== INPUTS PROVIDED ===

1) demographics (deterministic hard facts from the Person node):
{demographics_block}

2) timeline (unified, deduplicated events with provenance refs):
{timeline_context}

3) episodes (treatment episodes derived from the timeline):
{episodes_context}

4) background (pre-cancer baseline merged across chunks, incl. prior_surgeries[]):
{background_block}

5) deterministic_context (graph-search hits for key clinical variables):
{deterministic_context}

6) imaging_date_vector (Stanford OMOP-derived in-house imaging dates — sanity-check signal, NOT source of truth; the canonical CT history is in the unified timeline):
{ct_date_vector}

7) source_text (last 150k chars of serialized graph text — narrative semantics):
{source_text}

=== YOUR FOCUS: {focus_section} ===

{focus_section_schema}

=== STRICT OUTPUT FORMAT ===

Return JSON only, enclosed in ```json ... ``` markers, with EXACTLY this shape:

```json
{{
  "{focus_section}": {{
    ...fields per schema above...
  }}
}}
```

No other top-level keys. No prose. No commentary."""


SUMMARY_PROMPT_TEMPLATE = """You are writing a 4-line Stanford Epic tumor board note for the following patient. The note is shown verbatim on the dashboard.

=== PATIENT INFO (already extracted) ===
```json
{patient_info}
```

=== STRICT OUTPUT FORMAT ===

Return JSON only:
```json
{{"summary": "AIGen: [LASTNAME]: [AGEGENDER] with h/o [CANCER TYPE incl. detailed pathology, staging, molecular features]\\nPrior therapy: [DETAILED therapy history with dates/responses/toxicities]\\nTumor board question: [SPECIFIC clinical question]\\nTumor board decision: [DETAILED decisions]"}}
```

CRITICAL:
- Use ONLY information from the patient_info above. Do not invent.
- Target 750-999 characters total.
- 4 lines separated by \\n.
- Be detailed and specific."""


def fill_section_prompt(focus_section: str, evidence: dict, schema_blob: str, tumor_type: str) -> str:
    return SECTION_PROMPT_TEMPLATE.format(
        focus_section=focus_section,
        focus_section_schema=schema_blob,
        tumor_type=tumor_type,
        demographics_block=json.dumps(evidence["demographics"], indent=2),
        timeline_context=json.dumps(evidence["events"], indent=2),
        episodes_context=json.dumps(evidence["episodes"].get("episodes", []), indent=2),
        background_block=json.dumps(evidence["background"], indent=2),
        deterministic_context=evidence["det_context"],
        ct_date_vector=json.dumps(evidence["ct_dates"]),
        source_text=evidence["source_text"],
    )


def run_section(section: str, prompt: str, model: str, max_tokens: int) -> tuple[str, dict, float]:
    t0 = time.time()
    resp = gsgpt.chat(prompt, model=model, max_tokens=max_tokens)
    obj = parse_json_response(resp)
    if section not in obj:
        raise ValueError(f"section '{section}' missing in response (keys: {list(obj.keys())})")
    return section, obj[section], time.time() - t0


def run_summary(patient_info: dict, model: str, max_tokens: int) -> tuple[str, float]:
    t0 = time.time()
    prompt = SUMMARY_PROMPT_TEMPLATE.format(patient_info=json.dumps(patient_info, indent=2))
    resp = gsgpt.chat(prompt, model=model, max_tokens=max_tokens)
    obj = parse_json_response(resp)
    return obj.get("summary", ""), time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pid", required=True)
    ap.add_argument("--model", default="claude-opus-4-6")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--tumor-type", default="Thoracic Oncology")
    ap.add_argument("--include-source-text", action="store_true", default=True)
    ap.add_argument("--no-source-text", dest="include_source_text", action="store_false")
    ap.add_argument("--source-text-tail-chars", type=int, default=150_000)
    ap.add_argument("--output-suffix", default="_v3")
    ap.add_argument("--runs-root", default=str(ROOT / "agentic_toa" / "runs"))
    ap.add_argument("--graph-store-root", default=str(ROOT / "agentic_toa" / "temp_graphs"))
    args = ap.parse_args()

    runs_dir = Path(args.runs_root) / args.pid
    graphml_path = Path(args.graph_store_root) / f"{args.pid}.graphml"
    if not graphml_path.exists():
        print(f"ERROR: no graphml at {graphml_path}", file=sys.stderr); return 2
    graph = TOAGraph.load(str(graphml_path))

    evidence = build_evidence(args, runs_dir, graph)
    v2_prompt = V2_PROMPT_PATH.read_text()
    section_schemas = {s: extract_section_schema(v2_prompt, s) for s in SECTIONS}

    # 3 parallel section calls
    t_total = time.time()
    section_results = {}
    with ThreadPoolExecutor(max_workers=len(SECTIONS)) as pool:
        futures = {
            pool.submit(run_section, s,
                        fill_section_prompt(s, evidence, section_schemas[s], args.tumor_type),
                        args.model, args.max_tokens): s
            for s in SECTIONS
        }
        for fut in as_completed(futures):
            try:
                section, obj, dt = fut.result()
                section_results[section] = (obj, dt)
                print(f"  ✓ {section:<22} {dt:.1f}s", file=sys.stderr)
            except Exception as e:
                section = futures[fut]
                print(f"  ✗ {section} FAILED: {e}", file=sys.stderr)
                section_results[section] = ({}, 0.0)
    t_sections = time.time() - t_total

    patient_info = {s: section_results[s][0] for s in SECTIONS}

    # Sequential summary
    try:
        summary_text, t_sum = run_summary(patient_info, args.model, args.max_tokens)
        print(f"  ✓ summary               {t_sum:.1f}s", file=sys.stderr)
    except Exception as e:
        print(f"  ✗ summary FAILED: {e}", file=sys.stderr)
        summary_text, t_sum = "", 0.0

    t_elapsed = time.time() - t_total
    sfx = args.output_suffix
    pi_name = f"patient_info{sfx}.json"
    sm_name = f"summary{sfx}.json"
    (runs_dir / pi_name).write_text(json.dumps(patient_info, indent=2))
    (runs_dir / sm_name).write_text(json.dumps({"summary": summary_text}, indent=2))
    print(f"OK: pid={args.pid} elapsed={t_elapsed:.1f}s "
          f"(sections {t_sections:.1f}s parallel, summary {t_sum:.1f}s) "
          f"-> {pi_name} + {sm_name}", file=sys.stderr)
    if summary_text:
        print(f"  TB note ({len(summary_text)} chars):", file=sys.stderr)
        print(f"  {summary_text[:300]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
