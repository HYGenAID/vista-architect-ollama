"""
RAG baseline extraction for VISTA Architect manuscript comparison.

Pipeline (per patient):
  1. Resolve TB-date-truncated XML (same ground-truth input VISTA is evaluated on).
  2. Build BM25 index over XML <entry> blocks.
  3. For each of 16 MTB variables, pose a focused natural-language query,
     retrieve top-k chunks under a token budget, and call the answer LLM
     with a minimal prompt (no context infiltration, no elaborate prompting).
  4. Serialize the 16 answers into the patient_info.json schema that
     quick_eval.py --eval-from-snapshot expects.

Usage:
    python rag_baseline_eval.py --pid 135982524 --answer-model gpt-4.1
    python rag_baseline_eval.py --pids-from clinician_validation/_pid_mapping.json \
        --output-dir temp_jsons_rag_gpt5 --answer-model gpt-5
"""
import argparse
import json
import pickle
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).parent))
import gsgpt
from eval_variables_v2 import EVAL_VARIABLES_V2, VARIABLE_NAMES
from rag_baseline_queries import QUERIES, SYSTEM_PROMPT, build_user_prompt, TIMELINE_QUERY

CACHE_DIR = Path("rag_cache")
CACHE_DIR.mkdir(exist_ok=True)


def load_pids(args) -> list[str]:
    if args.pid:
        return [args.pid]
    if args.pids_from:
        with open(args.pids_from) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return list(data.values())
        return list(data)
    raise SystemExit("Need --pid or --pids-from")


def resolve_truncated_xml(patient_id: str, cohort_name: str = None) -> Path:
    from quick_eval import _resolve_truncated_xml
    return _resolve_truncated_xml(patient_id, cohort_name)


def build_bm25(pid: str, xml_text: str, rebuild: bool = False):
    """One chunk per <entry>; tokenize on word chars.

    Deliberately does NOT apply any noise-code filtering or domain cleaning —
    a vanilla RAG baseline takes the raw record as input.
    """
    idx_path = CACHE_DIR / f"bm25_{pid}.pkl"
    if idx_path.exists() and not rebuild:
        try:
            return pickle.loads(idx_path.read_bytes())
        except Exception:
            pass

    root = ET.fromstring(xml_text)
    chunks_tokens = []
    chunks_text = []
    for entry in root.findall(".//entry"):
        ts = entry.get("timestamp", "")
        parts = [ts] if ts else []
        for ev in entry.iter():
            name = ev.get("name", "") or ""
            txt = ev.text or ""
            if name:
                parts.append(name)
            if txt.strip():
                parts.append(txt.strip())
        raw = " ".join(parts).strip()
        if not raw:
            continue
        tokens = re.findall(r"\w+", raw.lower())
        if not tokens:
            continue
        chunks_tokens.append(tokens)
        chunks_text.append(raw)

    if not chunks_tokens:
        raise RuntimeError(f"No <entry> chunks extracted for {pid}")

    bm25 = BM25Okapi(chunks_tokens)
    meta = {"chunks": chunks_text}
    idx_path.write_bytes(pickle.dumps((bm25, meta)))
    return bm25, meta


def pack_context(question: str, bm25, meta, max_tokens: int = 4096, k: int = 20) -> str:
    """Top-k BM25 retrieval; pack chunks under a whitespace-token budget."""
    scores = bm25.get_scores(re.findall(r"\w+", question.lower()))
    top_idx = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:k]
    out_parts = []
    taken = 0
    for i in top_idx:
        chunk = meta["chunks"][i]
        n = len(chunk.split())
        if taken + n > max_tokens:
            break
        out_parts.append(chunk)
        taken += n
    return "\n\n".join(out_parts)


def ask_llm(context: str, query: str, format_hint: str, model: str,
            max_retries: int = 5) -> tuple[str, float]:
    """Call the LLM with a minimal prompt. Returns (answer, elapsed_seconds).

    Reasoning models (gpt-5*) share a `max_completion_tokens` budget between
    hidden reasoning and visible output; with no cap, the API default can be
    exhausted mid-reasoning and produce empty content on large contexts. We
    pass a generous 16K cap for gpt-5* to avoid that failure mode.

    Retries on 429 (rate limit) and transient 5xx with exponential backoff.
    """
    import random
    user_prompt = build_user_prompt(context, query, format_hint)
    max_tokens = 16000 if model.startswith("gpt-5") else None
    t0 = time.time()
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = gsgpt.chat(user_prompt, model=model, system=SYSTEM_PROMPT, max_tokens=max_tokens)
            return (resp or "").strip(), time.time() - t0
        except Exception as e:
            last_err = e
            msg = str(e)
            transient = ("429" in msg) or ("503" in msg) or ("504" in msg) or ("timeout" in msg.lower())
            if attempt < max_retries - 1 and transient:
                backoff = (2 ** attempt) + random.random()
                time.sleep(backoff)
                continue
            break
    return f"[ERROR: {last_err}]", time.time() - t0


def serialize_answers_to_patient_info(answers: dict[str, str]) -> dict:
    """Assemble the per-variable RAG answers into the 3-section schema
    quick_eval.extract_variables_from_snapshot expects."""
    demo = {}
    tumor = {}
    treatments = {}

    def section(name):
        return {"PATIENT DEMOGRAPHICS": demo, "TUMOR INFORMATION": tumor, "TREATMENTS": treatments}[name]

    def _is_negative(ans: str, negatives: tuple) -> bool:
        s = (ans or "").strip().strip(".").strip().lower()
        return s in negatives

    for var_name, spec in QUERIES.items():
        ans = answers.get(var_name, "Unknown")
        sec_name, key = spec["json_location"]
        sec = section(sec_name)

        if key == "driver_mutations":
            # Dict expected; wrap RAG free-text answer so the extractor iterates it.
            if ans and not _is_negative(ans, ("not tested", "unknown", "none", "")):
                sec[key] = {"rag_answer": ans}
            else:
                sec[key] = {}
        elif key == "allergies":
            if ans and not _is_negative(ans, ("nkda", "none", "unknown", "no known drug allergies", "")):
                sec[key] = [ans]
            else:
                sec[key] = "NKDA"
        elif key in ("previous", "current"):
            if ans and not _is_negative(ans, ("no", "none", "unknown", "")):
                sec[key] = [ans]
            else:
                sec[key] = []
        else:
            sec[key] = ans

    return {
        "PATIENT DEMOGRAPHICS": demo,
        "TUMOR INFORMATION": tumor,
        "TREATMENTS": treatments,
        "_rag_baseline": True,
    }


def process_patient(pid: str, out_dir: Path, answer_model: str,
                    top_k: int, max_ctx_tokens: int, rebuild: bool,
                    build_timeline: bool = False, query_workers: int = 16) -> dict:
    print(f"\n{'='*60}\nRAG baseline: {pid} (model={answer_model})\n{'='*60}")

    xml_path = resolve_truncated_xml(pid)
    if xml_path is None or not xml_path.exists():
        print(f"  ERROR: no XML resolved for {pid}")
        return {"pid": pid, "error": "no_xml"}
    xml_text = Path(xml_path).read_text(encoding="utf-8")

    bm25, meta = build_bm25(pid, xml_text, rebuild=rebuild)
    n_chunks = len(meta["chunks"])
    print(f"  BM25 index: {n_chunks} <entry> chunks from {xml_path.name}")

    answers = {}
    per_query_latency = {}

    def _run_one(var_name):
        spec = QUERIES[var_name]
        ctx = pack_context(spec["query"], bm25, meta, max_tokens=max_ctx_tokens, k=top_k)
        ans, elapsed = ask_llm(ctx, spec["query"], spec["format_hint"], model=answer_model)
        return var_name, ans, elapsed

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=query_workers) as pool:
        for var_name, ans, elapsed in pool.map(_run_one, VARIABLE_NAMES):
            answers[var_name] = ans
            per_query_latency[var_name] = elapsed
            short_ans = ans.replace("\n", " ")[:80]
            print(f"    [{elapsed:5.2f}s] {var_name:32s} -> {short_ans}")
    wall_parallel = time.time() - t_start
    print(f"  Parallel wall time for 16 queries: {wall_parallel:.1f}s (sum-of-query-latency: {sum(per_query_latency.values()):.1f}s)")

    patient_info = serialize_answers_to_patient_info(answers)
    out_patient_dir = out_dir / pid
    out_patient_dir.mkdir(parents=True, exist_ok=True)
    with open(out_patient_dir / "patient_info.json", "w") as f:
        json.dump(patient_info, f, indent=2)

    meta_out = {
        "pid": pid,
        "answer_model": answer_model,
        "xml_source": str(xml_path),
        "xml_chars": len(xml_text),
        "n_bm25_chunks": n_chunks,
        "top_k": top_k,
        "max_ctx_tokens": max_ctx_tokens,
        "total_latency_sec": sum(per_query_latency.values()),
        "wall_time_sec": wall_parallel,
        "per_query_latency": per_query_latency,
        "answers": answers,
    }
    with open(out_patient_dir / "rag_meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    if build_timeline:
        print(f"  Building RAG timeline...")
        q = TIMELINE_QUERY["query"]
        fmt = TIMELINE_QUERY["format_hint"]
        ctx = pack_context(q, bm25, meta, max_tokens=max_ctx_tokens, k=top_k)
        ans, elapsed = ask_llm(ctx, q, fmt, model=answer_model)
        timeline_out = {
            "pid": pid,
            "answer_model": answer_model,
            "elapsed_sec": elapsed,
            "top_k": top_k,
            "max_ctx_tokens": max_ctx_tokens,
            "timeline_text": ans,
        }
        with open(out_patient_dir / "rag_timeline.json", "w") as f:
            json.dump(timeline_out, f, indent=2)
        print(f"    [{elapsed:5.2f}s] Timeline generated ({len(ans)} chars)")

    print(f"  Wrote {out_patient_dir}/patient_info.json (total {meta_out['total_latency_sec']:.1f}s)")
    return meta_out


def main():
    parser = argparse.ArgumentParser(description="RAG baseline for VISTA Architect MTB variables.")
    parser.add_argument("--pid", type=str, help="Single patient ID")
    parser.add_argument("--pids-from", type=str, help="JSON file with patient IDs (dict values or list)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write {pid}/patient_info.json")
    parser.add_argument("--answer-model", type=str, default="gpt-4.1",
                        help="LLM model for answer generation (gpt-5, gpt-4.1, etc.)")
    parser.add_argument("--top-k", type=int, default=20, help="BM25 top-k retrieval")
    parser.add_argument("--max-ctx-tokens", type=int, default=4096,
                        help="Max whitespace-token budget for retrieved context")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="Force BM25 index rebuild even if cached")
    parser.add_argument("--build-timeline", action="store_true",
                        help="Also produce a RAG-built chronological timeline per patient")
    parser.add_argument("--query-workers", type=int, default=16,
                        help="Max concurrent LLM calls within a patient (default: 16, one per variable)")
    parser.add_argument("--patient-workers", type=int, default=4,
                        help="Max patients processed in parallel (default: 4)")
    args = parser.parse_args()

    pids = load_pids(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    t_start = time.time()

    def _run_patient(pid):
        try:
            return process_patient(
                pid, out_dir,
                answer_model=args.answer_model,
                top_k=args.top_k,
                max_ctx_tokens=args.max_ctx_tokens,
                rebuild=args.rebuild_index,
                build_timeline=args.build_timeline,
                query_workers=args.query_workers,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"pid": pid, "error": str(e)}

    if args.patient_workers <= 1:
        for pid in pids:
            summary.append(_run_patient(pid))
    else:
        with ThreadPoolExecutor(max_workers=args.patient_workers) as pool:
            for res in pool.map(_run_patient, pids):
                summary.append(res)

    total_elapsed = time.time() - t_start
    summary_path = out_dir / "_run_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "answer_model": args.answer_model,
            "n_patients": len(pids),
            "total_elapsed_sec": total_elapsed,
            "top_k": args.top_k,
            "max_ctx_tokens": args.max_ctx_tokens,
            "per_patient": summary,
        }, f, indent=2)
    print(f"\nDone: {len(pids)} patients in {total_elapsed/60:.1f} min")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
