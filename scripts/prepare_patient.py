#!/usr/bin/env python3
"""
Prepare Patient CLI - Complete dashboard preparation pipeline.

This tool orchestrates the full patient data preparation pipeline:
1. TOA Timeline Extraction: Extract structured timeline from EHR XML
   - Chunk extraction: GPT-4.1 (fast, granular event extraction)
   - Episode generation: GPT-4.1 (fast clinical narrative synthesis)
2. Patient Info + Note: Combined patient_info.json + summary.json generation
   - Uses GPT-5 (better clinical reasoning for driver mutations, TNM, note quality)
   - Single LLM call produces both outputs

LLM routing: routed through the pluggable vista_llm backend (configured via
the VISTA_LLM_BACKEND env var — see SETUP.md).
Provenance linking is enabled by default (MEDS graph build + source refs).

This is the ONE COMMAND to prepare a patient for the dashboard.

Examples:
    # Prepare single patient (GPT-4.1 chunks/episodes, GPT-5 info+note - default)
    python prepare_patient.py --pid toa_demo1

    # Prepare with custom models
    python prepare_patient.py --pid demo1 --chunk-model gpt-4.1 --episode-model gpt-4.1 --info-model gpt-5

    # Batch preparation from cohort manifest
    python prepare_patient.py --cohort cohort1_manifest.json

    # Force regeneration (ignore existing cache)
    python prepare_patient.py --pid demo1 --force
"""

import argparse
import sys
import os
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import TUMOR_TYPE, NOISE_CODES_PATH, MAX_TOKENS, MODEL_MAX_TOKENS, AVAILABLE_MODELS
from toa.backend import check_patient_ready, extract_patient_from_file, extract_patient_from_graph
from toa.io.ehr_xml import iter_chunks
import gsgpt


# ========== Configuration ==========

# Default model assignments
DEFAULT_CHUNK_MODEL = "gemma4:31b-it-q8_0"      # Fast chunk extraction
DEFAULT_EPISODE_MODEL = "gemma4:31b-it-q8_0"    # Fast episode synthesis (changed from gpt-5 for batched processing)
DEFAULT_JSON_MODEL = "gemma4:31b-it-q8_0"       # Fast retrieval-based JSON generation
DEFAULT_INFO_MODEL = "gemma4:31b-it-q8_0"         # Combined patient_info + note (benefits from stronger reasoning)

# Retry settings
MAX_RETRIES_TIMEOUT = 3          # Timeouts: give up sooner (likely a real problem)
MAX_RETRIES_RATE_LIMIT = 15      # 429s: keep trying — shared API key, bucket will refill
RETRY_WAIT_BASE = 30             # Base wait (seconds), increases with backoff
RETRY_WAIT_MAX = 120             # Cap backoff at 2 minutes
REQUEST_TIMEOUT = 300            # seconds


# ========== Chat Client Factory ==========

def create_chat_client(model: str):
    """
    Create a chat client routed through the pluggable vista_llm backend.

    Args:
        model: Model name (gpt-4.1, gpt-5, etc.)

    Returns:
        Chat client function with signature (system_prompt, user_prompt) -> str
    """
    max_tokens = MODEL_MAX_TOKENS.get(model, MAX_TOKENS)

    def chat_client(system_prompt: str, user_prompt: str) -> str:
        """Send prompt via gsgpt and return response.

        Retry strategy:
        - 429 rate limits: keep retrying with exponential backoff (up to ~15 min).
          Shared API key means the quota will refill — just wait it out.
        - Timeouts: retry a few times, then give up (likely a real connection issue).
        """
        max_attempts = max(MAX_RETRIES_TIMEOUT, MAX_RETRIES_RATE_LIMIT) + 1
        timeout_attempts = 0
        rate_limit_attempts = 0

        for attempt in range(max_attempts):
            try:
                return gsgpt.chat(
                    user_prompt,
                    system=system_prompt if system_prompt else None,
                    model=model,
                    max_tokens=max_tokens
                )
            except Exception as e:
                error_str = str(e)
                is_timeout = 'timeout' in error_str.lower() or 'timed out' in error_str.lower()
                is_rate_limit = '429' in error_str

                if is_rate_limit:
                    rate_limit_attempts += 1
                    if rate_limit_attempts > MAX_RETRIES_RATE_LIMIT:
                        raise
                    wait = min(RETRY_WAIT_BASE * (1 + rate_limit_attempts * 0.5), RETRY_WAIT_MAX)
                    print(f"  Rate limit hit, waiting {wait:.0f}s... (attempt {rate_limit_attempts}/{MAX_RETRIES_RATE_LIMIT})")
                    time.sleep(wait)
                elif is_timeout:
                    timeout_attempts += 1
                    if timeout_attempts > MAX_RETRIES_TIMEOUT:
                        raise
                    wait = RETRY_WAIT_BASE
                    print(f"  Request timed out, retrying in {wait}s... ({timeout_attempts}/{MAX_RETRIES_TIMEOUT})")
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError("Failed to get LLM response after retries")

    return chat_client


# ========== Prompt & JSON Helpers ==========

PROMPTS_DIR = "prompts"

def load_prompt(fn: str) -> str:
    """Load prompt from prompts directory with TUMOR_TYPE substitution."""
    with open(os.path.join(PROMPTS_DIR, fn), encoding="utf-8") as f:
        prompt = f.read().replace("{TUMOR_TYPE}", TUMOR_TYPE)

        # Escape all literal { } to {{ }}
        prompt = prompt.replace('{', '{{').replace('}', '}}')

        # Unescape known dynamic placeholders
        known_placeholders = [
            'current_timeline', 'current_summary', 'xml_chunk', 'is_final_chunk',
            'current_patient_info', 'current_treatment_options', 'TUMOR_TYPE',
            'timeline_context'
        ]
        for ph in known_placeholders:
            prompt = prompt.replace('{{' + ph + '}}', '{' + ph + '}')

        return prompt


def load_prompts() -> Dict[str, str]:
    """Load all prompts from prompts directory."""
    prompts = {}
    for f in os.listdir(PROMPTS_DIR):
        if f.endswith(".txt") and f not in ["chat_system.txt", "tumor_board_note.txt", "patient_info_and_note.txt"]:
            prompts[os.path.splitext(f)[0]] = load_prompt(f)
    return prompts


def fix_patient_info_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise patient_info section keys to the Finnish names the app expects:
    "POTILAAN PERUSTIEDOT", "RAAJAN JA VERISUONTEN TILA", "HOIDOT".

    Tolerates underscores, mixed spaces/underscores and stray whitespace
    (LLMs produce all of these), and maps the old English section names to
    Finnish as a fallback in case the model ignores the language instruction.
    Also strips leading/trailing whitespace from nested keys.
    """
    # Canonical section names (with spaces, as written in the prompt)
    SECTION_PERUSTIEDOT = "POTILAAN PERUSTIEDOT"
    SECTION_RAAJA = "RAAJAN JA VERISUONTEN TILA"
    SECTION_HOIDOT = "HOIDOT"

    # Keys are compared after: strip, upper-case, underscores -> spaces, collapse spaces
    key_mapping = {
        # Finnish (canonical + variants)
        "POTILAAN PERUSTIEDOT": SECTION_PERUSTIEDOT,
        "PERUSTIEDOT": SECTION_PERUSTIEDOT,
        "RAAJAN JA VERISUONTEN TILA": SECTION_RAAJA,
        "VERISUONTEN TILA": SECTION_RAAJA,
        "RAAJAN TILA": SECTION_RAAJA,
        "HOIDOT": SECTION_HOIDOT,
        # English fallbacks (old schema / model slipped back to English)
        "PATIENT DEMOGRAPHICS": SECTION_PERUSTIEDOT,
        "LIMB AND VASCULAR STATUS": SECTION_RAAJA,
        "VASCULAR STATUS": SECTION_RAAJA,
        "TREATMENTS": SECTION_HOIDOT,
    }

    def _norm(key: str) -> str:
        return re.sub(r"\s+", " ", key.strip().upper().replace("_", " "))

    fixed = {}
    for key, value in data.items():
        fixed_key = key_mapping.get(_norm(key), key.strip())
        # Also strip whitespace from keys in nested dicts
        if isinstance(value, dict):
            value = {k.strip(): v for k, v in value.items()}
        fixed[fixed_key] = value

    return fixed


# ========== Pipeline Steps ==========

def step_toa_extraction(
    pid: str,
    xml_path: str,
    output_dir: str,
    chunk_model: str,
    episode_model: str,
    force: bool = False,
    force_clean: bool = False,
    enable_provenance: bool = False,
    from_graph_store: bool = False,
    store_path: str = "graph_store",
    max_chunk_chars: int = 120_000,
    max_final_chunk_chars: Optional[int] = None,
):
    """
    Step 1: TOA Timeline Extraction.

    Args:
        pid: Patient ID
        xml_path: Path to patient XML file
        output_dir: Output directory for JSONs
        chunk_model: Model for chunk extraction (default: gpt-4.1)
        episode_model: Model for episode generation (default: gpt-5)
        force: Force regeneration even if cache exists (keeps old files)
        force_clean: Force clean regeneration (removes temp_jsons and graphs first)
        enable_provenance: Build Lumia graph for provenance tracking
        from_graph_store: Use pre-built graph from CohortGraphStore
        store_path: Path to graph store directory

    Returns:
        True if from_graph_store=False (backward compat),
        or (True, final_chunk) if from_graph_store=True (for Step 2).
        False on failure.
    """
    print(f"\n{'='*80}")
    print("STEP 1: TOA TIMELINE EXTRACTION")
    print(f"{'='*80}")
    print(f"  Patient: {pid}")
    print(f"  Chunk extraction model: {chunk_model}")
    print(f"  Episode generation model: {episode_model}")
    print()

    # Clean if force_clean
    if force_clean:
        import shutil
        print("🧹 Force clean enabled - removing existing files...")

        # Remove temp_jsons folder
        temp_json_dir = Path(output_dir)
        if temp_json_dir.exists():
            shutil.rmtree(temp_json_dir)
            print(f"  ✅ Removed: {temp_json_dir}")

        # Remove graph files
        graphs_dir = Path("graphs")
        if graphs_dir.exists():
            for graph_file in graphs_dir.glob(f"{pid}*.graphml"):
                graph_file.unlink()
                print(f"  ✅ Removed: {graph_file}")

        print()

    # Check if already extracted
    if not force and not force_clean and check_patient_ready(pid):
        print("✅ TOA timeline already extracted")
        print("   Files found:")
        for fname in ["timeline.json", "timeline_objects.jsonl", "episodes.json"]:
            fpath = Path(output_dir) / fname
            if fpath.exists():
                size = fpath.stat().st_size
                print(f"   - {fname}: {size:,} bytes")
        print()
        print("⏭️  Skipping extraction (use --force to regenerate)")

        # When using graph store, re-serialize to get final_chunk + deterministic_context
        # (needed for Step 2 JSON generation even when TOA is cached)
        if from_graph_store:
            try:
                from toa.graph_feeding import prepare_from_graph
                prepared = prepare_from_graph(pid, store_path=store_path)
                return True, prepared['final_chunk'], prepared.get('deterministic_context')
            except Exception as e:
                print(f"  ⚠️  Graph re-serialization failed: {e}")
                return True, None, None
        return True, None, None

    # Create chat clients
    chunk_client = create_chat_client(chunk_model)
    episode_client = create_chat_client(episode_model)

    print("Starting TOA extraction...")
    start_time = time.time()

    try:
        if from_graph_store:
            # Graph-first path: load from CohortGraphStore, no XML needed
            print(f"  Mode: Graph-first (store: {store_path})")
            result = extract_patient_from_graph(
                pid=pid,
                chat_client=chunk_client,
                store_path=store_path,
                profile="thoracic_tumor_board",
                tumor_type=TUMOR_TYPE,
                output_dir=output_dir,
                episodes_mode="llm",
                chunk_chat_client=chunk_client,
                episode_chat_client=episode_client,
                max_chunk_chars=max_chunk_chars,
                max_final_chunk_chars=max_final_chunk_chars,
            )
        else:
            # XML path: standard extraction from file
            result = extract_patient_from_file(
                pid=pid,
                xml_path=xml_path,
                chat_client=chunk_client,  # Default (not used when stage-specific provided)
                chunk_chat_client=chunk_client,
                episode_chat_client=episode_client,
                profile="thoracic_tumor_board",
                tumor_type=TUMOR_TYPE,
                output_dir=output_dir,
                episodes_mode="llm",
                enable_provenance=enable_provenance
            )

        duration = time.time() - start_time

        print()
        if result.success:
            print(f"✅ TOA extraction completed in {duration:.1f}s")
            print(f"   Events: {result.num_events}")
            print(f"   Episodes: {result.num_episodes}")
            print()
            print("   Files created:")
            for name, path in result.to_dict()["files_created"].items():
                if Path(path).exists():
                    size = Path(path).stat().st_size
                    print(f"   - {Path(path).name}: {size:,} bytes")

            # Return (success, final_chunk, deterministic_context) as 3-tuple
            det_ctx = getattr(result, 'deterministic_context', None)
            if from_graph_store and hasattr(result, 'final_chunk') and result.final_chunk:
                return True, result.final_chunk, det_ctx
            return True, None, det_ctx
        else:
            print(f"❌ TOA extraction failed: {result.error}")
            return False, None, None

    except Exception as e:
        print(f"❌ Error during TOA extraction: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def step_json_generation(
    pid: str,
    xml: str,
    output_dir: str,
    info_model: str,
    force: bool = False,
    deterministic_context: Optional[str] = None,
    from_graph_store: bool = False,
    store_path: str = "graph_store",
) -> bool:
    """
    Step 2: Combined patient_info + summary generation (single LLM call).

    Uses prompts/patient_info_and_note.txt to produce both patient_info.json
    and summary.json in one call. Uses gpt-5 by default for better clinical
    reasoning (driver mutations, TNM staging, note quality).

    Args:
        pid: Patient ID
        xml: Final chunk text (XML or graph-serialized)
        output_dir: Output directory
        info_model: Model for combined generation (default: gpt-5)
        force: Force regeneration even if cache exists
        deterministic_context: Graph search hits for clinical variable guidance
        from_graph_store: Whether running from graph store (for ct_date_vector)
        store_path: Path to graph store

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print("STEP 2: PATIENT INFO + NOTE GENERATION (COMBINED)")
    print(f"{'='*80}")
    print(f"  Patient: {pid}")
    print(f"  Model: {info_model}")
    print()

    patient_info_path = os.path.join(output_dir, "patient_info.json")
    summary_path = os.path.join(output_dir, "summary.json")

    # Check if both already exist
    if not force and os.path.exists(patient_info_path) and os.path.exists(summary_path):
        print(f"  ⏭️  patient_info.json + summary.json already exist (skipping)")
        return True

    # Load timeline for context
    timeline_path = os.path.join(output_dir, "timeline.json")
    if not os.path.exists(timeline_path):
        print(f"  ❌ Timeline not found at {timeline_path}")
        print("     Please run TOA extraction first!")
        return False

    with open(timeline_path, 'r', encoding='utf-8') as f:
        timeline = json.load(f)
    print(f"  Loaded timeline ({len(timeline.get('events', []))} events)")

    # Load episodes for context
    episodes_path = os.path.join(output_dir, "episodes.json")
    episodes_text = "[]"
    if os.path.exists(episodes_path):
        with open(episodes_path, 'r', encoding='utf-8') as f:
            episodes = json.load(f)
        episodes_text = json.dumps(episodes, indent=2)
        print(f"  Loaded episodes ({len(episodes)} episodes)")

    # Compute ct_date_vector from graph store
    ct_date_vector_text = "[]"
    try:
        from toa.graph_store import CohortGraphStore
        from toa.deterministic_retrieval import DeterministicRetriever
        store = CohortGraphStore(store_path)
        graph = store.load_patient_graph(pid)
        if graph is not None:
            retriever = DeterministicRetriever(graph)
            ct_dates = retriever.get_ct_date_vector()
            if ct_dates:
                ct_date_vector_text = json.dumps(ct_dates)
                print(f"  CT date vector: {ct_dates[:3]}{'...' if len(ct_dates) > 3 else ''}")
    except Exception as e:
        print(f"  ⚠️  CT date vector unavailable: {e}")

    # Always prepend structured demographics from XML <person> tag
    # This is the authoritative source for DOB, sex, race — never skip it
    import re
    source_text = xml
    xml_file = f"patient_records/thoracic_xmls/{pid}.xml"
    if os.path.exists(xml_file):
        try:
            import xml.etree.ElementTree as ET
            person = ET.parse(xml_file).getroot().find(".//person")
            if person is not None:
                demo_text = f"=== DEMOGRAPHICS ===\n{ET.tostring(person, encoding='unicode')}\n"
                source_text = demo_text + "\n" + source_text
                print(f"  Prepended demographics from XML ({len(demo_text)} chars)")
        except Exception:
            pass

    # Load combined prompt template
    prompt_path = os.path.join(PROMPTS_DIR, "patient_info_and_note.txt")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    # Fill placeholders via simple string replacement (avoids escaping issues with JSON braces)
    prompt = prompt_template.replace("{TUMOR_TYPE}", TUMOR_TYPE)
    # Cap source_text: keep demographics header (if present) + last 80k chars of clinical text
    # Most recent clinical data (ECOG, latest imaging, current treatment) is at the END
    if len(source_text) > 80_000:
        # Preserve demographics block at the start if present
        if source_text.startswith("=== DEMOGRAPHICS"):
            demo_end = source_text.find("\n=== ", len("=== DEMOGRAPHICS"))
            if demo_end > 0:
                demo_block = source_text[:demo_end]
                remaining = source_text[demo_end:]
                source_text_capped = demo_block + remaining[-(80_000 - len(demo_block)):]
            else:
                source_text_capped = source_text[-80_000:]
        else:
            source_text_capped = source_text[-80_000:]
    else:
        source_text_capped = source_text
    prompt = prompt.replace("{source_text}", source_text_capped)
    prompt = prompt.replace("{timeline_context}", json.dumps(timeline))
    prompt = prompt.replace("{episodes_context}", episodes_text)
    prompt = prompt.replace("{deterministic_context}", deterministic_context or "Not available")
    prompt = prompt.replace("{ct_date_vector}", ct_date_vector_text)

    print(f"  Prompt size: {len(prompt):,} chars (~{len(prompt)//4:,} tokens)")
    print()
    print(f"  📝 Generating patient_info + summary...")

    # Create chat client with info model
    chat_client = create_chat_client(info_model)

    try:
        raw = chat_client("", prompt)

        # Parse JSON from response (handle ```json ... ``` markers and double-JSON)
        json_str = raw.strip()
        if json_str.startswith("```"):
            # Remove ```json and trailing ```
            json_str = json_str.split("\n", 1)[1] if "\n" in json_str else json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
        json_str = json_str.strip()

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            # Try extracting first complete JSON object (handles double-JSON from concurrent LLM)
            from toa.llm_first_pass import _extract_first_json_object, _repair_json
            json_str = _extract_first_json_object(json_str)
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                parsed = json.loads(_repair_json(json_str))

        # Extract patient_info
        patient_info = parsed.get("patient_info", {})
        patient_info = fix_patient_info_keys(patient_info)

        with open(patient_info_path, 'w', encoding='utf-8') as f:
            json.dump(patient_info, f, indent=2, ensure_ascii=False)
        print(f"     ✅ patient_info.json ({len(patient_info)} sections)")

        # Extract summary
        summary_text = parsed.get("summary", "")
        summary_data = {"summary": summary_text.strip()}

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"     ✅ summary.json ({len(summary_text)} chars)")

    except json.JSONDecodeError as e:
        print(f"     ❌ Failed to parse JSON: {e}")
        print(f"     Raw response (first 500 chars): {raw[:500]}")
        return False
    except Exception as e:
        print(f"     ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Mark complete
    flag = os.path.join(output_dir, ".cache_complete")
    open(flag, 'w').close()

    return True


# ========== Main Pipeline ==========

def prepare_single_patient(args) -> Dict[str, Any]:
    """
    Prepare a single patient through the complete pipeline.

    Returns:
        Dict with success status and timing metrics
    """
    pid = args.pid
    from_graph_store = getattr(args, 'from_graph_store', False)
    store_path = getattr(args, 'store_path', 'graph_store')
    xml_path = f"{args.xml_dir}/{pid}.xml"
    output_dir = f"temp_jsons/{pid}"

    print(f"\n{'='*80}")
    print(f"PATIENT PREPARATION PIPELINE: {pid}")
    print(f"{'='*80}")
    print()
    print("Configuration:")
    if from_graph_store:
        print(f"  Source: Graph store ({store_path})")
    else:
        print(f"  XML: {xml_path}")
    print(f"  Output: {output_dir}")
    # print(f"  LLM: gsgpt (proxy: {gsgpt.PROXY_BASE})")
    print(f"  Chunk model: {args.chunk_model}")
    print(f"  Episode model: {args.episode_model}")
    print(f"  Info model: {args.info_model}")
    print(f"  Force regeneration: {args.force}")
    if from_graph_store:
        print(f"  Mode: graph-first (no XML needed)")
        chunk_chars = getattr(args, 'chunk_chars', 120_000)
        final_chars = getattr(args, 'final_chunk_chars', None) or chunk_chars
        if chunk_chars != 120_000 or final_chars != chunk_chars:
            print(f"  Chunk sizes: early={chunk_chars:,} final={final_chars:,}")
    else:
        print(f"  Provenance linking: {'enabled' if args.enable_provenance else 'disabled'}")
    print()

    # Initialize metrics
    pipeline_start = time.time()
    metrics = {
        "patient_id": pid,
        "success": False,
        "total_time": 0,
        "toa_time": 0,
        "json_time": 0,
        "xml_chars": 0,
        "num_events": 0,
        "num_episodes": 0,
        "models": {
            "chunk": args.chunk_model,
            "episode": args.episode_model,
            "info": args.info_model
        },
        "timestamp": datetime.now().isoformat()
    }

    # Verify source exists
    if not from_graph_store and not os.path.exists(xml_path):
        print(f"❌ XML file not found: {xml_path}")
        metrics["error"] = "XML file not found"
        metrics["total_time"] = time.time() - pipeline_start
        return metrics

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load XML (only needed for XML-based path)
    xml_full = None
    if not from_graph_store:
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_full = f.read()

    # Optional Step 0: Build Lumia Base Graph (if provenance enabled, XML path only)
    if not from_graph_store and args.enable_provenance:
        print(f"\n{'='*80}")
        print("STEP 0 (OPTIONAL): LUMIA BASE GRAPH")
        print(f"{'='*80}")
        print()

        try:
            from toa.xml_to_graph_hierarchical import build_hierarchical_graph

            lumia_graph_path = Path("graphs") / f"{pid}_lumia.graphml"

            # Create graphs directory if needed
            Path("graphs").mkdir(exist_ok=True)

            if not lumia_graph_path.exists() or args.force:
                print(f"Building Lumia graph from XML...")
                lumia_graph = build_hierarchical_graph(pid, xml_path)
                lumia_graph.save(str(lumia_graph_path))

                # Print stats
                stats = lumia_graph.stats()
                print(f"  ✅ Lumia graph: {lumia_graph_path}")
                print(f"     Nodes: {stats['total_nodes']}")
                print(f"     Edges: {stats['total_edges']}")
            else:
                print(f"  ✅ Lumia graph exists (cached): {lumia_graph_path}")

        except Exception as e:
            print(f"  ⚠️  Lumia graph build failed: {e}")
            print(f"     (Continuing without provenance)")

    # Step 1: TOA Timeline Extraction
    toa_start = time.time()
    step1_result = step_toa_extraction(
        pid=pid,
        xml_path=xml_path,
        output_dir=output_dir,
        chunk_model=args.chunk_model,
        episode_model=args.episode_model,
        force=args.force or args.force_clean,  # force_clean implies force
        force_clean=args.force_clean,
        enable_provenance=args.enable_provenance,
        from_graph_store=from_graph_store,
        store_path=store_path,
        max_chunk_chars=args.chunk_chars,
        max_final_chunk_chars=args.final_chunk_chars,
    )
    metrics["toa_time"] = time.time() - toa_start

    # Parse step1 result: always (success, final_chunk, deterministic_context) tuple
    success, graph_final_chunk, deterministic_context = step1_result

    if not success:
        print("\n❌ Pipeline failed at Step 1 (TOA Extraction)")
        metrics["error"] = "TOA extraction failed"
        metrics["total_time"] = time.time() - pipeline_start
        return metrics

    # Step 2: Dashboard JSON Generation
    if from_graph_store and graph_final_chunk:
        # Graph-first: use serialized graph text as context
        xml_final_chunk = graph_final_chunk
        print(f"\nJSON generation will use graph-serialized context: {len(xml_final_chunk):,} chars")
    else:
        # XML path: extract final chunk using same logic as TOA
        chunks = list(iter_chunks(xml_full, max_chars=120_000, strategy="streamlined"))
        xml_final_chunk = chunks[-1] if chunks else xml_full

    print(f"\nJSON generation source: {len(xml_final_chunk):,} chars (~{len(xml_final_chunk)//4:,} tokens)")

    json_start = time.time()
    success = step_json_generation(
        pid=pid,
        xml=xml_final_chunk,
        output_dir=output_dir,
        info_model=args.info_model,
        force=args.force,
        deterministic_context=deterministic_context,
        from_graph_store=from_graph_store,
        store_path=store_path,
    )
    metrics["json_time"] = time.time() - json_start

    if not success:
        print("\n❌ Pipeline failed at Step 2 (JSON Generation)")
        metrics["error"] = "JSON generation failed"
        metrics["total_time"] = time.time() - pipeline_start
        return metrics

    # Success!
    metrics["success"] = True
    metrics["total_time"] = time.time() - pipeline_start

    print(f"\n{'='*80}")
    print(f"✅ PATIENT PREPARATION COMPLETE: {pid}")
    print(f"{'='*80}")
    print()
    print(f"Dashboard ready at: {output_dir}")
    print()
    print("⏱️  Timing Metrics:")
    print(f"  TOA extraction: {metrics['toa_time']:.1f}s")
    print(f"  JSON generation: {metrics['json_time']:.1f}s")
    print(f"  Total time: {metrics['total_time']:.1f}s")
    print()
    print("Files created:")
    for fname in ["timeline.json", "timeline_objects.jsonl", "episodes.json",
                  "patient_info.json", "summary.json", "reference_map.json"]:
        fpath = Path(output_dir) / fname
        if fpath.exists():
            size = fpath.stat().st_size
            print(f"  ✅ {fname:<30} {size:>10,} bytes")
        else:
            print(f"  ⚠️  {fname:<30} (missing)")

    return metrics


def prepare_batch_patients(args) -> bool:
    """Prepare multiple patients through the complete pipeline."""
    # Read patient list - either from text file or cohort manifest
    if hasattr(args, 'batch') and args.batch:
        with open(args.batch, 'r') as f:
            patient_ids = [line.strip() for line in f if line.strip()]
        source = f"Patient list: {args.batch}"
        log_prefix = Path(args.batch).stem
    elif hasattr(args, 'cohort') and args.cohort:
        with open(args.cohort, 'r') as f:
            manifest = json.load(f)
        # Extract patient IDs from cohort manifest
        patient_ids = [p['patient_id'] for p in manifest['patients']]
        cohort_name = manifest['cohort_name']
        # Graph-store mode uses raw PIDs; XML mode uses cohort-prefixed truncated XMLs
        if not getattr(args, 'from_graph_store', False):
            patient_ids = [f"{cohort_name}_{pid}" for pid in patient_ids]
        source = f"Cohort manifest: {args.cohort} ({cohort_name})"
        log_prefix = cohort_name
    else:
        raise ValueError("Either --batch or --cohort must be specified")

    print(f"\n{'='*80}")
    print(f"BATCH PATIENT PREPARATION - {len(patient_ids)} patients")
    print(f"{'='*80}")
    print()
    print("Configuration:")
    print(f"  {source}")
    print(f"  Chunk model: {args.chunk_model}")
    print(f"  Episode model: {args.episode_model}")
    print(f"  Info model: {args.info_model}")
    print()

    batch_start = time.time()
    all_metrics = []

    for i, pid in enumerate(patient_ids, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(patient_ids)}] PROCESSING: {pid}")
        print(f"{'='*80}")

        # Create temp args for single patient
        class TempArgs:
            pass

        temp_args = TempArgs()
        temp_args.pid = pid
        temp_args.xml_dir = args.xml_dir
        temp_args.chunk_model = args.chunk_model
        temp_args.episode_model = args.episode_model
        temp_args.info_model = args.info_model
        temp_args.force = args.force
        temp_args.force_clean = args.force_clean
        temp_args.enable_provenance = args.enable_provenance
        temp_args.from_graph_store = getattr(args, 'from_graph_store', False)
        temp_args.store_path = getattr(args, 'store_path', 'graph_store')
        temp_args.chunk_chars = getattr(args, 'chunk_chars', 120_000)
        temp_args.final_chunk_chars = getattr(args, 'final_chunk_chars', None)

        metrics = prepare_single_patient(temp_args)
        all_metrics.append(metrics)

        if not metrics["success"]:
            print(f"\n⚠️  Patient {pid} failed - continuing with remaining patients...")

    batch_time = time.time() - batch_start

    # Save metrics log
    log_filename = f"{log_prefix}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_filename, 'w') as f:
        json.dump({
            "batch_start": datetime.now().isoformat(),
            "batch_duration": batch_time,
            "total_patients": len(patient_ids),
            "configuration": {
                "chunk_model": args.chunk_model,
                "episode_model": args.episode_model,
                "info_model": args.info_model
            },
            "patients": all_metrics
        }, f, indent=2)

    # Summary
    print(f"\n{'='*80}")
    print("BATCH PREPARATION SUMMARY")
    print(f"{'='*80}")

    successful = [m for m in all_metrics if m["success"]]
    failed = [m for m in all_metrics if not m["success"]]

    print(f"Total: {len(patient_ids)}")
    print(f"✅ Succeeded: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    print(f"⏱️  Total batch time: {batch_time:.1f}s ({batch_time/60:.1f} minutes)")

    if successful:
        avg_toa = sum(m["toa_time"] for m in successful) / len(successful)
        avg_json = sum(m["json_time"] for m in successful) / len(successful)
        avg_total = sum(m["total_time"] for m in successful) / len(successful)
        print(f"\n⏱️  Average per patient (successful):")
        print(f"  TOA extraction: {avg_toa:.1f}s")
        print(f"  JSON generation: {avg_json:.1f}s")
        print(f"  Total: {avg_total:.1f}s")

    if failed:
        print(f"\n❌ Failed patients:")
        for m in failed:
            error = m.get("error", "Unknown error")
            print(f"  - {m['patient_id']}: {error}")

    print(f"\n📊 Metrics saved to: {log_filename}")

    return len(failed) == 0


# ========== CLI ==========

def main():
    parser = argparse.ArgumentParser(
        description="Prepare patient data for VistaDash (TOA + JSONs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--pid",
        help="Patient ID (single patient mode)"
    )
    mode_group.add_argument(
        "--batch",
        help="Patient list file, one ID per line (batch mode)"
    )
    mode_group.add_argument(
        "--cohort",
        help="Cohort manifest JSON file (batch mode using cohort patients)"
    )

    # Model configuration
    parser.add_argument(
        "--chunk-model",
        default=DEFAULT_CHUNK_MODEL,
        choices=AVAILABLE_MODELS,
        help=f"Model for chunk extraction (default: {DEFAULT_CHUNK_MODEL})"
    )
    parser.add_argument(
        "--episode-model",
        default=DEFAULT_EPISODE_MODEL,
        choices=AVAILABLE_MODELS,
        help=f"Model for episode generation (default: {DEFAULT_EPISODE_MODEL})"
    )
    parser.add_argument(
        "--info-model",
        default=DEFAULT_INFO_MODEL,
        choices=AVAILABLE_MODELS,
        help=f"Model for patient info + note generation (default: {DEFAULT_INFO_MODEL})"
    )

    # Common arguments
    parser.add_argument(
        "--xml-dir",
        default="patient_records/thoracic_xmls",
        help="Directory containing {pid}.xml files (default: patient_records/thoracic_xmls)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration (ignore existing cache, but keep old files)"
    )
    parser.add_argument(
        "--force-clean",
        action="store_true",
        help="Force clean regeneration (remove temp_jsons and graphs first)"
    )
    parser.add_argument(
        "--enable-provenance",
        action="store_true",
        default=True,
        help="Enable provenance linking (default: enabled)"
    )
    parser.add_argument(
        "--no-provenance",
        action="store_true",
        help="Disable provenance linking"
    )
    parser.add_argument(
        "--from-graph-store",
        action="store_true",
        help="Use pre-built graph from CohortGraphStore (no XML needed)"
    )
    parser.add_argument(
        "--store-path",
        default="graph_store",
        help="Path to CohortGraphStore directory (default: graph_store)"
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=120_000,
        help="Max characters per early chunk for graph-first extraction (default: 120000)"
    )
    parser.add_argument(
        "--final-chunk-chars",
        type=int,
        default=None,
        help="Max characters for final chunk (default: same as --chunk-chars). Keep smaller for accuracy."
    )

    args = parser.parse_args()

    # --no-provenance overrides --enable-provenance
    if args.no_provenance:
        args.enable_provenance = False

    try:
        if args.pid:
            metrics = prepare_single_patient(args)
            # Save metrics for single patient too
            log_filename = f"{args.pid}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_filename, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"\n📊 Metrics saved to: {log_filename}")
            success = metrics["success"]
        elif args.batch or args.cohort:
            success = prepare_batch_patients(args)
        else:
            parser.print_help()
            return 1

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
