
from __future__ import annotations
from typing import List, Dict, Any, Callable, Optional
import json
import re

COMPACT_PROMPT_PATH = __package__.replace(".", "/") + "/prompts/timeline_compact.txt"

def _load_prompt() -> str:
    with open(COMPACT_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _extract_first_json_object(text: str) -> str:
    """Extract the first complete JSON object from text using bracket counting.

    Handles the 'Extra data' problem where LLMs (especially under high concurrency)
    produce multiple JSON objects concatenated together. The greedy regex
    r'{[\\s\\S]*}' would capture everything from first { to last }, creating
    invalid JSON. This function correctly isolates just the first object.
    """
    start = text.find('{')
    if start == -1:
        raise ValueError("No JSON object found in text")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == '\\':
            if in_string:
                escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    # If we get here, brackets aren't balanced — return everything from start
    # (the repair function will try to fix truncation)
    return text[start:]


def _repair_json(json_str: str) -> str:
    """
    Attempt to repair common JSON issues from LLM output.
    - Trailing commas before ] or }
    - Missing quotes around keys
    - Single quotes instead of double quotes
    - Incomplete/truncated JSON (missing closing brackets)
    """
    # Remove trailing commas before ] or }
    json_str = re.sub(r',\s*(\]|\})', r'\1', json_str)

    # Replace single quotes with double quotes (careful with apostrophes)
    # Only if the string doesn't already have balanced double quotes
    if json_str.count('"') < json_str.count("'"):
        json_str = json_str.replace("'", '"')

    # Try to fix incomplete/truncated JSON by adding missing closing brackets
    # Count open vs close brackets
    open_braces = json_str.count('{') - json_str.count('}')
    open_brackets = json_str.count('[') - json_str.count(']')

    # If truncated mid-object/array, try to close them
    if open_braces > 0 or open_brackets > 0:
        # Remove trailing incomplete object if we're mid-property
        # Look for trailing pattern like: "key":  or "key": " or "key": {
        json_str = re.sub(r',\s*"[^"]*":\s*[^,\]\}]*$', '', json_str)
        json_str = re.sub(r',\s*$', '', json_str)  # Remove trailing comma

        # Add missing brackets
        json_str += ']' * open_brackets
        json_str += '}' * open_braces

    return json_str


def extract_chunk_events(
    xml_chunk: str,
    system_prompt: Optional[str],
    chat_client: Callable[[str, str], str],
    lumia_context: Optional[str] = None,
    deterministic_context: Optional[str] = None,
    imaging_date_vector: Optional[List[str]] = None,
    is_final_chunk: bool = False
) -> List[Dict[str, Any]]:
    """
    chat_client(system_prompt, user_prompt) -> model_output (string)
    model_output must contain a top-level {'events': [...]} JSON.

    Args:
        xml_chunk: XML text to extract events from
        system_prompt: System prompt for LLM
        chat_client: Chat function
        lumia_context: Optional Lumia graph context with source event references [E1], [E2], etc.
        deterministic_context: Optional graph search hits for LLM attention guidance
        imaging_date_vector: Optional list of confirmed CT/PET-CT dates (YYYY-MM-DD) from
            structured institutional records for this chunk's time window. Used to ground
            imaging event dates to exact structured data.
        is_final_chunk: Whether this is the last chunk (add extra guidance for recent imaging)
    """
    prompt_template = _load_prompt()

    # Inject context layers before XML chunk
    # Order: imaging_vector -> deterministic_context -> lumia_context -> XML chunk
    context_prefix = ""
    if imaging_date_vector:
        vector_str = "=== IMAGING DATE VECTOR (ground truth from structured institutional records) ===\n"
        vector_str += "These are confirmed imaging study dates from THIS INSTITUTION for this time window.\n"
        vector_str += "When extracting imaging events, use the EXACT date from this vector instead of the\n"
        vector_str += "approximate date mentioned in the clinical note text. Match each in-house imaging\n"
        vector_str += "mention to the nearest date+modality in this vector.\n"
        vector_str += "Outside/external imaging (from referring hospitals) will NOT appear here — extract\n"
        vector_str += "those dates from the text as-is and mark them as outside in the description.\n"
        # Format as structured list: date modality (site)
        if isinstance(imaging_date_vector[0], dict):
            lines = []
            for entry in imaging_date_vector:
                site = f" ({entry['site']})" if entry.get('site') else ""
                lines.append(f"  {entry['date']}  {entry['modality']}{site}")
            vector_str += "Studies:\n" + "\n".join(lines) + "\n"
        else:
            # Fallback for plain date list (ct_date_vector format)
            vector_str += f"Dates: {', '.join(imaging_date_vector)}\n"
        if is_final_chunk:
            vector_str += "\n** FINAL CHUNK — RECENT IMAGING ALERT **\n"
            vector_str += "The most recent study in this vector may not yet have a finalized radiology report.\n"
            vector_str += "It may only be mentioned in a clinician note (e.g., 'CT on Friday showed...', 'PET-CT results pending').\n"
            vector_str += "You MUST still create a timeline imaging event for EVERY study in this vector,\n"
            vector_str += "even if the only evidence is a brief mention in a clinical note rather than a full radiology report.\n"
        context_prefix += vector_str + "\n"
    if deterministic_context:
        context_prefix += deterministic_context + "\n\n"
    if lumia_context:
        context_prefix += lumia_context + "\n\n"
    if context_prefix:
        prompt_template = context_prefix + prompt_template

    user_prompt = prompt_template.replace("{xml_chunk}", xml_chunk[:120_000])
    raw = chat_client(system_prompt or "", user_prompt)
    # Extract the first complete JSON object (handles double-JSON from concurrent LLM calls)
    try:
        json_str = _extract_first_json_object(raw)
    except ValueError:
        raise ValueError("LLM output did not contain JSON")

    # Try parsing, if it fails try to repair
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Attempt repair
        repaired = _repair_json(json_str)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError as e:
            # Save the raw output for debugging
            with open("/tmp/llm_json_error.txt", "w") as f:
                f.write(f"Original:\n{json_str}\n\nRepaired:\n{repaired}\n\nError: {e}")
            raise ValueError(f"LLM output contained invalid JSON even after repair: {e}")

    events = data.get("events", [])
    if not isinstance(events, list):
        raise ValueError("Invalid events payload")
    return events
