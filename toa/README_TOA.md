
# TOA (Timeline-Object Augmentation) – Scaffold

This package implements the first-pass **events → episodes** pipeline and a back-compat projection to the existing `timeline.json`.

## Public API
```python
from toa import extract, project_major_events
events, episodes = extract(pid, xml_text, chat_client=my_chat_fn, system_prompt=system_prompt_text)
timeline_json = project_major_events(events, episodes)
```

- `chat_client(system_prompt, user_prompt) -> str` should return the raw LLM string output.
- See `prompts/timeline_compact.txt` for the first-pass extraction prompt.

## Files
- `engine.py` – orchestration
- `llm_first_pass.py` – chunk-level extraction call
- `normalize.py` – imaging/lab/background roll-up helpers
- `dedupe_rollup.py` – deterministic day roll-up + sorting + event hashing
- `episodes.py` – rule-based episode construction
- `io/jsonio.py` – JSON helpers
- `io/ehr_xml.py` – simple chunker (char-based)
- `prompts/timeline_compact.txt` – compact extraction prompt

## Integration hint (app.py)
Feature flag:
```python
USE_TOA_TIMELINE = os.getenv("USE_TOA_TIMELINE","true").lower()=="true"
...
if USE_TOA_TIMELINE:
    from toa import extract
    events, episodes = extract(pid, xml_text, chat_client=secure_chat, system_prompt=SYSTEM_PROMPT_VISTA)
    jsonio.write_jsonl(f"temp_jsons/{pid}/timeline_objects.jsonl", events)
    jsonio.write_json (f"temp_jsons/{pid}/episodes.json", episodes)
    timeline_json = project_major_events(events, episodes)
    jsonio.write_json (f"temp_jsons/{pid}/timeline.json", timeline_json)
else:
    # existing path
    ...
```

## Notes
- Deterministic hashing uses BLAKE2b (12-byte hex) over normalized event signatures.
- Episodes are intentionally conservative; refine rules as clinicians review.
