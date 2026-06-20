"""Minimal Anthropic-backend round-trip.

Prereq:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    export VISTA_LLM_BACKEND=anthropic

Run:
    python vista_llm/examples/quickstart_anthropic.py
"""
import os
os.environ["VISTA_LLM_BACKEND"] = "anthropic"

import vista_llm
print(vista_llm.info())
print(vista_llm.chat("In one sentence: what is the TNM staging system?",
                     model="claude-sonnet-4-5", max_tokens=200))
