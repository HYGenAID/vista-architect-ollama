"""Minimal OpenAI-backend round-trip.

Prereq:
    pip install openai
    export OPENAI_API_KEY=sk-...
    export VISTA_LLM_BACKEND=openai

Run:
    python vista_llm/examples/quickstart_openai.py
"""
import os
os.environ["VISTA_LLM_BACKEND"] = "openai"

import vista_llm
print(vista_llm.info())
print(vista_llm.chat("In one sentence: what is the TNM staging system?",
                     model="gpt-4.1", max_tokens=200))
