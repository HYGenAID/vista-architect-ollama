"""Minimal public-Gemini round-trip (Google AI Studio).

Prereq:
    pip install google-genai
    export GOOGLE_API_KEY=AIza...
    export VISTA_LLM_BACKEND=gemini

Run:
    python vista_llm/examples/quickstart_gemini.py
"""
import os
os.environ["VISTA_LLM_BACKEND"] = "gemini"

import vista_llm
print(vista_llm.info())
print(vista_llm.chat("In one sentence: what is the TNM staging system?",
                     model="gemini-2.5-flash", max_tokens=200))
