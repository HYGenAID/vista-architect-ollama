"""Minimal Vertex AI round-trip — the configuration used in the paper.

Prereq:
    pip install google-genai "anthropic[vertex]"
    gcloud auth application-default login
    export GOOGLE_CLOUD_PROJECT=your-gcp-project
    export GOOGLE_CLOUD_LOCATION=us-central1
    export VISTA_LLM_BACKEND=vertex

Run:
    python vista_llm/examples/quickstart_vertex.py
"""
import os
os.environ["VISTA_LLM_BACKEND"] = "vertex"

import vista_llm
print(vista_llm.info())

# Gemini on Vertex
print("--- Gemini ---")
print(vista_llm.chat("In one sentence: what is the TNM staging system?",
                     model="gemini-2.5-flash", max_tokens=200))

# Claude on Vertex (if you have access)
# print("--- Claude ---")
# print(vista_llm.chat("In one sentence: what is the TNM staging system?",
#                      model="claude-sonnet-4-5", max_tokens=200))
