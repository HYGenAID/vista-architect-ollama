#!/usr/bin/env bash
# run_demo.sh — end-to-end one-patient walkthrough.
#
# Builds a MEDS Graph from examples/demo1.xml (deterministic), runs the
# TOA event-extraction + episode-synthesis pipeline against the configured
# LLM backend, and emits patient_info.json + summary.json.
#
# Backend is selected via VISTA_LLM_BACKEND (see SETUP.md).
# Total wall time ≈ 2 minutes against any of the four backends.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${VISTA_LLM_BACKEND:-}" ]]; then
    echo "ERROR: VISTA_LLM_BACKEND is not set." >&2
    echo "Pick one of: openai, anthropic, gemini, vertex (see SETUP.md)." >&2
    exit 1
fi

PID=demo1
WORKDIR=examples/run_demo_out
rm -rf "$WORKDIR" && mkdir -p "$WORKDIR"

echo "================================================================"
echo "VISTA Architect — one-patient demo"
echo "  patient ID:      $PID"
echo "  source XML:      examples/demo1.xml"
echo "  output dir:      $WORKDIR"
echo "  backend:         $VISTA_LLM_BACKEND"
echo "================================================================"
echo

echo "[1/3] Building MEDS Graph (deterministic, no LLM call) …"
python -c "
import sys
sys.path.insert(0, '.')
from toa.xml_to_graph_hierarchical import build_hierarchical_graph
g = build_hierarchical_graph('987654321','examples/demo1.xml')
g.save('$WORKDIR/${PID}_meds.graphml')
print(f'  MEDS graph: {g.G.number_of_nodes()} nodes, {g.G.number_of_edges()} edges')
"
echo

echo "[2/3] Running TOA pipeline (chunk extraction → episodes → patient_info) …"
python scripts/prepare_patient.py \
    --pid "$PID" \
    --xml-dir examples/ \
    --store-path "$WORKDIR" \
    --info-model "gemma4:31b-it-q8_0" \
    --force \
  || { echo "ERROR: pipeline failed"; exit 1; }
echo

echo "[3/3] Done. Outputs:"
ls -lh "$WORKDIR"/
echo
echo "Inspect: $WORKDIR/patient_info.json"
