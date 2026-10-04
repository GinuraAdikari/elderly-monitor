#!/usr/bin/env bash
# End-to-end execution of the Elderly Bed-Activity Monitor pipeline
set -e

echo "=========================================================="
echo " Elderly Bed-Activity Monitor — End-to-End Execution"
echo "=========================================================="

echo "[1/4] Running Unit Test Suite (pytest)..."
pytest -v

echo "[2/4] Generating Architecture Diagram..."
python scripts/render_architecture.py

echo "[3/4] Running Full Evaluation Suite across all videos..."
python scripts/run_eval_suite.py

echo "[4/4] Verification of Key Outputs..."
ls -lh outputs/gmdcsa_01/viterbi/
ls -lh outputs/gmdcsa_01/eval/

echo "=========================================================="
echo " Elderly Bed-Activity Monitor — PoC Execution & Evaluation Suite Complete!"
echo "=========================================================="

