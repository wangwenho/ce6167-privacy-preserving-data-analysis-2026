#!/usr/bin/env bash
set -e

echo "========================================="
echo "  Step 4a: Batch Evaluation → outputs/results.csv"
echo "========================================="

cd "$(dirname "$0")/.."

uv run python scripts/batch_eval.py \
    --batch-size 16

echo ""
echo "========================================="
echo "  Step 4b: Generate Figures → outputs/xxx.png"
echo "========================================="

uv run python scripts/plot.py

echo ""
echo "Done. Results: outputs/results.csv  |  Figures: outputs/"