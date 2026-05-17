#!/usr/bin/env bash
set -e

echo "========================================="
echo "  Step 4a: Batch Evaluation → CSV"
echo "========================================="

cd "$(dirname "$0")/.."

uv run python scripts/batch_eval.py -o results.csv --batch-size 16

echo ""
echo "========================================="
echo "  Step 4b: Generate Figures → assets/"
echo "========================================="

uv run python scripts/plot.py

echo ""
echo "Done. Results: results.csv  |  Figures: assets/"