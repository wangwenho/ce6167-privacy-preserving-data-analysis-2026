#!/usr/bin/env bash
set -e

echo "========================================="
echo "  Evaluate Results"
echo "========================================="

START_TIME=$(date +%s)
echo "Start: $(date)"
echo ""

cd "$(dirname "$0")/.."

uv run python scripts/eval_results.py --batch-size 16

echo ""
echo "End: $(date)"
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Elapsed: $((ELAPSED / 60)) min $((ELAPSED % 60)) sec"

echo ""
echo "--- Verification ---"
[ -f "outputs/results.csv" ] && echo "[OK] outputs/results.csv" || echo "[MISSING] outputs/results.csv"
echo ""