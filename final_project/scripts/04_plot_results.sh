#!/usr/bin/env bash
set -e

echo "========================================="
echo "  Plot Results"
echo "========================================="

START_TIME=$(date +%s)
echo "Start: $(date)"
echo ""

cd "$(dirname "$0")/.."

uv run python scripts/quantitative_plot.py
uv run python scripts/qualitative_plot.py

echo ""
echo "End: $(date)"
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Elapsed: $((ELAPSED / 60)) min $((ELAPSED % 60)) sec"

echo ""
echo "--- Verification ---"
FIG_COUNT=$(ls -1 outputs/fig*.png 2>/dev/null | wc -l)
echo "[OK] $FIG_COUNT figures in outputs/"
echo ""