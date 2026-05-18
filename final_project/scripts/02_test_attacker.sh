#!/usr/bin/env bash
set -e

echo "========================================="
echo "  Test Attacker with Noise"
echo "========================================="

START_TIME=$(date +%s)
echo "Start: $(date)"
echo ""

cd "$(dirname "$0")/.."

NOISE_SCALES=(0.0 0.001 0.002 0.005 0.008 0.01 0.015 0.02 0.03 0.04 0.05 0.075 0.1 0.25 0.5)

for noise in "${NOISE_SCALES[@]}"; do
    LOG_FILE="models/attacker_gpt2_large_personachat_mpnet_beam_noise_${noise}.log"
    echo ""
    echo "--- noise_scale = $noise ---"
    echo "    Log: $LOG_FILE"

    uv run python scripts/test_attacker.py \
        --dataset personachat \
        --embed_model mpnet \
        --decode beam \
        --batch_size 16 \
        --noise_scale "$noise"

    if [ -f "$LOG_FILE" ]; then
        echo "    [OK] Saved ($(du -h "$LOG_FILE" | cut -f1))"
    else
        echo "    [MISSING] $LOG_FILE"
    fi
done

echo ""
echo "End: $(date)"
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Elapsed: $((ELAPSED / 60)) min $((ELAPSED % 60)) sec"

echo ""
echo "--- Verification ---"
LOG_COUNT=$(ls -1 models/attacker_gpt2_large_personachat_mpnet_beam_noise_*.log 2>/dev/null | wc -l)
echo "[OK] $LOG_COUNT log files in models/"
echo ""