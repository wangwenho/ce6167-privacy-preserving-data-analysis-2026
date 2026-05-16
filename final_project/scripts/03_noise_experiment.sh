#!/usr/bin/env bash
set -e

echo "========================================="
echo "  Noise Experiment Suite"
echo "========================================="

cd "$(dirname "$0")/.."

NOISE_SCALES=(0.0 0.001 0.005 0.01 0.05 0.1 0.5)

for noise in "${NOISE_SCALES[@]}"; do
    LOG_FILE="models/attacker_gpt2_large_personachat_mpnet_beam_noise_${noise}.log"
    echo ""
    echo "--- noise_scale = $noise ---"
    echo "    Log: $LOG_FILE"

    uv run python scripts/attacker_noise_experiment.py \
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
echo "Noise experiments complete."