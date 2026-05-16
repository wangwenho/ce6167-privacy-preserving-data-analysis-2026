#!/usr/bin/env bash
set -e

echo "========================================="
echo "  Baseline Test (beam, noise=0)"
echo "========================================="

cd "$(dirname "$0")/.."

LOG_FILE="models/attacker_gpt2_large_personachat_mpnet_beam.log"
echo "Log: $LOG_FILE"
echo ""

uv run python attacker.py \
    --dataset personachat \
    --data_type test \
    --batch_size 16 \
    --embed_model mpnet \
    --decode beam

echo ""
[ -f "$LOG_FILE" ] && echo "[OK] Log created: $LOG_FILE ($(du -h "$LOG_FILE" | cut -f1))" || echo "[MISSING] $LOG_FILE"
echo ""