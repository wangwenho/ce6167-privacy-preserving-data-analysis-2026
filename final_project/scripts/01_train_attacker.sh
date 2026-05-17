#!/usr/bin/env bash
set -e

echo "========================================="
echo "  Training Attacker"
echo "  Dataset: personachat"
echo "  Victim:  mpnet (all-mpnet-base-v1)"
echo "  Epochs:  5"
echo "========================================="

START_TIME=$(date +%s)
echo "Start: $(date)"
echo ""

cd "$(dirname "$0")/.."

uv run python attacker.py \
    --dataset personachat \
    --data_type train \
    --num_epochs 5 \
    --batch_size 16 \
    --embed_model mpnet

echo ""
echo "End: $(date)"
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Elapsed: $((ELAPSED / 60)) min $((ELAPSED % 60)) sec"

echo ""
echo "--- Verification ---"
MODEL_DIR="models/attacker_gpt2_large_personachat_mpnet"
PROJ_FILE="models/projection_gpt2_large_personachat_mpnet"

[ -d "$MODEL_DIR" ] && echo "[OK] Attacker model: $MODEL_DIR" || echo "[MISSING] $MODEL_DIR"
[ -f "$PROJ_FILE" ] && echo "[OK] Projection: $PROJ_FILE"    || echo "[MISSING] $PROJ_FILE"
echo ""