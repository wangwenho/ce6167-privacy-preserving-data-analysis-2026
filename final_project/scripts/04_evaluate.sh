#!/usr/bin/env bash
set -e

echo "========================================="
echo "  Evaluation Instructions"
echo "========================================="
echo ""
echo "Eval scripts have hardcoded log paths."
echo "Edit them manually before running."
echo ""

echo "--- Available log files ---"
for f in models/attacker_gpt2_large_personachat_mpnet_beam*.log; do
    [ -f "$f" ] && echo "  $f ($(du -h "$f" | cut -f1))"
done
echo ""

echo "--- How to evaluate (generation metrics) ---"
echo ""
echo "  1. Open eval_generation.py"
echo "  2. Scroll to the bottom (if __name__ == '__main__':)"
echo "  3. Replace hardcoded paths with your log file:"
echo ""
echo "       log_path = 'models/attacker_gpt2_large_personachat_mpnet_beam.log'"
echo "       with open(log_path) as f:"
echo "           data = json.load(f)"
echo "       report_metrics(data)"
echo ""
echo "  4. Run: uv run python eval_generation.py"
echo ""

echo "--- How to evaluate (classification metrics) ---"
echo ""
echo "  1. Open eval_classification.py"
echo "  2. Scroll to the bottom (if __name__ == '__main__':)"
echo "  3. Replace hardcoded paths:"
echo ""
echo "       log_path = 'models/attacker_gpt2_large_personachat_mpnet_beam.log'"
echo "       logger.info(f'====={log_path}=====')"
echo "       metric_token(log_path)"
echo ""
echo "  4. Run: uv run python eval_classification.py"
echo ""

echo "--- Noise experiment logs to evaluate ---"
echo ""
echo "  models/attacker_gpt2_large_personachat_mpnet_beam.log          (baseline)"
for noise in 0.0 0.001 0.005 0.01 0.05 0.1 0.5; do
    echo "  models/attacker_gpt2_large_personachat_mpnet_beam_noise_${noise}.log"
done
echo ""