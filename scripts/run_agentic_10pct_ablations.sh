#!/usr/bin/env bash
set -euo pipefail

unset OMP_NUM_THREADS MKL_NUM_THREADS
cd "$(dirname "$0")/.."

CACHE="runs/agentic_cache_10pct_seed42"
for variant in no_memory myopic; do
  .venv/bin/livebridge agent-train \
    --cache-dir "$CACHE" \
    --output-dir "runs/agentic_10pct/${variant}_seed42" \
    --variant "$variant" \
    --epochs 10 --warmup-epochs 5 --learning-rate 0.001 \
    --kl-beta 0.1 --entropy-coef 0.005 \
    --users-per-epoch 1024 --group-size 4 --max-steps 8 --seed 42
done
