#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
source .venv/bin/activate

while pgrep -f "livebridge strong-train.*scale_full_lightgcn_seed42" >/dev/null; do
  sleep 20
done

test -s runs/scale_full_lightgcn_seed42/metrics.json

for seed in 43 44; do
  output="runs/scale_full_lightgcn_seed${seed}"
  mkdir -p "${output}"
  date +%s > "${output}/start.txt"
  livebridge strong-train --model lightgcn \
    --processed-dir data/processed/klm3_temporal_full_seed42 \
    --output-dir "${output}" --epochs 30 --seed "${seed}" \
    --batch-size 4096 --embedding-dim 64 > "${output}/run.log" 2>&1
  date +%s > "${output}/end.txt"
done
