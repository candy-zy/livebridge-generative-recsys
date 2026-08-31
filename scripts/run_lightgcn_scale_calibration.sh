#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
source .venv/bin/activate

run_timed() {
  local name="$1"
  shift
  local output="runs/${name}"
  mkdir -p "${output}"
  date +%s > "${output}/start.txt"
  "$@" --output-dir "${output}" > "${output}/run.log" 2>&1
  date +%s > "${output}/end.txt"
}

# Let the already-running Target-BPR seeds retain exclusive access to the GPU.
while pgrep -f "livebridge train.*scale_full_target_seed" >/dev/null; do
  sleep 20
done

# Measure convergence and runtime on 10% before committing to a full-scale run.
run_timed lightgcn_optimized_10pct_30ep \
  livebridge strong-train --model lightgcn \
  --processed-dir data/processed/klm3_temporal_10pct_seed42 \
  --epochs 30 --seed 42 --batch-size 4096 --embedding-dim 64

# Isolate one full-data epoch plus full-sort evaluation for a reliable scale estimate.
run_timed lightgcn_optimized_full_1ep \
  livebridge strong-train --model lightgcn \
  --processed-dir data/processed/klm3_temporal_full_seed42 \
  --epochs 1 --seed 42 --batch-size 4096 --embedding-dim 64
