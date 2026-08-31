#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
source .venv/bin/activate

for seed in 42 43 44; do
  processed="${PROJECT_DIR}/data/processed/klm3_temporal_1pct_seed${seed}"
  run_root="${PROJECT_DIR}/runs/strong_suite_seed${seed}"
  test -s "${processed}/live.csv"
  test -s "${processed}/photo_author.csv"
  mkdir -p "${run_root}"

  livebridge popularity --processed-dir "${processed}" --output-dir "${run_root}/popularity" \
    | tee "${run_root}/popularity.log"
  livebridge train --processed-dir "${processed}" --output-dir "${run_root}/target" \
    --mode target --epochs 30 --seed "${seed}" | tee "${run_root}/target.log"
  for model in lightgcn sasrec emcdr; do
    livebridge strong-train --model "${model}" --processed-dir "${processed}" \
      --output-dir "${run_root}/${model}" --epochs 30 --seed "${seed}" \
      --batch-size 4096 --embedding-dim 64 \
      | tee "${run_root}/${model}.log"
  done
  livebridge train --processed-dir "${processed}" --output-dir "${run_root}/bridge" \
    --mode bridge --epochs 30 --seed "${seed}" | tee "${run_root}/bridge.log"
done

python scripts/summarize_strong_suite.py \
  --runs-root "${PROJECT_DIR}/runs" \
  --output "${PROJECT_DIR}/runs/strong_suite_summary.json"
