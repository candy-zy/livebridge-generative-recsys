#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
source .venv/bin/activate
mkdir -p runs/grpo_suite runs/grpo_ablation_seed42

for seed in 42 43 44; do
  livebridge grpo-train \
    --processed-dir "data/processed/klm3_temporal_1pct_seed${seed}" \
    --bridge-checkpoint "runs/strong_suite_seed${seed}/bridge/model.pt" \
    --output-dir "runs/grpo_suite/seed${seed}" \
    --epochs 30 --seed "${seed}" --candidate-pool 50 --slate-size 10 \
    --group-size 8 --learning-rate 0.01 --residual-scale 0.35 \
    --source-weight 0.10 --longtail-weight 0.05 \
    --profile-weight 0.0 \
    | tee "runs/grpo_suite/seed${seed}.log"
done

for variant in no_source no_longtail; do
  source_weight=0.10
  longtail_weight=0.05
  if [[ "${variant}" == "no_source" ]]; then source_weight=0.0; fi
  if [[ "${variant}" == "no_longtail" ]]; then longtail_weight=0.0; fi
  livebridge grpo-train \
    --processed-dir data/processed/klm3_temporal_1pct_seed42 \
    --bridge-checkpoint runs/strong_suite_seed42/bridge/model.pt \
    --output-dir "runs/grpo_ablation_seed42/${variant}" \
    --epochs 30 --seed 42 --candidate-pool 50 --slate-size 10 \
    --group-size 8 --learning-rate 0.01 --residual-scale 0.35 \
    --source-weight "${source_weight}" --longtail-weight "${longtail_weight}" \
    --profile-weight 0.0 \
    | tee "runs/grpo_ablation_seed42/${variant}.log"
done
