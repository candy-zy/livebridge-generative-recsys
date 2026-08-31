#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROCESSED_DIR="${PROCESSED_DIR:-${PROJECT_DIR}/data/processed/klm3_temporal_full_seed42}"
EPOCHS="${EPOCHS:-30}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-64}"

cd "${PROJECT_DIR}"
source .venv/bin/activate
test -s "${PROCESSED_DIR}/live.csv"
test -s "${PROCESSED_DIR}/photo_author.csv"

echo "REMAINING_SEEDS_START=$(date -Is)"
for seed in 43 44; do
  run_root="runs/scale_full_seed${seed}"
  bridge_run="${run_root}/bridge"
  grpo_run="${run_root}/grpo"
  mkdir -p "${bridge_run}" "${grpo_run}"

  echo "SEED_${seed}_START=$(date -Is)"
  if [[ ! -s "${bridge_run}/model.pt" ]]; then
    echo "SEED_${seed}_BRIDGE_START=$(date -Is)"
    livebridge train \
      --processed-dir "${PROCESSED_DIR}" --output-dir "${bridge_run}" \
      --mode bridge --epochs "${EPOCHS}" --seed "${seed}"
    echo "SEED_${seed}_BRIDGE_END=$(date -Is)"
  else
    echo "SEED_${seed}_BRIDGE_REUSE=${bridge_run}/model.pt"
  fi

  echo "SEED_${seed}_GRPO_START=$(date -Is)"
  livebridge grpo-train \
    --processed-dir "${PROCESSED_DIR}" \
    --bridge-checkpoint "${bridge_run}/model.pt" \
    --output-dir "${grpo_run}" \
    --epochs "${EPOCHS}" --seed "${seed}" \
    --candidate-pool 50 --slate-size 10 --group-size 8 \
    --learning-rate 0.01 --residual-scale 0.35 \
    --source-weight 0.10 --profile-weight 0.0 --longtail-weight 0.05 \
    --score-batch-size "${SCORE_BATCH_SIZE}"
  echo "SEED_${seed}_GRPO_END=$(date -Is)"
  echo "SEED_${seed}_END=$(date -Is)"
done

python scripts/summarize_scale_full.py \
  --runs-root runs --output runs/scale_full_multiseed_summary.json
echo "REMAINING_SEEDS_END=$(date -Is)"
