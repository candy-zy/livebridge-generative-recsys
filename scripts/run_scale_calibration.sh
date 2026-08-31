#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RATIO="${RATIO:-0.10}"
SEED="${SEED:-42}"
TAG="${TAG:-10pct}"
EPOCHS="${EPOCHS:-30}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-128}"

cd "${PROJECT_DIR}"
source .venv/bin/activate

processed="data/processed/klm3_temporal_${TAG}_seed${SEED}"
bridge_run="runs/scale_${TAG}_seed${SEED}/bridge"
grpo_run="runs/scale_${TAG}_seed${SEED}/grpo"
mkdir -p "${processed}" "${bridge_run}" "${grpo_run}"

echo "PIPELINE_START=$(date -Is)"
echo "RATIO=${RATIO} SEED=${SEED} TAG=${TAG} EPOCHS=${EPOCHS}"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
df -h "${PROJECT_DIR}"

if [[ ! -s "${processed}/live.csv" || ! -s "${processed}/photo_author.csv" ]]; then
  echo "STAGE_PREPARE_START=$(date -Is)"
  livebridge prepare \
    --data-dir data/KuaiLive-M3 \
    --output-dir "${processed}" \
    --sample-ratio "${RATIO}" --seed "${SEED}" --source-mode temporal
  echo "STAGE_PREPARE_END=$(date -Is)"
else
  echo "STAGE_PREPARE_REUSE=${processed}"
fi

if [[ ! -s "${bridge_run}/model.pt" ]]; then
  echo "STAGE_BRIDGE_START=$(date -Is)"
  livebridge train \
    --processed-dir "${processed}" --output-dir "${bridge_run}" \
    --mode bridge --epochs "${EPOCHS}" --seed "${SEED}"
  echo "STAGE_BRIDGE_END=$(date -Is)"
else
  echo "STAGE_BRIDGE_REUSE=${bridge_run}/model.pt"
fi

echo "STAGE_GRPO_START=$(date -Is)"
livebridge grpo-train \
  --processed-dir "${processed}" \
  --bridge-checkpoint "${bridge_run}/model.pt" \
  --output-dir "${grpo_run}" \
  --epochs "${EPOCHS}" --seed "${SEED}" \
  --candidate-pool 50 --slate-size 10 --group-size 8 \
  --learning-rate 0.01 --residual-scale 0.35 \
  --source-weight 0.10 --profile-weight 0.0 --longtail-weight 0.05 \
  --score-batch-size "${SCORE_BATCH_SIZE}"
echo "STAGE_GRPO_END=$(date -Is)"

nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "PIPELINE_END=$(date -Is)"
