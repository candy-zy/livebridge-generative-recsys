#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_DIR="${1:-data/KuaiLive-M3}"
PROCESSED_DIR="${2:-data/processed/klm3_temporal_1pct_seed42}"
BRIDGE_CHECKPOINT="${3:-runs/strong_suite_seed42/bridge/model.pt}"
OUTPUT_DIR="${4:-runs/generative_stage1_1pct_seed42}"

mkdir -p "$OUTPUT_DIR/alignment" "$OUTPUT_DIR/id" "$OUTPUT_DIR/content"

.venv/bin/livebridge content-align \
  --data-dir "$DATA_DIR" \
  --processed-dir "$PROCESSED_DIR" \
  --output-dir "$OUTPUT_DIR/alignment" \
  --epochs 100 --seed 42 \
  >"$OUTPUT_DIR/alignment/train.log" 2>&1

.venv/bin/livebridge generative-train \
  --processed-dir "$PROCESSED_DIR" \
  --bridge-checkpoint "$BRIDGE_CHECKPOINT" \
  --content-path "$OUTPUT_DIR/alignment/aligned_author_content.npz" \
  --output-dir "$OUTPUT_DIR/id" \
  --variant id --epochs 50 --seed 42 \
  >"$OUTPUT_DIR/id/train.log" 2>&1

.venv/bin/livebridge generative-train \
  --processed-dir "$PROCESSED_DIR" \
  --bridge-checkpoint "$BRIDGE_CHECKPOINT" \
  --content-path "$OUTPUT_DIR/alignment/aligned_author_content.npz" \
  --output-dir "$OUTPUT_DIR/content" \
  --variant content --epochs 50 --seed 42 \
  >"$OUTPUT_DIR/content/train.log" 2>&1

set +e
.venv/bin/python scripts/summarize_generative_gate.py \
  --alignment "$OUTPUT_DIR/alignment" \
  --id-run "$OUTPUT_DIR/id" \
  --content-run "$OUTPUT_DIR/content" \
  --output "$OUTPUT_DIR/gate.json" \
  >"$OUTPUT_DIR/gate.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" >"$OUTPUT_DIR/exit_code.txt"
exit "$status"
