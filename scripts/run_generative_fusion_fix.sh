#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PROCESSED_DIR="${1:-data/processed/klm3_temporal_1pct_seed42}"
BRIDGE_CHECKPOINT="${2:-runs/klm3_temporal_1pct_seed42/bridge/model.pt}"
STAGE1_DIR="${3:-runs/generative_stage1_1pct_seed42}"
OUTPUT_DIR="${4:-runs/generative_fusion_1pct_seed42}"

mkdir -p "$OUTPUT_DIR"

.venv/bin/livebridge generative-train \
  --processed-dir "$PROCESSED_DIR" \
  --bridge-checkpoint "$BRIDGE_CHECKPOINT" \
  --content-path "$STAGE1_DIR/alignment/aligned_author_content.npz" \
  --output-dir "$OUTPUT_DIR" \
  --variant fusion --epochs 50 --seed 42 \
  >"$OUTPUT_DIR/train.log" 2>&1

.venv/bin/python scripts/summarize_generative_gate.py \
  --alignment "$STAGE1_DIR/alignment" \
  --id-run "$STAGE1_DIR/id" \
  --content-run "$OUTPUT_DIR" \
  --output "$OUTPUT_DIR/gate.json" \
  >"$OUTPUT_DIR/gate.log" 2>&1

printf '0\n' >"$OUTPUT_DIR/exit_code.txt"
