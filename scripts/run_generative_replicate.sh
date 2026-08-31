#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SEED="${1:?seed is required}"
PROCESSED_DIR="${2:?processed directory is required}"
BRIDGE_CHECKPOINT="${3:?bridge checkpoint is required}"
OUTPUT_DIR="${4:?output directory is required}"
DATA_DIR="${5:-data/KuaiLive-M3}"

mkdir -p "$OUTPUT_DIR/alignment" "$OUTPUT_DIR/id" "$OUTPUT_DIR/fusion"

.venv/bin/livebridge content-align \
  --data-dir "$DATA_DIR" --processed-dir "$PROCESSED_DIR" \
  --output-dir "$OUTPUT_DIR/alignment" --epochs 100 --seed "$SEED" \
  >"$OUTPUT_DIR/alignment/train.log" 2>&1

.venv/bin/livebridge generative-train \
  --processed-dir "$PROCESSED_DIR" --bridge-checkpoint "$BRIDGE_CHECKPOINT" \
  --content-path "$OUTPUT_DIR/alignment/aligned_author_content.npz" \
  --output-dir "$OUTPUT_DIR/id" --variant id --epochs 50 --seed "$SEED" \
  >"$OUTPUT_DIR/id/train.log" 2>&1

.venv/bin/livebridge generative-train \
  --processed-dir "$PROCESSED_DIR" --bridge-checkpoint "$BRIDGE_CHECKPOINT" \
  --content-path "$OUTPUT_DIR/alignment/aligned_author_content.npz" \
  --output-dir "$OUTPUT_DIR/fusion" --variant fusion --epochs 50 --seed "$SEED" \
  >"$OUTPUT_DIR/fusion/train.log" 2>&1

.venv/bin/python scripts/summarize_generative_gate.py \
  --alignment "$OUTPUT_DIR/alignment" --id-run "$OUTPUT_DIR/id" \
  --content-run "$OUTPUT_DIR/fusion" --output "$OUTPUT_DIR/gate.json" \
  >"$OUTPUT_DIR/gate.log" 2>&1

printf '0\n' >"$OUTPUT_DIR/exit_code.txt"
