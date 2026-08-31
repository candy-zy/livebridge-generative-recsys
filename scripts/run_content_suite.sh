#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
source .venv/bin/activate
mkdir -p runs/content_suite

for seed in 42 43 44; do
  livebridge content-eval \
    --processed-dir "data/processed/klm3_temporal_1pct_seed${seed}" \
    --author-profile data/KuaiLive-M3/author_profile.csv \
    --bridge-checkpoint "runs/strong_suite_seed${seed}/bridge/model.pt" \
    --output-dir "runs/content_suite/seed${seed}" \
    | tee "runs/content_suite/seed${seed}.log"
done
