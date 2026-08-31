#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
source .venv/bin/activate
processed=data/processed/klm3_temporal_1pct_seed42
mkdir -p runs/sasrec_tune
for lr in 0.0005 0.001 0.002; do
  tag="lr${lr/./p}"
  livebridge strong-train --model sasrec --processed-dir "${processed}" \
    --output-dir "runs/sasrec_tune/${tag}" --epochs 100 --seed 42 \
    --batch-size 4096 --embedding-dim 64 --max-length 50 --learning-rate "${lr}" \
    | tee "runs/sasrec_tune/${tag}.log"
done
python - <<'PY'
import json
from pathlib import Path
for path in sorted(Path('runs/sasrec_tune').glob('*/metrics.json')):
    data=json.loads(path.read_text())
    print(path.parent.name, data['valid']['recall@10'], data['valid']['ndcg@10'], data['test']['recall@10'])
PY
