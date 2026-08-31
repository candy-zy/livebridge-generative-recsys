#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
source .venv/bin/activate
mkdir -p runs/sasrec_tuned

for seed in 42 43 44; do
  livebridge strong-train --model sasrec \
    --processed-dir "data/processed/klm3_temporal_1pct_seed${seed}" \
    --output-dir "runs/sasrec_tuned/seed${seed}" \
    --epochs 100 --seed "${seed}" --batch-size 4096 \
    --embedding-dim 64 --max-length 50 --learning-rate 0.0005 \
    | tee "runs/sasrec_tuned/seed${seed}.log"
done

python - <<'PY'
import json
from pathlib import Path

for path in sorted(Path("runs/sasrec_tuned").glob("seed*/metrics.json")):
    data = json.loads(path.read_text())
    print(
        path.parent.name,
        "valid_r10=", data["valid"]["recall@10"],
        "test_r10=", data["test"]["recall@10"],
        "test_n10=", data["test"]["ndcg@10"],
    )
PY
