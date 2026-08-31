#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data/KuaiLive-M3}"
PROCESSED_DIR="${PROJECT_DIR}/data/processed/klm3_temporal_1pct_seed42"
RUN_ROOT="${PROJECT_DIR}/runs/klm3_temporal_1pct_seed42"

cd "${PROJECT_DIR}"
source .venv/bin/activate
mkdir -p "${RUN_ROOT}"
test ! -e "${DATA_DIR}/photo_play.parquet.aria2"
test "$(stat -c %s "${DATA_DIR}/photo_play.parquet")" -gt 2000000000

livebridge prepare \
  --data-dir "${DATA_DIR}" \
  --output-dir "${PROCESSED_DIR}" \
  --sample-ratio 0.01 --seed 42 \
  --source-mode temporal | tee "${RUN_ROOT}/prepare.log"

for mode in target bridge; do
  livebridge train \
    --processed-dir "${PROCESSED_DIR}" \
    --output-dir "${RUN_ROOT}/${mode}" \
    --mode "${mode}" --epochs 30 | tee "${RUN_ROOT}/${mode}.log"
done

export RUN_ROOT
python - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ["RUN_ROOT"])
target = json.loads((root / "target/metrics.json").read_text())
bridge = json.loads((root / "bridge/metrics.json").read_text())
summary = {"target": target["test"], "bridge": bridge["test"]}
summary["delta"] = {k: bridge["test"][k] - target["test"][k] for k in target["test"] if k != "users"}
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
