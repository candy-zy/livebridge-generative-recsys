#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data/KuaiLive-M3}"
PROCESSED_DIR="${PROCESSED_DIR:-${PROJECT_DIR}/data/processed/klm3_1pct_seed42}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/klm3_1pct_seed42}"

cd "${PROJECT_DIR}"
source .venv/bin/activate
mkdir -p "${RUN_ROOT}"

# A partial multi-GB CSV is still a regular file, so existence-only validation
# is insufficient after an interrupted cloud download.
test ! -e "${DATA_DIR}/live_interaction.csv.aria2"
test "$(stat -c %s "${DATA_DIR}/photo_interaction.csv")" -gt 3000000000
test "$(stat -c %s "${DATA_DIR}/live_interaction.csv")" -gt 4000000000

livebridge validate-data --data-dir "${DATA_DIR}"
livebridge prepare \
  --data-dir "${DATA_DIR}" \
  --output-dir "${PROCESSED_DIR}" \
  --sample-ratio 0.01 \
  --seed 42 | tee "${RUN_ROOT}/prepare.log"

livebridge train \
  --processed-dir "${PROCESSED_DIR}" \
  --output-dir "${RUN_ROOT}/target" \
  --mode target \
  --epochs 30 | tee "${RUN_ROOT}/target.log"

livebridge train \
  --processed-dir "${PROCESSED_DIR}" \
  --output-dir "${RUN_ROOT}/bridge" \
  --mode bridge \
  --epochs 30 | tee "${RUN_ROOT}/bridge.log"

export RUN_ROOT
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
target = json.loads((root / "target/metrics.json").read_text())
bridge = json.loads((root / "bridge/metrics.json").read_text())
summary = {"target": target["test"], "bridge": bridge["test"]}
summary["delta"] = {
    key: bridge["test"][key] - target["test"][key]
    for key in target["test"] if key != "users"
}
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
