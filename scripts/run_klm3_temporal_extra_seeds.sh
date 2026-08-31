#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data/KuaiLive-M3}"
cd "${PROJECT_DIR}"
source .venv/bin/activate

for seed in 43 44; do
  processed="${PROJECT_DIR}/data/processed/klm3_temporal_1pct_seed${seed}"
  run_root="${PROJECT_DIR}/runs/klm3_temporal_1pct_seed${seed}"
  mkdir -p "${run_root}"
  livebridge prepare --data-dir "${DATA_DIR}" --output-dir "${processed}" \
    --sample-ratio 0.01 --seed "${seed}" --source-mode temporal \
    | tee "${run_root}/prepare.log"
  for mode in target bridge; do
    livebridge train --processed-dir "${processed}" --output-dir "${run_root}/${mode}" \
      --mode "${mode}" --epochs 30 --seed "${seed}" | tee "${run_root}/${mode}.log"
  done
done

export PROJECT_DIR
python - <<'PY'
import json
import os
import statistics
from pathlib import Path

base = Path(os.environ["PROJECT_DIR"]) / "runs"
records = {}
for seed in (42, 43, 44):
    root = base / f"klm3_temporal_1pct_seed{seed}"
    records[seed] = {
        mode: json.loads((root / mode / "metrics.json").read_text())["test"]
        for mode in ("target", "bridge")
    }
metrics = [k for k in records[42]["target"] if k != "users"]
summary = {"seeds": records, "aggregate": {}}
for mode in ("target", "bridge"):
    summary["aggregate"][mode] = {
        key: {
            "mean": statistics.mean(records[s][mode][key] for s in records),
            "std": statistics.stdev(records[s][mode][key] for s in records),
        }
        for key in metrics
    }
summary["aggregate"]["delta"] = {
    key: {
        "mean": statistics.mean(records[s]["bridge"][key] - records[s]["target"][key] for s in records),
        "std": statistics.stdev(records[s]["bridge"][key] - records[s]["target"][key] for s in records),
    }
    for key in metrics
}
out = base / "klm3_temporal_multiseed_summary.json"
out.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary["aggregate"], indent=2))
PY
