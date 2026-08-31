#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"
source .venv/bin/activate
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

processed="data/processed/klm3_temporal_full_seed42"
tune_root="runs/lightgcn_tune_seed42"
mkdir -p "${tune_root}"

run_one() {
  local output="$1"
  local seed="$2"
  local learning_rate="$3"
  local epochs="$4"
  mkdir -p "${output}"
  date +%s > "${output}/start.txt"
  livebridge strong-train --model lightgcn \
    --processed-dir "${processed}" --output-dir "${output}" \
    --epochs "${epochs}" --seed "${seed}" --learning-rate "${learning_rate}" \
    --batch-size 4096 --embedding-dim 64 > "${output}/run.log" 2>&1
  date +%s > "${output}/end.txt"
}

for spec in \
  "lr0p005_e80 0.005 80" \
  "lr0p01_e80 0.01 80" \
  "lr0p005_e160 0.005 160" \
  "lr0p01_e160 0.01 160"
do
  read -r name learning_rate epochs <<< "${spec}"
  run_one "${tune_root}/${name}" 42 "${learning_rate}" "${epochs}"
done

best_spec="$(python - "${tune_root}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in root.glob("*/metrics.json"):
    metrics = json.loads(path.read_text())
    cfg = metrics["config"]
    rows.append((
        float(metrics["valid"]["ndcg@10"]),
        float(metrics["valid"]["recall@10"]),
        path.parent.name,
        float(cfg["learning_rate"]),
        int(cfg["epochs"]),
    ))
best = max(rows)
print(best[2], best[3], best[4])
PY
)"
read -r best_name best_lr best_epochs <<< "${best_spec}"

final_seed42="runs/scale_full_lightgcn_tuned_seed42"
test ! -e "${final_seed42}"
cp -a "${tune_root}/${best_name}" "${final_seed42}"

for seed in 43 44; do
  run_one "runs/scale_full_lightgcn_tuned_seed${seed}" \
    "${seed}" "${best_lr}" "${best_epochs}"
done

python - "${best_name}" "${best_lr}" "${best_epochs}" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

best_name, best_lr, best_epochs = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
seeds = [42, 43, 44]
rows = []
for seed in seeds:
    path = Path(f"runs/scale_full_lightgcn_tuned_seed{seed}/metrics.json")
    rows.append(json.loads(path.read_text()))

keys = ("recall@10", "ndcg@10", "recall@20", "ndcg@20", "recall@40", "ndcg@40")
aggregate = {}
for key in keys:
    values = np.asarray([row["test"][key] for row in rows], dtype=float)
    aggregate[key] = {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "values": values.tolist(),
    }

summary = {
    "selection_split": "valid",
    "selection_metric": "ndcg@10",
    "best_tuning_run": best_name,
    "learning_rate": best_lr,
    "epochs": best_epochs,
    "seeds": seeds,
    "aggregate": aggregate,
}
Path("runs/scale_full_lightgcn_tuned_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(json.dumps(summary, indent=2))
PY
