#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p runs/agentic_budget_v3

run_one() {
  local scale="$1"
  local cache="$2"
  local epochs="$3"
  local out="runs/agentic_budget_v3/${scale}_seed42"
  mkdir -p "$out"
  set +e
  .venv/bin/python scripts/train_budgeted_escalation.py \
    --cache-dir "$cache" \
    --output-dir "$out" \
    --epochs "$epochs" \
    --seed 42 \
    >"$out/train.log" 2>&1
  local status=$?
  set -e
  printf '%s\n' "$status" >"$out/exit_code.txt"
  return "$status"
}

# The 1% run is an engineering sanity.  Effect gates are decided on 10%, where
# the rare positive escalation class has enough examples for user holdout.
run_one sanity_1pct runs/agentic_cache_1pct_seed42 80 || true
run_one gate_10pct runs/agentic_cache_10pct_seed42 200
