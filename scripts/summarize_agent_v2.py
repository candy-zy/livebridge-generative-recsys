#!/usr/bin/env python3
"""Create structured three-seed and ablation summaries for Agent V2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    "session_return", "recall@10", "ndcg@10", "logged_watch@10",
    "action_switch_rate", "action_entropy", "catalog_coverage@10",
    "exposure_gini@10", "policy_latency_ms_p95",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    root = args.root
    runs = [
        ("fixed_action_1", 42, "baseline", root / "main_10pct_seed42/fixed_action_1/metrics.json"),
        ("agent_v2_seed42", 42, "main", root / "main_10pct_seed42/agent_v2/metrics.json"),
        ("agent_v2_seed43", 43, "main", root / "main_10pct_seed43/agent_v2/metrics.json"),
        ("agent_v2_seed44", 44, "main", root / "main_10pct_seed44/agent_v2/metrics.json"),
        ("dense_only", 42, "ablation", root / "ablations_10pct_seed42/dense_only/metrics.json"),
        ("no_memory", 42, "ablation", root / "ablations_10pct_seed42/no_memory_v2/metrics.json"),
    ]
    rows = []
    documents = {}
    for name, seed, role, path in runs:
        document = load(path)
        documents[name] = document
        checkpoint = document.get("selected_checkpoint", "not_applicable")
        if document.get("config", {}).get("epochs") == 0 and document.get("config", {}).get("warmup_epochs", 0) > 0:
            checkpoint = "reward_warm_start"
        for split in ("valid", "test"):
            overall = document[split]["overall"]
            row = {
                "run": name,
                "seed": seed,
                "role": role,
                "split": split,
                "selected_checkpoint": checkpoint,
            }
            row.update({metric: overall[metric] for metric in METRICS})
            row["action_counts"] = json.dumps(overall["action_counts"])
            rows.append(row)
    output_csv = root / "summary_v2.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    baseline = documents["fixed_action_1"]["test"]["overall"]
    main_docs = [documents[f"agent_v2_seed{seed}"] for seed in (42, 43, 44)]
    aggregate = {}
    for metric in METRICS:
        values = np.asarray([doc["test"]["overall"][metric] for doc in main_docs], dtype=float)
        base = float(baseline[metric])
        aggregate[metric] = {
            "baseline": base,
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "relative_delta_percent": float((values.mean() / base - 1.0) * 100.0) if base else None,
            "values": values.tolist(),
        }
    baseline_users = pd.read_csv(root / "main_10pct_seed42/fixed_action_1/per_user_metrics.csv").set_index("user_id")
    agent_users = [
        pd.read_csv(root / f"main_10pct_seed{seed}/agent_v2/per_user_metrics.csv").set_index("user_id")
        for seed in (42, 43, 44)
    ]
    shared_users = baseline_users.index
    for frame in agent_users:
        shared_users = shared_users.intersection(frame.index)
    rng = np.random.default_rng(20260830)
    bootstrap = {}
    for metric in ("session_return", "recall@10", "ndcg@10"):
        baseline_values = baseline_users.loc[shared_users, metric].to_numpy(dtype=float)
        agent_values = np.stack([
            frame.loc[shared_users, metric].to_numpy(dtype=float) for frame in agent_users
        ]).mean(axis=0)
        paired_delta = agent_values - baseline_values
        samples = np.empty(2000, dtype=float)
        for iteration in range(len(samples)):
            indices = rng.integers(0, len(paired_delta), size=len(paired_delta))
            samples[iteration] = paired_delta[indices].mean()
        bootstrap[metric] = {
            "users": int(len(paired_delta)),
            "mean_paired_delta": float(paired_delta.mean()),
            "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
            "probability_delta_positive": float(np.mean(samples > 0)),
            "per_user_win_rate": float(np.mean(paired_delta > 0)),
        }

    def selected_checkpoint(document: dict) -> str:
        if document.get("config", {}).get("epochs") == 0 and document.get("config", {}).get("warmup_epochs", 0) > 0:
            return "reward_warm_start"
        return document.get("selected_checkpoint", "not_applicable")

    result = {
        "seeds": [42, 43, 44],
        "main_seed_count": 3,
        "counterfactual_grpo_selected_seeds": [
            seed for seed, doc in zip((42, 43, 44), main_docs)
            if selected_checkpoint(doc) == "grpo_final"
        ],
        "test_aggregate": aggregate,
        "paired_user_bootstrap": bootstrap,
        "ablations_test": {
            name: {
                "selected_checkpoint": selected_checkpoint(documents[name]),
                **{metric: documents[name]["test"]["overall"][metric] for metric in METRICS},
            }
            for name in ("dense_only", "no_memory")
        },
        "claim_boundary": "offline logged-positive replay; no online CTR or revenue claim",
    }
    (root / "aggregate_v2.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
