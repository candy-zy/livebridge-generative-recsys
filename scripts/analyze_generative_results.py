#!/usr/bin/env python3
"""Aggregate three-seed generative retrieval results with paired uncertainty."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
METRICS = ("recall@10", "ndcg@10", "recall@20", "ndcg@20", "recall@40", "ndcg@40")


def run_dir(seed: int, variant: str) -> Path:
    base = ROOT / "runs" / f"generative_stage1_1pct_seed{seed}"
    if seed == 42 and variant == "fusion":
        return ROOT / "runs" / "generative_fusion_1pct_seed42"
    return base / variant


def main() -> None:
    rows: list[dict[str, object]] = []
    paired: dict[str, list[np.ndarray]] = {metric: [] for metric in METRICS}
    split_audit: dict[str, bool] = {}
    for seed in SEEDS:
        for variant in ("id", "fusion"):
            result = json.loads((run_dir(seed, variant) / "metrics.json").read_text())
            rows.append({"seed": seed, "model": variant, **result["test"]})
        sasrec = json.loads(
            (ROOT / "runs" / "sasrec_tuned" / f"seed{seed}" / "metrics.json").read_text()
        )
        rows.append({"seed": seed, "model": "sasrec_tuned", **sasrec["test"]})
        id_users = pd.read_csv(run_dir(seed, "id") / "per_user_metrics.csv")
        fusion_users = pd.read_csv(run_dir(seed, "fusion") / "per_user_metrics.csv")
        sasrec_users = pd.read_csv(
            ROOT / "runs" / "sasrec_tuned" / f"seed{seed}" / "per_user_metrics.csv"
        )
        keys = ["user_id", "train_interactions", "bucket"]
        split_audit[str(seed)] = id_users[keys].equals(fusion_users[keys]) and id_users[keys].equals(
            sasrec_users[keys]
        )
        for metric in METRICS:
            paired[metric].append(
                fusion_users[metric].to_numpy() - id_users[metric].to_numpy()
            )

    raw = pd.DataFrame(rows)
    out = ROOT / "runs" / "generative_summary_1pct"
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "raw_three_seed.csv", index=False)

    aggregate_rows = []
    for model, group in raw.groupby("model", sort=False):
        for metric in METRICS:
            aggregate_rows.append({
                "model": model,
                "metric": metric,
                "mean": group[metric].mean(),
                "std": group[metric].std(ddof=1),
            })
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(out / "mean_std.csv", index=False)

    rng = np.random.default_rng(20260830)
    bootstrap = {}
    for metric, seed_differences in paired.items():
        observed = float(np.mean([values.mean() for values in seed_differences]))
        samples = np.empty(10_000, dtype=np.float64)
        for index in range(len(samples)):
            per_seed = [
                values[rng.integers(0, len(values), len(values))].mean()
                for values in seed_differences
            ]
            samples[index] = np.mean(per_seed)
        bootstrap[metric] = {
            "fusion_minus_id_mean": observed,
            "ci95_low": float(np.quantile(samples, 0.025)),
            "ci95_high": float(np.quantile(samples, 0.975)),
            "all_seed_directions_positive": bool(
                all(values.mean() > 0 for values in seed_differences)
            ),
        }

    means = raw.groupby("model")[list(METRICS)].mean()
    relative = {}
    for baseline in ("id", "sasrec_tuned"):
        relative[baseline] = {
            metric: float((means.loc["fusion", metric] / means.loc[baseline, metric] - 1.0))
            for metric in METRICS
        }
    summary = {
        "seeds": list(SEEDS),
        "sampling": "1% users per seed; chronological source-train/valid/test split",
        "split_audit_same_users_train_counts_buckets": split_audit,
        "paired_hierarchical_user_bootstrap": bootstrap,
        "relative_gain_of_fusion": relative,
    }
    (out / "statistics.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Generative retrieval 1% three-seed summary",
        "",
        "All values are test metrics; dispersion is sample standard deviation across seeds 42/43/44.",
        "",
        "| Model | Recall@10 | NDCG@10 | Recall@40 | NDCG@40 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in ("sasrec_tuned", "id", "fusion"):
        group = raw[raw.model == model]
        cell = lambda metric: f"{group[metric].mean():.4f} ± {group[metric].std(ddof=1):.4f}"
        lines.append(
            f"| {model} | {cell('recall@10')} | {cell('ndcg@10')} | "
            f"{cell('recall@40')} | {cell('ndcg@40')} |"
        )
    lines.extend(["", "## Paired fusion - ID bootstrap", ""])
    for metric in ("recall@10", "ndcg@10", "recall@40", "ndcg@40"):
        item = bootstrap[metric]
        lines.append(
            f"- {metric}: {item['fusion_minus_id_mean']:+.4f}, "
            f"95% CI [{item['ci95_low']:+.4f}, {item['ci95_high']:+.4f}]"
        )
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
