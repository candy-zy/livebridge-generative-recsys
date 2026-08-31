#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


MODELS = ("popularity", "target", "lightgcn", "sasrec", "emcdr", "bridge")
SEEDS = (42, 43, 44)


def mean_std(values):
    return {"mean": statistics.mean(values), "std": statistics.stdev(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--sasrec-tuned-root",
        help="Optional directory containing seed{42,43,44}/metrics.json; replaces the initial SASRec runs.",
    )
    args = parser.parse_args()
    root = Path(args.runs_root)
    records = {
        seed: {
            model: json.loads((root / f"strong_suite_seed{seed}" / model / "metrics.json").read_text())
            for model in MODELS
        }
        for seed in SEEDS
    }
    if args.sasrec_tuned_root:
        tuned_root = Path(args.sasrec_tuned_root)
        for seed in SEEDS:
            records[seed]["sasrec"] = json.loads(
                (tuned_root / f"seed{seed}" / "metrics.json").read_text()
            )
    metrics = [key for key in records[42]["target"]["test"] if key != "users"]
    aggregate = {
        model: {key: mean_std([records[s][model]["test"][key] for s in SEEDS]) for key in metrics}
        for model in MODELS
    }
    aggregate["bridge_minus_target"] = {
        key: mean_std([records[s]["bridge"]["test"][key] - records[s]["target"]["test"][key] for s in SEEDS])
        for key in metrics
    }
    buckets = {}
    for model in MODELS:
        bucket_names = sorted({name for seed in SEEDS for name in records[seed][model].get("test_buckets", {})})
        buckets[model] = {}
        for bucket in bucket_names:
            buckets[model][bucket] = {}
            for metric in metrics:
                values = [records[s][model]["test_buckets"][bucket][metric]
                          for s in SEEDS if bucket in records[s][model].get("test_buckets", {})]
                if len(values) >= 2:
                    buckets[model][bucket][metric] = mean_std(values)
            buckets[model][bucket]["users"] = [
                records[s][model]["test_buckets"][bucket]["users"]
                for s in SEEDS if bucket in records[s][model].get("test_buckets", {})
            ]
    output = {"seeds": list(SEEDS), "models": list(MODELS), "aggregate": aggregate, "buckets": buckets}
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
