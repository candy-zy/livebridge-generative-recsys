#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SEEDS = (42, 43, 44)


def stats(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "std": statistics.stdev(values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--suite-dir", default="grpo_suite")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.runs_root)
    records = {
        seed: json.loads((root / args.suite_dir / f"seed{seed}" / "metrics.json").read_text())
        for seed in SEEDS
    }
    accuracy_keys = [key for key in records[42]["test"] if key != "users"]
    exposure_keys = [key for key in records[42]["test_exposure"] if key != "users"]
    output = {
        "seeds": list(SEEDS),
        "reference": {
            key: stats([records[seed]["reference_test"][key] for seed in SEEDS])
            for key in accuracy_keys
        },
        "grpo": {
            key: stats([records[seed]["test"][key] for seed in SEEDS])
            for key in accuracy_keys
        },
        "grpo_minus_reference": {
            key: stats([
                records[seed]["test"][key] - records[seed]["reference_test"][key]
                for seed in SEEDS
            ])
            for key in accuracy_keys
        },
        "reference_exposure": {
            key: stats([records[seed]["reference_exposure"][key] for seed in SEEDS])
            for key in exposure_keys
        },
        "grpo_exposure": {
            key: stats([records[seed]["test_exposure"][key] for seed in SEEDS])
            for key in exposure_keys
        },
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
