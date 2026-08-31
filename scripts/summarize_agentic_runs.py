"""Create an auditable flat table from Agentic RL metrics.json artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FIELDS = (
    "session_return", "recall@10", "ndcg@10", "logged_watch@10",
    "longtail_share@10", "repeat_rate@10", "action_switch_rate",
    "action_entropy", "catalog_coverage@10", "exposure_gini@10",
    "policy_latency_ms_p50", "policy_latency_ms_p95",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.rglob("metrics.json")):
        payload = json.loads(path.read_text())
        for split in ("valid", "test"):
            metrics = payload[split]["overall"]
            row = {
                "run": str(path.parent.relative_to(args.root)),
                "variant": payload["variant"],
                "seed": payload["config"]["seed"],
                "split": split,
                "selected_checkpoint": payload.get("selected_checkpoint", "legacy"),
            }
            row.update({field: metrics.get(field) for field in FIELDS})
            row["action_counts"] = json.dumps(metrics.get("action_counts", []))
            rows.append(row)
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
