"""Non-personalized target-domain popularity baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from livebridge_rl.evaluation import evaluate_full_sort, write_evaluation


def run_popularity(processed_dir: str | Path, output_dir: str | Path) -> dict[str, object]:
    live = pd.read_csv(Path(processed_dir) / "live.csv").sort_values("timestamp")
    users = sorted(live["user_id"].unique())
    authors = sorted(live["author_id"].unique())
    u_map, a_map = {value: idx for idx, value in enumerate(users)}, {value: idx for idx, value in enumerate(authors)}
    live["u"] = live["user_id"].map(u_map)
    live["a"] = live["author_id"].map(a_map)
    train, valid, test = (live[live["split"] == name] for name in ("train", "valid", "test"))
    popularity = np.bincount(train["a"].astype(int), minlength=len(authors)).astype(np.float64)
    popularity = np.log1p(popularity)
    candidates = np.arange(len(authors), dtype=np.int64)
    train_counts = train.groupby("u").size().astype(int).to_dict()
    train_seen = {int(uid): set(group["a"].astype(int)) for uid, group in train.groupby("u")}

    def truth(frame: pd.DataFrame) -> dict[int, set[int]]:
        return {int(uid): set(group["a"].astype(int)) for uid, group in frame.groupby("u")}

    scorer = lambda _uid, _items: popularity
    valid_result, valid_rows = evaluate_full_sort(
        truth(valid), train_seen, candidates, scorer, train_counts
    )
    test_seen = {uid: set(items) for uid, items in train_seen.items()}
    for uid, group in valid.groupby("u"):
        test_seen.setdefault(int(uid), set()).update(group["a"].astype(int))
    test_result, test_rows = evaluate_full_sort(
        truth(test), test_seen, candidates, scorer, train_counts
    )
    result = {
        "model": "popularity",
        "valid": valid_result["overall"],
        "test": test_result["overall"],
        "valid_buckets": valid_result["buckets"],
        "test_buckets": test_result["buckets"],
        "users": len(users),
        "authors": len(authors),
    }
    out = Path(output_dir)
    write_evaluation(out, result, test_rows)
    valid_rows.to_csv(out / "valid_per_user_metrics.csv", index=False)
    print(json.dumps(result, indent=2), flush=True)
    return result
