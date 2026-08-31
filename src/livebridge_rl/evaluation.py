"""Shared full-sort evaluation and cold-start bucket reporting."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd


def history_bucket(count: int) -> str:
    if count <= 10:
        return "5-10"
    if count <= 30:
        return "11-30"
    return "31+"


def ranking_row(ranking: Iterable[int], truth: set[int], ks=(10, 20, 40)) -> dict[str, float]:
    ranked = list(ranking)
    row: dict[str, float] = {}
    for k in ks:
        hits = [1 if item in truth else 0 for item in ranked[:k]]
        row[f"recall@{k}"] = sum(hits) / len(truth) if truth else 0.0
        dcg = sum(hit / math.log2(position + 2) for position, hit in enumerate(hits))
        ideal = sum(1 / math.log2(position + 2) for position in range(min(len(truth), k)))
        row[f"ndcg@{k}"] = dcg / ideal if ideal else 0.0
    return row


def evaluate_full_sort(
    user_truth: dict[int, set[int]],
    seen: dict[int, set[int]],
    candidates: np.ndarray,
    score_fn: Callable[[int, np.ndarray], np.ndarray],
    train_counts: dict[int, int],
    ks=(10, 20, 40),
) -> tuple[dict[str, object], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    top_k = min(max(ks), len(candidates))
    for user_id, truth in user_truth.items():
        if not truth:
            continue
        scores = np.asarray(score_fn(user_id, candidates), dtype=np.float64).copy()
        if scores.shape != (len(candidates),):
            raise ValueError(f"score_fn returned {scores.shape}; expected {(len(candidates),)}")
        blocked = seen.get(user_id, set()) - truth
        if blocked:
            scores[np.fromiter(blocked, dtype=np.int64)] = -np.inf
        if top_k == len(candidates):
            order = np.argsort(-scores)
        else:
            shortlist = np.argpartition(-scores, top_k - 1)[:top_k]
            order = shortlist[np.argsort(-scores[shortlist])]
        count = int(train_counts.get(user_id, 0))
        row: dict[str, object] = {
            "user_id": user_id,
            "train_interactions": count,
            "bucket": history_bucket(count),
        }
        row.update(ranking_row(order.tolist(), truth, ks))
        rows.append(row)
    per_user = pd.DataFrame(rows)
    metric_names = [f"{name}@{k}" for k in ks for name in ("recall", "ndcg")]
    overall = {name: float(per_user[name].mean()) for name in metric_names}
    overall["users"] = len(per_user)
    buckets: dict[str, dict[str, float | int]] = {}
    for bucket, group in per_user.groupby("bucket", sort=False):
        buckets[str(bucket)] = {name: float(group[name].mean()) for name in metric_names}
        buckets[str(bucket)]["users"] = len(group)
    return {"overall": overall, "buckets": buckets}, per_user


def evaluate_rankings(
    user_truth: dict[int, set[int]],
    rankings: dict[int, Iterable[int]],
    train_counts: dict[int, int],
    ks=(10, 20, 40),
    candidate_count: int | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate precomputed rankings without materializing full item-score vectors.

    This is equivalent to :func:`evaluate_full_sort` when each supplied ranking
    contains at least ``max(ks)`` items from the full-sort order.  It is useful
    for rerankers that only modify a frozen Top-M candidate pool where M is at
    least the largest requested cutoff.
    """
    if not ks:
        raise ValueError("ks must not be empty")
    rows: list[dict[str, object]] = []
    for user_id, truth in user_truth.items():
        if not truth:
            continue
        ranking = list(rankings.get(user_id, []))
        count = int(train_counts.get(user_id, 0))
        row: dict[str, object] = {
            "user_id": user_id,
            "train_interactions": count,
            "bucket": history_bucket(count),
        }
        row.update(ranking_row(ranking, truth, ks))
        rows.append(row)
    per_user = pd.DataFrame(rows)
    metric_names = [f"{name}@{k}" for k in ks for name in ("recall", "ndcg")]
    overall = {name: float(per_user[name].mean()) for name in metric_names}
    overall["users"] = len(per_user)
    buckets: dict[str, dict[str, float | int]] = {}
    for bucket, group in per_user.groupby("bucket", sort=False):
        buckets[str(bucket)] = {name: float(group[name].mean()) for name in metric_names}
        buckets[str(bucket)]["users"] = len(group)
    return {"overall": overall, "buckets": buckets}, per_user


def write_evaluation(output_dir: str | Path, result: dict[str, object], per_user: pd.DataFrame) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    per_user.to_csv(out / "per_user_metrics.csv", index=False)
