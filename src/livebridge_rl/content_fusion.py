"""Leakage-safe author-profile content fusion over a frozen Bridge ranker."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from livebridge_rl.baseline import BridgeBPR
from livebridge_rl.evaluation import evaluate_full_sort


PROFILE_COLUMNS = ("gender", "age_segment", "fans_user_num", "is_photo_author", "is_live_author")


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    std = float(values.std())
    return np.zeros_like(values) if std < 1e-8 else (values - float(values.mean())) / std


def run_profile_fusion(
    processed_dir: str | Path,
    author_profile_path: str | Path,
    bridge_checkpoint: str | Path,
    output_dir: str | Path,
    alpha_grid: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5),
) -> dict[str, object]:
    root, out = Path(processed_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    live = pd.read_csv(root / "live.csv").sort_values("timestamp")
    checkpoint = torch.load(bridge_checkpoint, map_location="cpu", weights_only=False)
    u_map = {int(key): int(value) for key, value in checkpoint["user_map"].items()}
    a_map = {int(key): int(value) for key, value in checkpoint["author_map"].items()}
    live = live[live.user_id.isin(u_map) & live.author_id.isin(a_map)].copy()
    live["u"] = live.user_id.map(u_map).astype(int)
    live["a"] = live.author_id.map(a_map).astype(int)
    users, authors = len(u_map), len(a_map)

    profile = pd.read_csv(author_profile_path)
    available = [column for column in PROFILE_COLUMNS if column in profile.columns]
    if not available:
        raise ValueError(f"author profile has none of the supported columns: {PROFILE_COLUMNS}")
    profile = profile[profile.author_id.isin(a_map)].copy()
    encoded = pd.get_dummies(profile[available].astype("string").fillna("unknown"), dummy_na=False)
    features = np.zeros((authors, encoded.shape[1]), dtype=np.float32)
    for index, author_id in enumerate(profile.author_id.astype(int)):
        features[a_map[author_id]] = encoded.iloc[index].to_numpy(dtype=np.float32)
    feature_norm = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(feature_norm, 1.0)

    train, valid, test = (live[live.split == name] for name in ("train", "valid", "test"))
    user_profile = np.zeros((users, features.shape[1]), dtype=np.float32)
    for uid, group in train.groupby("u"):
        item_ids = group.a.astype(int).to_numpy()
        weights = np.log1p(pd.to_numeric(group.play_duration, errors="coerce").fillna(0).to_numpy())
        if weights.sum() <= 0:
            weights = np.ones(len(group), dtype=np.float32)
        user_profile[int(uid)] = np.average(features[item_ids], axis=0, weights=weights)
    user_profile /= np.maximum(np.linalg.norm(user_profile, axis=1, keepdims=True), 1e-8)
    content_scores = user_profile @ features.T

    state = checkpoint["state_dict"]
    dim = int(state["user.weight"].shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bridge = BridgeBPR(users, authors, dim).to(device)
    bridge.load_state_dict(state)
    bridge.eval()
    all_items = torch.arange(authors, device=device)
    base_scores: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for uid in sorted(live.u.unique()):
            user_tensor = torch.full((authors,), int(uid), device=device)
            base_scores[int(uid)] = bridge.score(user_tensor, all_items).cpu().numpy()

    candidates = np.arange(authors, dtype=np.int64)
    counts = train.groupby("u").size().astype(int).to_dict()
    train_seen = {int(uid): set(group.a.astype(int)) for uid, group in train.groupby("u")}
    valid_truth = {int(uid): set(group.a.astype(int)) for uid, group in valid.groupby("u")}
    test_truth = {int(uid): set(group.a.astype(int)) for uid, group in test.groupby("u")}
    test_seen = {uid: set(items) for uid, items in train_seen.items()}
    for uid, group in valid.groupby("u"):
        test_seen.setdefault(int(uid), set()).update(group.a.astype(int))

    def scorer(alpha: float):
        return lambda uid, item_ids: (
            _zscore(base_scores[uid])[item_ids] + alpha * content_scores[uid, item_ids]
        )

    validation_grid: list[dict[str, float]] = []
    best_alpha, best_key = 0.0, (-float("inf"), -float("inf"))
    for alpha in alpha_grid:
        evaluation, _ = evaluate_full_sort(valid_truth, train_seen, candidates, scorer(alpha), counts)
        overall = evaluation["overall"]
        validation_grid.append({"alpha": alpha, **overall})
        key = (float(overall["ndcg@10"]), float(overall["recall@10"]))
        if key > best_key:
            best_alpha, best_key = alpha, key

    reference_valid, _ = evaluate_full_sort(valid_truth, train_seen, candidates, scorer(0.0), counts)
    fused_valid, valid_rows = evaluate_full_sort(valid_truth, train_seen, candidates, scorer(best_alpha), counts)
    reference_test, _ = evaluate_full_sort(test_truth, test_seen, candidates, scorer(0.0), counts)
    fused_test, test_rows = evaluate_full_sort(test_truth, test_seen, candidates, scorer(best_alpha), counts)
    result: dict[str, object] = {
        "model": "creator_bridge_profile_fusion",
        "device": str(device),
        "profile_columns": available,
        "encoded_features": int(features.shape[1]),
        "authors_with_profile": int(np.count_nonzero(np.linalg.norm(features, axis=1))),
        "alpha_grid": validation_grid,
        "selected_alpha": best_alpha,
        "reference_valid": reference_valid["overall"],
        "valid": fused_valid["overall"],
        "valid_buckets": fused_valid["buckets"],
        "reference_test": reference_test["overall"],
        "test": fused_test["overall"],
        "test_buckets": fused_test["buckets"],
        "audit": {
            "selection_split": "valid",
            "final_evaluation_split": "test",
            "profile_is_static": True,
            "future_live_content_used": False,
        },
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    valid_rows.to_csv(out / "valid_per_user_metrics.csv", index=False)
    test_rows.to_csv(out / "per_user_metrics.csv", index=False)
    print(json.dumps(result, indent=2), flush=True)
    return result
