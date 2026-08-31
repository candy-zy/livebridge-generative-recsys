"""Constrained group-relative policy optimization for long-horizon live reranking.

The policy reranks a frozen Creator-Bridge candidate pool.  Training uses only
logged validation positives; test interactions remain untouched until final
evaluation.  This is a logged-positive policy experiment, not unbiased OPE.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from livebridge_rl.baseline import BridgeBPR
from livebridge_rl.evaluation import evaluate_rankings


@dataclass
class GRPOConfig:
    epochs: int = 30
    learning_rate: float = 1e-2
    candidate_pool: int = 50
    slate_size: int = 10
    group_size: int = 8
    temperature: float = 0.8
    clip_epsilon: float = 0.2
    kl_beta: float = 0.02
    residual_scale: float = 0.35
    discount: float = 0.9
    relevance_weight: float = 1.0
    watch_weight: float = 0.25
    source_weight: float = 0.10
    profile_weight: float = 0.05
    longtail_weight: float = 0.05
    score_batch_size: int = 128
    seed: int = 42


@dataclass
class CandidateFeatures:
    """Compact per-user Top-M cache used by both training and evaluation."""

    items: np.ndarray
    features: np.ndarray


class ResidualPolicy(nn.Module):
    """A deliberately small, auditable residual over a frozen reference ranker."""

    FEATURE_NAMES = ("base", "source_affinity", "profile_affinity", "popularity", "longtail")

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(len(self.FEATURE_NAMES)))
        self.bias = nn.Parameter(torch.zeros(()))

    def logits(
        self,
        reference_logits: torch.Tensor,
        features: torch.Tensor,
        residual_scale: float,
    ) -> torch.Tensor:
        residual = torch.tanh(features @ self.weight + self.bias)
        return reference_logits + residual_scale * residual


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    std = float(values.std())
    if std < 1e-8:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


def _build_sparse_source_index(
    photo: pd.DataFrame,
    u_map: dict[int, int],
    a_map: dict[int, int],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Store only observed source affinities instead of a users×authors matrix."""
    photo = photo[photo.user_id.isin(u_map) & photo.author_id.isin(a_map)].copy()
    if photo.empty:
        return {}
    photo["u"] = photo.user_id.map(u_map).astype(np.int64)
    photo["a"] = photo.author_id.map(a_map).astype(np.int64)
    values = np.log1p(pd.to_numeric(photo.engagement, errors="coerce").fillna(0).to_numpy())
    maximum = float(values.max()) or 1.0
    photo["source_value"] = (values / maximum).astype(np.float32)
    index: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for uid, group in photo.groupby("u", sort=False):
        ordered = group.sort_values("a")
        index[int(uid)] = (
            ordered.a.to_numpy(dtype=np.int64),
            ordered.source_value.to_numpy(dtype=np.float32),
        )
    return index


def _sparse_source_values(
    index: dict[int, tuple[np.ndarray, np.ndarray]], uid: int, items: np.ndarray
) -> np.ndarray:
    output = np.zeros(len(items), dtype=np.float32)
    entry = index.get(uid)
    if entry is None or not len(items):
        return output
    source_items, source_values = entry
    positions = np.searchsorted(source_items, items)
    valid = positions < len(source_items)
    if valid.any():
        valid_rows = np.flatnonzero(valid)
        matched = source_items[positions[valid]] == items[valid]
        rows = valid_rows[matched]
        output[rows] = source_values[positions[rows]]
    return output


def _build_candidate_cache(
    reference: BridgeBPR,
    user_ids: list[int],
    seen: dict[int, set[int]],
    candidate_pool: int,
    score_batch_size: int,
    source_index: dict[int, tuple[np.ndarray, np.ndarray]],
    user_profile: np.ndarray | None,
    author_features: np.ndarray | None,
    pop_feature: np.ndarray,
    longtail: np.ndarray,
    device: torch.device,
    independent_tool_pools: bool = False,
) -> dict[int, CandidateFeatures]:
    """GPU-batch full-sort scoring with optional true multi-tool recall.

    The legacy cache retained only the BridgeBPR Top-M and let every "tool"
    rerank that same set.  With ``independent_tool_pools=True``, Top-M is drawn
    independently from bridge, source, profile/content, popularity, and
    long-tail generators, then unioned.  This gives routing actions actual
    retrieval authority instead of five views of an already-fixed pool.
    ``candidate_pool`` is the per-tool budget in that mode.
    """
    cache: dict[int, CandidateFeatures] = {}
    author_weight = reference.author.weight.detach()
    author_bias = reference.author_bias.weight.detach().squeeze(-1)
    authors = author_weight.shape[0]
    batch_size = max(1, score_batch_size)
    popular_order = np.argsort(-pop_feature)

    def unseen_prefix(order: np.ndarray, blocked: set[int], limit: int) -> list[int]:
        output: list[int] = []
        for item in order:
            value = int(item)
            if value in blocked:
                continue
            output.append(value)
            if len(output) >= limit:
                break
        return output

    with torch.no_grad():
        for start in range(0, len(user_ids), batch_size):
            batch = user_ids[start:start + batch_size]
            uid_tensor = torch.as_tensor(batch, device=device, dtype=torch.long)
            raw = reference.user(uid_tensor) @ author_weight.T + author_bias.unsqueeze(0)
            means = raw.mean(dim=1, keepdim=True)
            stds = raw.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-8)
            for row, uid in enumerate(batch):
                blocked = seen.get(uid, set())
                if blocked:
                    blocked_ids = torch.as_tensor(
                        list(blocked), device=device, dtype=torch.long
                    )
                    raw[row, blocked_ids] = -torch.inf
            pool_size = min(candidate_pool, authors)
            top_values, top_items = torch.topk(raw, pool_size, dim=1)
            base_values = (top_values - means) / stds
            for row, uid in enumerate(batch):
                finite = torch.isfinite(top_values[row])
                base_items = top_items[row][finite].cpu().numpy().astype(np.int64, copy=False)
                if independent_tool_pools:
                    blocked = seen.get(uid, set())
                    generators: list[list[int]] = [base_items.tolist()]
                    source_entry = source_index.get(uid)
                    if source_entry is not None:
                        source_items, source_values = source_entry
                        source_order = source_items[np.argsort(-source_values)]
                        generators.append(unseen_prefix(source_order, blocked, pool_size))
                    if user_profile is not None and author_features is not None:
                        profile_scores = author_features @ user_profile[uid]
                        profile_order = np.argsort(-profile_scores)
                        generators.append(unseen_prefix(profile_order, blocked, pool_size))
                    generators.append(unseen_prefix(popular_order, blocked, pool_size))
                    tail_scores = raw[row].clone()
                    tail_mask = torch.as_tensor(longtail <= 0, device=device)
                    tail_scores[tail_mask] = -torch.inf
                    tail_items = torch.topk(tail_scores, pool_size).indices
                    tail_items = tail_items[torch.isfinite(tail_scores[tail_items])]
                    generators.append(tail_items.cpu().numpy().astype(np.int64).tolist())
                    ordered_union = list(dict.fromkeys(
                        item for generator in generators for item in generator
                    ))
                    items = np.asarray(ordered_union, dtype=np.int64)
                    item_tensor = torch.as_tensor(items, device=device, dtype=torch.long)
                    base = (
                        (raw[row, item_tensor] - means[row, 0]) / stds[row, 0]
                    ).cpu().numpy().astype(np.float32, copy=False)
                else:
                    items = base_items
                    base = base_values[row][finite].cpu().numpy().astype(np.float32, copy=False)
                source = _sparse_source_values(source_index, uid, items)
                if user_profile is not None and author_features is not None:
                    profile = (author_features[items] @ user_profile[uid]).astype(np.float32)
                else:
                    profile = np.zeros(len(items), dtype=np.float32)
                features = np.column_stack((
                    base, source, profile, pop_feature[items], longtail[items]
                )).astype(np.float32, copy=False)
                cache[uid] = CandidateFeatures(items=items, features=features)
            del raw, means, stds, top_values, top_items, base_values
    return cache


def _plackett_log_prob(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Mean per-position log probability for ordered samples without replacement."""
    if actions.ndim == 1:
        actions = actions.unsqueeze(0)
    expanded = logits.unsqueeze(0).expand(actions.shape[0], -1)
    masked = expanded.clone()
    result = torch.zeros(actions.shape[0], device=logits.device)
    for position in range(actions.shape[1]):
        selected = actions[:, position]
        result += torch.log_softmax(masked, dim=1).gather(1, selected[:, None]).squeeze(1)
        masked = masked.scatter(1, selected[:, None], -torch.inf)
    return result / max(1, actions.shape[1])


def _sample_slates(logits: torch.Tensor, group_size: int, slate_size: int) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=0)
    samples = [torch.multinomial(probabilities, slate_size, replacement=False) for _ in range(group_size)]
    return torch.stack(samples)


def _group_rewards(
    actions: torch.Tensor,
    truth_mask: torch.Tensor,
    watch_value: torch.Tensor,
    source_value: torch.Tensor,
    profile_value: torch.Tensor,
    longtail_value: torch.Tensor,
    cfg: GRPOConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    positions = torch.arange(actions.shape[1], device=actions.device)
    discount = cfg.discount ** positions
    hit = truth_mask[actions]
    ideal_hits = min(int(truth_mask.sum().item()), actions.shape[1])
    ideal = discount[:ideal_hits].sum().clamp_min(1e-8)
    relevance = (hit * discount).sum(1) / ideal
    watch = (watch_value[actions] * hit * discount).sum(1) / ideal
    source = source_value[actions].mean(1)
    longtail = longtail_value[actions].mean(1)
    total = (
        cfg.relevance_weight * relevance
        + cfg.watch_weight * watch
        + cfg.source_weight * source
        + cfg.profile_weight * profile_value[actions].mean(1)
        + cfg.longtail_weight * longtail
    )
    components = {
        "reward": float(total.mean().item()),
        "relevance": float(relevance.mean().item()),
        "watch": float(watch.mean().item()),
        "source": float(source.mean().item()),
        "profile": float(profile_value[actions].mean().item()),
        "longtail": float(longtail.mean().item()),
    }
    return total, components


def _exposure_gini(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.sum() <= 0:
        return 0.0
    ordered = np.sort(counts)
    n = len(ordered)
    return float((2 * np.dot(np.arange(1, n + 1), ordered) / (n * ordered.sum())) - (n + 1) / n)


def _slate_metrics(
    rankings: dict[int, list[int]],
    truth: dict[int, set[int]],
    duration: dict[tuple[int, int], float],
    candidate_cache: dict[int, CandidateFeatures],
    authors: int,
    k: int = 10,
    discount: float = 0.9,
) -> dict[str, float | int]:
    exposures = np.zeros(authors, dtype=np.int64)
    utilities, watches, affinities, profile_affinities, tails = [], [], [], [], []
    for uid, relevant in truth.items():
        slate = rankings.get(uid, [])[:k]
        if not slate:
            continue
        exposures[slate] += 1
        weights = discount ** np.arange(len(slate))
        ideal = weights[: min(len(relevant), k)].sum() or 1.0
        hit = np.asarray([item in relevant for item in slate], dtype=np.float32)
        utilities.append(float((hit * weights).sum() / ideal))
        watch = np.asarray([math.log1p(duration.get((uid, item), 0.0)) for item in slate])
        watches.append(float((hit * watch * weights).sum() / ideal))
        candidate = candidate_cache[uid]
        positions = {int(item): index for index, item in enumerate(candidate.items)}
        rows = np.asarray([positions[item] for item in slate], dtype=np.int64)
        affinities.append(float(candidate.features[rows, 1].mean()))
        profile_affinities.append(float(candidate.features[rows, 2].mean()))
        tails.append(float(candidate.features[rows, 4].mean()))
    return {
        "logged_discounted_utility@10": float(np.mean(utilities)) if utilities else 0.0,
        "logged_discounted_watch@10": float(np.mean(watches)) if watches else 0.0,
        "source_affinity@10": float(np.mean(affinities)) if affinities else 0.0,
        "profile_affinity@10": float(np.mean(profile_affinities)) if profile_affinities else 0.0,
        "longtail_share@10": float(np.mean(tails)) if tails else 0.0,
        "catalog_coverage@10": float(np.count_nonzero(exposures) / max(1, authors)),
        "exposure_gini@10": _exposure_gini(exposures),
        "users": len(utilities),
    }


def train_grpo_reranker(
    processed_dir: str | Path,
    bridge_checkpoint: str | Path,
    output_dir: str | Path,
    config: GRPOConfig | None = None,
    author_profile_path: str | Path | None = None,
) -> dict[str, object]:
    cfg = config or GRPOConfig()
    if cfg.group_size < 2:
        raise ValueError("group_size must be at least 2")
    _seed_everything(cfg.seed)
    root, out = Path(processed_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    live = pd.read_csv(root / "live.csv").sort_values("timestamp")
    photo = pd.read_csv(root / "photo_author.csv")
    checkpoint = torch.load(bridge_checkpoint, map_location="cpu", weights_only=False)
    u_map = {int(key): int(value) for key, value in checkpoint["user_map"].items()}
    a_map = {int(key): int(value) for key, value in checkpoint["author_map"].items()}
    inverse_author = {value: key for key, value in a_map.items()}
    live = live[live.user_id.isin(u_map) & live.author_id.isin(a_map)].copy()
    live["u"] = live.user_id.map(u_map).astype(int)
    live["a"] = live.author_id.map(a_map).astype(int)
    users, authors = len(u_map), len(a_map)

    state = checkpoint["state_dict"]
    dim = int(state["user.weight"].shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference = BridgeBPR(users, authors, dim).to(device)
    reference.load_state_dict(state)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    target_popularity = np.zeros(authors, dtype=np.float32)
    train = live[live.split == "train"]
    for author, count in train.author_id.value_counts().items():
        target_popularity[a_map[int(author)]] = float(count)
    pop_feature = _zscore(np.log1p(target_popularity))
    positive_pop = target_popularity[target_popularity > 0]
    tail_cutoff = float(np.median(positive_pop)) if len(positive_pop) else 0.0
    longtail = (target_popularity <= tail_cutoff).astype(np.float32)

    source_index = _build_sparse_source_index(photo, u_map, a_map)
    del photo

    author_features: np.ndarray | None = None
    user_profile: np.ndarray | None = None
    profile_columns: list[str] = []
    if author_profile_path:
        author_profile = pd.read_csv(author_profile_path)
        supported = ("gender", "age_segment", "fans_user_num", "is_photo_author", "is_live_author")
        profile_columns = [column for column in supported if column in author_profile.columns]
        author_profile = author_profile[author_profile.author_id.isin(a_map)].copy()
        if profile_columns and not author_profile.empty:
            encoded = pd.get_dummies(
                author_profile[profile_columns].astype("string").fillna("unknown"), dummy_na=False
            )
            author_features = np.zeros((authors, encoded.shape[1]), dtype=np.float32)
            for index, author_id in enumerate(author_profile.author_id.astype(int)):
                author_features[a_map[author_id]] = encoded.iloc[index].to_numpy(dtype=np.float32)
            author_features /= np.maximum(np.linalg.norm(author_features, axis=1, keepdims=True), 1.0)
            user_profile = np.zeros((users, author_features.shape[1]), dtype=np.float32)
            for uid, group in train.groupby("u"):
                ids = group.a.astype(int).to_numpy()
                weights = np.log1p(pd.to_numeric(group.play_duration, errors="coerce").fillna(0).to_numpy())
                if weights.sum() <= 0:
                    weights = np.ones(len(ids), dtype=np.float32)
                user_profile[int(uid)] = np.average(author_features[ids], axis=0, weights=weights)
            user_profile /= np.maximum(np.linalg.norm(user_profile, axis=1, keepdims=True), 1e-8)

    train_seen = {int(uid): set(group.a.astype(int)) for uid, group in train.groupby("u")}
    valid = live[live.split == "valid"]
    valid_truth = {int(uid): set(group.a.astype(int)) for uid, group in valid.groupby("u")}
    valid_duration = {
        (int(row.u), int(row.a)): float(row.play_duration) for row in valid.itertuples(index=False)
    }
    max_watch = max((math.log1p(value) for value in valid_duration.values()), default=1.0)
    valid_users = sorted(valid_truth)
    ranking_pool = max(cfg.candidate_pool, 40)
    valid_cache = _build_candidate_cache(
        reference, valid_users, train_seen, ranking_pool, cfg.score_batch_size,
        source_index, user_profile, author_features, pop_feature, longtail, device,
    )

    policy = ResidualPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    rng = np.random.default_rng(cfg.seed)
    history: list[dict[str, float]] = []
    for epoch in range(cfg.epochs):
        rng.shuffle(valid_users)
        old_weight = policy.weight.detach().clone()
        old_bias = policy.bias.detach().clone()
        totals = {key: 0.0 for key in ("loss", "reward", "relevance", "watch", "source", "profile", "longtail", "kl")}
        steps = 0
        for uid in valid_users:
            candidate = valid_cache[uid]
            pool_size = min(cfg.candidate_pool, len(candidate.items))
            slate_size = min(cfg.slate_size, pool_size)
            if slate_size < 2:
                continue
            pool = candidate.items[:pool_size]
            pool_features = torch.as_tensor(candidate.features[:pool_size], device=device)
            reference_logits = pool_features[:, 0] / cfg.temperature
            old_residual = torch.tanh(pool_features @ old_weight + old_bias)
            old_logits = reference_logits + cfg.residual_scale * old_residual
            with torch.no_grad():
                actions = _sample_slates(old_logits, cfg.group_size, slate_size)
                old_log_prob = _plackett_log_prob(old_logits, actions)
            current_logits = policy.logits(reference_logits, pool_features, cfg.residual_scale)
            current_log_prob = _plackett_log_prob(current_logits, actions)
            truth_mask = torch.as_tensor(
                np.asarray([item in valid_truth[uid] for item in pool], dtype=np.float32), device=device
            )
            watch_value = torch.as_tensor(
                np.asarray([math.log1p(valid_duration.get((uid, item), 0.0)) / max_watch for item in pool], dtype=np.float32),
                device=device,
            )
            source_value = pool_features[:, 1]
            profile_value = pool_features[:, 2]
            tail_value = pool_features[:, 4]
            rewards, components = _group_rewards(
                actions, truth_mask, watch_value, source_value, profile_value, tail_value, cfg
            )
            advantage = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-6)
            ratio = torch.exp((current_log_prob - old_log_prob).clamp(-5, 5))
            unclipped = ratio * advantage
            clipped = ratio.clamp(1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * advantage
            current_log_softmax = torch.log_softmax(current_logits, dim=0)
            kl = torch.sum(torch.softmax(current_logits, dim=0) * (current_log_softmax - torch.log_softmax(reference_logits, dim=0)))
            loss = -torch.minimum(unclipped, clipped).mean() + cfg.kl_beta * kl
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss.item())
            totals["kl"] += float(kl.item())
            for key, value in components.items():
                totals[key] += value
            steps += 1
        record = {key: value / max(1, steps) for key, value in totals.items()}
        record["epoch"] = epoch + 1
        history.append(record)
        print(
            f"grpo epoch={epoch + 1:03d} loss={record['loss']:.6f} "
            f"reward={record['reward']:.6f} kl={record['kl']:.6f}", flush=True
        )

    valid_items = {int(uid): set(group.a.astype(int)) for uid, group in valid.groupby("u")}
    test_seen = {uid: set(items) for uid, items in train_seen.items()}
    for uid, items in valid_items.items():
        test_seen.setdefault(uid, set()).update(items)
    counts = train.groupby("u").size().astype(int).to_dict()
    truth = lambda frame: {int(uid): set(group.a.astype(int)) for uid, group in frame.groupby("u")}
    test = live[live.split == "test"]
    test_truth = truth(test)
    test_cache = _build_candidate_cache(
        reference, sorted(test_truth), test_seen, ranking_pool, cfg.score_batch_size,
        source_index, user_profile, author_features, pop_feature, longtail, device,
    )

    def make_rankings(
        cache: dict[int, CandidateFeatures], use_policy: bool
    ) -> dict[int, list[int]]:
        rankings: dict[int, list[int]] = {}
        for uid, candidate in cache.items():
            items = candidate.items
            pool_size = min(cfg.candidate_pool, len(items))
            pool = items[:pool_size]
            if use_policy and pool_size:
                features = torch.as_tensor(candidate.features[:pool_size], device=device)
                reference_logits = features[:, 0] / cfg.temperature
                with torch.no_grad():
                    logits = policy.logits(
                        reference_logits, features, cfg.residual_scale
                    ).cpu().numpy()
                pool = pool[np.argsort(-logits)]
            rankings[uid] = np.concatenate((pool, items[pool_size:])).tolist()
        return rankings

    reference_valid_rankings = make_rankings(valid_cache, False)
    policy_valid_rankings = make_rankings(valid_cache, True)
    reference_test_rankings = make_rankings(test_cache, False)
    policy_test_rankings = make_rankings(test_cache, True)
    reference_valid, _ = evaluate_rankings(
        valid_truth, reference_valid_rankings, counts, candidate_count=authors
    )
    policy_valid, valid_rows = evaluate_rankings(
        valid_truth, policy_valid_rankings, counts, candidate_count=authors
    )
    reference_test, _ = evaluate_rankings(
        test_truth, reference_test_rankings, counts, candidate_count=authors
    )
    policy_test, test_rows = evaluate_rankings(
        test_truth, policy_test_rankings, counts, candidate_count=authors
    )
    test_duration = {(int(row.u), int(row.a)): float(row.play_duration) for row in test.itertuples(index=False)}
    reference_exposure = _slate_metrics(
        reference_test_rankings, test_truth, test_duration, test_cache,
        authors, discount=cfg.discount,
    )
    policy_exposure = _slate_metrics(
        policy_test_rankings, test_truth, test_duration, test_cache,
        authors, discount=cfg.discount,
    )
    result: dict[str, object] = {
        "model": "creator_bridge_grpo",
        "device": str(device),
        "config": asdict(cfg),
        "feature_names": list(ResidualPolicy.FEATURE_NAMES),
        "profile_columns": profile_columns,
        "policy_weights": policy.weight.detach().cpu().tolist(),
        "policy_bias": float(policy.bias.detach().cpu().item()),
        "reference_valid": reference_valid["overall"],
        "valid": policy_valid["overall"],
        "valid_buckets": policy_valid["buckets"],
        "reference_test": reference_test["overall"],
        "test": policy_test["overall"],
        "test_buckets": policy_test["buckets"],
        "reference_exposure": reference_exposure,
        "test_exposure": policy_exposure,
        "training_history": history,
        "audit": {
            "policy_training_split": "valid",
            "final_evaluation_split": "test",
            "unbiased_ope": False,
            "claim_boundary": "logged-positive offline reranking; no online CTR/revenue claim",
            "tail_popularity_cutoff": tail_cutoff,
            "author_profile_used": bool(profile_columns),
            "raw_author_ids_preserved": len(inverse_author),
            "candidate_cache_items_per_user": ranking_pool,
            "dense_user_author_features": False,
        },
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    valid_rows.to_csv(out / "valid_per_user_metrics.csv", index=False)
    test_rows.to_csv(out / "per_user_metrics.csv", index=False)
    torch.save({"state_dict": policy.state_dict(), "config": asdict(cfg)}, out / "policy.pt")
    print(json.dumps(result, indent=2), flush=True)
    return result
