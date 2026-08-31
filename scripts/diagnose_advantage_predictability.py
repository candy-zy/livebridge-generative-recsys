#!/usr/bin/env python3
"""Cross-fitted audit of whether rare positive route overrides are predictable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from livebridge_rl.agentic_rl import (
    AgentConfig,
    LiveSessionEnv,
    ROUTING_ACTIONS,
    _counterfactual_action_rewards,
    load_agent_sessions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--baseline-action", type=int, default=1)
    parser.add_argument("--no-route-signatures", action="store_true")
    return parser.parse_args()


def _jaccard(left: list[int], right: list[int]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / max(1, len(a | b))


def _state_features(
    env: LiveSessionEnv, baseline_action: int, include_route_signatures: bool
) -> np.ndarray:
    observation = env._observation()  # serving-safe causal state
    if not include_route_signatures:
        return observation
    positions = {int(item): row for row, item in enumerate(env.session.items)}
    slates = [
        env.registry.retrieve(
            env.session,
            action,
            previously_recommended=env.previously_recommended,
        )[0]
        for action in range(len(ROUTING_ACTIONS))
    ]
    baseline = slates[baseline_action]
    signatures: list[float] = []
    for slate in slates:
        rows = [positions[item] for item in slate if item in positions]
        values = env.session.features[rows] if rows else np.zeros((1, 5), dtype=np.float32)
        signatures.extend([
            _jaccard(baseline, slate),
            *values.mean(axis=0).tolist(),
            *values.max(axis=0).tolist(),
            *values.std(axis=0).tolist(),
        ])
    return np.concatenate((observation, np.asarray(signatures, dtype=np.float32)))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores)
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / positives)


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return 0.5
    # Pairwise form is exact here and small enough for this diagnostic.
    wins = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(positive) * len(negative)))


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    sessions = load_agent_sessions(args.cache_dir / "valid_sessions.npz")
    cfg = AgentConfig(
        max_steps=args.max_steps,
        seed=args.seed,
        source_weight=0.0,
        longtail_weight=0.0,
    )
    vectors: list[np.ndarray] = []
    labels: list[int] = []
    users: list[int] = []
    advantages: list[float] = []
    for session in sessions:
        env = LiveSessionEnv(session, cfg)
        env.reset(seed=args.seed)
        done = False
        while not done:
            vectors.append(_state_features(
                env, args.baseline_action, not args.no_route_signatures
            ))
            rewards = np.asarray(_counterfactual_action_rewards(env), dtype=np.float32)
            advantage = float(rewards.max() - rewards[args.baseline_action])
            labels.append(int(advantage > 1e-8))
            advantages.append(advantage)
            users.append(session.user_id)
            _, _, done, _, _ = env.step(args.baseline_action)

    x = np.stack(vectors).astype(np.float32)
    y = np.asarray(labels, dtype=np.float32)
    user = np.asarray(users, dtype=np.int64)
    advantage = np.asarray(advantages, dtype=np.float32)
    # Stable user-level fold assignment prevents states from the same session
    # appearing on both sides of one cross-fit.
    unique_users = np.unique(user)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_users)
    user_fold = {int(value): index % args.folds for index, value in enumerate(unique_users)}
    folds = np.asarray([user_fold[int(value)] for value in user], dtype=np.int64)
    oof = np.zeros(len(y), dtype=np.float32)
    fold_records = []
    for fold in range(args.folds):
        train = folds != fold
        held = folds == fold
        mean = x[train].mean(axis=0, keepdims=True)
        std = x[train].std(axis=0, keepdims=True).clip(min=1e-4)
        x_train = torch.as_tensor((x[train] - mean) / std)
        y_train = torch.as_tensor(y[train, None])
        x_held = torch.as_tensor((x[held] - mean) / std)
        model = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(64, 1),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
        positives = float(y_train.sum().item())
        negatives = float(len(y_train) - positives)
        positive_weight = torch.tensor([min(50.0, negatives / max(1.0, positives))])
        for _ in range(args.epochs):
            model.train()
            logits = model(x_train)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y_train, pos_weight=positive_weight
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(x_held)).squeeze(1).numpy()
        oof[held] = scores
        fold_records.append({
            "fold": fold,
            "train_states": int(train.sum()),
            "heldout_states": int(held.sum()),
            "heldout_positives": int(y[held].sum()),
        })

    prevalence = float(y.mean())
    budget = max(1, int(round(len(y) * prevalence)))
    selected = np.argsort(-oof)[:budget]
    selected_positive = int(y[selected].sum())
    random_expected = budget * prevalence
    result = {
        "states": len(y),
        "users": len(unique_users),
        "positive_override_states": int(y.sum()),
        "positive_override_prevalence": prevalence,
        "baseline_action": args.baseline_action,
        "feature_dim": int(x.shape[1]),
        "route_signatures_in_policy_input": not args.no_route_signatures,
        "cross_fit_folds": args.folds,
        "oof_roc_auc": _roc_auc(y, oof),
        "oof_average_precision": _average_precision(y, oof),
        "top_prevalence_budget": budget,
        "top_prevalence_precision": selected_positive / budget,
        "top_prevalence_recall": selected_positive / max(1, int(y.sum())),
        "precision_lift_over_random": (selected_positive / budget) / max(prevalence, 1e-12),
        "selected_oracle_advantage_mean_all_states": float(advantage[selected].sum() / len(y)),
        "random_expected_positives_at_budget": random_expected,
        "folds": fold_records,
        "audit": {
            "features_are_causal": True,
            "labels_use_logged_target": True,
            "test_split_read": False,
            "claim": "predictability diagnostic only; not final policy performance",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
