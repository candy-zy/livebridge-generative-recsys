#!/usr/bin/env python3
"""Train a conservative bridge->source escalation critic.

The critic sees only the causal state available before an expensive retrieval
call.  It defaults to bridge-only (action 0) and escalates to the strongest
fixed source route (action 1) when a calibrated probability threshold is met.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from livebridge_rl.agentic_rl import (
    AgentConfig,
    AgentSession,
    LiveSessionEnv,
    _counterfactual_action_rewards,
    load_agent_sessions,
)


class EscalationCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--selection-fraction", type=float, default=0.20)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--quality-tolerance", type=float, default=0.01)
    parser.add_argument("--minimum-call-reduction", type=float, default=0.30)
    parser.add_argument("--minimum-selection-auc", type=float, default=0.62)
    parser.add_argument(
        "--objective",
        choices=("binary", "advantage_regression"),
        default="binary",
        help="Predict a positive event or the signed source-minus-bridge advantage.",
    )
    parser.add_argument(
        "--advantage-weight",
        type=float,
        default=20.0,
        help="Extra Huber weight for large-magnitude advantage states.",
    )
    return parser.parse_args()


def split_users(
    sessions: list[AgentSession], fraction: float, seed: int
) -> tuple[list[AgentSession], list[AgentSession]]:
    if len(sessions) < 2 or not 0 < fraction < 1:
        raise ValueError("need at least two sessions and 0 < selection_fraction < 1")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sessions))
    count = min(len(sessions) - 1, max(1, int(round(len(sessions) * fraction))))
    selection = [sessions[int(index)] for index in order[:count]]
    training = [sessions[int(index)] for index in order[count:]]
    assert not ({s.user_id for s in training} & {s.user_id for s in selection})
    return training, selection


def build_examples(
    sessions: list[AgentSession], cfg: AgentConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    labels: list[float] = []
    advantages: list[float] = []
    for session in sessions:
        env = LiveSessionEnv(session, cfg)
        observation, _ = env.reset(seed=cfg.seed)
        done = False
        while not done:
            rewards = _counterfactual_action_rewards(env)
            advantage = float(rewards[1] - rewards[0])
            observations.append(observation.copy())
            labels.append(float(advantage > 1e-8))
            advantages.append(advantage)
            observation, _, done, _, _ = env.step(0)
    return (
        np.stack(observations).astype(np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(advantages, dtype=np.float32),
    )


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels > 0.5]
    negative = scores[labels <= 0.5]
    if not len(positive) or not len(negative):
        return 0.5
    wins = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(positive) * len(negative)))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores)
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / positives)


def train_critic(
    x: np.ndarray,
    y: np.ndarray,
    advantages: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[EscalationCritic, np.ndarray, np.ndarray, float]:
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).clip(min=1e-4).astype(np.float32)
    normalized = (x - mean) / std
    model = EscalationCritic(x.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if args.objective == "binary":
        targets = y.astype(np.float32)
        target_scale = 1.0
        positives = float(y.sum())
        positive_weight = min(50.0, (len(y) - positives) / max(1.0, positives))
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(positive_weight, device=device)
        )
        sample_weights = np.ones_like(targets)
    else:
        nonzero = np.abs(advantages[np.abs(advantages) > 1e-8])
        target_scale = float(np.quantile(nonzero, 0.95)) if len(nonzero) else 1.0
        target_scale = max(target_scale, 1e-4)
        targets = (advantages / target_scale).astype(np.float32)
        magnitude = np.minimum(1.0, np.abs(advantages) / target_scale)
        sample_weights = (1.0 + args.advantage_weight * magnitude).astype(np.float32)
        loss_fn = nn.SmoothL1Loss(reduction="none", beta=0.25)
    generator = torch.Generator().manual_seed(args.seed)
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(normalized),
        torch.as_tensor(targets),
        torch.as_tensor(sample_weights),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for features, target, weight in loader:
            features, target, weight = (
                features.to(device), target.to(device), weight.to(device)
            )
            per_example = loss_fn(model(features), target)
            loss = per_example if per_example.ndim == 0 else (per_example * weight).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item()) * len(target)
        if epoch in {0, args.epochs - 1} or (epoch + 1) % 50 == 0:
            print(
                f"epoch={epoch + 1:03d} loss={epoch_loss / len(dataset):.6f}",
                flush=True,
            )
    return model, mean, std, target_scale


def predict(
    model: EscalationCritic,
    observations: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    objective: str = "binary",
    target_scale: float = 1.0,
) -> np.ndarray:
    model.eval()
    tensor = torch.as_tensor((observations - mean) / std, device=device)
    with torch.no_grad():
        output = model(tensor)
        if objective == "binary":
            output = torch.sigmoid(output)
        else:
            output = output * target_scale
        return output.cpu().numpy()


def evaluate(
    sessions: list[AgentSession],
    cfg: AgentConfig,
    mode: str,
    model: EscalationCritic | None,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
    device: torch.device,
    objective: str = "binary",
    target_scale: float = 1.0,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    rows = []
    for session in sessions:
        env = LiveSessionEnv(session, cfg)
        observation, _ = env.reset(seed=cfg.seed)
        done = False
        rewards: list[float] = []
        hits: list[float] = []
        ndcgs: list[float] = []
        watches: list[float] = []
        escalations = extra_calls = 0
        while not done:
            if mode == "bridge":
                action = 0
            elif mode == "source":
                action = 1
            elif mode == "critic":
                if model is None:
                    raise ValueError("critic mode requires a model")
                score = float(predict(
                    model, observation[None, :], mean, std, device,
                    objective, target_scale,
                )[0])
                action = 1 if score >= threshold else 0
            else:
                raise ValueError(f"unknown mode: {mode}")
            observation, reward, done, _, info = env.step(action)
            rewards.append(float(reward))
            position = int(info["hit_position"])
            hits.append(float(position >= 0))
            ndcgs.append(1.0 / math.log2(position + 2) if position >= 0 else 0.0)
            watches.append(float(info["watch_reward"]))
            calls = len(info["calls"])
            escalations += int(action == 1)
            extra_calls += max(0, calls - 1)
        steps = max(1, len(rewards))
        rows.append({
            "user_id": session.user_id,
            "steps": len(rewards),
            "session_return": float(sum(
                cfg.trajectory_gamma**index * reward
                for index, reward in enumerate(rewards)
            )),
            "recall@10": float(np.mean(hits)),
            "ndcg@10": float(np.mean(ndcgs)),
            "logged_watch@10": float(np.mean(watches)),
            "escalation_rate": escalations / steps,
            "extra_tool_calls_per_step": extra_calls / steps,
        })
    frame = pd.DataFrame(rows)
    names = (
        "session_return", "recall@10", "ndcg@10", "logged_watch@10",
        "escalation_rate", "extra_tool_calls_per_step",
    )
    overall = {name: float(frame[name].mean()) for name in names}
    overall["users"] = len(frame)
    return overall, frame


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_sessions = load_agent_sessions(args.cache_dir / "valid_sessions.npz")
    test_sessions = load_agent_sessions(args.cache_dir / "test_sessions.npz")
    train_sessions, selection_sessions = split_users(
        valid_sessions, args.selection_fraction, args.seed
    )
    cfg = AgentConfig(
        seed=args.seed,
        max_steps=args.max_steps,
        source_weight=0.0,
        longtail_weight=0.0,
    )
    x_train, y_train, train_advantage = build_examples(train_sessions, cfg)
    x_selection, y_selection, selection_advantage = build_examples(
        selection_sessions, cfg
    )
    model, mean, std, target_scale = train_critic(
        x_train, y_train, train_advantage, args, device
    )
    selection_scores = predict(
        model, x_selection, mean, std, device, args.objective, target_scale
    )
    selection_auc = roc_auc(y_selection, selection_scores)
    selection_ap = average_precision(y_selection, selection_scores)

    bridge_selection, _ = evaluate(
        selection_sessions, cfg, "bridge", None, mean, std, 1.0, device
    )
    source_selection, _ = evaluate(
        selection_sessions, cfg, "source", None, mean, std, 0.0, device
    )
    # Raw BCE probabilities are ranking scores, not calibrated probabilities.
    # A fixed [0.05, 0.95] sweep can accidentally omit the entire useful
    # 20%-100% escalation region.  Add score quantiles so model selection
    # explicitly covers call budgets, plus both fixed-policy endpoints.
    target_escalation_rates = np.linspace(0.05, 1.0, 20)
    quantile_thresholds = np.quantile(
        selection_scores,
        np.clip(1.0 - target_escalation_rates, 0.0, 1.0),
    )
    thresholds = np.unique(np.concatenate((
        np.asarray([0.0]),
        np.linspace(0.05, 0.95, 19),
        np.asarray([0.97, 0.98, 0.99, 0.995, 1.0]),
        quantile_thresholds,
    )))
    threshold_records = []
    for threshold in thresholds:
        metrics, _ = evaluate(
            selection_sessions, cfg, "critic", model, mean, std,
            float(threshold), device, args.objective, target_scale,
        )
        return_floor = source_selection["session_return"] * (1 - args.quality_tolerance)
        recall_floor = source_selection["recall@10"] * (1 - args.quality_tolerance)
        source_calls = max(1e-12, float(source_selection["extra_tool_calls_per_step"]))
        call_reduction = 1 - float(metrics["extra_tool_calls_per_step"]) / source_calls
        feasible = (
            metrics["session_return"] >= return_floor
            and metrics["recall@10"] >= recall_floor
        )
        threshold_records.append({
            "threshold": float(threshold),
            **metrics,
            "call_reduction_vs_always_source": call_reduction,
            "quality_feasible": bool(feasible),
        })
    feasible = [row for row in threshold_records if row["quality_feasible"]]
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (
                row["call_reduction_vs_always_source"], row["session_return"]
            ),
        )
    else:
        selected = max(threshold_records, key=lambda row: row["session_return"])
    threshold = float(selected["threshold"])

    bridge_test, bridge_frame = evaluate(
        test_sessions, cfg, "bridge", None, mean, std, 1.0, device
    )
    source_test, source_frame = evaluate(
        test_sessions, cfg, "source", None, mean, std, 0.0, device
    )
    critic_test, critic_frame = evaluate(
        test_sessions, cfg, "critic", model, mean, std, threshold, device,
        args.objective, target_scale,
    )
    test_call_reduction = 1 - (
        float(critic_test["extra_tool_calls_per_step"])
        / max(1e-12, float(source_test["extra_tool_calls_per_step"]))
    )
    gate_pass = bool(
        selection_auc >= args.minimum_selection_auc
        and selected["quality_feasible"]
        and selected["call_reduction_vs_always_source"] >= args.minimum_call_reduction
    )
    result = {
        "model": f"budgeted_bridge_to_source_{args.objective}_critic",
        "device": str(device),
        "config": vars(args) | {"cache_dir": str(args.cache_dir), "output_dir": str(args.output_dir)},
        "agent_config": asdict(cfg),
        "data": {
            "router_train_users": len(train_sessions),
            "router_selection_users": len(selection_sessions),
            "test_users": len(test_sessions),
            "train_states": len(y_train),
            "train_positive_rate": float(y_train.mean()),
            "train_advantage_mean": float(train_advantage.mean()),
            "train_advantage_abs_mean": float(np.abs(train_advantage).mean()),
            "selection_states": len(y_selection),
            "selection_positive_rate": float(y_selection.mean()),
            "selection_positive_advantage_mean": float(
                selection_advantage[y_selection > 0.5].mean()
            ) if y_selection.sum() else 0.0,
        },
        "selection_classifier": {
            "roc_auc": selection_auc,
            "average_precision": selection_ap,
        },
        "selection": {
            "bridge": bridge_selection,
            "always_source": source_selection,
            "threshold_sweep": threshold_records,
            "selected": selected,
        },
        "test": {
            "bridge": bridge_test,
            "always_source": source_test,
            "critic": critic_test,
            "critic_call_reduction_vs_always_source": test_call_reduction,
        },
        "gate": {
            "passed": gate_pass,
            "minimum_selection_auc": args.minimum_selection_auc,
            "quality_tolerance": args.quality_tolerance,
            "minimum_call_reduction": args.minimum_call_reduction,
        },
        "audit": {
            "policy_features": "causal cheap state only",
            "checkpoint_selection_split": "valid/router_selection",
            "test_read_after_threshold_selection": True,
            "unbiased_ope": False,
            "claim_boundary": "logged-positive replay; no online CTR/revenue claim",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    torch.save({
        "state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "threshold": threshold,
        "objective": args.objective,
        "target_scale": target_scale,
        "input_dim": x_train.shape[1],
        "hidden_dim": args.hidden_dim,
    }, args.output_dir / "policy.pt")
    bridge_frame.add_prefix("bridge_").to_csv(
        args.output_dir / "bridge_test_per_user.csv", index=False
    )
    source_frame.add_prefix("source_").to_csv(
        args.output_dir / "source_test_per_user.csv", index=False
    )
    critic_frame.to_csv(args.output_dir / "critic_test_per_user.csv", index=False)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
