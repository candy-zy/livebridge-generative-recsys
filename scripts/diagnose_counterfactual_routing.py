#!/usr/bin/env python3
"""Measure whether logged replay provides learnable route-level supervision."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np

from livebridge_rl.agentic_rl import AgentConfig, LiveSessionEnv, ROUTING_ACTIONS, load_agent_sessions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sessions", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sessions = load_agent_sessions(args.cache_dir / f"{args.split}_sessions.npz")
    rng = np.random.default_rng(args.seed)
    if len(sessions) > args.max_sessions:
        chosen = rng.choice(len(sessions), size=args.max_sessions, replace=False)
        sessions = [sessions[int(index)] for index in chosen]

    cfg = AgentConfig(max_steps=args.max_steps, seed=args.seed)
    counts: Counter[int] = Counter()
    informative = ties = switches = transitions = 0
    margins: list[float] = []
    oracle_returns: list[float] = []
    fixed_returns: list[float] = []

    for session in sessions:
        env = LiveSessionEnv(session, cfg)
        env.reset(seed=args.seed)
        rewards: list[float] = []
        actions: list[int] = []
        done = False
        while not done:
            action_rewards = []
            for action in range(len(ROUTING_ACTIONS)):
                probe = copy.deepcopy(env)
                _, reward, _, _, _ = probe.step(action)
                action_rewards.append(float(reward))
            order = np.argsort(-np.asarray(action_rewards))
            best = int(order[0])
            margin = action_rewards[best] - action_rewards[int(order[1])]
            informative += int(np.ptp(action_rewards) > 1e-8)
            ties += int(abs(margin) <= 1e-8)
            margins.append(float(margin))
            counts[best] += 1
            actions.append(best)
            _, reward, done, _, _ = env.step(best)
            rewards.append(float(reward))
        switches += sum(a != b for a, b in zip(actions, actions[1:]))
        transitions += max(0, len(actions) - 1)
        oracle_returns.append(sum(cfg.trajectory_gamma**i * r for i, r in enumerate(rewards)))

        fixed = LiveSessionEnv(session, cfg)
        fixed.reset(seed=args.seed)
        fixed_rewards: list[float] = []
        fixed_done = False
        while not fixed_done:
            _, reward, fixed_done, _, _ = fixed.step(1)
            fixed_rewards.append(float(reward))
        fixed_returns.append(sum(cfg.trajectory_gamma**i * r for i, r in enumerate(fixed_rewards)))

    decisions = sum(counts.values())
    result = {
        "split": args.split,
        "sessions": len(sessions),
        "decisions": decisions,
        "informative_state_rate": informative / max(1, decisions),
        "tie_rate": ties / max(1, decisions),
        "mean_best_second_margin": float(np.mean(margins)) if margins else 0.0,
        "stepwise_oracle_action_counts": [counts[index] for index in range(len(ROUTING_ACTIONS))],
        "stepwise_oracle_switch_rate": switches / max(1, transitions),
        "stepwise_oracle_session_return": float(np.mean(oracle_returns)) if oracle_returns else 0.0,
        "fixed_action_1_session_return": float(np.mean(fixed_returns)) if fixed_returns else 0.0,
        "diagnostic_only": True,
        "warning": "The oracle reads the next logged target and is an upper-bound teacher, not a deployable policy.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
