#!/usr/bin/env python3
"""Audit whether a safe baseline-override router has enough actionable states.

Unlike the oracle diagnostic, state visitation follows the fixed production
baseline.  Labels may inspect the current logged target, but only to measure an
offline upper bound; no target-derived value is emitted as a serving feature.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

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
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-action", type=int, default=1)
    parser.add_argument("--max-sessions", type=int, default=1000000)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _jaccard(left: list[int], right: list[int]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / max(1, len(a | b))


def main() -> int:
    args = parse_args()
    if not 0 <= args.baseline_action < len(ROUTING_ACTIONS):
        raise ValueError("baseline-action is out of range")
    sessions = load_agent_sessions(args.cache_dir / f"{args.split}_sessions.npz")
    rng = np.random.default_rng(args.seed)
    if len(sessions) > args.max_sessions:
        chosen = rng.choice(len(sessions), size=args.max_sessions, replace=False)
        sessions = [sessions[int(index)] for index in chosen]

    # Match the final V2 accuracy-only experiment.  Leaving the legacy source
    # and long-tail shaping defaults enabled makes the diagnostic declare the
    # long-tail route an oracle even when it does not retrieve the logged item.
    cfg = AgentConfig(
        max_steps=args.max_steps,
        seed=args.seed,
        source_weight=0.0,
        longtail_weight=0.0,
    )
    decisions = actionable = positive_override = oracle_nonbaseline = 0
    best_counts: Counter[int] = Counter()
    best_advantages: list[float] = []
    max_slate_distances: list[float] = []
    positive_advantages: list[float] = []

    for session in sessions:
        env = LiveSessionEnv(session, cfg)
        env.reset(seed=args.seed)
        done = False
        while not done:
            slates = [
                env.registry.retrieve(
                    session,
                    action,
                    previously_recommended=env.previously_recommended,
                )[0]
                for action in range(len(ROUTING_ACTIONS))
            ]
            baseline_slate = slates[args.baseline_action]
            distances = [1.0 - _jaccard(baseline_slate, slate) for slate in slates]
            max_distance = max(distances)
            actionable += int(max_distance > 1e-8)
            max_slate_distances.append(max_distance)

            rewards = np.asarray(_counterfactual_action_rewards(env), dtype=np.float64)
            advantages = rewards - rewards[args.baseline_action]
            best = int(np.argmax(rewards))
            best_advantage = float(advantages[best])
            best_counts[best] += 1
            oracle_nonbaseline += int(best != args.baseline_action)
            positive_override += int(best != args.baseline_action and best_advantage > 1e-8)
            best_advantages.append(best_advantage)
            if best != args.baseline_action and best_advantage > 1e-8:
                positive_advantages.append(best_advantage)

            _, _, done, _, _ = env.step(args.baseline_action)
            decisions += 1

    positive = np.asarray(positive_advantages, dtype=np.float64)
    result = {
        "split": args.split,
        "state_visitation": f"fixed_action_{args.baseline_action}",
        "sessions": len(sessions),
        "decisions": decisions,
        "actionable_slate_disagreement_rate": actionable / max(1, decisions),
        "mean_max_slate_jaccard_distance": float(np.mean(max_slate_distances)) if decisions else 0.0,
        "oracle_nonbaseline_rate": oracle_nonbaseline / max(1, decisions),
        "strict_positive_override_rate": positive_override / max(1, decisions),
        "mean_best_advantage": float(np.mean(best_advantages)) if decisions else 0.0,
        "positive_override_advantage_mean": float(positive.mean()) if len(positive) else 0.0,
        "positive_override_advantage_p50": float(np.percentile(positive, 50)) if len(positive) else 0.0,
        "positive_override_advantage_p90": float(np.percentile(positive, 90)) if len(positive) else 0.0,
        "positive_override_advantage_p99": float(np.percentile(positive, 99)) if len(positive) else 0.0,
        "positive_override_advantage_gt_0p01_rate": float(np.mean(positive > 0.01)) if len(positive) else 0.0,
        "oracle_best_action_counts": [best_counts[index] for index in range(len(ROUTING_ACTIONS))],
        "diagnostic_only": True,
        "warning": "Target-derived advantages are offline labels/upper bounds, never serving features.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
