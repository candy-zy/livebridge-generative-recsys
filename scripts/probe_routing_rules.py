"""Cheap diagnostic for state-dependent routing rules on an agent cache."""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from livebridge_rl.agentic_rl import AgentConfig, LiveSessionEnv, load_agent_sessions


def choose(rule: str, env: LiveSessionEnv) -> int:
    if rule == "source_then_balanced":
        return 4 if env.miss_streak >= 1 else 1
    if rule == "source_then_bridge":
        return 0 if env.miss_streak >= 1 else 1
    if rule == "source_then_bridge2":
        return 0 if env.miss_streak >= 2 else 1
    if rule == "source_then_bridge3":
        return 0 if env.miss_streak >= 3 else 1
    if rule == "source_then_tail2":
        return 3 if env.miss_streak >= 2 else 1
    if rule == "bucket_source_balanced":
        return 1 if env.session.train_count < 31 else 4
    if rule == "first_source_then_balanced":
        return 1 if env.step_index == 0 else 4
    if rule == "cycle":
        return (1, 4, 3)[env.step_index % 3]
    raise ValueError(rule)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache")
    parser.add_argument("--split", default="test", choices=("valid", "test"))
    args = parser.parse_args()
    sessions = load_agent_sessions(f"{args.cache}/{args.split}_sessions.npz")
    rules = (
        "source_then_balanced", "source_then_bridge", "source_then_bridge2",
        "source_then_bridge3", "source_then_tail2",
        "bucket_source_balanced", "first_source_then_balanced", "cycle",
    )
    for rule in rules:
        totals = defaultdict(float)
        actions = np.zeros(5, dtype=np.int64)
        for session in sessions:
            env = LiveSessionEnv(session, AgentConfig(max_steps=8))
            _, _ = env.reset()
            rewards = []
            done = False
            while not done:
                action = choose(rule, env)
                actions[action] += 1
                _, reward, done, _, info = env.step(action)
                rewards.append(reward)
                totals["hit"] += float(info["hit"])
                totals["ndcg"] += (
                    1.0 / np.log2(int(info["hit_position"]) + 2)
                    if int(info["hit_position"]) >= 0 else 0.0
                )
                totals["steps"] += 1
            totals["return"] += sum(0.95 ** i * r for i, r in enumerate(rewards))
        print({
            "rule": rule,
            "session_return": totals["return"] / max(1, len(sessions)),
            "recall@10": totals["hit"] / max(1, totals["steps"]),
            "ndcg@10": totals["ndcg"] / max(1, totals["steps"]),
            "actions": actions.tolist(),
        })

    totals = defaultdict(float)
    actions = np.zeros(5, dtype=np.int64)
    for session in sessions:
        candidates = []
        for action in range(5):
            env = LiveSessionEnv(session, AgentConfig(max_steps=8))
            env.reset()
            rewards, hits, ndcgs = [], [], []
            done = False
            while not done:
                _, reward, done, _, info = env.step(action)
                rewards.append(reward)
                hits.append(float(info["hit"]))
                position = int(info["hit_position"])
                ndcgs.append(1.0 / np.log2(position + 2) if position >= 0 else 0.0)
            candidates.append((
                sum(0.95 ** i * r for i, r in enumerate(rewards)),
                float(np.mean(hits)), float(np.mean(ndcgs)), len(rewards), action,
            ))
        best = max(candidates, key=lambda row: row[0])
        totals["return"] += best[0]
        totals["recall"] += best[1]
        totals["ndcg"] += best[2]
        actions[best[4]] += best[3]
    print({
        "rule": "per_session_oracle_upper_bound",
        "session_return": totals["return"] / max(1, len(sessions)),
        "recall@10": totals["recall"] / max(1, len(sessions)),
        "ndcg@10": totals["ndcg"] / max(1, len(sessions)),
        "actions": actions.tolist(),
    })


if __name__ == "__main__":
    main()
