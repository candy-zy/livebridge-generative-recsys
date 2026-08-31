"""Benchmark hierarchical serving and deterministic failure recovery."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from livebridge_rl.agentic_rl import (
    AgentConfig,
    LiveSessionEnv,
    SessionPolicy,
    ToolRegistry,
    load_agent_sessions,
)


def percentiles(values: list[float]) -> dict[str, float]:
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    sessions = load_agent_sessions(args.cache_dir / "test_sessions.npz")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = AgentConfig(**saved["config"])
    policy = SessionPolicy(LiveSessionEnv.OBSERVATION_DIM, cfg.hidden_dim, recurrent=True)
    policy.load_state_dict(saved["state_dict"])
    policy.eval()
    registry = ToolRegistry(cfg.slate_size)

    observations = []
    actions = []
    for session in sessions:
        env = LiveSessionEnv(session, cfg)
        observation, _ = env.reset()
        observations.append(observation)
        with torch.no_grad():
            logits, _ = policy.step(torch.as_tensor(observation))
        actions.append(int(torch.argmax(logits).item()))

    policy_latency, cached_ranker_latency, synchronous_latency = [], [], []
    for request in range(args.requests):
        index = request % len(sessions)
        observation, session = observations[index], sessions[index]
        start = time.perf_counter()
        with torch.no_grad():
            logits, _ = policy.step(torch.as_tensor(observation))
        action = int(torch.argmax(logits).item())
        policy_end = time.perf_counter()
        registry.retrieve(session, action)
        end = time.perf_counter()
        policy_latency.append((policy_end - start) * 1000)
        synchronous_latency.append((end - start) * 1000)

        start = time.perf_counter()
        registry.retrieve(session, actions[index])
        cached_ranker_latency.append((time.perf_counter() - start) * 1000)

    failure = {}
    sample_count = min(512, len(sessions))
    chosen = rng.choice(len(sessions), size=sample_count, replace=False)
    for rate in (0.10, 0.30):
        invalid_before = invalid_after = fallback_success = exposed = 0
        for index in chosen:
            session = sessions[int(index)]
            count = max(1, int(len(session.items) * rate))
            unavailable = set(int(x) for x in rng.choice(session.items, size=count, replace=False))
            action = actions[int(index)]
            unfiltered, _ = registry.retrieve(session, action)
            filtered, _ = registry.retrieve(session, action, unavailable=unavailable)
            invalid_before += sum(item in unavailable for item in unfiltered)
            invalid_after += sum(item in unavailable for item in filtered)
            exposed += len(unfiltered)
            # A failed non-bridge tool degrades to the always-available bridge route.
            fallback, _ = registry.retrieve(session, 0, unavailable=unavailable)
            fallback_success += int(bool(fallback) and not any(x in unavailable for x in fallback))
        failure[str(rate)] = {
            "invalid_exposure_rate_without_filter": invalid_before / max(1, exposed),
            "invalid_exposure_rate_with_filter": invalid_after / max(1, exposed),
            "fallback_success_rate": fallback_success / sample_count,
        }

    intervals = rng.exponential(scale=5.0, size=args.requests)
    ttl = {
        str(seconds): {
            "policy_cache_hit_rate": float(np.mean(intervals <= seconds)),
            "refresh_rate": float(np.mean(intervals > seconds)),
        }
        for seconds in (10, 30, 60)
    }
    result = {
        "requests": args.requests,
        "sessions": len(sessions),
        "device": "cpu",
        "policy_update": percentiles(policy_latency),
        "cached_fast_ranker": percentiles(cached_ranker_latency),
        "synchronous_policy_plus_ranker": percentiles(synchronous_latency),
        "ttl": ttl,
        "failure_injection": failure,
        "architecture_boundary": {
            "training_in_hot_path": False,
            "llm_in_hot_path": False,
            "policy_refresh": "asynchronous TTL cache",
            "request_path": "cached action -> live-status filter -> fast ranker",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
