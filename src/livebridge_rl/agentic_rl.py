"""Session-level tool-routing agent for logged-positive live recommendation.

The implementation deliberately keeps an LLM out of the serving hot path.  A
small recurrent policy chooses a retrieval recipe; deterministic tools and a
fast ranker materialize the slate.  Training is offline logged-positive replay,
not unbiased OPE and not an online CTR claim.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from torch import nn

from livebridge_rl.baseline import BridgeBPR
from livebridge_rl.evaluation import history_bucket
from livebridge_rl.grpo_reranker import (
    CandidateFeatures,
    ResidualPolicy,
    _build_candidate_cache,
    _build_sparse_source_index,
    _exposure_gini,
    _zscore,
)


@dataclass(frozen=True)
class RoutingAction:
    name: str
    quotas: tuple[float, float, float, float, float]
    weights: tuple[float, float, float, float, float]


ROUTING_ACTIONS = (
    RoutingAction("bridge_exploit", (1.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0, 0.0)),
    RoutingAction("source_transfer", (0.35, 0.50, 0.0, 0.0, 0.15), (0.65, 0.75, 0.0, 0.0, 0.15)),
    RoutingAction("content_match", (0.35, 0.0, 0.50, 0.0, 0.15), (0.65, 0.0, 0.75, 0.0, 0.15)),
    RoutingAction("longtail_explore", (0.30, 0.10, 0.10, 0.0, 0.50), (0.55, 0.10, 0.10, -0.10, 0.80)),
    RoutingAction("balanced", (0.45, 0.20, 0.15, 0.05, 0.15), (0.75, 0.30, 0.20, 0.05, 0.20)),
)


@dataclass
class AgentSession:
    user_id: int
    items: np.ndarray
    features: np.ndarray
    targets: np.ndarray
    watch: np.ndarray
    timestamps: np.ndarray
    train_count: int
    author_count: int


@dataclass
class AgentConfig:
    epochs: int = 30
    learning_rate: float = 3e-3
    hidden_dim: int = 48
    group_size: int = 4
    slate_size: int = 10
    max_steps: int = 8
    users_per_epoch: int = 2048
    clip_epsilon: float = 0.2
    kl_beta: float = 0.02
    entropy_coef: float = 0.01
    warmup_epochs: int = 2
    warmup_strategy: str = "causal_fallback"
    counterfactual_temperature: float = 0.05
    selection_fraction: float = 0.20
    trajectory_gamma: float = 0.95
    relevance_weight: float = 1.0
    watch_weight: float = 0.25
    source_weight: float = 0.05
    longtail_weight: float = 0.05
    repeat_penalty: float = 0.10
    fixed_action: int = 0
    seed: int = 42


class ToolRegistry:
    """Deterministic retrieval tools over one shared, auditable candidate budget."""

    TOOL_NAMES = ("bridge", "source", "content", "popular", "longtail")

    def __init__(self, slate_size: int = 10) -> None:
        self.slate_size = slate_size

    @staticmethod
    def _tool_score(features: np.ndarray, tool: int) -> np.ndarray:
        base = features[:, 0]
        if tool == 0:
            return base
        if tool == 1:
            return features[:, 1] + 0.05 * base
        if tool == 2:
            return features[:, 2] + 0.05 * base
        if tool == 3:
            return features[:, 3] + 0.05 * base
        return features[:, 4] + 0.05 * base

    def retrieve(
        self,
        session: AgentSession,
        action_id: int,
        unavailable: set[int] | None = None,
        previously_recommended: set[int] | None = None,
    ) -> tuple[list[int], dict[str, object]]:
        action = ROUTING_ACTIONS[int(action_id)]
        unavailable = unavailable or set()
        previously_recommended = previously_recommended or set()
        # Live feeds should not recycle the same slate across adjacent steps.
        # Previously exposed authors are filtered before tool retrieval; only
        # if the candidate cache is exhausted do we fall back to allowing them.
        valid = np.asarray([
            item not in unavailable and item not in previously_recommended
            for item in session.items
        ], dtype=bool)
        if int(valid.sum()) < self.slate_size:
            valid = np.asarray([item not in unavailable for item in session.items], dtype=bool)
        tool_orders = []
        for tool in range(len(self.TOOL_NAMES)):
            score = self._tool_score(session.features, tool)
            order = np.argsort(-score)
            tool_orders.append([int(index) for index in order if valid[index]])

        quotas = np.floor(np.asarray(action.quotas) * self.slate_size).astype(int)
        quotas[0] += self.slate_size - int(quotas.sum())
        selected_rows: list[int] = []
        selected_set: set[int] = set()
        calls: list[dict[str, object]] = []
        for tool, quota in enumerate(quotas):
            if quota <= 0:
                continue
            added = 0
            for row in tool_orders[tool]:
                item = int(session.items[row])
                if item in selected_set:
                    continue
                selected_rows.append(row)
                selected_set.add(item)
                added += 1
                if added >= quota:
                    break
            if quota:
                calls.append({"tool": self.TOOL_NAMES[tool], "quota": int(quota), "returned": added})

        for row in tool_orders[0]:
            if len(selected_rows) >= self.slate_size:
                break
            item = int(session.items[row])
            if item not in selected_set:
                selected_rows.append(row)
                selected_set.add(item)

        if not selected_rows:
            return [], {"action": action.name, "calls": calls, "fallback": True}
        rows = np.asarray(selected_rows, dtype=np.int64)
        rank_score = session.features[rows] @ np.asarray(action.weights, dtype=np.float32)
        repeat = np.asarray(
            [int(session.items[row]) in previously_recommended for row in rows], dtype=np.float32
        )
        rank_score -= 0.20 * repeat
        rows = rows[np.argsort(-rank_score)]
        slate = [int(session.items[row]) for row in rows[: self.slate_size]]
        return slate, {"action": action.name, "calls": calls, "fallback": False}


class LiveSessionEnv(gym.Env[np.ndarray, int]):
    """Gymnasium replay environment with explicit logged-positive boundaries."""

    metadata = {"render_modes": []}
    STATIC_DIM = 16
    DYNAMIC_DIM = 8
    # Previous consumed-item features, previous slate features, and normalized
    # inter-event time make interest drift observable without exposing the
    # current/future target to the serving policy.
    FEEDBACK_DIM = 11
    OBSERVATION_DIM = STATIC_DIM + DYNAMIC_DIM + FEEDBACK_DIM + len(ROUTING_ACTIONS)

    def __init__(self, session: AgentSession, config: AgentConfig | None = None):
        super().__init__()
        self.session = session
        self.cfg = config or AgentConfig()
        self.action_space = spaces.Discrete(len(ROUTING_ACTIONS))
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.OBSERVATION_DIM,), dtype=np.float32
        )
        self.registry = ToolRegistry(self.cfg.slate_size)
        self._reset_state()

    def _reset_state(self) -> None:
        self.step_index = 0
        self.last_hit = 0.0
        self.last_watch = 0.0
        self.hits = 0.0
        self.watch_total = 0.0
        self.longtail_total = 0.0
        self.repeat_total = 0.0
        self.miss_streak = 0
        self.last_action = -1
        self.last_target_features = np.zeros(5, dtype=np.float32)
        self.last_slate_features = np.zeros(5, dtype=np.float32)
        self.last_time_gap = 0.0
        self.previously_recommended: set[int] = set()

    def _observation(self) -> np.ndarray:
        features = self.session.features
        if len(features):
            distribution = np.concatenate((
                features.mean(axis=0), features.max(axis=0), features.std(axis=0)
            )).astype(np.float32)
        else:
            distribution = np.zeros(15, dtype=np.float32)
        static = np.concatenate((np.asarray([
            min(math.log1p(self.session.train_count) / 8.0, 1.5),
        ], dtype=np.float32), distribution))
        denominator = max(1, self.step_index)
        dynamic = np.asarray([
            self.step_index / max(1, min(len(self.session.targets), self.cfg.max_steps)),
            self.last_hit,
            self.last_watch,
            self.hits / denominator,
            self.watch_total / denominator,
            self.longtail_total / denominator,
            self.repeat_total / denominator,
            min(self.miss_streak / 5.0, 1.0),
        ], dtype=np.float32)
        previous = np.zeros(len(ROUTING_ACTIONS), dtype=np.float32)
        if self.last_action >= 0:
            previous[self.last_action] = 1.0
        feedback = np.concatenate((
            self.last_target_features,
            self.last_slate_features,
            np.asarray([self.last_time_gap], dtype=np.float32),
        ))
        return np.concatenate((static, dynamic, feedback, previous)).astype(np.float32, copy=False)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_state()
        return self._observation(), {"user_id": self.session.user_id}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid routing action: {action}")
        if self.step_index >= min(len(self.session.targets), self.cfg.max_steps):
            raise RuntimeError("step called after episode termination")
        slate, trace = self.registry.retrieve(
            self.session, int(action), previously_recommended=self.previously_recommended
        )
        target = int(self.session.targets[self.step_index])
        watch = float(self.session.watch[self.step_index])
        hit_position = slate.index(target) if target in slate else -1
        relevance = self.cfg.trajectory_gamma ** hit_position if hit_position >= 0 else 0.0
        watch_reward = relevance * min(math.log1p(max(0.0, watch)) / 10.0, 1.0)
        positions = {int(item): row for row, item in enumerate(self.session.items)}
        rows = [positions[item] for item in slate if item in positions]
        source = float(np.mean(self.session.features[rows, 1])) if rows else 0.0
        longtail = float(np.mean(self.session.features[rows, 4])) if rows else 0.0
        repeated = sum(item in self.previously_recommended for item in slate) / max(1, len(slate))
        # Gate auxiliary shaping by logged relevance.  A small exploration
        # prior remains when there is no hit, but the policy cannot maximize
        # return simply by emitting unrelated source/tail candidates.
        relevance_gate = 0.25 + 0.75 * relevance
        reward = (
            self.cfg.relevance_weight * relevance
            + self.cfg.watch_weight * watch_reward
            + self.cfg.source_weight * source * relevance_gate
            + self.cfg.longtail_weight * longtail * relevance_gate
            - self.cfg.repeat_penalty * repeated
        )
        self.previously_recommended.update(slate)
        self.last_hit = float(hit_position >= 0)
        self.last_watch = watch_reward
        self.hits += self.last_hit
        self.watch_total += watch_reward
        self.longtail_total += longtail
        self.repeat_total += repeated
        self.miss_streak = 0 if self.last_hit else self.miss_streak + 1
        self.last_action = int(action)
        target_row = positions.get(target)
        self.last_target_features = (
            self.session.features[target_row].astype(np.float32, copy=True)
            if target_row is not None else np.zeros(5, dtype=np.float32)
        )
        self.last_slate_features = (
            self.session.features[rows].mean(axis=0).astype(np.float32, copy=False)
            if rows else np.zeros(5, dtype=np.float32)
        )
        if self.step_index > 0:
            delta = max(0, int(self.session.timestamps[self.step_index]) - int(self.session.timestamps[self.step_index - 1]))
            self.last_time_gap = min(math.log1p(delta) / 20.0, 1.5)
        else:
            self.last_time_gap = 0.0
        self.step_index += 1
        terminated = self.step_index >= min(len(self.session.targets), self.cfg.max_steps)
        info = {
            **trace,
            "target": target,
            "slate": slate,
            "hit": self.last_hit,
            "hit_position": hit_position,
            "watch_reward": watch_reward,
            "source_affinity": source,
            "longtail_share": longtail,
            "repeat_rate": repeated,
        }
        return self._observation(), float(reward), terminated, False, info


class SessionPolicy(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int, recurrent: bool = True):
        super().__init__()
        self.recurrent = recurrent
        self.hidden_dim = hidden_dim
        if recurrent:
            self.encoder = nn.GRUCell(observation_dim, hidden_dim)
        else:
            self.encoder = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.Tanh())
        self.head = nn.Linear(hidden_dim, len(ROUTING_ACTIONS))

    def step(
        self, observation: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if self.recurrent:
            if hidden is None:
                hidden = torch.zeros(
                    observation.shape[0], self.hidden_dim,
                    device=observation.device, dtype=observation.dtype,
                )
            hidden = self.encoder(observation, hidden)
        else:
            hidden = self.encoder(observation)
        return self.head(hidden), hidden


@dataclass
class Trajectory:
    observations: list[np.ndarray]
    actions: list[int]
    old_log_prob: float
    rewards: list[float]
    infos: list[dict[str, object]]

    def discounted_return(self, gamma: float) -> float:
        return float(sum((gamma ** step) * reward for step, reward in enumerate(self.rewards)))


def _masked_observation(observation: np.ndarray, variant: str) -> np.ndarray:
    output = observation.copy()
    if variant == "no_memory":
        output[LiveSessionEnv.STATIC_DIM:] = 0.0
    return output


def _rollout(
    session: AgentSession,
    policy: SessionPolicy | None,
    cfg: AgentConfig,
    variant: str,
    sample: bool,
    device: torch.device,
) -> Trajectory:
    env = LiveSessionEnv(session, cfg)
    observation, _ = env.reset(seed=cfg.seed)
    hidden = None
    observations, actions, rewards, infos = [], [], [], []
    log_prob = 0.0
    terminated = False
    while not terminated:
        model_observation = _masked_observation(observation, variant)
        observations.append(model_observation)
        if variant.startswith("fixed"):
            action = cfg.fixed_action
        elif variant == "no_routing":
            action = len(ROUTING_ACTIONS) - 1
        else:
            if policy is None:
                raise ValueError(f"variant {variant} requires a policy")
            tensor = torch.as_tensor(model_observation, device=device)
            with torch.no_grad():
                logits, hidden = policy.step(tensor, hidden)
                distribution = torch.distributions.Categorical(logits=logits.squeeze(0))
                chosen = distribution.sample() if sample else torch.argmax(logits.squeeze(0))
                action = int(chosen.item())
                log_prob += float(distribution.log_prob(chosen).item())
        observation, reward, terminated, _, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        infos.append(info)
    return Trajectory(observations, actions, log_prob / max(1, len(actions)), rewards, infos)


def _trajectory_log_prob_and_kl(
    policy: SessionPolicy,
    reference_policy: SessionPolicy,
    trajectory: Trajectory,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = None
    reference_hidden = None
    log_probs, kls, entropies = [], [], []
    for observation, action in zip(trajectory.observations, trajectory.actions):
        tensor = torch.as_tensor(observation, device=device)
        logits, hidden = policy.step(tensor, hidden)
        logits = logits.squeeze(0)
        log_softmax = torch.log_softmax(logits, dim=0)
        log_probs.append(log_softmax[action])
        probability = torch.softmax(logits, dim=0)
        with torch.no_grad():
            reference_logits, reference_hidden = reference_policy.step(tensor, reference_hidden)
            reference_log_probability = torch.log_softmax(reference_logits.squeeze(0), dim=0)
        kls.append(torch.sum(probability * (log_softmax - reference_log_probability)))
        entropies.append(-torch.sum(probability * log_softmax))
    return (
        torch.stack(log_probs).mean(),
        torch.stack(kls).mean(),
        torch.stack(entropies).mean(),
    )


def _counterfactual_action_rewards(env: LiveSessionEnv) -> list[float]:
    """Return exact one-step rewards for the small deterministic action set."""
    rewards = []
    for candidate_action in range(len(ROUTING_ACTIONS)):
        slate, _ = env.registry.retrieve(
            env.session,
            candidate_action,
            previously_recommended=env.previously_recommended,
        )
        target = int(env.session.targets[env.step_index])
        watch = float(env.session.watch[env.step_index])
        hit_position = slate.index(target) if target in slate else -1
        relevance = env.cfg.trajectory_gamma**hit_position if hit_position >= 0 else 0.0
        watch_reward = relevance * min(math.log1p(max(0.0, watch)) / 10.0, 1.0)
        positions = {int(item): row for row, item in enumerate(env.session.items)}
        rows = [positions[item] for item in slate if item in positions]
        source = float(np.mean(env.session.features[rows, 1])) if rows else 0.0
        longtail = float(np.mean(env.session.features[rows, 4])) if rows else 0.0
        repeated = sum(item in env.previously_recommended for item in slate) / max(1, len(slate))
        relevance_gate = 0.25 + 0.75 * relevance
        rewards.append(float(
            env.cfg.relevance_weight * relevance
            + env.cfg.watch_weight * watch_reward
            + env.cfg.source_weight * source * relevance_gate
            + env.cfg.longtail_weight * longtail * relevance_gate
            - env.cfg.repeat_penalty * repeated
        ))
    return rewards


def _reward_warm_start(
    policy: SessionPolicy,
    sessions: list[AgentSession],
    cfg: AgentConfig,
    variant: str,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    """Warm-start from a training-split router before trajectory-level GRPO.

    ``causal_fallback`` only consumes feedback that has already occurred.
    ``trajectory_oracle`` labels each policy-training session with its best
    constant route under logged replay, which is a long-horizon offline target.
    ``counterfactual`` evaluates all five deterministic routes at the current
    training state and distils their dense reward distribution. The current
    target is used only by the training teacher; it is never present in the
    policy observation, and held-out test sessions remain untouched.
    Neither strategy reads held-out test sessions. GRPO then improves complete
    trajectory return while a frozen copy supplies Reference KL.
    """
    history: list[dict[str, float]] = []
    expert_actions: dict[int, int] = {}
    class_weights = None
    if cfg.warmup_strategy == "trajectory_oracle":
        for index, session in enumerate(sessions):
            action_returns = []
            for candidate_action in range(len(ROUTING_ACTIONS)):
                probe = LiveSessionEnv(session, cfg)
                probe.reset(seed=cfg.seed)
                probe_rewards = []
                probe_done = False
                while not probe_done:
                    _, probe_reward, probe_done, _, _ = probe.step(candidate_action)
                    probe_rewards.append(probe_reward)
                action_returns.append(sum(
                    cfg.trajectory_gamma ** step * reward
                    for step, reward in enumerate(probe_rewards)
                ))
            expert_actions[index] = int(np.argmax(action_returns))
        counts = np.bincount(list(expert_actions.values()), minlength=len(ROUTING_ACTIONS))
        weights = max(1, counts.max()) / np.maximum(counts, 1)
        weights = np.minimum(weights, 12.0)
        weights /= weights.mean()
        class_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
        print(f"warmup expert counts={counts.tolist()}", flush=True)
    for epoch in range(cfg.warmup_epochs):
        count = min(len(sessions), cfg.users_per_epoch)
        selected = rng.choice(len(sessions), size=count, replace=False)
        total_loss, decisions = 0.0, 0
        for index in selected:
            session = sessions[int(index)]
            env = LiveSessionEnv(session, cfg)
            observation, _ = env.reset(seed=cfg.seed + epoch)
            hidden = None
            terminated = False
            losses = []
            while not terminated:
                model_observation = _masked_observation(observation, variant)
                if cfg.warmup_strategy == "trajectory_oracle":
                    target = expert_actions[int(index)]
                elif cfg.warmup_strategy == "causal_fallback":
                    target = 0 if env.miss_streak >= 1 else 1
                elif cfg.warmup_strategy == "counterfactual":
                    action_rewards = _counterfactual_action_rewards(env)
                    rewards_tensor = torch.as_tensor(action_rewards, dtype=torch.float32, device=device)
                    target_distribution = torch.softmax(
                        (rewards_tensor - rewards_tensor.max())
                        / max(cfg.counterfactual_temperature, 1e-4),
                        dim=0,
                    )
                    target = int(torch.argmax(rewards_tensor).item())
                else:
                    raise ValueError(f"unknown warmup strategy: {cfg.warmup_strategy}")
                logits, hidden = policy.step(
                    torch.as_tensor(model_observation, device=device), hidden
                )
                if cfg.warmup_strategy == "counterfactual":
                    reward_span = (rewards_tensor.max() - rewards_tensor.min()).detach()
                    confidence = torch.clamp(reward_span / 0.05, min=0.05, max=1.0)
                    decision_loss = -(
                        target_distribution * torch.log_softmax(logits.squeeze(0), dim=0)
                    ).sum() * confidence
                else:
                    target_tensor = torch.tensor([target], device=device)
                    decision_loss = torch.nn.functional.cross_entropy(
                        logits, target_tensor, reduction="none"
                    ).mean()
                if class_weights is not None:
                    decision_loss = decision_loss * class_weights[target]
                losses.append(decision_loss)
                observation, _, terminated, _, _ = env.step(target)
                decisions += 1
            if losses:
                loss = torch.stack(losses).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.item())
        record = {
            "warmup_epoch": float(epoch + 1),
            "warmup_loss": total_loss / max(1, count),
            "warmup_decisions": float(decisions),
        }
        history.append(record)
        print(
            f"warmup epoch={epoch + 1:03d} loss={record['warmup_loss']:.6f} "
            f"decisions={decisions}", flush=True,
        )
    return history


def _counterfactual_grpo(
    policy: SessionPolicy,
    reference_policy: SessionPolicy,
    sessions: list[AgentSession],
    cfg: AgentConfig,
    variant: str,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    """Optimize exact action groups instead of hoping sampled groups differ.

    The five deterministic routes form the GRPO group at every visited state.
    Rewards are normalized within that group, and the old-policy probability
    weights make the enumerated PPO objective match an expectation over the
    behavior policy.  State visitation follows the old greedy policy, so the
    learner is trained on states it will actually encounter at inference.
    """
    history: list[dict[str, float]] = []
    for epoch in range(cfg.epochs):
        old_policy = copy.deepcopy(policy).eval()
        count = min(len(sessions), cfg.users_per_epoch)
        selected = rng.choice(len(sessions), size=count, replace=False)
        totals = {"loss": 0.0, "return": 0.0, "kl": 0.0, "entropy": 0.0}
        updates = 0
        for index in selected:
            env = LiveSessionEnv(sessions[int(index)], cfg)
            observation, _ = env.reset(seed=cfg.seed + epoch)
            hidden = old_hidden = reference_hidden = None
            losses: list[torch.Tensor] = []
            behavior_rewards: list[float] = []
            kls: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []
            terminated = False
            while not terminated:
                model_observation = _masked_observation(observation, variant)
                tensor = torch.as_tensor(model_observation, device=device)
                logits, hidden = policy.step(tensor, hidden)
                logits = logits.squeeze(0)
                log_probability = torch.log_softmax(logits, dim=0)
                probability = torch.softmax(logits, dim=0)
                with torch.no_grad():
                    old_logits, old_hidden = old_policy.step(tensor, old_hidden)
                    old_log_probability = torch.log_softmax(old_logits.squeeze(0), dim=0)
                    old_probability = torch.softmax(old_logits.squeeze(0), dim=0)
                    reference_logits, reference_hidden = reference_policy.step(
                        tensor, reference_hidden
                    )
                    reference_log_probability = torch.log_softmax(
                        reference_logits.squeeze(0), dim=0
                    )
                    rewards = torch.as_tensor(
                        _counterfactual_action_rewards(env),
                        dtype=torch.float32, device=device,
                    )
                    advantage = (rewards - rewards.mean()) / rewards.std(
                        unbiased=False
                    ).clamp_min(1e-6)
                ratio = torch.exp((log_probability - old_log_probability).clamp(-5, 5))
                objective = torch.minimum(
                    ratio * advantage,
                    ratio.clamp(1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * advantage,
                )
                kl = torch.sum(probability * (log_probability - reference_log_probability))
                entropy = -torch.sum(probability * log_probability)
                losses.append(
                    -(old_probability * objective).sum()
                    + cfg.kl_beta * kl
                    - cfg.entropy_coef * entropy
                )
                kls.append(kl)
                entropies.append(entropy)
                behavior_action = int(torch.argmax(old_logits.squeeze(0)).item())
                observation, reward, terminated, _, _ = env.step(behavior_action)
                behavior_rewards.append(float(reward))
            if losses:
                loss = torch.stack(losses).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                totals["loss"] += float(loss.item())
                totals["return"] += float(sum(
                    cfg.trajectory_gamma**step * reward
                    for step, reward in enumerate(behavior_rewards)
                ))
                totals["kl"] += float(torch.stack(kls).mean().item())
                totals["entropy"] += float(torch.stack(entropies).mean().item())
                updates += 1
        record = {key: value / max(1, updates) for key, value in totals.items()}
        record["epoch"] = float(epoch + 1)
        record["counterfactual_grpo"] = 1.0
        history.append(record)
        print(
            f"cf-grpo epoch={epoch + 1:03d} loss={record['loss']:.6f} "
            f"return={record['return']:.6f} kl={record['kl']:.6f}", flush=True,
        )
    return history


def train_session_agent(
    train_sessions: list[AgentSession],
    test_sessions: list[AgentSession],
    output_dir: str | Path,
    config: AgentConfig | None = None,
    variant: str = "agentic",
) -> dict[str, object]:
    cfg = config or AgentConfig()
    if variant not in {"fixed", "contextual", "agentic", "no_memory", "no_routing", "myopic"}:
        raise ValueError(f"unknown agent variant: {variant}")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recurrent = variant in {"agentic", "myopic"}
    needs_policy = variant not in {"fixed", "no_routing"}
    policy = SessionPolicy(LiveSessionEnv.OBSERVATION_DIM, cfg.hidden_dim, recurrent).to(device) if needs_policy else None
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate) if policy is not None else None
    rng = np.random.default_rng(cfg.seed)
    history: list[dict[str, float]] = []

    if not 0.0 < cfg.selection_fraction < 1.0:
        raise ValueError("selection_fraction must be strictly between 0 and 1")
    if len(train_sessions) < 2:
        raise ValueError("at least two policy-training sessions are required")
    # The public cache calls this split ``valid`` because it is held out from
    # the recommender backbone.  It must still be split again for the router:
    # using the same users for gradient updates and checkpoint selection made
    # the previous pilot selection-biased.  User-level separation also avoids
    # leaking later states from the same session into model selection.
    permutation = rng.permutation(len(train_sessions))
    selection_count = min(
        len(train_sessions) - 1,
        max(1, int(round(len(train_sessions) * cfg.selection_fraction))),
    )
    selection_indices = permutation[:selection_count]
    policy_indices = permutation[selection_count:]
    selection_sessions = [train_sessions[int(index)] for index in selection_indices]
    policy_train_sessions = [train_sessions[int(index)] for index in policy_indices]

    selected_checkpoint = "not_applicable"
    valid_result = None
    if policy is not None:
        history.extend(_reward_warm_start(
            policy, policy_train_sessions, cfg, variant, optimizer, device, rng
        ))
        warm_valid, _ = evaluate_session_agent(
            selection_sessions, policy, cfg, variant, device
        )
        warm_score = float(warm_valid["overall"]["session_return"])
        warm_state = copy.deepcopy(policy.state_dict())
        reference_policy = copy.deepcopy(policy).eval()
        if cfg.warmup_strategy == "counterfactual":
            history.extend(_counterfactual_grpo(
                policy, reference_policy, policy_train_sessions, cfg, variant,
                optimizer, device, rng,
            ))
        else:
            for epoch in range(cfg.epochs):
                old_policy = copy.deepcopy(policy).eval()
                count = min(len(policy_train_sessions), cfg.users_per_epoch)
                selected = rng.choice(
                    len(policy_train_sessions), size=count, replace=False
                )
                totals = {"loss": 0.0, "return": 0.0, "kl": 0.0, "entropy": 0.0}
                updates = 0
                for index in selected:
                    session = policy_train_sessions[int(index)]
                    trajectories = [
                        _rollout(session, old_policy, cfg, variant, True, device)
                        for _ in range(cfg.group_size)
                    ]
                    gamma = 0.0 if variant == "myopic" else cfg.trajectory_gamma
                    returns = torch.tensor(
                        [trajectory.discounted_return(gamma) for trajectory in trajectories],
                        dtype=torch.float32, device=device,
                    )
                    advantage = (returns - returns.mean()) / returns.std(unbiased=False).clamp_min(1e-6)
                    current, old, kls, entropies = [], [], [], []
                    for trajectory in trajectories:
                        log_prob, kl, entropy = _trajectory_log_prob_and_kl(
                            policy, reference_policy, trajectory, device
                        )
                        current.append(log_prob)
                        old.append(torch.tensor(trajectory.old_log_prob, device=device))
                        kls.append(kl)
                        entropies.append(entropy)
                    current_tensor = torch.stack(current)
                    old_tensor = torch.stack(old)
                    kl_tensor = torch.stack(kls).mean()
                    entropy_tensor = torch.stack(entropies).mean()
                    ratio = torch.exp((current_tensor - old_tensor).clamp(-5, 5))
                    objective = torch.minimum(
                        ratio * advantage,
                        ratio.clamp(1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * advantage,
                    )
                    loss = (
                        -objective.mean()
                        + cfg.kl_beta * kl_tensor
                        - cfg.entropy_coef * entropy_tensor
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    totals["loss"] += float(loss.item())
                    totals["return"] += float(returns.mean().item())
                    totals["kl"] += float(kl_tensor.item())
                    totals["entropy"] += float(entropy_tensor.item())
                    updates += 1
                record = {key: value / max(1, updates) for key, value in totals.items()}
                record["epoch"] = float(epoch + 1)
                history.append(record)
                print(
                    f"agent epoch={epoch + 1:03d} loss={record['loss']:.6f} "
                    f"return={record['return']:.6f} kl={record['kl']:.6f}", flush=True,
                )
        grpo_valid, _ = evaluate_session_agent(
            selection_sessions, policy, cfg, variant, device
        )
        grpo_score = float(grpo_valid["overall"]["session_return"])
        if cfg.epochs <= 0:
            policy.load_state_dict(warm_state)
            valid_result = warm_valid
            selected_checkpoint = "reward_warm_start"
        elif warm_score > grpo_score:
            policy.load_state_dict(warm_state)
            valid_result = warm_valid
            selected_checkpoint = "reward_warm_start"
        else:
            valid_result = grpo_valid
            selected_checkpoint = "grpo_final"
    if valid_result is None:
        valid_result, _ = evaluate_session_agent(
            selection_sessions, policy, cfg, variant, device
        )
    test_result, per_user = evaluate_session_agent(test_sessions, policy, cfg, variant, device)
    result: dict[str, object] = {
        "model": "livebridge_session_agent",
        "variant": variant,
        "device": str(device),
        "config": asdict(cfg),
        "actions": [asdict(action) for action in ROUTING_ACTIONS],
        "valid": valid_result,
        "test": test_result,
        "training_history": history,
        "selected_checkpoint": selected_checkpoint,
        "audit": {
            "policy_training_split": "valid/router_train",
            "checkpoint_selection_split": "valid/router_selection",
            "router_train_users": len(policy_train_sessions),
            "router_selection_users": len(selection_sessions),
            "router_user_overlap": 0,
            "final_evaluation_split": "test",
            "unbiased_ope": False,
            "llm_in_hot_path": False,
            "claim_boundary": "logged-positive session replay; no online CTR/revenue claim",
        },
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    per_user.to_csv(out / "per_user_metrics.csv", index=False)
    if policy is not None:
        torch.save({"state_dict": policy.state_dict(), "config": asdict(cfg), "variant": variant}, out / "policy.pt")
    print(json.dumps(result, indent=2), flush=True)
    return result


def evaluate_session_agent(
    sessions: list[AgentSession],
    policy: SessionPolicy | None,
    cfg: AgentConfig,
    variant: str,
    device: torch.device,
) -> tuple[dict[str, object], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    exposures = np.zeros(max((s.author_count for s in sessions), default=1), dtype=np.int64)
    action_counts = np.zeros(len(ROUTING_ACTIONS), dtype=np.int64)
    tool_calls = {name: 0 for name in ToolRegistry.TOOL_NAMES}
    latencies = []
    for session in sessions:
        start = time.perf_counter()
        trajectory = _rollout(session, policy, cfg, variant, False, device)
        latencies.append((time.perf_counter() - start) * 1000 / max(1, len(trajectory.actions)))
        hits, ndcgs, watches, tails, repeats = [], [], [], [], []
        for action, info in zip(trajectory.actions, trajectory.infos):
            action_counts[action] += 1
            for call in info["calls"]:
                tool_calls[str(call["tool"])] += 1
            slate = [int(item) for item in info["slate"]]
            if slate:
                exposures[slate] += 1
            position = int(info["hit_position"])
            hits.append(float(position >= 0))
            ndcgs.append(1.0 / math.log2(position + 2) if position >= 0 else 0.0)
            watches.append(float(info["watch_reward"]))
            tails.append(float(info["longtail_share"]))
            repeats.append(float(info["repeat_rate"]))
        switches = sum(a != b for a, b in zip(trajectory.actions, trajectory.actions[1:]))
        row = {
            "user_id": session.user_id,
            "train_interactions": session.train_count,
            "bucket": history_bucket(session.train_count),
            "steps": len(trajectory.actions),
            "session_return": trajectory.discounted_return(cfg.trajectory_gamma),
            "recall@10": float(np.mean(hits)) if hits else 0.0,
            "ndcg@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "logged_watch@10": float(np.mean(watches)) if watches else 0.0,
            "longtail_share@10": float(np.mean(tails)) if tails else 0.0,
            "repeat_rate@10": float(np.mean(repeats)) if repeats else 0.0,
            "action_switch_rate": switches / max(1, len(trajectory.actions) - 1),
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    metric_names = (
        "session_return", "recall@10", "ndcg@10", "logged_watch@10",
        "longtail_share@10", "repeat_rate@10", "action_switch_rate",
    )
    overall = {name: float(frame[name].mean()) if len(frame) else 0.0 for name in metric_names}
    overall.update({
        "users": len(frame),
        "catalog_coverage@10": float(np.count_nonzero(exposures) / max(1, len(exposures))),
        "exposure_gini@10": _exposure_gini(exposures),
        "policy_latency_ms_p50": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "policy_latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "action_counts": action_counts.tolist(),
        "action_entropy": _count_entropy(action_counts),
        "tool_calls": tool_calls,
    })
    buckets = {}
    if len(frame):
        for bucket, group in frame.groupby("bucket", sort=False):
            buckets[str(bucket)] = {name: float(group[name].mean()) for name in metric_names}
            buckets[str(bucket)]["users"] = len(group)
    return {"overall": overall, "buckets": buckets}, frame


def _count_entropy(counts: np.ndarray) -> float:
    probabilities = counts.astype(np.float64)
    probabilities /= max(1.0, probabilities.sum())
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log(probabilities))) if len(probabilities) else 0.0


def _pack_sessions(
    sessions: list[AgentSession], output: Path, split: str, feature_names: tuple[str, ...]
) -> None:
    users = len(sessions)
    max_candidates = max((len(session.items) for session in sessions), default=0)
    max_events = max((len(session.targets) for session in sessions), default=0)
    items = np.full((users, max_candidates), -1, dtype=np.int64)
    features = np.zeros((users, max_candidates, len(feature_names)), dtype=np.float32)
    targets = np.full((users, max_events), -1, dtype=np.int64)
    watch = np.zeros((users, max_events), dtype=np.float32)
    timestamps = np.zeros((users, max_events), dtype=np.int64)
    candidate_lengths = np.zeros(users, dtype=np.int32)
    event_lengths = np.zeros(users, dtype=np.int32)
    user_ids = np.zeros(users, dtype=np.int64)
    train_counts = np.zeros(users, dtype=np.int32)
    author_counts = np.zeros(users, dtype=np.int32)
    for row, session in enumerate(sessions):
        c, e = len(session.items), len(session.targets)
        items[row, :c] = session.items
        features[row, :c] = session.features
        targets[row, :e] = session.targets
        watch[row, :e] = session.watch
        timestamps[row, :e] = session.timestamps
        candidate_lengths[row], event_lengths[row] = c, e
        user_ids[row], train_counts[row], author_counts[row] = (
            session.user_id, session.train_count, session.author_count
        )
    np.savez_compressed(
        output,
        split=np.asarray(split), feature_names=np.asarray(feature_names),
        user_ids=user_ids, items=items, features=features, targets=targets,
        watch=watch, timestamps=timestamps, candidate_lengths=candidate_lengths,
        event_lengths=event_lengths, train_counts=train_counts, author_counts=author_counts,
    )


def load_agent_sessions(path: str | Path) -> list[AgentSession]:
    data = np.load(path, allow_pickle=False)
    sessions = []
    for row in range(len(data["user_ids"])):
        c, e = int(data["candidate_lengths"][row]), int(data["event_lengths"][row])
        sessions.append(AgentSession(
            user_id=int(data["user_ids"][row]),
            items=data["items"][row, :c].copy(),
            features=data["features"][row, :c].copy(),
            targets=data["targets"][row, :e].copy(),
            watch=data["watch"][row, :e].copy(),
            timestamps=data["timestamps"][row, :e].copy(),
            train_count=int(data["train_counts"][row]),
            author_count=int(data["author_counts"][row]),
        ))
    return sessions


def build_agent_cache(
    processed_dir: str | Path,
    bridge_checkpoint: str | Path,
    output_dir: str | Path,
    author_profile_path: str | Path | None = None,
    candidate_pool: int = 100,
    score_batch_size: int = 128,
    max_steps: int = 8,
    independent_tool_pools: bool = False,
) -> dict[str, object]:
    root, out = Path(processed_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    live = pd.read_csv(root / "live.csv").sort_values("timestamp")
    photo = pd.read_csv(root / "photo_author.csv")
    checkpoint = torch.load(bridge_checkpoint, map_location="cpu", weights_only=False)
    u_map = {int(key): int(value) for key, value in checkpoint["user_map"].items()}
    a_map = {int(key): int(value) for key, value in checkpoint["author_map"].items()}
    live = live[live.user_id.isin(u_map) & live.author_id.isin(a_map)].copy()
    live["u"] = live.user_id.map(u_map).astype(int)
    live["a"] = live.author_id.map(a_map).astype(int)
    users, authors = len(u_map), len(a_map)
    state = checkpoint["state_dict"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference = BridgeBPR(users, authors, int(state["user.weight"].shape[1])).to(device)
    reference.load_state_dict(state)
    reference.eval()

    train = live[live.split == "train"]
    popularity = np.zeros(authors, dtype=np.float32)
    for author, count in train.author_id.value_counts().items():
        popularity[a_map[int(author)]] = float(count)
    pop_feature = _zscore(np.log1p(popularity))
    positive = popularity[popularity > 0]
    cutoff = float(np.median(positive)) if len(positive) else 0.0
    longtail = (popularity <= cutoff).astype(np.float32)
    source_index = _build_sparse_source_index(photo, u_map, a_map)

    author_features = user_profile = None
    profile_columns: list[str] = []
    if author_profile_path:
        profile = pd.read_csv(author_profile_path)
        supported = ("gender", "age_segment", "fans_user_num", "is_photo_author", "is_live_author")
        profile_columns = [column for column in supported if column in profile.columns]
        profile = profile[profile.author_id.isin(a_map)].copy()
        if profile_columns and not profile.empty:
            encoded = pd.get_dummies(profile[profile_columns].astype("string").fillna("unknown"))
            author_features = np.zeros((authors, encoded.shape[1]), dtype=np.float32)
            for index, author_id in enumerate(profile.author_id.astype(int)):
                author_features[a_map[author_id]] = encoded.iloc[index].to_numpy(dtype=np.float32)
            author_features /= np.maximum(np.linalg.norm(author_features, axis=1, keepdims=True), 1.0)
            user_profile = np.zeros((users, author_features.shape[1]), dtype=np.float32)
            for uid, group in train.groupby("u"):
                ids = group.a.astype(int).to_numpy()
                weights = np.log1p(group.play_duration.to_numpy(dtype=np.float32))
                if weights.sum() <= 0:
                    weights = np.ones(len(ids), dtype=np.float32)
                user_profile[int(uid)] = np.average(author_features[ids], axis=0, weights=weights)
            user_profile /= np.maximum(np.linalg.norm(user_profile, axis=1, keepdims=True), 1e-8)

    train_seen = {int(uid): set(group.a.astype(int)) for uid, group in train.groupby("u")}
    counts = train.groupby("u").size().astype(int).to_dict()
    valid = live[live.split == "valid"]
    valid_users = sorted(int(uid) for uid in valid.u.unique())
    valid_cache = _build_candidate_cache(
        reference, valid_users, train_seen, candidate_pool, score_batch_size,
        source_index, user_profile, author_features, pop_feature, longtail, device,
        independent_tool_pools=independent_tool_pools,
    )
    test_seen = {uid: set(items) for uid, items in train_seen.items()}
    for uid, group in valid.groupby("u"):
        test_seen.setdefault(int(uid), set()).update(group.a.astype(int))
    test = live[live.split == "test"]
    test_users = sorted(int(uid) for uid in test.u.unique())
    test_cache = _build_candidate_cache(
        reference, test_users, test_seen, candidate_pool, score_batch_size,
        source_index, user_profile, author_features, pop_feature, longtail, device,
        independent_tool_pools=independent_tool_pools,
    )

    def make_sessions(frame: pd.DataFrame, cache: dict[int, CandidateFeatures]) -> list[AgentSession]:
        sessions = []
        for uid, group in frame.sort_values("timestamp").groupby("u", sort=False):
            candidate = cache.get(int(uid))
            if candidate is None or len(candidate.items) < 2:
                continue
            group = group.iloc[:max_steps]
            sessions.append(AgentSession(
                user_id=int(uid), items=candidate.items.copy(), features=candidate.features.copy(),
                targets=group.a.to_numpy(dtype=np.int64),
                watch=group.play_duration.to_numpy(dtype=np.float32),
                timestamps=group.timestamp.to_numpy(dtype=np.int64),
                train_count=int(counts.get(int(uid), 0)), author_count=authors,
            ))
        return sessions

    valid_sessions, test_sessions = make_sessions(valid, valid_cache), make_sessions(test, test_cache)
    _pack_sessions(valid_sessions, out / "valid_sessions.npz", "valid", ResidualPolicy.FEATURE_NAMES)
    _pack_sessions(test_sessions, out / "test_sessions.npz", "test", ResidualPolicy.FEATURE_NAMES)
    metadata = {
        "processed_dir": str(root), "bridge_checkpoint": str(bridge_checkpoint),
        "device": str(device), "users": users, "authors": authors,
        "valid_sessions": len(valid_sessions), "test_sessions": len(test_sessions),
        "candidate_pool": candidate_pool, "max_steps": max_steps,
        "independent_tool_pools": independent_tool_pools,
        "profile_columns": profile_columns,
        "audit": {"valid_for_training": True, "test_for_final_evaluation_only": True},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
