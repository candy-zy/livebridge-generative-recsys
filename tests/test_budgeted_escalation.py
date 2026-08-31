from pathlib import Path

import numpy as np
import torch

from livebridge_rl.agentic_rl import AgentConfig
from scripts.train_budgeted_escalation import (
    EscalationCritic,
    average_precision,
    build_examples,
    roc_auc,
    split_users,
    train_critic,
)
from test_agentic_rl import _session


def test_user_split_has_no_overlap():
    sessions = [_session(index) for index in range(10)]
    train, selection = split_users(sessions, 0.2, 42)
    assert len(train) == 8
    assert len(selection) == 2
    assert not ({s.user_id for s in train} & {s.user_id for s in selection})


def test_examples_are_causal_and_binary():
    x, y, advantage = build_examples(
        [_session(0)],
        AgentConfig(max_steps=4, slate_size=5, source_weight=0, longtail_weight=0),
    )
    assert x.shape == (4, 40)
    assert set(np.unique(y)).issubset({0.0, 1.0})
    assert np.array_equal(y, (advantage > 1e-8).astype(np.float32))


def test_metrics_and_critic_shapes():
    labels = np.asarray([0, 0, 1, 1], dtype=np.float32)
    scores = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    assert roc_auc(labels, scores) == 1.0
    assert average_precision(labels, scores) == 1.0
    model = EscalationCritic(40, hidden_dim=8)
    assert model(torch.zeros(3, 40)).shape == (3,)


def test_advantage_regression_critic_trains():
    from argparse import Namespace

    rng = np.random.default_rng(42)
    x = rng.normal(size=(32, 4)).astype(np.float32)
    advantage = (0.5 * x[:, 0] - 0.2 * x[:, 1]).astype(np.float32)
    y = (advantage > 0).astype(np.float32)
    args = Namespace(
        hidden_dim=8,
        learning_rate=1e-2,
        weight_decay=0.0,
        objective="advantage_regression",
        advantage_weight=4.0,
        batch_size=16,
        epochs=3,
        seed=42,
    )
    model, mean, std, scale = train_critic(
        x, y, advantage, args, torch.device("cpu")
    )
    assert isinstance(model, EscalationCritic)
    assert mean.shape == std.shape == (4,)
    assert scale > 0
