from pathlib import Path
import copy

import numpy as np
from gymnasium.utils.env_checker import check_env

from livebridge_rl.agentic_rl import (
    AgentConfig,
    AgentSession,
    LiveSessionEnv,
    ToolRegistry,
    load_agent_sessions,
    train_session_agent,
    _counterfactual_action_rewards,
    _pack_sessions,
)
from livebridge_rl.grpo_reranker import ResidualPolicy


def _session(user_id: int = 0) -> AgentSession:
    items = np.arange(20, dtype=np.int64)
    features = np.zeros((20, 5), dtype=np.float32)
    features[:, 0] = np.linspace(1, -1, 20)
    features[:, 1] = np.linspace(-1, 1, 20)
    features[:, 2] = np.sin(np.arange(20))
    features[:, 3] = np.linspace(1, 0, 20)
    features[:, 4] = (np.arange(20) >= 10).astype(np.float32)
    return AgentSession(
        user_id=user_id,
        items=items,
        features=features,
        targets=np.asarray([0, 19, 1, 18], dtype=np.int64),
        watch=np.asarray([20, 30, 10, 40], dtype=np.float32),
        timestamps=np.arange(4, dtype=np.int64),
        train_count=7,
        author_count=20,
    )


def test_environment_and_tools_are_valid():
    session = _session()
    env = LiveSessionEnv(session, AgentConfig(max_steps=4, slate_size=5))
    check_env(env, skip_render_check=True)
    observation, _ = env.reset(seed=1)
    assert observation.shape == (LiveSessionEnv.OBSERVATION_DIM,)
    next_observation, _, _, _, _ = env.step(1)
    feedback_start = LiveSessionEnv.STATIC_DIM + LiveSessionEnv.DYNAMIC_DIM
    assert np.any(next_observation[feedback_start:feedback_start + 10] != 0)
    slate, trace = ToolRegistry(5).retrieve(session, 1)
    assert len(slate) == 5
    assert len(set(slate)) == 5
    assert trace["action"] == "source_transfer"


def test_cache_roundtrip_and_agent_training(tmp_path: Path):
    sessions = [_session(index) for index in range(4)]
    cache = tmp_path / "sessions.npz"
    _pack_sessions(sessions, cache, "valid", ResidualPolicy.FEATURE_NAMES)
    loaded = load_agent_sessions(cache)
    assert len(loaded) == 4
    assert np.array_equal(loaded[0].targets, sessions[0].targets)
    result = train_session_agent(
        loaded,
        loaded,
        tmp_path / "run",
        AgentConfig(
            epochs=1, users_per_epoch=2, group_size=2,
            max_steps=3, slate_size=5, hidden_dim=8, seed=1,
        ),
        variant="agentic",
    )
    assert result["variant"] == "agentic"
    assert result["audit"]["unbiased_ope"] is False
    assert result["audit"]["checkpoint_selection_split"] == "valid/router_selection"
    assert result["audit"]["router_train_users"] == 3
    assert result["audit"]["router_selection_users"] == 1
    assert result["audit"]["router_user_overlap"] == 0
    assert result["test"]["overall"]["users"] == 4
    assert (tmp_path / "run" / "metrics.json").is_file()
    assert (tmp_path / "run" / "per_user_metrics.csv").is_file()


def test_counterfactual_warm_start_runs(tmp_path: Path):
    sessions = [_session(index) for index in range(4)]
    result = train_session_agent(
        sessions,
        sessions,
        tmp_path / "counterfactual",
        AgentConfig(
            epochs=1, warmup_epochs=1, warmup_strategy="counterfactual",
            users_per_epoch=4, group_size=2, max_steps=4,
            slate_size=5, hidden_dim=8, seed=3,
        ),
        variant="agentic",
    )
    assert result["selected_checkpoint"] in {"reward_warm_start", "grpo_final"}
    assert result["config"]["warmup_strategy"] == "counterfactual"
    assert any(record.get("counterfactual_grpo") == 1.0 for record in result["training_history"])
    assert (tmp_path / "counterfactual" / "policy.pt").is_file()


def test_counterfactual_rewards_match_environment_step():
    env = LiveSessionEnv(_session(), AgentConfig(max_steps=4, slate_size=5))
    env.reset(seed=5)
    direct = _counterfactual_action_rewards(env)
    replay = [copy.deepcopy(env).step(action)[1] for action in range(5)]
    assert np.allclose(direct, replay)
