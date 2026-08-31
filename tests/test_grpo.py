from pathlib import Path

import pandas as pd

from livebridge_rl.baseline import TrainConfig, train_baseline
from livebridge_rl.grpo_reranker import GRPOConfig, train_grpo_reranker
from livebridge_rl.content_fusion import run_profile_fusion
from livebridge_rl.evaluation import evaluate_full_sort, evaluate_rankings
from livebridge_rl.preprocess import prepare_sample


def test_compact_ranking_evaluation_matches_full_sort():
    truth = {0: {2, 4}, 1: {1}}
    seen = {0: {0}, 1: {3}}
    candidates = pd.Series(range(6)).to_numpy()
    scores = {
        0: pd.Series([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]).to_numpy(),
        1: pd.Series([0.1, 0.9, 0.8, 0.7, 0.6, 0.5]).to_numpy(),
    }
    full, _ = evaluate_full_sort(
        truth, seen, candidates, lambda uid, items: scores[uid][items],
        {0: 5, 1: 5}, ks=(2, 3),
    )
    rankings = {0: [1, 2, 3], 1: [1, 2, 4]}
    compact, _ = evaluate_rankings(
        truth, rankings, {0: 5, 1: 5}, ks=(2, 3), candidate_count=6,
    )
    assert compact == full


def test_grpo_pipeline(tmp_path: Path):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    pd.DataFrame({"user_id": [1, 2, 3]}).to_csv(raw / "user_id_set.csv", index=False)
    live_rows = []
    for user in (1, 2, 3):
        for index in range(8):
            live_rows.append({
                "user_id": user,
                "author_id": (index + user) % 8,
                "live_play_start_timestamp": f"2026-02-{index + 1:02d}",
                "play_duration": 5 + index,
            })
    pd.DataFrame(live_rows).to_csv(raw / "live_interaction.csv", index=False)
    pd.DataFrame([
        {
            "user_id": user, "author_id": author, "show_cnt": 1,
            "complete_play_cnt": int(author % 2 == 0), "play_progress": 1,
            "like_cnt": int(author == user), "direct_comment_cnt": 0,
            "reply_comment_cnt": 0, "follow_cnt": 0, "share_cnt": 0,
        }
        for user in (1, 2, 3) for author in range(8)
    ]).to_csv(raw / "photo_interaction.csv", index=False)
    prepare_sample(
        raw, processed, sample_ratio=1.0, min_live_user=1,
        min_live_author=1, source_mode="aggregate",
    )
    bridge_dir = tmp_path / "bridge"
    train_baseline(
        processed, bridge_dir, "bridge",
        TrainConfig(embedding_dim=8, epochs=1, batch_size=16, seed=1),
    )
    pd.DataFrame([
        {
            "author_id": author, "is_photo_author": 1, "is_live_author": 1,
            "gender": "M" if author % 2 else "F", "age_segment": "20-30",
            "fans_user_num": "1w-10w",
        }
        for author in range(8)
    ]).to_csv(raw / "author_profile.csv", index=False)
    result = train_grpo_reranker(
        processed, bridge_dir / "model.pt", tmp_path / "grpo",
        GRPOConfig(
            epochs=1, group_size=3, candidate_pool=6, slate_size=3,
            learning_rate=0.01, seed=1,
        ), author_profile_path=raw / "author_profile.csv",
    )
    assert result["model"] == "creator_bridge_grpo"
    assert result["audit"]["unbiased_ope"] is False
    assert "catalog_coverage@10" in result["test_exposure"]
    assert (tmp_path / "grpo" / "metrics.json").is_file()
    assert (tmp_path / "grpo" / "per_user_metrics.csv").is_file()
    content = run_profile_fusion(
        processed, raw / "author_profile.csv", bridge_dir / "model.pt", tmp_path / "content"
    )
    assert content["model"] == "creator_bridge_profile_fusion"
    assert content["audit"]["future_live_content_used"] is False
