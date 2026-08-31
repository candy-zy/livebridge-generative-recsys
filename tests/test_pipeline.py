from pathlib import Path

import pandas as pd

from livebridge_rl.preprocess import prepare_sample
from livebridge_rl.baseline import TrainConfig, train_baseline
from livebridge_rl.popularity import run_popularity
from livebridge_rl.strong_baselines import StrongConfig, train_strong


def test_prepare_sample(tmp_path: Path):
    data, out = tmp_path / "raw", tmp_path / "processed"
    data.mkdir()
    pd.DataFrame({"user_id": [1, 2]}).to_csv(data / "user_id_set.csv", index=False)
    rows = []
    for user in (1, 2):
        for i in range(6):
            rows.append({"user_id": user, "author_id": i, "live_play_start_timestamp": f"2026-01-{i+1:02d}", "play_duration": 10})
    pd.DataFrame(rows).to_csv(data / "live_interaction.csv", index=False)
    pd.DataFrame([
        {"user_id": u, "author_id": a, "show_cnt": 1, "complete_play_cnt": 1,
         "play_progress": 1, "like_cnt": 0, "direct_comment_cnt": 0,
         "reply_comment_cnt": 0, "follow_cnt": 0, "share_cnt": 0}
        for u in (1, 2) for a in range(6)
    ]).to_csv(data / "photo_interaction.csv", index=False)
    report = prepare_sample(data, out, sample_ratio=1.0, min_live_user=1, min_live_author=1)
    assert report["retained_users"] == 2
    prepared = pd.read_csv(out / "live.csv")
    assert set(prepared["split"]) == {"train", "valid", "test"}
    assert (prepared.groupby("user_id")["split"].nunique() == 3).all()
    assert report["shared_authors"] == 6
    result = train_baseline(out, tmp_path / "run", "bridge", TrainConfig(epochs=1, batch_size=8))
    assert (tmp_path / "run" / "metrics.json").is_file()
    assert result["mode"] == "bridge"
    popularity = run_popularity(out, tmp_path / "popularity")
    assert popularity["model"] == "popularity"
    strong_cfg = StrongConfig(
        embedding_dim=8, epochs=1, batch_size=8, layers=1, max_length=5, seed=1
    )
    for model in ("lightgcn", "sasrec", "emcdr"):
        strong = train_strong(model, out, tmp_path / model, strong_cfg)
        assert strong["model"] == model
        assert "5-10" in strong["test_buckets"]

    pd.DataFrame([
        {"user_id": u, "author_id": a, "enter_timestamp": when,
         "leave_timestamp": pd.Timestamp(when) + pd.Timedelta(seconds=5),
         "is_complete_play": 1, "like_status_type": 0,
         "is_follow_before_play": 0, "is_follow_after_play": 0}
        for u in (1, 2)
        for a, when in ((0, "2026-01-02"), (5, "2026-01-10"))
    ]).to_parquet(data / "photo_play.parquet", index=False)
    temporal_out = tmp_path / "temporal"
    temporal = prepare_sample(
        data, temporal_out, sample_ratio=1.0, min_live_user=1,
        min_live_author=1, source_mode="temporal"
    )
    assert temporal["source_mode"] == "temporal"
    assert set(pd.read_csv(temporal_out / "photo_author.csv")["author_id"]) == {0}
