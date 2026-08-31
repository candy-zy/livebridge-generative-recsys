from pathlib import Path

import numpy as np
import pandas as pd

from livebridge_rl.baseline import TrainConfig, train_baseline
from livebridge_rl.generative_retrieval import (
    AlignmentConfig,
    GeneratorConfig,
    _retrieval_metrics,
    aggregate_author_content,
    fuse_cached_generative_candidates,
    residual_kmeans,
    train_content_alignment,
    train_generative_retriever,
)
from livebridge_rl.preprocess import prepare_sample


def _fixture(tmp_path: Path):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    pd.DataFrame({"user_id": [1, 2, 3, 4]}).to_csv(
        raw / "user_id_set.csv", index=False
    )
    live_rows = []
    for user in range(1, 5):
        for index in range(8):
            live_rows.append({
                "user_id": user,
                "author_id": (index + user) % 8,
                "live_play_start_timestamp": f"2026-03-{index + 1:02d}",
                "play_duration": 10 + index,
            })
    pd.DataFrame(live_rows).to_csv(raw / "live_interaction.csv", index=False)
    photo_rows = []
    photo_meta = []
    photo_emb = []
    live_meta = []
    live_emb = []
    rng = np.random.default_rng(7)
    latent = rng.normal(size=(8, 16)).astype(np.float32)
    for user in range(1, 5):
        for author in range(8):
            photo_rows.append({
                "user_id": user, "photo_id": author, "author_id": author,
                "show_cnt": 1, "complete_play_cnt": 1, "play_progress": 1,
                "like_cnt": 0, "direct_comment_cnt": 0, "reply_comment_cnt": 0,
                "follow_cnt": 0, "share_cnt": 0,
            })
    for author in range(8):
        photo_meta.append({"photo_id": author, "author_id": author})
        photo_emb.append({
            "photo_id": author,
            "feature": np.pad(latent[author], (0, 112)).tolist(),
        })
        live_meta.append({"live_id": author, "author_id": author})
        live_emb.append({
            "live_id": author,
            "embedding": np.pad(latent[author], (0, 48)).tolist(),
        })
    pd.DataFrame(photo_rows).to_csv(raw / "photo_interaction.csv", index=False)
    pd.DataFrame(photo_meta).to_parquet(raw / "photo_meta.parquet", index=False)
    pd.DataFrame(photo_emb).to_parquet(raw / "photo_emb_128.parquet", index=False)
    pd.DataFrame(live_meta).to_parquet(raw / "live_room_meta.parquet", index=False)
    pd.DataFrame(live_emb).to_parquet(raw / "live_emb_64.parquet", index=False)
    prepare_sample(
        raw, processed, sample_ratio=1.0, min_live_user=1,
        min_live_author=1, source_mode="aggregate",
    )
    return raw, processed


def test_residual_kmeans_is_deterministic():
    vectors = np.arange(48, dtype=np.float32).reshape(8, 6)
    first, books = residual_kmeans(vectors, levels=2, codebook_size=4, seed=3)
    second, _ = residual_kmeans(vectors, levels=2, codebook_size=4, seed=3)
    assert np.array_equal(first, second)
    assert first.shape == (8, 2)
    assert len(books) == 2


def test_retrieval_metrics_exact_rank_without_full_matrix():
    vectors = np.eye(5, dtype=np.float32)
    metrics = _retrieval_metrics(vectors, vectors)
    assert metrics == {"recall@1": 1.0, "recall@10": 1.0, "mrr": 1.0}


def test_alignment_and_generator_pipeline(tmp_path: Path):
    raw, processed = _fixture(tmp_path)
    aligned = tmp_path / "aligned"
    audit = aggregate_author_content(raw, processed, aligned, parquet_batch_size=3)
    assert audit["paired_authors"] == 8
    alignment = train_content_alignment(
        aligned / "author_content_raw.npz", aligned,
        AlignmentConfig(
            output_dim=8, hidden_dim=16, epochs=3, batch_size=4, seed=1
        ),
    )
    assert alignment["audit"]["candidate_universe_reduced"] is False
    bridge = tmp_path / "bridge"
    train_baseline(
        processed, bridge, "bridge",
        TrainConfig(embedding_dim=8, epochs=1, batch_size=16, seed=1),
    )
    generated = train_generative_retriever(
        processed, bridge / "model.pt", aligned / "aligned_author_content.npz",
        tmp_path / "generated", "id",
        GeneratorConfig(
            hidden_dim=8, codebook_size=4, code_levels=2, epochs=1,
            batch_size=8, max_length=6, beam_width=8, seed=1,
        ),
    )
    assert generated["candidate_authors"] == 8
    assert generated["audit"]["candidate_universe_reduced"] is False
    assert (tmp_path / "generated" / "semantic_ids.csv").is_file()
    assert (tmp_path / "generated" / "valid_base_candidates.npz").is_file()
    assert (tmp_path / "generated" / "training_checkpoint.pt").is_file()
    progress = pd.read_json(
        tmp_path / "generated" / "training_progress.json", typ="series"
    )
    assert progress["completed_epochs"] == 1
    fused = fuse_cached_generative_candidates(
        processed, aligned / "aligned_author_content.npz",
        tmp_path / "generated", tmp_path / "fused",
    )
    assert fused["audit"]["generator_retrained"] is False
    assert fused["audit"]["beam_decoding_repeated"] is False
