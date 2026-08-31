"""Streaming preprocessing for the KuaiLive-M3 cross-domain task."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


LIVE_COLUMNS = ["user_id", "author_id", "live_play_start_timestamp", "play_duration"]
PHOTO_COLUMNS = [
    "user_id", "author_id", "show_cnt", "complete_play_cnt", "play_progress",
    "like_cnt", "direct_comment_cnt", "reply_comment_cnt", "follow_cnt", "share_cnt",
]
PHOTO_PLAY_COLUMNS = [
    "user_id", "author_id", "enter_timestamp", "leave_timestamp",
    "is_complete_play", "like_status_type", "is_follow_before_play", "is_follow_after_play",
]


def _read_selected(path: Path, columns: list[str], users: set[int], chunksize: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False):
        hit = chunk[chunk["user_id"].isin(users)]
        if not hit.empty:
            parts.append(hit)
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)


def _kcore(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    previous = -1
    while previous != len(df):
        previous = len(df)
        uc = df["user_id"].value_counts()
        ic = df["author_id"].value_counts()
        df = df[df["user_id"].isin(uc[uc >= min_user].index)]
        df = df[df["author_id"].isin(ic[ic >= min_item].index)]
    return df.reset_index(drop=True)


def _read_temporal_photo(
    path: Path, users: set[int], train_cutoff: dict[int, int], batch_size: int
) -> pd.DataFrame:
    """Read source events that occurred no later than each user's target train cutoff."""
    import pyarrow.parquet as pq

    parts: list[pd.DataFrame] = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=PHOTO_PLAY_COLUMNS, batch_size=batch_size):
        frame = batch.to_pandas()
        frame = frame[frame["user_id"].isin(users)]
        if frame.empty:
            continue
        enter = pd.to_datetime(frame["enter_timestamp"], errors="coerce")
        leave = pd.to_datetime(frame["leave_timestamp"], errors="coerce")
        frame["timestamp"] = enter.astype("int64") // 10**9
        frame["cutoff"] = frame["user_id"].map(train_cutoff)
        watch_seconds = (leave - enter).dt.total_seconds().fillna(0)
        frame = frame[(frame["timestamp"] > 0) & (frame["timestamp"] <= frame["cutoff"]) & (watch_seconds > 1)]
        if frame.empty:
            continue
        for column in ("is_complete_play", "like_status_type", "is_follow_before_play", "is_follow_after_play"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
        frame["engagement"] = (
            1.0 + 2.0 * frame["is_complete_play"].clip(lower=0)
            + 3.0 * frame["like_status_type"].clip(lower=0)
            + 5.0 * (frame["is_follow_after_play"] - frame["is_follow_before_play"]).clip(lower=0)
        ).astype("float32")
        # Reduce repeated user-author events inside each parquet batch before
        # retaining them.  On full KuaiLive-M3 this avoids keeping tens of
        # millions of event-level rows alive until the final aggregation.
        parts.append(
            frame.groupby(["user_id", "author_id"], as_index=False)["engagement"].sum()
        )
    if not parts:
        return pd.DataFrame(columns=["user_id", "author_id", "engagement"])
    photo = pd.concat(parts, ignore_index=True)
    return photo.groupby(["user_id", "author_id"], as_index=False)["engagement"].sum()


def prepare_sample(
    data_dir: str | Path,
    output_dir: str | Path,
    sample_ratio: float = 0.01,
    seed: int = 42,
    chunksize: int = 500_000,
    min_live_user: int = 5,
    min_live_author: int = 2,
    source_mode: str = "aggregate",
) -> dict[str, object]:
    """Create a deterministic user-level sample without loading raw CSVs into RAM."""
    root, out = Path(data_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    users_df = pd.read_csv(root / "user_id_set.csv", usecols=["user_id"])
    all_users = users_df["user_id"].dropna().astype("int64").unique()
    if not 0 < sample_ratio <= 1:
        raise ValueError("sample_ratio must be in (0, 1]")
    rng = np.random.default_rng(seed)
    count = max(1, round(len(all_users) * sample_ratio))
    sampled = set(rng.choice(all_users, size=count, replace=False).tolist())

    live = _read_selected(root / "live_interaction.csv", LIVE_COLUMNS, sampled, chunksize)
    live["timestamp"] = pd.to_datetime(
        live.pop("live_play_start_timestamp"), errors="coerce"
    ).astype("int64") // 10**9
    live["play_duration"] = pd.to_numeric(live["play_duration"], errors="coerce").fillna(0)
    live = live[(live["timestamp"] > 0) & (live["play_duration"] > 0)]
    live = live.sort_values("timestamp").drop_duplicates(["user_id", "author_id"], keep="last")
    live = _kcore(live, min_live_user, min_live_author)

    splits: list[pd.DataFrame] = []
    for _, group in live.sort_values("timestamp").groupby("user_id", sort=False):
        n = len(group)
        # k-core guarantees n>=5. Reserve at least one chronological event
        # for both validation and test even when a user's history is short.
        valid_end = min(max(2, int(n * 0.9)), n - 1)
        train_end = min(max(1, int(n * 0.8)), valid_end - 1)
        labels = np.full(n, "train", dtype=object)
        labels[train_end:valid_end] = "valid"
        labels[valid_end:] = "test"
        part = group.copy()
        part["split"] = labels
        splits.append(part)
    live = pd.concat(splits, ignore_index=True) if splits else live.assign(split=[])

    retained_users = set(live["user_id"].unique().tolist())
    if source_mode == "aggregate":
        photo = _read_selected(root / "photo_interaction.csv", PHOTO_COLUMNS, retained_users, chunksize)
        numeric = [c for c in PHOTO_COLUMNS if c not in ("user_id", "author_id")]
        photo[numeric] = photo[numeric].apply(pd.to_numeric, errors="coerce").fillna(0)
        photo["engagement"] = (
            np.log1p(photo["show_cnt"].clip(lower=0))
            + 2.0 * photo["complete_play_cnt"].clip(lower=0)
            + 3.0 * photo["like_cnt"].clip(lower=0)
            + 4.0 * (photo["direct_comment_cnt"] + photo["reply_comment_cnt"]).clip(lower=0)
            + 5.0 * photo["follow_cnt"].clip(lower=0)
            + 5.0 * photo["share_cnt"].clip(lower=0)
        ).astype("float32")
        photo = photo[photo["engagement"] > 0]
        photo = photo.groupby(["user_id", "author_id"], as_index=False)["engagement"].sum()
    elif source_mode == "temporal":
        cutoff = live[live["split"] == "train"].groupby("user_id")["timestamp"].max().to_dict()
        photo = _read_temporal_photo(root / "photo_play.parquet", retained_users, cutoff, chunksize)
    else:
        raise ValueError("source_mode must be aggregate or temporal")

    live.to_csv(out / "live.csv", index=False)
    photo.to_csv(out / "photo_author.csv", index=False)
    metadata = {
        "seed": seed,
        "sample_ratio": sample_ratio,
        "source_mode": source_mode,
        "sampled_users": len(sampled),
        "retained_users": int(live["user_id"].nunique()),
        "live_authors": int(live["author_id"].nunique()),
        "live_interactions": len(live),
        "photo_author_interactions": len(photo),
        "cross_domain_users": int(len(set(photo["user_id"]) & set(live["user_id"]))),
        "shared_authors": int(len(set(photo["author_id"]) & set(live["author_id"]))),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
