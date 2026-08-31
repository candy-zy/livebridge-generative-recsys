"""Leakage-safe content alignment and Semantic-ID generative retrieval.

The dataset supplies precomputed multimodal vectors.  This module aligns those
vectors at author level; it does not train or claim a raw-video MLLM.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from livebridge_rl.evaluation import evaluate_rankings


@dataclass
class AlignmentConfig:
    output_dim: int = 64
    hidden_dim: int = 128
    epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 1e-3
    temperature: float = 0.07
    selection_fraction: float = 0.20
    parquet_batch_size: int = 131_072
    seed: int = 42


@dataclass
class GeneratorConfig:
    hidden_dim: int = 64
    codebook_size: int = 32
    code_levels: int = 3
    epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 1e-3
    max_length: int = 50
    beam_width: int = 100
    seed: int = 42


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _vector_matrix(values: pd.Series, expected_dim: int) -> np.ndarray:
    rows = [np.asarray(value, dtype=np.float32) for value in values]
    if not rows:
        return np.empty((0, expected_dim), dtype=np.float32)
    matrix = np.stack(rows)
    if matrix.shape[1] != expected_dim:
        raise ValueError(f"expected {expected_dim}D vectors, got {matrix.shape}")
    return matrix


def _stream_author_means(
    parquet_path: Path,
    id_column: str,
    vector_column: str,
    expected_dim: int,
    id_to_author: dict[int, int],
    candidate_index: dict[int, int],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq

    sums = np.zeros((len(candidate_index), expected_dim), dtype=np.float64)
    counts = np.zeros(len(candidate_index), dtype=np.int64)
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(
        columns=[id_column, vector_column], batch_size=batch_size
    ):
        frame = batch.to_pandas()
        authors = frame[id_column].map(id_to_author)
        keep = authors.isin(candidate_index)
        if not keep.any():
            continue
        vectors = _vector_matrix(frame.loc[keep, vector_column], expected_dim)
        rows = authors[keep].map(candidate_index).to_numpy(dtype=np.int64)
        np.add.at(sums, rows, vectors)
        np.add.at(counts, rows, 1)
    means = np.zeros_like(sums, dtype=np.float32)
    present = counts > 0
    means[present] = (sums[present] / counts[present, None]).astype(np.float32)
    return means, counts


def aggregate_author_content(
    data_dir: str | Path,
    processed_dir: str | Path,
    output_dir: str | Path,
    parquet_batch_size: int = 131_072,
) -> dict[str, object]:
    raw, processed, out = Path(data_dir), Path(processed_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidate_authors = np.sort(
        pd.read_csv(processed / "live.csv", usecols=["author_id"])
        .author_id.dropna().astype(np.int64).unique()
    )
    candidate_index = {int(author): index for index, author in enumerate(candidate_authors)}
    photo_meta = pd.read_parquet(raw / "photo_meta.parquet", columns=["photo_id", "author_id"])
    photo_meta = photo_meta[photo_meta.author_id.isin(candidate_index)]
    live_meta = pd.read_parquet(raw / "live_room_meta.parquet", columns=["live_id", "author_id"])
    live_meta = live_meta[live_meta.author_id.isin(candidate_index)]
    photo_map = dict(zip(photo_meta.photo_id.astype(int), photo_meta.author_id.astype(int)))
    live_map = dict(zip(live_meta.live_id.astype(int), live_meta.author_id.astype(int)))
    photo, photo_count = _stream_author_means(
        raw / "photo_emb_128.parquet", "photo_id", "feature", 128,
        photo_map, candidate_index, parquet_batch_size,
    )
    live, live_count = _stream_author_means(
        raw / "live_emb_64.parquet", "live_id", "embedding", 64,
        live_map, candidate_index, parquet_batch_size,
    )
    np.savez_compressed(
        out / "author_content_raw.npz",
        author_ids=candidate_authors,
        photo=photo,
        live=live,
        photo_count=photo_count,
        live_count=live_count,
    )
    audit = {
        "candidate_authors": len(candidate_authors),
        "photo_present": int((photo_count > 0).sum()),
        "live_present": int((live_count > 0).sum()),
        "paired_authors": int(((photo_count > 0) & (live_count > 0)).sum()),
        "candidate_universe_reduced": False,
        "future_segment_embeddings_used": False,
        "photo_schema": {"id": "photo_id", "vector": "feature", "dim": 128},
        "live_schema": {"id": "live_id", "vector": "embedding", "dim": 64},
    }
    (out / "aggregation_audit.json").write_text(json.dumps(audit, indent=2))
    return audit


class Projection(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(x), dim=-1)


def _retrieval_metrics(query: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if not len(query):
        return {"recall@1": 0.0, "recall@10": 0.0, "mrr": 0.0}
    if len(query) != len(target):
        raise ValueError("paired retrieval evaluation requires equal query/target rows")
    # Full-data evaluation can contain hundreds of thousands of paired authors.
    # Materialising the complete N x N score and argsort matrices would require
    # hundreds of gigabytes, so compute the exact positive rank in bounded
    # query blocks. Continuous embeddings make score ties vanishingly rare.
    max_score_elements = 16_000_000  # about 64 MiB of float32 scores
    block_size = max(1, min(1024, max_score_elements // len(target)))
    positions = np.empty(len(query), dtype=np.int64)
    target_t = target.T
    for start in range(0, len(query), block_size):
        stop = min(start + block_size, len(query))
        scores = query[start:stop] @ target_t
        positive = scores[np.arange(stop - start), np.arange(start, stop)]
        positions[start:stop] = np.sum(scores > positive[:, None], axis=1)
    return {
        "recall@1": float(np.mean(positions < 1)),
        "recall@10": float(np.mean(positions < min(10, len(query)))),
        "mrr": float(np.mean(1.0 / (positions + 1))),
    }


def train_content_alignment(
    aggregate_path: str | Path,
    output_dir: str | Path,
    cfg: AlignmentConfig | None = None,
) -> dict[str, object]:
    cfg = cfg or AlignmentConfig()
    _seed(cfg.seed)
    data, out = np.load(aggregate_path), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    author_ids = data["author_ids"]
    photo, live = data["photo"], data["live"]
    photo_present, live_present = data["photo_count"] > 0, data["live_count"] > 0
    paired = np.flatnonzero(photo_present & live_present)
    if len(paired) < 4:
        raise RuntimeError("need at least four paired authors for alignment")
    rng = np.random.default_rng(cfg.seed)
    paired = rng.permutation(paired)
    selection_count = min(len(paired) - 2, max(2, round(len(paired) * cfg.selection_fraction)))
    selection, training = paired[:selection_count], paired[selection_count:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    photo_encoder = Projection(128, cfg.hidden_dim, cfg.output_dim).to(device)
    live_encoder = Projection(64, cfg.hidden_dim, cfg.output_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(photo_encoder.parameters()) + list(live_encoder.parameters()),
        lr=cfg.learning_rate, weight_decay=1e-4,
    )
    losses = []
    for epoch in range(cfg.epochs):
        order = rng.permutation(training)
        total = 0.0
        for start in range(0, len(order), cfg.batch_size):
            rows = order[start:start + cfg.batch_size]
            p = photo_encoder(torch.as_tensor(photo[rows], device=device))
            l = live_encoder(torch.as_tensor(live[rows], device=device))
            logits = p @ l.T / cfg.temperature
            labels = torch.arange(len(rows), device=device)
            loss = 0.5 * (
                nn.functional.cross_entropy(logits, labels)
                + nn.functional.cross_entropy(logits.T, labels)
            )
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss.item()) * len(rows)
        losses.append(total / max(1, len(training)))
    photo_encoder.eval(); live_encoder.eval()
    with torch.no_grad():
        photo_aligned = photo_encoder(torch.as_tensor(photo, device=device)).cpu().numpy()
        live_aligned = live_encoder(torch.as_tensor(live, device=device)).cpu().numpy()
    fixed_photo = photo[selection, :64]
    fixed_photo /= np.linalg.norm(fixed_photo, axis=1, keepdims=True).clip(1e-8)
    fixed_live = live[selection]
    fixed_live /= np.linalg.norm(fixed_live, axis=1, keepdims=True).clip(1e-8)
    fixed_metrics = _retrieval_metrics(fixed_photo, fixed_live)
    learned_metrics = _retrieval_metrics(
        photo_aligned[selection], live_aligned[selection]
    )
    content = np.zeros((len(author_ids), cfg.output_dim), dtype=np.float32)
    both = photo_present & live_present
    content[both] = photo_aligned[both] + live_aligned[both]
    content[photo_present & ~live_present] = photo_aligned[photo_present & ~live_present]
    content[live_present & ~photo_present] = live_aligned[live_present & ~photo_present]
    norms = np.linalg.norm(content, axis=1, keepdims=True)
    nonzero = norms[:, 0] > 0
    content[nonzero] /= norms[nonzero]
    np.savez_compressed(
        out / "aligned_author_content.npz",
        author_ids=author_ids,
        content=content,
        content_present=photo_present | live_present,
        photo_present=photo_present,
        live_present=live_present,
    )
    torch.save({
        "photo_encoder": photo_encoder.state_dict(),
        "live_encoder": live_encoder.state_dict(),
        "config": asdict(cfg),
    }, out / "alignment.pt")
    result = {
        "model": "author_cross_modal_infonce",
        "device": str(device),
        "config": asdict(cfg),
        "training_pairs": len(training),
        "selection_pairs": len(selection),
        "content_coverage": float((photo_present | live_present).mean()),
        "fixed_projection": fixed_metrics,
        "learned_alignment": learned_metrics,
        "gate_passed": learned_metrics["recall@10"] > fixed_metrics["recall@10"],
        "loss": losses,
        "audit": {
            "author_disjoint_selection": True,
            "candidate_universe_reduced": False,
            "dataset_precomputed_embeddings": True,
            "raw_video_encoder_trained": False,
            "future_segment_embeddings_used": False,
        },
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def residual_kmeans(
    vectors: np.ndarray, levels: int, codebook_size: int, seed: int, iterations: int = 20
) -> tuple[np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    residual = vectors.astype(np.float32).copy()
    codes = np.zeros((len(vectors), levels), dtype=np.int64)
    codebooks = []
    for level in range(levels):
        k = min(codebook_size, len(vectors))
        centers = residual[rng.choice(len(vectors), size=k, replace=False)].copy()
        for _ in range(iterations):
            distances = (
                (residual**2).sum(1, keepdims=True)
                - 2 * residual @ centers.T
                + (centers**2).sum(1)[None, :]
            )
            assignment = distances.argmin(1)
            updated = centers.copy()
            for cluster in range(k):
                members = residual[assignment == cluster]
                if len(members):
                    updated[cluster] = members.mean(0)
            if np.array_equal(assignment, codes[:, level]) and level == 0:
                centers = updated
                break
            centers = updated
        codes[:, level] = assignment
        residual -= centers[assignment]
        if k < codebook_size:
            centers = np.pad(centers, ((0, codebook_size - k), (0, 0)))
        codebooks.append(centers.astype(np.float32))
    return codes, codebooks


class SemanticIDGenerator(nn.Module):
    def __init__(self, cfg: GeneratorConfig):
        super().__init__()
        self.cfg = cfg
        self.tokens = nn.ModuleList([
            nn.Embedding(cfg.codebook_size, cfg.hidden_dim)
            for _ in range(cfg.code_levels)
        ])
        self.position = nn.Embedding(cfg.max_length, cfg.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            cfg.hidden_dim, nhead=4, dim_feedforward=cfg.hidden_dim * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.history = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.start = nn.Parameter(torch.zeros(cfg.hidden_dim))
        self.decoder = nn.GRUCell(cfg.hidden_dim, cfg.hidden_dim)
        self.heads = nn.ModuleList([
            nn.Linear(cfg.hidden_dim, cfg.codebook_size)
            for _ in range(cfg.code_levels)
        ])

    def encode(self, history_codes: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        state = sum(
            embedding(history_codes[:, :, level])
            for level, embedding in enumerate(self.tokens)
        )
        positions = torch.arange(state.shape[1], device=state.device)[None, :]
        state = state + self.position(positions)
        causal = torch.triu(torch.ones(
            state.shape[1], state.shape[1], dtype=torch.bool, device=state.device
        ), diagonal=1)
        encoded = self.history(state, mask=causal, src_key_padding_mask=padding_mask)
        last = (~padding_mask).sum(1).sub(1).clamp_min(0)
        return self.norm(encoded[torch.arange(len(encoded), device=state.device), last])

    def teacher_logits(
        self, history_codes: torch.Tensor, padding_mask: torch.Tensor, target: torch.Tensor
    ) -> list[torch.Tensor]:
        hidden = self.encode(history_codes, padding_mask)
        decoder_input = self.start[None, :].expand(len(hidden), -1)
        outputs = []
        for level in range(self.cfg.code_levels):
            hidden = self.decoder(decoder_input, hidden)
            outputs.append(self.heads[level](hidden))
            decoder_input = self.tokens[level](target[:, level])
        return outputs


def _pad_code_history(sequence: list[int], codes: np.ndarray, max_length: int):
    values = sequence[-max_length:]
    output = np.zeros((max_length, codes.shape[1]), dtype=np.int64)
    mask = np.ones(max_length, dtype=bool)
    if values:
        output[:len(values)] = codes[np.asarray(values)]
        mask[:len(values)] = False
    return output, mask


def _beam_codes(
    model: SemanticIDGenerator,
    history: list[int],
    codes: np.ndarray,
    device: torch.device,
    beam_width: int,
) -> list[tuple[tuple[int, ...], float]]:
    return _beam_codes_batch(
        model, [history], codes, device, beam_width
    )[0]


def _beam_codes_batch(
    model: SemanticIDGenerator,
    histories: list[list[int]],
    codes: np.ndarray,
    device: torch.device,
    beam_width: int,
) -> list[list[tuple[tuple[int, ...], float]]]:
    """Vectorized beam search for many users to avoid tiny per-user kernels."""
    if not histories:
        return []
    padded = [_pad_code_history(row, codes, model.cfg.max_length) for row in histories]
    packed = np.stack([row[0] for row in padded])
    mask = np.stack([row[1] for row in padded])
    batch_size = len(histories)
    with torch.no_grad():
        hidden = model.encode(
            torch.as_tensor(packed, device=device),
            torch.as_tensor(mask, device=device),
        )
        states = hidden[:, None, :]
        decoder_inputs = model.start[None, None, :].expand(batch_size, 1, -1)
        scores = torch.zeros((batch_size, 1), device=device)
        prefixes = torch.empty(
            (batch_size, 1, 0), dtype=torch.long, device=device
        )
        for level in range(model.cfg.code_levels):
            width = states.shape[1]
            next_states = model.decoder(
                decoder_inputs.reshape(-1, model.cfg.hidden_dim),
                states.reshape(-1, model.cfg.hidden_dim),
            ).reshape(batch_size, width, model.cfg.hidden_dim)
            logp = torch.log_softmax(model.heads[level](next_states), dim=-1)
            combined = scores[:, :, None] + logp
            next_width = min(beam_width, width * model.cfg.codebook_size)
            scores, selected = torch.topk(
                combined.reshape(batch_size, -1), next_width, dim=1
            )
            parent = selected // model.cfg.codebook_size
            token = selected % model.cfg.codebook_size
            gather_state = parent[:, :, None].expand(-1, -1, model.cfg.hidden_dim)
            states = torch.gather(next_states, 1, gather_state)
            if level:
                gather_prefix = parent[:, :, None].expand(-1, -1, level)
                prefixes = torch.gather(prefixes, 1, gather_prefix)
            else:
                prefixes = prefixes.expand(-1, next_width, -1)
            prefixes = torch.cat((prefixes, token[:, :, None]), dim=2)
            decoder_inputs = model.tokens[level](token)
    prefix_rows = prefixes.cpu().tolist()
    score_rows = scores.cpu().tolist()
    return [
        [(tuple(map(int, prefix)), float(score)) for prefix, score in zip(p, s)]
        for p, s in zip(prefix_rows, score_rows)
    ]


def _content_rerank(
    ranking: list[int],
    history: list[int],
    content_vectors: np.ndarray,
    present: np.ndarray,
    alpha: float,
) -> list[int]:
    if alpha <= 0 or not ranking:
        return list(ranking)
    usable = [author for author in history if present[author]]
    if not usable:
        return list(ranking)
    centroid = content_vectors[usable].mean(axis=0)
    centroid /= np.linalg.norm(centroid).clip(1e-8)
    candidates = np.asarray(ranking, dtype=np.int64)
    semantic = content_vectors[candidates] @ centroid
    base = np.linspace(1.0, 0.0, len(candidates), dtype=np.float32)
    score = (1.0 - alpha) * base + alpha * (semantic + 1.0) / 2.0
    return candidates[np.argsort(-score, kind="stable")].tolist()


def _load_bridge_vectors(checkpoint: str | Path, author_ids: np.ndarray) -> np.ndarray:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    author_map = {int(key): int(value) for key, value in payload["author_map"].items()}
    weights = payload["state_dict"]["author.weight"].cpu().numpy()
    vectors = np.stack([weights[author_map[int(author)]] for author in author_ids])
    return vectors.astype(np.float32)


def train_generative_retriever(
    processed_dir: str | Path,
    bridge_checkpoint: str | Path,
    content_path: str | Path,
    output_dir: str | Path,
    variant: str = "content",
    cfg: GeneratorConfig | None = None,
) -> dict[str, object]:
    if variant not in {"id", "content", "fusion"}:
        raise ValueError("variant must be id, content or fusion")
    cfg = cfg or GeneratorConfig()
    _seed(cfg.seed)
    root, out = Path(processed_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    live = pd.read_csv(root / "live.csv").sort_values("timestamp")
    authors = np.sort(live.author_id.unique().astype(np.int64))
    a_map = {int(author): index for index, author in enumerate(authors)}
    users = np.sort(live.user_id.unique().astype(np.int64))
    u_map = {int(user): index for index, user in enumerate(users)}
    live["u"] = live.user_id.map(u_map); live["a"] = live.author_id.map(a_map)
    bridge = _load_bridge_vectors(bridge_checkpoint, authors)
    bridge /= np.linalg.norm(bridge, axis=1, keepdims=True).clip(1e-8)
    content_data = np.load(content_path)
    if not np.array_equal(content_data["author_ids"], authors):
        raise ValueError("content author universe does not match processed candidates")
    present = content_data["content_present"].astype(bool)
    if variant == "content":
        representation = np.concatenate((
            bridge, content_data["content"], present[:, None].astype(np.float32)
        ), axis=1)
    else:
        # ID and late-fusion share exactly the same behavioural Semantic IDs.
        # This isolates the effect of content at ranking time instead of
        # confounding it with a changed discrete target space.
        representation = bridge
    codes, codebooks = residual_kmeans(
        representation, cfg.code_levels, cfg.codebook_size, cfg.seed
    )
    code_to_authors: dict[tuple[int, ...], list[int]] = defaultdict(list)
    popularity = live[live.split == "train"].a.value_counts().to_dict()
    for author, code in enumerate(codes):
        code_to_authors[tuple(map(int, code))].append(author)
    for values in code_to_authors.values():
        values.sort(key=lambda item: (-popularity.get(item, 0), item))
    train = live[live.split == "train"]
    histories = {
        int(uid): group.a.astype(int).tolist()
        for uid, group in train.groupby("u", sort=False)
    }
    example_count = sum(max(0, len(sequence) - 1) for sequence in histories.values())
    if not example_count:
        raise RuntimeError("no sequential training examples")
    # One contiguous uint8 tensor is substantially smaller than millions of
    # Python tuples containing separate int64 arrays. Codes are bounded by the
    # configured codebook size (currently 32), so uint8 is lossless.
    example_history = np.zeros(
        (example_count, cfg.max_length, cfg.code_levels), dtype=np.uint8
    )
    example_mask = np.ones((example_count, cfg.max_length), dtype=bool)
    example_target = np.zeros((example_count, cfg.code_levels), dtype=np.uint8)
    cursor = 0
    for sequence in histories.values():
        for index in range(1, len(sequence)):
            values = sequence[max(0, index - cfg.max_length):index]
            length = len(values)
            example_history[cursor, :length] = codes[np.asarray(values)]
            example_mask[cursor, :length] = False
            example_target[cursor] = codes[sequence[index]]
            cursor += 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SemanticIDGenerator(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
    rng, losses = np.random.default_rng(cfg.seed), []
    checkpoint_path = out / "training_checkpoint.pt"
    start_epoch = 0
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        losses = list(map(float, checkpoint["losses"]))
        rng.bit_generator.state = checkpoint["rng_state"]
    started_at = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        order = rng.permutation(example_count); total = 0.0
        model.train()
        for start in range(0, len(order), cfg.batch_size):
            rows = order[start:start + cfg.batch_size]
            history = torch.as_tensor(
                example_history[rows], dtype=torch.long, device=device
            )
            mask = torch.as_tensor(example_mask[rows], device=device)
            target = torch.as_tensor(
                example_target[rows], dtype=torch.long, device=device
            )
            logits = model.teacher_logits(history, mask, target)
            loss = sum(
                nn.functional.cross_entropy(level_logits, target[:, level])
                for level, level_logits in enumerate(logits)
            )
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.item()) * len(rows)
        losses.append(total / example_count)
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "losses": losses,
            "rng_state": rng.bit_generator.state,
            "config": asdict(cfg),
        }
        temporary_checkpoint = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary_checkpoint)
        temporary_checkpoint.replace(checkpoint_path)
        (out / "training_progress.json").write_text(json.dumps({
            "completed_epochs": epoch + 1,
            "total_epochs": cfg.epochs,
            "latest_loss": losses[-1],
            "elapsed_seconds_this_process": time.time() - started_at,
            "batch_size": cfg.batch_size,
            "checkpoint": str(checkpoint_path),
        }, indent=2))
    model.eval()
    valid_items = {
        int(uid): group.a.astype(int).tolist()
        for uid, group in live[live.split == "valid"].groupby("u")
    }
    popularity_order = sorted(range(len(authors)), key=lambda x: (-popularity.get(x, 0), x))
    base_ranking_cache: dict[bool, dict[int, list[int]]] = {}

    content_vectors = content_data["content"].astype(np.float32)

    def rankings(include_valid: bool, fusion_alpha: float = 0.0) -> dict[int, list[int]]:
        if include_valid not in base_ranking_cache:
            generated = {}
            target_users = list(map(int, live[
                live.split == ("test" if include_valid else "valid")
            ].u.unique()))
            inference_batch_size = 128
            for batch_start in range(0, len(target_users), inference_batch_size):
                batch_users = target_users[batch_start:batch_start + inference_batch_size]
                batch_histories = []
                for uid in batch_users:
                    history = list(histories.get(uid, []))
                    if include_valid:
                        history += valid_items.get(uid, [])
                    batch_histories.append(history)
                decoded = _beam_codes_batch(
                    model, batch_histories, codes, device, cfg.beam_width
                )
                for uid, history, beam in zip(batch_users, batch_histories, decoded):
                    seen = set(history)
                    ranking = []
                    ranking_set = set()
                    for code, _ in beam:
                        for author in code_to_authors.get(code, []):
                            if author not in seen and author not in ranking_set:
                                ranking.append(author)
                                ranking_set.add(author)
                    for author in popularity_order:
                        if len(ranking) >= 200:
                            break
                        if author not in seen and author not in ranking_set:
                            ranking.append(author)
                            ranking_set.add(author)
                    generated[uid] = ranking
                (out / f"{'test' if include_valid else 'valid'}_decode_progress.json").write_text(
                    json.dumps({
                        "completed_users": min(
                            batch_start + inference_batch_size, len(target_users)
                        ),
                        "total_users": len(target_users),
                        "batch_size": inference_batch_size,
                    }, indent=2)
                )
            base_ranking_cache[include_valid] = generated
        output = {}
        for uid, base_ranking in base_ranking_cache[include_valid].items():
            ranking = list(base_ranking)
            if variant == "fusion" and fusion_alpha > 0 and ranking:
                history = list(histories.get(uid, []))
                if include_valid:
                    history += valid_items.get(uid, [])
                ranking = _content_rerank(
                    ranking, history, content_vectors, present, fusion_alpha
                )
            output[uid] = ranking[:40]
        return output

    counts = train.groupby("u").size().astype(int).to_dict()
    truth = lambda split: {
        int(uid): set(group.a.astype(int))
        for uid, group in live[live.split == split].groupby("u")
    }
    fusion_sweep: dict[str, dict[str, float]] = {}
    selected_alpha = 0.0
    if variant == "fusion":
        for alpha in (0.0, 0.05, 0.10, 0.20, 0.35, 0.50):
            evaluated, _ = evaluate_rankings(
                truth("valid"), rankings(False, alpha), counts,
                candidate_count=len(authors),
            )
            fusion_sweep[str(alpha)] = evaluated["overall"]
        selected_alpha = max(
            map(float, fusion_sweep),
            key=lambda alpha: (
                fusion_sweep[str(alpha)]["recall@40"],
                fusion_sweep[str(alpha)]["ndcg@40"],
                -alpha,
            ),
        )
    valid_eval, valid_rows = evaluate_rankings(
        truth("valid"), rankings(False, selected_alpha), counts,
        candidate_count=len(authors),
    )
    test_eval, test_rows = evaluate_rankings(
        truth("test"), rankings(True, selected_alpha), counts,
        candidate_count=len(authors),
    )
    collision_sizes = np.asarray([len(values) for values in code_to_authors.values()])
    for include_valid, split_name in ((False, "valid"), (True, "test")):
        cached = base_ranking_cache[include_valid]
        cached_users = np.asarray(sorted(cached), dtype=np.int32)
        cache_width = min(
            200,
            max((len(cached[int(uid)]) for uid in cached_users), default=0),
        )
        cached_candidates = np.full(
            (len(cached_users), cache_width), -1, dtype=np.int32
        )
        cached_lengths = np.zeros(len(cached_users), dtype=np.int32)
        for row_index, uid in enumerate(cached_users):
            values = cached[int(uid)][:cache_width]
            cached_lengths[row_index] = len(values)
            cached_candidates[row_index, :len(values)] = values
        np.savez_compressed(
            out / f"{split_name}_base_candidates.npz",
            user_ids=cached_users,
            candidates=cached_candidates,
            lengths=cached_lengths,
        )
    result = {
        "model": "semantic_id_generative_retriever",
        "variant": variant,
        "device": str(device),
        "config": asdict(cfg),
        "valid": valid_eval["overall"], "test": test_eval["overall"],
        "valid_buckets": valid_eval["buckets"], "test_buckets": test_eval["buckets"],
        "candidate_authors": len(authors),
        "content_coverage": float(present.mean()),
        "unique_codes": len(code_to_authors),
        "collision_rate": float(1 - len(code_to_authors) / len(authors)),
        "max_collision": int(collision_sizes.max()),
        "training_examples": example_count,
        "training_storage": "contiguous_uint8",
        "selected_fusion_alpha": selected_alpha,
        "fusion_valid_sweep": fusion_sweep,
        "loss": losses,
        "audit": {
            "candidate_universe_reduced": False,
            "test_interactions_used_for_training": False,
            "fallback": "train-popularity only after generated codes",
            "dataset_precomputed_embeddings": True,
            "content_fusion": (
                "validation-selected cached author-centroid reranking"
                if variant == "fusion" else "none"
            ),
            "beam_generation_cached_across_fusion_sweep": variant == "fusion",
        },
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    valid_rows.to_csv(out / "valid_per_user_metrics.csv", index=False)
    test_rows.to_csv(out / "per_user_metrics.csv", index=False)
    pd.DataFrame(codes, columns=[f"code_{i}" for i in range(cfg.code_levels)]).assign(
        author_id=authors
    ).to_csv(out / "semantic_ids.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)}, out / "model.pt")
    np.savez_compressed(out / "codebooks.npz", **{
        f"level_{index}": value for index, value in enumerate(codebooks)
    })
    return result


def fuse_cached_generative_candidates(
    processed_dir: str | Path,
    content_path: str | Path,
    id_run_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Rerank cached ID-generator candidates without retraining or decoding."""
    root, id_run, out = Path(processed_dir), Path(id_run_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    live = pd.read_csv(root / "live.csv").sort_values("timestamp")
    authors = np.sort(live.author_id.unique().astype(np.int64))
    a_map = {int(author): index for index, author in enumerate(authors)}
    users = np.sort(live.user_id.unique().astype(np.int64))
    u_map = {int(user): index for index, user in enumerate(users)}
    live["u"] = live.user_id.map(u_map); live["a"] = live.author_id.map(a_map)
    content_data = np.load(content_path)
    if not np.array_equal(content_data["author_ids"], authors):
        raise ValueError("content author universe does not match processed candidates")
    content_vectors = content_data["content"].astype(np.float32)
    present = content_data["content_present"].astype(bool)
    train = live[live.split == "train"]
    histories = {
        int(uid): group.a.astype(int).tolist()
        for uid, group in train.groupby("u", sort=False)
    }
    valid_items = {
        int(uid): group.a.astype(int).tolist()
        for uid, group in live[live.split == "valid"].groupby("u")
    }
    counts = train.groupby("u").size().astype(int).to_dict()

    cache: dict[bool, dict[int, list[int]]] = {}
    for include_valid, split_name in ((False, "valid"), (True, "test")):
        payload = np.load(id_run / f"{split_name}_base_candidates.npz")
        cache[include_valid] = {
            int(uid): row[:int(length)].astype(int).tolist()
            for uid, row, length in zip(
                payload["user_ids"],
                payload["candidates"],
                payload["lengths"] if "lengths" in payload else
                np.full(len(payload["user_ids"]), payload["candidates"].shape[1]),
            )
        }

    def rankings(include_valid: bool, alpha: float) -> dict[int, list[int]]:
        output = {}
        for uid, candidates in cache[include_valid].items():
            history = list(histories.get(uid, []))
            if include_valid:
                history += valid_items.get(uid, [])
            output[uid] = _content_rerank(
                candidates, history, content_vectors, present, alpha
            )[:40]
        return output

    def truth(split: str) -> dict[int, set[int]]:
        return {
            int(uid): set(group.a.astype(int))
            for uid, group in live[live.split == split].groupby("u")
        }

    fusion_sweep: dict[str, dict[str, float]] = {}
    for alpha in (0.0, 0.05, 0.10, 0.20, 0.35, 0.50):
        evaluated, _ = evaluate_rankings(
            truth("valid"), rankings(False, alpha), counts,
            candidate_count=len(authors),
        )
        fusion_sweep[str(alpha)] = evaluated["overall"]
    selected_alpha = max(
        map(float, fusion_sweep),
        key=lambda alpha: (
            fusion_sweep[str(alpha)]["recall@40"],
            fusion_sweep[str(alpha)]["ndcg@40"],
            -alpha,
        ),
    )
    valid_eval, valid_rows = evaluate_rankings(
        truth("valid"), rankings(False, selected_alpha), counts,
        candidate_count=len(authors),
    )
    test_eval, test_rows = evaluate_rankings(
        truth("test"), rankings(True, selected_alpha), counts,
        candidate_count=len(authors),
    )
    id_metrics = json.loads((id_run / "metrics.json").read_text())
    result = {
        "model": "semantic_id_cached_content_fusion",
        "variant": "fusion_cached",
        "selected_fusion_alpha": selected_alpha,
        "fusion_valid_sweep": fusion_sweep,
        "valid": valid_eval["overall"], "test": test_eval["overall"],
        "valid_buckets": valid_eval["buckets"], "test_buckets": test_eval["buckets"],
        "candidate_authors": len(authors),
        "content_coverage": float(present.mean()),
        "unique_codes": id_metrics["unique_codes"],
        "collision_rate": id_metrics["collision_rate"],
        "max_collision": id_metrics["max_collision"],
        "audit": {
            "candidate_universe_reduced": False,
            "test_interactions_used_for_training": False,
            "fusion_weight_selected_on_valid_only": True,
            "generator_retrained": False,
            "beam_decoding_repeated": False,
            "reused_id_run": str(id_run),
            "dataset_precomputed_embeddings": True,
        },
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    valid_rows.to_csv(out / "valid_per_user_metrics.csv", index=False)
    test_rows.to_csv(out / "per_user_metrics.csv", index=False)
    (out / "model_reference.json").write_text(json.dumps({
        "model": str(id_run / "model.pt"),
        "semantic_ids": str(id_run / "semantic_ids.csv"),
        "valid_candidates": str(id_run / "valid_base_candidates.npz"),
        "test_candidates": str(id_run / "test_base_candidates.npz"),
    }, indent=2))
    return result
