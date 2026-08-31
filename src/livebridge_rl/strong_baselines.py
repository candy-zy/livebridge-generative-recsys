"""In-project LightGCN, SASRec and EMCDR baselines with shared evaluation."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from livebridge_rl.evaluation import evaluate_full_sort


@dataclass
class StrongConfig:
    embedding_dim: int = 64
    epochs: int = 30
    batch_size: int = 512
    learning_rate: float = 2e-3
    layers: int = 3
    max_length: int = 50
    seed: int = 42


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load(processed_dir: str | Path):
    root = Path(processed_dir)
    live = pd.read_csv(root / "live.csv").sort_values("timestamp")
    photo = pd.read_csv(root / "photo_author.csv")
    users, authors = sorted(live.user_id.unique()), sorted(live.author_id.unique())
    u_map, a_map = {value: idx for idx, value in enumerate(users)}, {value: idx for idx, value in enumerate(authors)}
    live["u"], live["a"] = live.user_id.map(u_map), live.author_id.map(a_map)
    photo = photo[photo.user_id.isin(u_map)].copy()
    photo["u"] = photo.user_id.map(u_map)
    return live, photo, u_map, a_map


def _pairs(frame: pd.DataFrame) -> list[tuple[int, int]]:
    return list(zip(frame.u.astype(int), frame.a.astype(int)))


def _evaluate(
    model_name: str, live: pd.DataFrame, candidates: np.ndarray,
    valid_scorer, test_scorer, output_dir: Path, extra: dict[str, object],
) -> dict[str, object]:
    train, valid, test = (live[live.split == name] for name in ("train", "valid", "test"))
    counts = train.groupby("u").size().astype(int).to_dict()
    seen = {int(uid): set(group.a.astype(int)) for uid, group in train.groupby("u")}
    truth = lambda frame: {int(uid): set(group.a.astype(int)) for uid, group in frame.groupby("u")}
    valid_eval, valid_rows = evaluate_full_sort(truth(valid), seen, candidates, valid_scorer, counts)
    test_seen = {uid: set(items) for uid, items in seen.items()}
    for uid, group in valid.groupby("u"):
        test_seen.setdefault(int(uid), set()).update(group.a.astype(int))
    test_eval, test_rows = evaluate_full_sort(truth(test), test_seen, candidates, test_scorer, counts)
    result = {
        "model": model_name,
        "valid": valid_eval["overall"], "test": test_eval["overall"],
        "valid_buckets": valid_eval["buckets"], "test_buckets": test_eval["buckets"],
        **extra,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    valid_rows.to_csv(output_dir / "valid_per_user_metrics.csv", index=False)
    test_rows.to_csv(output_dir / "per_user_metrics.csv", index=False)
    print(json.dumps(result, indent=2), flush=True)
    return result


class LightGCN(nn.Module):
    def __init__(
        self,
        users: int,
        items: int,
        dim: int,
        layers: int,
        edge_users: np.ndarray,
        edge_items: np.ndarray,
        device: torch.device,
    ):
        super().__init__()
        self.users, self.items, self.layers = users, items, layers
        self.user = nn.Embedding(users, dim)
        self.item = nn.Embedding(items, dim)
        nn.init.normal_(self.user.weight, std=0.05)
        nn.init.normal_(self.item.weight, std=0.05)
        edge_u = torch.as_tensor(edge_users, dtype=torch.long, device=device)
        edge_i = torch.as_tensor(edge_items, dtype=torch.long, device=device) + users
        indices = torch.stack((
            torch.cat((edge_u, edge_i)),
            torch.cat((edge_i, edge_u)),
        ))
        degree = torch.bincount(indices[0], minlength=users + items).float().clamp_min(1)
        values = (degree[indices[0]] * degree[indices[1]]).rsqrt()
        self.register_buffer("adj_indices", indices)
        self.register_buffer("adj_values", values)

    def propagated(self):
        adjacency = torch.sparse_coo_tensor(
            self.adj_indices, self.adj_values, (self.users + self.items,) * 2
        ).coalesce()
        state = torch.cat((self.user.weight, self.item.weight))
        layers = [state]
        for _ in range(self.layers):
            state = torch.sparse.mm(adjacency, state)
            layers.append(state)
        final = torch.stack(layers).mean(0)
        return final[:self.users], final[self.users:]


def train_lightgcn(processed_dir: str | Path, output_dir: str | Path, cfg: StrongConfig):
    _seed(cfg.seed)
    live, _, u_map, a_map = _load(processed_dir)
    train = live[live.split == "train"]
    pair_users = train.u.to_numpy(dtype=np.int64, copy=True)
    pair_items = train.a.to_numpy(dtype=np.int64, copy=True)
    num_pairs = len(pair_users)
    if num_pairs == 0:
        raise RuntimeError("no LightGCN training interactions after preprocessing")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightGCN(
        len(u_map), len(a_map), cfg.embedding_dim, cfg.layers,
        pair_users, pair_items, device,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    rng, losses = np.random.default_rng(cfg.seed), []
    for epoch in range(cfg.epochs):
        # Propagating the complete graph once per mini-batch makes the cost
        # quadratic in the number of interactions.  Compute the propagated
        # embeddings once, accumulate the exact BPR gradient on detached leaf
        # tensors in mini-batches, then backpropagate that accumulated gradient
        # through the graph convolution once.  This is one full-batch optimizer
        # step per epoch and preserves gradients through every LightGCN layer.
        permutation = rng.permutation(num_pairs)
        optimizer.zero_grad()
        ue, ie = model.propagated()
        ue_leaf = ue.detach().requires_grad_(True)
        ie_leaf = ie.detach().requires_grad_(True)
        total = 0.0
        for start in range(0, num_pairs, cfg.batch_size):
            indices = permutation[start:start + cfg.batch_size]
            users = torch.as_tensor(pair_users[indices], dtype=torch.long, device=device)
            positive = torch.as_tensor(pair_items[indices], dtype=torch.long, device=device)
            negative = torch.randint(0, len(a_map), (len(indices),), device=device)
            batch_loss = -torch.nn.functional.logsigmoid(
                (ue_leaf[users] * ie_leaf[positive]).sum(-1)
                - (ue_leaf[users] * ie_leaf[negative]).sum(-1)
            ).sum()
            (batch_loss / num_pairs).backward()
            total += batch_loss.item()
        torch.autograd.backward((ue, ie), (ue_leaf.grad, ie_leaf.grad))
        optimizer.step()
        losses.append(total / num_pairs)
        print(f"lightgcn epoch={epoch+1:03d} loss={losses[-1]:.6f}", flush=True)
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model.propagated()
    def scorer(uid: int, items: np.ndarray) -> np.ndarray:
        # Keep full-sort dot products on the GPU.  Copying the complete item
        # table to NumPy makes full-data evaluation memory-bandwidth bound on
        # the CPU and repeats that scan once per user.
        with torch.no_grad():
            item_ids = torch.as_tensor(items, dtype=torch.long, device=device)
            return (item_emb[item_ids] @ user_emb[uid]).cpu().numpy()
    out = Path(output_dir)
    result = _evaluate("lightgcn", live, np.arange(len(a_map)), scorer, scorer, out,
                       {"device": str(device), "config": asdict(cfg), "loss": losses,
                        "optimizer_schedule": "one_full_graph_step_per_epoch"})
    torch.save(model.state_dict(), out / "model.pt")
    return result


class SASRec(nn.Module):
    def __init__(self, items: int, cfg: StrongConfig):
        super().__init__()
        self.item = nn.Embedding(items + 1, cfg.embedding_dim, padding_idx=0)
        self.position = nn.Embedding(cfg.max_length, cfg.embedding_dim)
        layer = nn.TransformerEncoderLayer(
            cfg.embedding_dim, nhead=4, dim_feedforward=cfg.embedding_dim * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(cfg.embedding_dim)
        nn.init.normal_(self.item.weight, std=0.02)
        nn.init.normal_(self.position.weight, std=0.02)
        with torch.no_grad():
            self.item.weight[0].zero_()

    def encode(self, sequences: torch.Tensor) -> torch.Tensor:
        length = sequences.shape[1]
        positions = torch.arange(length, device=sequences.device).unsqueeze(0)
        state = self.item(sequences) + self.position(positions)
        causal = torch.triu(torch.ones(length, length, device=sequences.device, dtype=torch.bool), diagonal=1)
        state = self.encoder(state, mask=causal, src_key_padding_mask=sequences.eq(0))
        last = sequences.ne(0).sum(1).sub(1).clamp_min(0)
        return self.norm(state[torch.arange(len(sequences), device=sequences.device), last])


def _pad_sequence(sequence: list[int], max_length: int) -> list[int]:
    values = [item + 1 for item in sequence[-max_length:]]
    return values + [0] * (max_length - len(values))


def train_sasrec(processed_dir: str | Path, output_dir: str | Path, cfg: StrongConfig):
    _seed(cfg.seed)
    live, _, u_map, a_map = _load(processed_dir)
    train = live[live.split == "train"].sort_values("timestamp")
    histories = {int(uid): group.a.astype(int).tolist() for uid, group in train.groupby("u")}
    examples: list[tuple[list[int], int]] = []
    for sequence in histories.values():
        for index in range(1, len(sequence)):
            examples.append((_pad_sequence(sequence[:index], cfg.max_length), sequence[index]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SASRec(len(a_map), cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
    rng, losses = np.random.default_rng(cfg.seed), []
    for epoch in range(cfg.epochs):
        rng.shuffle(examples); total = 0.0
        model.train()
        for start in range(0, len(examples), cfg.batch_size):
            batch = examples[start:start + cfg.batch_size]
            sequences = torch.tensor([x[0] for x in batch], device=device)
            positive = torch.tensor([x[1] + 1 for x in batch], device=device)
            negative = torch.randint(1, len(a_map) + 1, (len(batch),), device=device)
            state = model.encode(sequences)
            loss = -torch.nn.functional.logsigmoid(
                (state * model.item(positive)).sum(-1) - (state * model.item(negative)).sum(-1)
            ).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * len(batch)
        losses.append(total / max(1, len(examples)))
        print(f"sasrec epoch={epoch+1:03d} loss={losses[-1]:.6f}", flush=True)
    valid_items = {int(uid): group.a.astype(int).tolist() for uid, group in live[live.split == "valid"].groupby("u")}
    model.eval()
    item_emb = model.item.weight[1:].detach()
    def make_scorer(include_valid: bool):
        def scorer(uid: int, items: np.ndarray) -> np.ndarray:
            sequence = list(histories.get(uid, []))
            if include_valid: sequence += valid_items.get(uid, [])
            tensor = torch.tensor([_pad_sequence(sequence, cfg.max_length)], device=device)
            with torch.no_grad():
                item_ids = torch.as_tensor(items, device=device)
                return (item_emb[item_ids] @ model.encode(tensor)[0]).cpu().numpy()
        return scorer
    out = Path(output_dir)
    result = _evaluate("sasrec", live, np.arange(len(a_map)), make_scorer(False), make_scorer(True), out,
                       {"device": str(device), "config": asdict(cfg), "loss": losses})
    torch.save(model.state_dict(), out / "model.pt")
    return result


class MF(nn.Module):
    def __init__(self, users: int, items: int, dim: int):
        super().__init__()
        self.user, self.item = nn.Embedding(users, dim), nn.Embedding(items, dim)
        nn.init.normal_(self.user.weight, std=0.05); nn.init.normal_(self.item.weight, std=0.05)


def _fit_mf(model: MF, pairs: list[tuple[int, int]], items: int, cfg: StrongConfig, device) -> list[float]:
    optimizer, rng, losses = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate), np.random.default_rng(cfg.seed), []
    for _ in range(cfg.epochs):
        rng.shuffle(pairs); total = 0.0
        for start in range(0, len(pairs), cfg.batch_size):
            batch = pairs[start:start + cfg.batch_size]
            users = torch.tensor([x[0] for x in batch], device=device)
            pos = torch.tensor([x[1] for x in batch], device=device)
            neg = torch.randint(0, items, (len(batch),), device=device)
            loss = -torch.nn.functional.logsigmoid(
                (model.user(users) * model.item(pos)).sum(-1) - (model.user(users) * model.item(neg)).sum(-1)
            ).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step(); total += loss.item() * len(batch)
        losses.append(total / len(pairs))
    return losses


def train_emcdr(processed_dir: str | Path, output_dir: str | Path, cfg: StrongConfig):
    _seed(cfg.seed)
    live, photo, u_map, target_map = _load(processed_dir)
    source_authors = sorted(photo.author_id.unique())
    source_map = {value: idx for idx, value in enumerate(source_authors)}
    photo["a"] = photo.author_id.map(source_map)
    target_pairs, source_pairs = _pairs(live[live.split == "train"]), _pairs(photo)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = MF(len(u_map), len(target_map), cfg.embedding_dim).to(device)
    source = MF(len(u_map), len(source_map), cfg.embedding_dim).to(device)
    target_loss = _fit_mf(target, target_pairs, len(target_map), cfg, device)
    source_loss = _fit_mf(source, source_pairs, len(source_map), cfg, device)
    mapper = nn.Linear(cfg.embedding_dim, cfg.embedding_dim).to(device)
    optimizer = torch.optim.Adam(mapper.parameters(), lr=cfg.learning_rate)
    source_users, target_users = source.user.weight.detach(), target.user.weight.detach()
    map_loss = []
    for _ in range(300):
        loss = torch.nn.functional.mse_loss(mapper(source_users), target_users)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); map_loss.append(loss.item())
    mapped = mapper(source_users).detach().cpu().numpy()
    target_items = target.item.weight.detach().cpu().numpy()
    scorer = lambda uid, items: target_items[items] @ mapped[uid]
    out = Path(output_dir)
    result = _evaluate("emcdr", live, np.arange(len(target_map)), scorer, scorer, out, {
        "device": str(device), "config": asdict(cfg), "target_loss": target_loss,
        "source_loss": source_loss, "mapping_loss": map_loss,
    })
    torch.save({"source": source.state_dict(), "target": target.state_dict(), "mapper": mapper.state_dict()}, out / "model.pt")
    return result


def train_strong(model: str, processed_dir: str | Path, output_dir: str | Path, cfg: StrongConfig):
    if model == "lightgcn": return train_lightgcn(processed_dir, output_dir, cfg)
    if model == "sasrec": return train_sasrec(processed_dir, output_dir, cfg)
    if model == "emcdr": return train_emcdr(processed_dir, output_dir, cfg)
    raise ValueError(f"unknown model: {model}")
