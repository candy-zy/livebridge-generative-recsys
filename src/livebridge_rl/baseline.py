"""GPU BPR baselines for target-only and creator-bridge transfer."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from livebridge_rl.evaluation import evaluate_full_sort


@dataclass
class TrainConfig:
    embedding_dim: int = 64
    epochs: int = 30
    batch_size: int = 4096
    learning_rate: float = 2e-3
    source_weight: float = 0.35
    seed: int = 42


class BridgeBPR(nn.Module):
    def __init__(self, users: int, authors: int, dim: int):
        super().__init__()
        self.user = nn.Embedding(users, dim)
        self.author = nn.Embedding(authors, dim)
        self.user_bias = nn.Embedding(users, 1)
        self.author_bias = nn.Embedding(authors, 1)
        for emb in (self.user, self.author):
            nn.init.normal_(emb.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.author_bias.weight)

    def score(self, users: torch.Tensor, authors: torch.Tensor) -> torch.Tensor:
        return (self.user(users) * self.author(authors)).sum(-1) + self.author_bias(authors).squeeze(-1)


def _seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _metrics(model: BridgeBPR, split: pd.DataFrame, seen: dict[int, set[int]],
             num_authors: int, device: torch.device, ks=(10, 20, 40)) -> dict[str, float]:
    totals = {f"recall@{k}": 0.0 for k in ks} | {f"ndcg@{k}": 0.0 for k in ks}
    evaluated = 0
    model.eval()
    with torch.no_grad():
        all_items = torch.arange(num_authors, device=device)
        for uid, group in split.groupby("u"):
            truth = set(group["a"].astype(int))
            if not truth: continue
            users = torch.full((num_authors,), int(uid), device=device)
            scores = model.score(users, all_items)
            blocked = seen.get(int(uid), set()) - truth
            if blocked:
                scores[torch.tensor(list(blocked), device=device)] = -torch.inf
            ranking = torch.topk(scores, min(max(ks), num_authors)).indices.cpu().tolist()
            for k in ks:
                hits = [1 if item in truth else 0 for item in ranking[:k]]
                totals[f"recall@{k}"] += sum(hits) / len(truth)
                dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
                ideal = sum(1 / math.log2(i + 2) for i in range(min(len(truth), k)))
                totals[f"ndcg@{k}"] += dcg / ideal if ideal else 0
            evaluated += 1
    return {key: value / max(1, evaluated) for key, value in totals.items()} | {"users": evaluated}


def train_baseline(processed_dir: str | Path, output_dir: str | Path,
                   mode: str = "bridge", config: TrainConfig | None = None) -> dict[str, object]:
    cfg = config or TrainConfig()
    if mode not in {"target", "bridge"}: raise ValueError("mode must be target or bridge")
    _seed_everything(cfg.seed)
    root, out = Path(processed_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    live, photo = pd.read_csv(root / "live.csv"), pd.read_csv(root / "photo_author.csv")
    users = sorted(set(live.user_id) | set(photo.user_id))
    # Candidate items must remain target-domain live authors. Source-only creators
    # are transfer evidence, not valid live recommendation candidates.
    authors = sorted(set(live.author_id))
    u_map, a_map = {v:i for i,v in enumerate(users)}, {v:i for i,v in enumerate(authors)}
    live["u"], live["a"] = live.user_id.map(u_map), live.author_id.map(a_map)
    photo = photo[photo.author_id.isin(a_map)]
    photo["u"], photo["a"] = photo.user_id.map(u_map), photo.author_id.map(a_map)
    train = live[live.split == "train"]
    positives = list(zip(train.u.astype(int), train.a.astype(int)))
    if mode == "bridge":
        repeat = np.maximum(1, np.ceil(np.log1p(photo.engagement) * cfg.source_weight).astype(int))
        source = list(zip(np.repeat(photo.u.astype(int), repeat), np.repeat(photo.a.astype(int), repeat)))
        positives += source
    if not positives: raise RuntimeError("no training interactions after preprocessing")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BridgeBPR(len(users), len(authors), cfg.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
    rng = np.random.default_rng(cfg.seed)
    history = []
    for epoch in range(cfg.epochs):
        rng.shuffle(positives)
        total = 0.0
        for start in range(0, len(positives), cfg.batch_size):
            batch = positives[start:start + cfg.batch_size]
            u = torch.tensor([x[0] for x in batch], device=device)
            pos = torch.tensor([x[1] for x in batch], device=device)
            neg = torch.randint(0, len(authors), (len(batch),), device=device)
            loss = -torch.nn.functional.logsigmoid(model.score(u, pos) - model.score(u, neg)).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * len(batch)
        history.append(total / len(positives))
        print(f"epoch={epoch+1:03d} loss={history[-1]:.6f}", flush=True)
    seen = {int(uid): set(g.a.astype(int)) for uid, g in train.groupby("u")}
    train_counts = train.groupby("u").size().astype(int).to_dict()
    candidates = np.arange(len(authors), dtype=np.int64)
    def truth(frame: pd.DataFrame) -> dict[int, set[int]]:
        return {int(uid): set(group.a.astype(int)) for uid, group in frame.groupby("u")}
    def scorer(uid: int, item_ids: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            item_tensor = torch.as_tensor(item_ids, device=device)
            user_tensor = torch.full(item_tensor.shape, uid, device=device)
            return model.score(user_tensor, item_tensor).detach().cpu().numpy()
    valid_eval, valid_rows = evaluate_full_sort(
        truth(live[live.split == "valid"]), seen, candidates, scorer, train_counts
    )
    test_seen = {k:set(v) for k,v in seen.items()}
    for uid, g in live[live.split == "valid"].groupby("u"):
        test_seen.setdefault(int(uid), set()).update(g.a.astype(int))
    test_eval, test_rows = evaluate_full_sort(
        truth(live[live.split == "test"]), test_seen, candidates, scorer, train_counts
    )
    result = {"mode": mode, "device": str(device), "config": asdict(cfg),
              "valid": valid_eval["overall"], "test": test_eval["overall"],
              "valid_buckets": valid_eval["buckets"], "test_buckets": test_eval["buckets"],
              "train_interactions": len(positives), "users": len(users), "authors": len(authors), "loss": history}
    torch.save({"state_dict": model.state_dict(), "user_map": u_map, "author_map": a_map}, out / "model.pt")
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    valid_rows.to_csv(out / "valid_per_user_metrics.csv", index=False)
    test_rows.to_csv(out / "per_user_metrics.csv", index=False)
    print(json.dumps(result, indent=2), flush=True)
    return result
