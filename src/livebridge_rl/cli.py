"""Command-line utilities for LiveBridge-RL."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from livebridge_rl.data_contract import validate_data_dir
from livebridge_rl.preprocess import prepare_sample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="livebridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate-data", help="validate a local KuaiLive-M3 directory"
    )
    validate.add_argument("--data-dir", required=True)
    prepare = subparsers.add_parser("prepare", help="stream a deterministic KuaiLive-M3 user sample")
    prepare.add_argument("--data-dir", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--sample-ratio", type=float, default=0.01)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--source-mode", choices=("aggregate", "temporal"), default="aggregate")
    train = subparsers.add_parser("train", help="train and evaluate a GPU BPR baseline")
    train.add_argument("--processed-dir", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--mode", choices=("target", "bridge"), default="bridge")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--seed", type=int, default=42)
    popularity = subparsers.add_parser("popularity", help="run the target popularity baseline")
    popularity.add_argument("--processed-dir", required=True)
    popularity.add_argument("--output-dir", required=True)
    strong = subparsers.add_parser("strong-train", help="train LightGCN, SASRec or EMCDR")
    strong.add_argument("--model", choices=("lightgcn", "sasrec", "emcdr"), required=True)
    strong.add_argument("--processed-dir", required=True)
    strong.add_argument("--output-dir", required=True)
    strong.add_argument("--epochs", type=int, default=30)
    strong.add_argument("--seed", type=int, default=42)
    strong.add_argument("--batch-size", type=int, default=4096)
    strong.add_argument("--embedding-dim", type=int, default=64)
    strong.add_argument("--layers", type=int, default=3)
    strong.add_argument("--max-length", type=int, default=50)
    strong.add_argument("--learning-rate", type=float, default=2e-3)
    grpo = subparsers.add_parser("grpo-train", help="train a constrained GRPO reranker over Creator-Bridge")
    grpo.add_argument("--processed-dir", required=True)
    grpo.add_argument("--bridge-checkpoint", required=True)
    grpo.add_argument("--author-profile")
    grpo.add_argument("--output-dir", required=True)
    grpo.add_argument("--epochs", type=int, default=30)
    grpo.add_argument("--seed", type=int, default=42)
    grpo.add_argument("--candidate-pool", type=int, default=50)
    grpo.add_argument("--slate-size", type=int, default=10)
    grpo.add_argument("--group-size", type=int, default=8)
    grpo.add_argument("--learning-rate", type=float, default=1e-2)
    grpo.add_argument("--residual-scale", type=float, default=0.35)
    grpo.add_argument("--source-weight", type=float, default=0.10)
    grpo.add_argument("--profile-weight", type=float, default=0.05)
    grpo.add_argument("--longtail-weight", type=float, default=0.05)
    grpo.add_argument("--score-batch-size", type=int, default=128)
    content = subparsers.add_parser("content-eval", help="evaluate leakage-safe author-profile content fusion")
    content.add_argument("--processed-dir", required=True)
    content.add_argument("--author-profile", required=True)
    content.add_argument("--bridge-checkpoint", required=True)
    content.add_argument("--output-dir", required=True)
    align = subparsers.add_parser(
        "content-align", help="aggregate and align official photo/live embeddings"
    )
    align.add_argument("--data-dir", required=True)
    align.add_argument("--processed-dir", required=True)
    align.add_argument("--output-dir", required=True)
    align.add_argument("--epochs", type=int, default=100)
    align.add_argument("--seed", type=int, default=42)
    align.add_argument("--batch-size", type=int, default=512)
    generate = subparsers.add_parser(
        "generative-train", help="train Semantic-ID autoregressive retrieval"
    )
    generate.add_argument("--processed-dir", required=True)
    generate.add_argument("--bridge-checkpoint", required=True)
    generate.add_argument("--content-path", required=True)
    generate.add_argument("--output-dir", required=True)
    generate.add_argument(
        "--variant", choices=("id", "content", "fusion"), default="content"
    )
    generate.add_argument("--epochs", type=int, default=50)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--batch-size", type=int, default=512)
    generate.add_argument("--hidden-dim", type=int, default=64)
    generate.add_argument("--codebook-size", type=int, default=32)
    generate.add_argument("--code-levels", type=int, default=3)
    generate.add_argument("--max-length", type=int, default=50)
    generate.add_argument("--beam-width", type=int, default=100)
    fuse_cache = subparsers.add_parser(
        "generative-fuse-cache",
        help="apply content fusion to cached generative candidates without retraining",
    )
    fuse_cache.add_argument("--processed-dir", required=True)
    fuse_cache.add_argument("--content-path", required=True)
    fuse_cache.add_argument("--id-run-dir", required=True)
    fuse_cache.add_argument("--output-dir", required=True)
    agent_cache = subparsers.add_parser("agent-cache", help="build reusable session-agent candidate caches")
    agent_cache.add_argument("--processed-dir", required=True)
    agent_cache.add_argument("--bridge-checkpoint", required=True)
    agent_cache.add_argument("--output-dir", required=True)
    agent_cache.add_argument("--author-profile")
    agent_cache.add_argument("--candidate-pool", type=int, default=100)
    agent_cache.add_argument("--score-batch-size", type=int, default=128)
    agent_cache.add_argument("--max-steps", type=int, default=8)
    agent_cache.add_argument("--independent-tool-pools", action="store_true")
    agent_train = subparsers.add_parser("agent-train", help="train or evaluate the session tool-routing agent")
    agent_train.add_argument("--cache-dir", required=True)
    agent_train.add_argument("--output-dir", required=True)
    agent_train.add_argument(
        "--variant", choices=("fixed", "contextual", "agentic", "no_memory", "no_routing", "myopic"),
        default="agentic",
    )
    agent_train.add_argument("--epochs", type=int, default=30)
    agent_train.add_argument("--seed", type=int, default=42)
    agent_train.add_argument("--learning-rate", type=float, default=3e-3)
    agent_train.add_argument("--hidden-dim", type=int, default=48)
    agent_train.add_argument("--group-size", type=int, default=4)
    agent_train.add_argument("--slate-size", type=int, default=10)
    agent_train.add_argument("--max-steps", type=int, default=8)
    agent_train.add_argument("--users-per-epoch", type=int, default=2048)
    agent_train.add_argument("--warmup-epochs", type=int, default=2)
    agent_train.add_argument(
        "--warmup-strategy",
        choices=("causal_fallback", "trajectory_oracle", "counterfactual"),
        default="causal_fallback",
    )
    agent_train.add_argument("--counterfactual-temperature", type=float, default=0.05)
    agent_train.add_argument("--selection-fraction", type=float, default=0.20)
    agent_train.add_argument("--entropy-coef", type=float, default=1e-2)
    agent_train.add_argument("--kl-beta", type=float, default=2e-2)
    agent_train.add_argument("--trajectory-gamma", type=float, default=0.95)
    agent_train.add_argument("--source-weight", type=float, default=0.05)
    agent_train.add_argument("--longtail-weight", type=float, default=0.05)
    agent_train.add_argument("--fixed-action", type=int, choices=range(5), default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-data":
        try:
            report = validate_data_dir(args.data_dir)
        except NotADirectoryError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
            return 2
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 0 if report.valid_minimal_cdr else 1
    if args.command == "prepare":
        report = prepare_sample(args.data_dir, args.output_dir, args.sample_ratio, args.seed, source_mode=args.source_mode)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "train":
        from livebridge_rl.baseline import TrainConfig, train_baseline
        train_baseline(
            args.processed_dir, args.output_dir, args.mode,
            TrainConfig(epochs=args.epochs, seed=args.seed),
        )
        return 0
    if args.command == "popularity":
        from livebridge_rl.popularity import run_popularity
        run_popularity(args.processed_dir, args.output_dir)
        return 0
    if args.command == "strong-train":
        from livebridge_rl.strong_baselines import StrongConfig, train_strong
        train_strong(
            args.model, args.processed_dir, args.output_dir,
            StrongConfig(
                embedding_dim=args.embedding_dim, epochs=args.epochs,
                batch_size=args.batch_size, layers=args.layers,
                max_length=args.max_length, learning_rate=args.learning_rate,
                seed=args.seed,
            ),
        )
        return 0
    if args.command == "grpo-train":
        from livebridge_rl.grpo_reranker import GRPOConfig, train_grpo_reranker
        train_grpo_reranker(
            args.processed_dir, args.bridge_checkpoint, args.output_dir,
            GRPOConfig(
                epochs=args.epochs, seed=args.seed, candidate_pool=args.candidate_pool,
                slate_size=args.slate_size, group_size=args.group_size,
                learning_rate=args.learning_rate, residual_scale=args.residual_scale,
                source_weight=args.source_weight, longtail_weight=args.longtail_weight,
                profile_weight=args.profile_weight,
                score_batch_size=args.score_batch_size,
            ),
            author_profile_path=args.author_profile,
        )
        return 0
    if args.command == "content-eval":
        from livebridge_rl.content_fusion import run_profile_fusion
        run_profile_fusion(
            args.processed_dir, args.author_profile,
            args.bridge_checkpoint, args.output_dir,
        )
        return 0
    if args.command == "content-align":
        from livebridge_rl.generative_retrieval import (
            AlignmentConfig, aggregate_author_content, train_content_alignment,
        )
        aggregate_author_content(
            args.data_dir, args.processed_dir, args.output_dir
        )
        train_content_alignment(
            Path(args.output_dir) / "author_content_raw.npz", args.output_dir,
            AlignmentConfig(
                epochs=args.epochs, batch_size=args.batch_size, seed=args.seed
            ),
        )
        return 0
    if args.command == "generative-train":
        from livebridge_rl.generative_retrieval import (
            GeneratorConfig, train_generative_retriever,
        )
        train_generative_retriever(
            args.processed_dir, args.bridge_checkpoint, args.content_path,
            args.output_dir, args.variant,
            GeneratorConfig(
                hidden_dim=args.hidden_dim, codebook_size=args.codebook_size,
                code_levels=args.code_levels, epochs=args.epochs,
                batch_size=args.batch_size, max_length=args.max_length,
                beam_width=args.beam_width, seed=args.seed,
            ),
        )
        return 0
    if args.command == "generative-fuse-cache":
        from livebridge_rl.generative_retrieval import fuse_cached_generative_candidates
        fuse_cached_generative_candidates(
            args.processed_dir, args.content_path,
            args.id_run_dir, args.output_dir,
        )
        return 0
    if args.command == "agent-cache":
        from livebridge_rl.agentic_rl import build_agent_cache
        build_agent_cache(
            args.processed_dir, args.bridge_checkpoint, args.output_dir,
            author_profile_path=args.author_profile,
            candidate_pool=args.candidate_pool,
            score_batch_size=args.score_batch_size,
            max_steps=args.max_steps,
            independent_tool_pools=args.independent_tool_pools,
        )
        return 0
    if args.command == "agent-train":
        from livebridge_rl.agentic_rl import AgentConfig, load_agent_sessions, train_session_agent
        cache = Path(args.cache_dir)
        train_session_agent(
            load_agent_sessions(cache / "valid_sessions.npz"),
            load_agent_sessions(cache / "test_sessions.npz"),
            args.output_dir,
            AgentConfig(
                epochs=args.epochs, seed=args.seed, learning_rate=args.learning_rate,
                hidden_dim=args.hidden_dim, group_size=args.group_size,
                slate_size=args.slate_size, max_steps=args.max_steps,
                users_per_epoch=args.users_per_epoch,
                warmup_epochs=args.warmup_epochs, entropy_coef=args.entropy_coef,
                warmup_strategy=args.warmup_strategy,
                counterfactual_temperature=args.counterfactual_temperature,
                selection_fraction=args.selection_fraction,
                fixed_action=args.fixed_action,
                kl_beta=args.kl_beta, trajectory_gamma=args.trajectory_gamma,
                source_weight=args.source_weight, longtail_weight=args.longtail_weight,
            ),
            variant=args.variant,
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
