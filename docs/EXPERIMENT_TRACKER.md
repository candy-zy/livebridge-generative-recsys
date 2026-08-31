# LiveBridge-GRPO 实验追踪

| Run | 阶段 | 配置 | 状态 | 产物/备注 |
|---|---|---|---|---|
| G000 | M0 | synthetic sanity | DONE | `tests/test_grpo.py`; local/server passed |
| G042 | M2 | constrained GRPO, seed 42 | DONE | `runs/grpo_suite/seed42/`; R10 0.1240 |
| G043 | M2 | constrained GRPO, seed 43 | DONE | `runs/grpo_suite/seed43/`; R10 0.0926 |
| G044 | M2 | constrained GRPO, seed 44 | DONE | `runs/grpo_suite/seed44/`; R10 0.1241 |
| A042-S | M3 | no source-affinity reward | DONE | `runs/grpo_ablation_seed42/no_source/` |
| A042-L | M3 | no long-tail reward | DONE | `runs/grpo_ablation_seed42/no_longtail/` |
| C042-44 | P1 | author-profile fusion, seeds 42/43/44 | DONE | `runs/content_suite/`; mean N10 0.1663 |
| GP042-44 | M3 | profile-aware GRPO, seeds 42/43/44 | DONE | `runs/grpo_profile_suite/`; trade-off extension |
