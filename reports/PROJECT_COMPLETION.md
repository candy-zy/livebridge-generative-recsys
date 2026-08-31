# 项目完成验收

验收日期：2026-08-30

## 交付状态

| 模块 | 状态 | 可核验产物 |
|---|---|---|
| KuaiLive-M3 数据契约与校验 | DONE | `data_contract.py`、`docs/DATA_CONTRACT.md` |
| 1% 流式预处理与逐用户时间门禁 | DONE | `preprocess.py`、3 个 processed seeds |
| Popularity/BPR/LightGCN/SASRec/EMCDR | DONE | `runs/strong_suite_seed{42,43,44}` |
| Creator-Bridge 跨域召回 | DONE | `reports/STRONG_BASELINES.md` |
| SASRec 初始化修复与调参 | DONE | `runs/sasrec_tune/`、`runs/sasrec_tuned/` |
| 作者内容画像融合 | DONE | `runs/content_suite/`、`reports/CONTENT_RESULTS.md` |
| 受约束长程 GRPO | DONE | `runs/grpo_suite/`、`reports/GRPO_RESULTS.md` |
| GRPO reward 消融 | DONE | `runs/grpo_ablation_seed42/` |
| Profile-aware GRPO 扩展 | DONE | `runs/grpo_profile_suite/` |
| Session Agentic Router V1/V2 | DONE | V1 失败证据在 `runs/agentic_10pct/`；V2 三种子与消融在 `runs/agentic_dense_v2/` |
| Agent 分层服务与故障注入 | DONE | `runs/agentic_10pct/serving_benchmark.json` |
| Semantic-ID 生成式召回 | DONE | 1% 三种子、10% seed42、全量 seed42、失败消融与修复见 `reports/GENERATIVE_RETRIEVAL_RESULTS.md` |
| 跨模态内容对齐 | DONE | 官方预计算 128D/64D 向量、author-disjoint InfoNCE 与内容缓存 |
| Dashboard | DONE | `dashboard/`；build/lint 通过 |
| 简历与面试边界 | DONE | `docs/RESUME_BULLETS.md` |

## 最终主结果

- 最强非项目基线 LightGCN：Recall@10 `0.0391±0.0125`，NDCG@10 `0.0611±0.0036`。
- Creator-Bridge：Recall@10 `0.1023±0.0168`，NDCG@10 `0.1640±0.0081`。
- Creator-Bridge + constrained GRPO：Recall@10 `0.1136±0.0182`，NDCG@10 `0.1717±0.0095`。
- GRPO 相比 Creator-Bridge：Recall@10 相对 `+11.0%`，NDCG@10 相对 `+4.7%`；三个 seed 方向一致。
- Long-tail Share@10：`0.3093 → 0.5411`；Catalog Coverage@10：`0.0617 → 0.0656`。
- Semantic-ID generator + content late fusion：1% 三种子 Recall@40 `0.1014±0.0296 → 0.1362±0.0363`，NDCG@40 `0.0781±0.0190 → 0.1144±0.0188`；paired bootstrap 95% CI 均不跨 0。
- 10% seed42（123,946 candidates / 2,169 test users）：Recall@40 `0.0325 → 0.0599`，NDCG@40 `0.0336 → 0.0564`。
- 全量 seed42（600,935 candidates / 21,825 test users）：Recall@40 `0.03587 → 0.04300`（`+19.9%`），NDCG@40 `0.04377 → 0.05231`（`+19.5%`）；完整训练、缓存解码与 Gate 退出码均为 0。

## 质量门禁

- Python：`17 passed`（2 个 PyTorch nested-tensor 非阻塞 warning）。
- Dashboard：production build 与 ESLint 通过。
- 所有主实验都有 seed/config/metrics JSON 和逐用户 CSV。
- test split 不参与模型或策略训练；GRPO 只使用 validation logged positives。
- 项目不保存 SSH 密码、API token 或原始公开数据。
- Agentic Router V1 失败结果被完整保留；V2 的 10% 三种子仅报告为小幅方向性 Pilot，Bootstrap CI 跨 0，不包装为统计显著或全量结论。

## 明确不声称

- 不声称线上 CTR、观看时长或收入提升；
- 不声称带 propensity 的无偏 OPE；
- 不声称全量数据 SOTA 或统计显著；
- 不把静态 `author_profile.csv` 描述为实时直播视频理解；
- 不把官方预计算内容 embedding 描述为自行训练的原始视频 MLLM；
- 主 GRPO 与三种子生成式统计来自公开数据 1% 用户样本；生成式分支另完成 10% seed42 和全量 seed42 验证，但全量仍是单种子离线结果。

## 一键入口

```bash
python -m pytest -q
bash scripts/run_strong_baseline_suite.sh
bash scripts/run_grpo_suite.sh
bash scripts/run_generative_replicate.sh 42 \
  data/processed/klm3_temporal_1pct_seed42 \
  runs/strong_suite_seed42/bridge/model.pt \
  runs/generative_stage1_1pct_seed42
cd dashboard && npm run dev
```
