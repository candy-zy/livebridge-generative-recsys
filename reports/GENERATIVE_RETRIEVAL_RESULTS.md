# 生成式推荐与多模态内容对齐结果

## 结论

本阶段完成了一个与现有 Creator-Bridge / GRPO 链路互补的生成式召回分支：使用 KuaiLive-M3 官方预计算短视频 128D 与直播 64D 内容向量，先做作者级跨模态 InfoNCE 对齐，再对 Creator-Bridge 行为表示执行 Residual K-Means，得到分层 Semantic ID；因果 Transformer 根据用户历史自回归生成下一位直播作者的 Semantic ID，最后以缓存内容向量对生成候选做轻量 late fusion。

首版“将内容直接拼入 Semantic ID 聚类”没有通过 valid 门控。修复版保持行为 Semantic ID 不变，只在生成后的 200 个候选中融合内容相似度，并仅用 valid 集选择融合权重。修复版在 1% 三随机种子、10% seed42 和全量 seed42 均通过门控。

## 实验协议

- 数据集：公开 KuaiLive-M3。
- 切分：逐用户直播行为 chronological 80/10/10；源域只读取训练门禁前证据。
- 评测：全候选集 Recall/NDCG@10/20/40；test 不参与训练或融合权重选择。
- 内容：数据集预计算 embedding，不声称训练原始音视频 MLLM；不使用带未来时间戳的直播片段向量。
- 对齐选择集：作者不重叠于 InfoNCE 训练作者。
- 1%：seeds 42/43/44；test 用户分别为 204/214/210。
- 10%：seed42；2,169 test 用户、123,946 个直播作者候选。
- 全量：seed42；7,870,064 条交互、6,265,495 个序列训练样本、21,825 个 test 用户、600,935 个直播作者候选。

## 1% 三随机种子原始结果

| Seed | 模型 | Recall@10 | NDCG@10 | Recall@40 | NDCG@40 |
|---:|---|---:|---:|---:|---:|
| 42 | ID-only generator | 0.0402 | 0.0555 | 0.1242 | 0.0871 |
| 42 | Content late fusion | 0.0718 | 0.0998 | 0.1706 | 0.1280 |
| 43 | ID-only generator | 0.0299 | 0.0501 | 0.0679 | 0.0563 |
| 43 | Content late fusion | 0.0486 | 0.0976 | 0.0983 | 0.0930 |
| 44 | ID-only generator | 0.0529 | 0.0772 | 0.1122 | 0.0909 |
| 44 | Content late fusion | 0.0706 | 0.1148 | 0.1398 | 0.1223 |

## 1% 聚合结果

| 模型 | Recall@10 | NDCG@10 | Recall@40 | NDCG@40 |
|---|---:|---:|---:|---:|
| Tuned SASRec | 0.0127 ± 0.0032 | 0.0194 ± 0.0017 | 0.0338 ± 0.0043 | 0.0252 ± 0.0018 |
| ID-only generator | 0.0410 ± 0.0115 | 0.0609 ± 0.0143 | 0.1014 ± 0.0296 | 0.0781 ± 0.0190 |
| Content late fusion | **0.0637 ± 0.0131** | **0.1041 ± 0.0093** | **0.1362 ± 0.0363** | **0.1144 ± 0.0188** |

相对 ID-only，Content late fusion 的三种子均值提升为 Recall@10 `+55.3%`、NDCG@10 `+70.8%`、Recall@40 `+34.3%`、NDCG@40 `+46.5%`。分层逐用户 paired bootstrap（10,000 次）得到：

- Recall@10 差值 `+0.0227`，95% CI `[+0.0168, +0.0287]`；
- NDCG@10 差值 `+0.0431`，95% CI `[+0.0337, +0.0529]`；
- Recall@40 差值 `+0.0348`，95% CI `[+0.0274, +0.0425]`；
- NDCG@40 差值 `+0.0363`，95% CI `[+0.0299, +0.0428]`。

SASRec、ID-only 与 fusion 的逐用户 ID、训练交互数和冷启动 bucket 在每个 seed 中逐行一致。这里的 SASRec 对比仍只代表本仓库同协议下的 tuned baseline，不声称公开榜单 SOTA。

## 10% 规模验证

| 模型 | Recall@10 | NDCG@10 | Recall@40 | NDCG@40 |
|---|---:|---:|---:|---:|
| ID-only generator | 0.0160 | 0.0369 | 0.0325 | 0.0336 |
| Content late fusion | **0.0284** | **0.0575** | **0.0599** | **0.0564** |
| 相对提升 | **+78.0%** | **+56.0%** | **+84.3%** | **+67.9%** |

10% valid Recall@40 从 `0.03322` 提升至 `0.05200`；valid 扫描的融合权重从 0 到 0.5 整体上升，最终选择 `α=0.5`。候选全集未缩减，test 未用于调参。

## 全量 seed42 验证

| 模型 | Recall@10 | NDCG@10 | Recall@40 | NDCG@40 |
|---|---:|---:|---:|---:|
| ID-only generator | 0.01757 | 0.05291 | 0.03587 | 0.04377 |
| Content late fusion | **0.02331** | **0.06465** | **0.04300** | **0.05231** |
| 相对提升 | **+32.7%** | **+22.2%** | **+19.9%** | **+19.5%** |

全量对齐覆盖 600,935 个候选作者中的 `99.51%`，作者不重叠 selection 上跨模态 Recall@10 从固定投影的 `0.00025` 提升到 `0.25148`。融合权重仍只在 valid 上选择，最终 `α=0.5`；验证和测试各 21,825 个用户均在完整 600,935 作者候选空间评估。训练采用连续 uint8 序列存储、16,384 大批次、逐 epoch 原子 checkpoint 和 128 用户批量 Beam Search；最终训练、解码、融合与 Gate 退出码均为 0。

## 失败消融与修复依据

初版内容 Semantic ID 将 unique codes 从 4,169 提升到 5,997、碰撞率从 `55.8%` 降至 `36.4%`，但 seed42 valid Recall@40 从 `0.1234` 降至 `0.1025`。这说明“更唯一的离散码”不等于“更可预测的行为码”：内容特征改变了目标码空间，使相似消费轨迹对应的下一个 token 更分散，生成损失也更高。

修复版把内容放到生成后的候选重排：行为 Semantic ID 和生成器与 ID-only 保持同口径，内容只改变候选顺序；`α=0` 被纳入 valid 网格，因此若内容无效可以退化为 ID-only。工程上缓存每个用户的一次 beam 结果，六个融合权重只重排缓存候选，避免重复自回归解码。

## 产物与退出状态

- 三种子汇总：`runs/generative_summary_1pct/`。
- 1% seed42 初版失败证据：`runs/generative_stage1_1pct_seed42/content/`，研究门控退出码 2。
- 1% 修复版：`runs/generative_fusion_1pct_seed42/`、`runs/generative_stage1_1pct_seed43/fusion/`、`runs/generative_stage1_1pct_seed44/fusion/`，runner 退出码均为 0。
- 10%：`runs/generative_stage1_10pct_seed42/`，runner 退出码 0。
- 全量 seed42：`runs/generative_full_seed42/`，训练与 launcher 退出码均为 0。
- 全仓测试：`17 passed`，2 个 PyTorch nested-tensor warning。

## 声明边界

- 这是公开数据上的离线结果，不是线上 CTR、观看时长或 GMV。
- 1% 有三随机种子；10% 和全量都只有 seed42，不能把全量单种子包装成统计显著或 SOTA。
- 内容编码器使用数据集提供的预计算向量，不能写成“自行训练多模态大模型”。
- 生成式召回分支是 Semantic-ID autoregressive retrieval + cached late fusion，不包装成 LLM Agent。
