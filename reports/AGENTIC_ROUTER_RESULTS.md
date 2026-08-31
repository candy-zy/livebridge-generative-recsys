# Session Agentic Router：10% Pilot 与服务边界

实验日期：2026-08-30
数据：KuaiLive-M3 temporal 10% seed 42，2,169 个 valid 用户与 2,169 个 test 用户
边界：策略只在 valid replay 上训练；test 只做最终评测；结果不是线上 CTR/GMV 或无偏 OPE。

## 结论先行

首版会话级 GRPO Router 在 10% Pilot 未超过固定路由。失败诊断显示，84.5% 的状态存在可区分的路由奖励、逐步 Oracle 切换率约 26.3%，但旧状态没有消费内容反馈，随机采样的轨迹组又经常产生近零相对优势。V2 增加因果内容反馈，并以五个可枚举路由构造稠密反事实监督和 Counterfactual-GRPO。**V2 在 10% 三种子上得到小幅、同向的 Return/Recall 提升，但按用户 Bootstrap 的 95% CI 仍跨 0，应描述为可复现的方向性 Pilot，而非统计显著或线上收益。**

同时，分层服务假设得到支持：策略和 Ranker 均远低于 50ms 原型阈值；LiveStatusFilter 能把注入的无效直播曝光降到 0，bridge fallback 在本实验中成功率为 100%。

## 10% 主结果

| 系统 | Test Session Return | Recall@10 | NDCG@10 | Long-tail Share@10 | 动作切换率 | 动作熵 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed source（最强准确率基线） | **0.16381** | **0.02615** | **0.01311** | 0.18711 | 0 | 0 |
| Fixed balanced | 0.14485 | 0.02393 | 0.01169 | 0.18619 | 0 | 0 |
| Contextual Router | 0.15428 | 0.02532 | 0.01299 | 0.16118 | 0.2197 | 0.4436 |
| Recurrent Agent + GRPO | 0.15428 | 0.02532 | 0.01299 | 0.16118 | 0.2197 | 0.4436 |
| Weighted trajectory warm-start Agent | 0.15221 | 0.02415 | 0.01073 | **0.25127** | 0 | 0.7701 |

相较 Fixed source，首版 recurrent Agent 的 Session Return / Recall / NDCG 分别低约 5.8% / 3.2% / 0.9%，且与 Contextual Router 完全相同，说明当前 replay 下循环记忆没有产生独立收益。加权 long-horizon warm-start 将 Long-tail Share 提升约 34.3%，但 Recall 下降约 7.6%，只能作为可解释的生态曝光权衡，不能包装成推荐质量提升。

验证集逐用户固定动作 oracle 的 Session Return 上界为 0.29506，而最强固定策略约 0.14030，说明动作空间本身存在潜力；目前失败点是仅凭公开缓存特征无法稳定预测逐用户 oracle，而不是缺少可选路由。这个上界使用未来 replay 标签，只用于诊断，绝不作为可部署结果。

## Agent V2：10% 三种子修复结果

V2 的 Reward 只保留 relevance 与 logged-watch proxy，以免辅助生态奖励掩盖准确率退化。固定 Action 1 也按相同 Reward 重跑。

| Test 指标 | Fixed Action 1 | Agent V2（3 seeds, mean±std） | 相对变化 |
|---|---:|---:|---:|
| Session Return | 0.136467 | 0.137291±0.000466 | +0.60% |
| Recall@10 | 0.026147 | 0.026339±0.000088 | +0.73% |
| NDCG@10 | 0.013109 | 0.013122±0.000022 | +0.10% |
| Logged Watch@10 | 0.019235 | 0.019345±0.000054 | +0.57% |
| Catalog Coverage@10 | 0.161482 | 0.162557±0.000964 | +0.67% |
| Action Switch Rate | 0 | 0.161690±0.109221 | 动态路由 |
| Policy P95 | 0.276ms | 0.850±0.007ms | 仍低于 1ms |

- Return 与 Recall 在 seeds 42/43/44 均高于固定基线；NDCG 均值基本持平。
- Counterfactual-GRPO checkpoint 在 seeds 42/44 被验证集选中，seed43 自动回退稠密蒸馏 checkpoint。
- 2,169 个配对用户 Bootstrap：Return delta 95% CI `[-0.00104, 0.00277]`，Recall delta `[-0.000096, 0.000499]`，NDCG delta `[-0.000177, 0.000186]`；均不能声称统计显著。
- dense-only 的 Test Return/Recall 为 `0.136779/0.026262`，弱于完整 V2 均值；no-memory 则精确退化到固定基线 `0.136467/0.026147`，支持消费内容反馈是动态路由的必要输入。

## 工程修复记录

- 修复重复 slate：检索前过滤已经曝光的主播，1% sanity 的 Repeat Rate 从约 74% 降至 0。
- 修复 Reference KL：参照冻结 warm-start policy，而不是硬编码 logits。
- 增加 session feature distribution、动作熵、工具调用计数、逐用户 CSV 与可复用候选 cache。
- V2 将上一条消费内容特征、上一 slate 统计与事件时间差纳入因果状态；当前目标仅用于训练教师打标签，不进入策略输入。
- 实现五动作稠密反事实蒸馏与 Counterfactual-GRPO，用完整动作组替代容易同质化的随机轨迹组。
- 发现并修复 PyTorch 单样本 weighted cross-entropy 权重被 reduction 抵消的问题。
- 所有最终结果保留 `metrics.json`、`summary.csv` 与明确的 audit boundary。

## 服务基准（10,000 请求，CPU）

| 路径 | P50 | P95 | P99 |
|---|---:|---:|---:|
| Policy update | 0.059ms | 0.061ms | 0.064ms |
| Cached fast ranker | 0.115ms | 0.118ms | 0.121ms |
| 同步 Policy + Ranker | 0.186ms | 0.190ms | 0.195ms |

- TTL=10/30/60 秒时，模拟 policy cache hit 分别为 85.86% / 99.66% / 100%。
- 注入 10%/30% 无效直播供给时，未过滤的无效曝光为 9.77%/30.10%，LiveStatusFilter 后均为 0。
- 两档故障下 bridge fallback 均生成合法 slate，成功率 100%。
- 热链路不运行训练，也不调用 LLM；推荐链路是 `cached action → live-status filter → fast ranker`。

## 最终主张边界

可以声称：实现并审计了 session Agentic RL 原型、动态工具路由、稠密反事实监督、Counterfactual-GRPO、异步 TTL 与故障降级；V2 在 10% 三种子离线 Pilot 上获得 Return/Recall 小幅同向提升，并通过消融验证消费内容反馈的必要性。

不能声称：提升达到统计显著、完成全量数据实验、带来线上 CTR/时长/收入提升，或达到生产系统的真实端到端延迟。
