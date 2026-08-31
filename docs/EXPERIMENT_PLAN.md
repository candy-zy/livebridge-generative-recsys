# LiveBridge-GRPO 实验计划

## 目标与边界

在已训练的 Creator-Bridge 召回模型上增加一个可审计的长程直播重排策略。策略把 Top-M 候选排序视为不放回的多步决策，使用 Group-Relative Policy Optimization（GRPO）优化折扣累计奖励。训练只读取目标域 `valid` 日志及其当时可见的短视频证据，最终指标只在未参与训练的 `test` 上计算。

这是一项离线 logged-positive 实验，不具备展示/点击 propensity，不能声称无偏 OPE、线上 CTR 或收入提升。

## 方法

- Reference policy：Creator-Bridge BPR 的冻结打分。
- Policy：Reference score 加一个小幅受限的线性 residual，特征为 base score、用户-创作者短视频亲和度、目标域流行度和长尾程度。
- Action：从 Top-M 候选中按 Plackett-Luce 策略依次选择 K 个不重复直播作者。
- Group sampling：同一用户从旧策略采样 G 条候选轨迹。
- Reward：折扣相关性/观看时长、短视频兴趣一致性、长尾曝光奖励，并保留对 reference policy 的 KL 约束。
- Advantage：同用户组内 reward 标准化；采用 clipped policy ratio 更新。

## 实验矩阵

| 阶段 | 系统 | Seed | 验收条件 |
|---|---|---:|---|
| M0 | synthetic sanity | 1 | loss/梯度/JSON/逐用户输出正常 |
| M1 | Creator-Bridge reference | 42/43/44 | 已完成 |
| M2 | constrained GRPO reranker | 42/43/44 | test Full-sort 指标、coverage、long-tail、affinity 均可复现 |
| M3 | no-source / no-long-tail ablation | 42 | 至少解释一个 reward 组件作用 |

## 主要指标

- Accuracy：Recall/NDCG@10/20/40。
- Long-horizon slate：logged discounted utility@10、source-affinity@10。
- Exposure：catalog coverage@10、long-tail share@10、exposure Gini@10。
- 分桶：目标域历史 `5-10`、`11-30`、`31+`。

## 成功标准

1. 三随机种子全部完成并保留 parseable JSON/CSV；
2. 相比 Creator-Bridge，NDCG@10 相对下降不超过 5%，同时 coverage 或 long-tail share 提升；或者准确率提升且曝光指标不恶化；
3. 所有结论明确区分官方 Full-sort 准确率与离线 reward proxy；
4. Dashboard、README 和简历数字均能追溯到运行产物。

## 计算预算

1% 数据、单张 RTX 5090：sanity 小于 2 分钟；主实验和消融预计合计小于 1 GPU-hour。长任务必须在 `screen` 中运行。
