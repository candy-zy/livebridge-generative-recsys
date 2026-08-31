# LiveBridge-RL V3 修复诊断：从五路由分类转向预算化工具升级

日期：2026-08-30
数据边界：KuaiLive-M3 temporal 10% valid；所有可预测性结果均为 user-level 5-fold out-of-fold，未读取 test。

## 结论

V2 的主要瓶颈不是 GRPO epoch 不够，而是决策定义不合理：策略被要求在每一步从五个路由中分类，但相较最强固定 source 路由，只有 `1.10%` 的状态存在严格正 override。大多数动作切换只能增加方差，无法增加准确率。

更合理的 V3 是 **Budgeted Tool-Escalation Agent**：

1. 默认执行廉价 bridge-only 路由；
2. 根据廉价、因果的会话状态预测是否值得升级调用 source/content 等工具；
3. 只有置信下界超过阈值时 escalation，否则 abstain；
4. 把剩余调用次数、累计实测延迟和会话步数放入状态，使动作改变未来预算，形成可辩护的 constrained sequential decision problem；
5. 目标由“盲目超过固定路由”改为“在准确率安全约束下减少昂贵工具调用/延迟”。

## 已核实的结构性问题

### 1. Router checkpoint selection 泄漏

旧实现使用同一批 `train_sessions` 既更新参数又选择 warm-start/GRPO checkpoint。现已按用户随机、可复现地拆为：

- `valid/router_train`
- `valid/router_selection`
- `test` 仅最终一次评估

远端全套测试：`10 passed, 1 warning`。

### 2. 五个“工具”原来共享同一候选池

旧 cache 先截断 BridgeBPR Top-100，再让 source/content/popular/long-tail 在这 100 个作者中重排。工具没有独立召回权限。现已实现 `--independent-tool-pools`：五个生成器分别从全量作者空间取 Top-M，再 union 成候选缓存。

但 1% sanity 显示，独立工具池仅把严格正 override 率从 `2.33%` 提高到 `2.42%`，未达到扩大 10% 训练的门槛。因此它是正确的工程修复，但不是单独的效果解法。

### 3. 相对最强 source 基线的收益事件过于稀少

10% valid、固定 Action 1 状态访问：

| 指标 | 数值 |
|---|---:|
| 状态数 | 14,770 |
| 严格正 override 状态 | 163 |
| 严格正 override 率 | 1.10% |
| OOF ROC-AUC | 0.572 |
| OOF Average Precision | 0.0160 |
| 同等 prevalence 预算下 Precision | 3.07% |

这解释了 V2 约 2% 的 per-user win rate 与跨零置信区间：不是简单调参能解决。

### 4. Bridge 默认 + 按需升级更可学习

10% valid、固定 Action 0 状态访问：

| 指标 | 使用全部 route signatures | 仅廉价因果状态 |
|---|---:|---:|
| 正 escalation 率 | 3.49% | 3.49% |
| OOF ROC-AUC | 0.644 | **0.665** |
| OOF Average Precision | 0.0624 | **0.0759** |
| 同等 3.49% 预算下 Precision | 9.13% | **13.01%** |
| Precision / random | 2.62× | **3.73×** |

只用廉价状态更好，说明无需先执行昂贵工具才能决定是否 escalation，避免了成本评估作弊。

## V3 方法定义

### 在线链路

```text
cheap session state + bridge uncertainty + remaining budget
                         |
                 escalation critic
                         |
        LCB(benefit) > threshold ?
              /                   \
            yes                    no
   call source/content tools     bridge-only
              \                   /
       live-status filter -> fast ranker -> slate
```

### 训练

- 在 `router_train` 枚举动作，标签为相对 bridge-only 的 advantage。
- 使用 focal/weighted ranking loss 处理 3.49% 稀有正例。
- 用户级 cross-fitting 校准 advantage uncertainty。
- 在 `router_selection` 选择阈值，使 Recall/Return 不低于固定 source 路由的安全容差，同时最小化额外工具调用。
- 只有 dense critic 通过 sanity 后，才在带剩余预算的 trajectory 上做 constrained GRPO；若 GRPO 不增益则删除。

### 必须新增的真实指标

- 每工具独立召回 P50/P95 延迟与缓存命中率；不能使用拍脑袋成本。
- Escalation rate、extra tool calls/session、累计检索延迟。
- 相对 always-source 的 Return/Recall 差值与 latency saving。
- 预算违反率、fallback 成功率、paired bootstrap CI。

## 下一阶段 Gate

1. 1%：训练 binary escalation critic，selection 上 OOF/held-out AUC > 0.62。
2. 1%：在不低于 always-source Recall 1% 相对容差下，额外工具调用减少至少 30%。
3. 达标后才跑 10% 三随机种子。
4. 若 cheap-state critic 无法保持准确率，停止 Agentic RL 扩展，并把项目诚实定位为 Agentic Router 原型。

## 主张边界

当前数据的未来 logged target 不随 agent 动作变化，因此不能把它描述为“优化用户行为转移的完整 RL”。V3 的可辩护 RL 部分来自**动作对剩余检索预算与未来工具可用性的影响**；若不实现预算状态和 constrained trajectory objective，更准确的名称是 offline contextual bandit router。

## 10% Gate 最终结果（2026-08-30）

工程链路与 user-separated selection 均已完成，远端全套测试为 `13 passed, 1 warning`。10% seed42 的 cheap-state critic 在 selection 达到 ROC-AUC `0.7298`、AP `0.0703`，证明它能学习到稀有升级事件的排序信号；但该信号不足以满足预注册的质量—成本联合约束。

| 10% test | Return | Recall@10 | NDCG@10 | Escalation | 调用减少 |
|---|---:|---:|---:|---:|---:|
| bridge-only | 0.11505 | 0.02039 | 0.01014 | 0% | 100% |
| always-source | 0.13647 | 0.02615 | 0.01311 | 100% | 0% |
| selection-calibrated critic | 0.13509 | 0.02597 | 0.01303 | 99.50% | 0.50% |

保持 selection Return 与 Recall 在 always-source 相对 `1%` 容差内时，选出的阈值只能减少 `0.52%` 工具调用，远低于 `30%` 目标。selection 上约 `70.7%` 调用减少的 Pareto 点，Recall 下降 `1.34%`，但 Return 下降 `7.73%`，仍不达标。

因此 V3-03 判定 **FAIL**，V3-04 seeds43/44 与 constrained GRPO 按 stop/go 规则停止。该结果不支持“预算 Agent 在几乎不损失效果时显著减少调用”的主张；可保留的证据是：可学习的排序信号存在，但在当前 cheap features、logged-feedback 和 sequential visitation 下不足以形成质量安全的稀疏调用策略。
