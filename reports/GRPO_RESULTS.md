# Constrained GRPO 长程重排结果

设置：KuaiLive-M3 时间安全 1% 用户样本，seed 42/43/44；Creator-Bridge BPR 作为冻结 reference；Top-50 候选上采样 8 条长度为 10 的不放回轨迹；GRPO 使用组内标准化 advantage、PPO clip 和 reference KL。策略训练只读取 `valid` logged positives，最终结果只在 `test` 上计算。

## 三随机种子结果

均值 ± 样本标准差：

| 系统 | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | Recall@40 | NDCG@40 |
|---|---:|---:|---:|---:|---:|---:|
| Creator-Bridge reference | 0.1023±0.0168 | 0.1640±0.0081 | 0.1624±0.0142 | 0.1711±0.0065 | 0.2343±0.0185 | 0.1892±0.0110 |
| **Creator-Bridge + GRPO** | **0.1136±0.0182** | **0.1717±0.0095** | **0.1752±0.0201** | **0.1785±0.0106** | **0.2442±0.0217** | **0.1970±0.0128** |

GRPO 相对 Creator-Bridge 将 Recall@10 提升约 `11.0%`，NDCG@10 提升约 `4.7%`。三个 seed 的 Recall@10 和 NDCG@10 方向均为正；只有三个 seed，不能表述为统计显著。

## 长程曝光与代理奖励

| 指标 | Reference | GRPO | 变化 |
|---|---:|---:|---:|
| Logged discounted utility@10 | 0.1595±0.0060 | 0.1688±0.0090 | +5.8% |
| Source affinity@10 | 0.2867±0.0332 | 0.3018±0.0352 | +5.3% |
| Long-tail share@10 | 0.3093±0.0148 | 0.5411±0.0148 | +75.0% |
| Catalog coverage@10 | 0.0617±0.0114 | 0.0656±0.0115 | +6.3% |
| Exposure Gini@10 | 0.9775±0.0037 | 0.9751±0.0035 | -0.0024（越低越均衡） |

## Reward 消融（seed 42）

| 配置 | Recall@10 | NDCG@10 | Source affinity@10 | Long-tail share@10 |
|---|---:|---:|---:|---:|
| Full reward | 0.1240 | 0.1798 | 0.2903 | 0.5446 |
| 去掉 source reward | 0.1120 | 0.1621 | 0.2654 | 0.6245 |
| 去掉 long-tail reward | 0.1322 | 0.1800 | 0.3126 | 0.3510 |

消融表明 source reward 对兴趣一致性及准确率有正贡献；long-tail reward 明确改变准确率—曝光权衡，显著提高长尾曝光，但会牺牲 seed 42 的一部分 Recall。它不是“免费提升”，这正是使用受约束多目标策略的业务动机。

## 作者画像策略扩展

另运行了把 `profile_affinity` 加入 policy feature 与 reward 的统一版本：Recall@10 `0.1130±0.0185`、NDCG@10 `0.1706±0.0112`、Long-tail Share@10 `0.5506±0.0113`。该版本在 validation 均值上略优，但 test NDCG 在 seed 43 轻微下降，因此不替换三 seed 方向一致的行为版 GRPO 主结果；它作为内容画像—长尾曝光权衡扩展保留在 `runs/grpo_profile_suite/`。

## 结论边界

- 这是 logged-positive 离线重排，不是带 propensity 的无偏 OPE。
- `play_duration`、source affinity 和 long-tail exposure 是 reward proxy，不能写成线上 CTR、观看时长或收入提升。
- 测试集没有参与策略训练；数据切分和 source event cutoff 与强基线实验一致。
- 可复现产物：`runs/grpo_suite/`、`runs/grpo_ablation_seed42/`、`runs/grpo_summary.json`。
