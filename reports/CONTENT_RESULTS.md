# 作者内容画像融合结果

数据：KuaiLive-M3 `author_profile.csv`。使用字段包括性别、年龄段、粉丝层级、是否为短视频作者、是否为直播作者；类别经 one-hot 编码后形成作者侧画像。用户画像只由目标域 `train` 交互作者按 `log1p(play_duration)` 加权聚合，融合系数只在 `valid` 网格中选择，`test` 只评一次。

## 三随机种子

| 系统 | Recall@10 | NDCG@10 |
|---|---:|---:|
| Creator-Bridge reference | 0.1023±0.0168 | 0.1640±0.0081 |
| Creator-Bridge + profile fusion | 0.1051±0.0162 | 0.1663±0.0089 |

三 seed 的 NDCG@10 均提升；Recall@10 平均绝对提升 `0.0028`，但 seed 42 有轻微下降，因此不把画像融合描述为稳定 Recall 增益。该实验的作用是证明项目确实读取公开内容/画像信息，并为统一 GRPO policy 提供可解释的 `profile_affinity` 特征。

## 边界

- 这是静态作者内容画像，不是实时直播视频理解。
- 当前结果没有使用 `live_emb_64.parquet` 或 `live_emb_128_ts/`，不得描述为动态片段建模。
- 测试期画像交互不会反向进入用户画像；融合权重仅由 validation 决定。
