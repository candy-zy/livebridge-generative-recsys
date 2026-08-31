# 简历项目表述

## 推荐项目名

**LiveBridge-GRPO｜短视频兴趣迁移驱动的直播跨域推荐与长程策略优化**　个人项目｜2026.08

**技术栈：** Python、PyTorch、Transformer、Semantic ID、InfoNCE、GRPO/PPO、LightGCN、SASRec、BPR、PyArrow、React

**项目架构图：** [`livebridge_architecture.svg`](livebridge_architecture.svg)（全链路）｜[`generative_retrieval_architecture.svg`](generative_retrieval_architecture.svg)（生成式推荐分支）｜[`livebridge_architecture.png`](../figures/livebridge_architecture.png)（简历/面试展示版）

## 可直接粘贴到简历（推荐版）

- 针对直播推荐中**新用户/低历史用户行为稀疏、短视频兴趣难以直接迁移、热门主播挤压长尾曝光**的问题，基于公开 **KuaiLive-M3** 从 0 到 1 搭建短视频→直播跨域推荐系统；实现多 GB CSV/Parquet 流式处理和逐用户时间门禁，并统一复现 Popularity、BPR、LightGCN、SASRec、EMCDR 等基线的 Full-sort 评测，规避源域未来行为泄漏。
- 设计共享 Creator 表征的 **Creator-Bridge** 跨域召回，将短视频作者偏好转化为直播候选信号；在 **1% 用户采样、3 个随机种子的可复现离线验证**中，相较同一协议下的**单域最强基线 LightGCN**，将 **Recall@10 从 0.0391 提升至 0.1023（+162%）**、**NDCG@10 从 0.0611 提升至 0.1640（+168%）**。
- 面向生成式推荐岗位，新增 **Semantic-ID Generative Retrieval** 分支：对官方短视频 128D / 直播 64D 预计算内容向量做作者级 InfoNCE 对齐，以 Residual K-Means 构造分层行为 Semantic ID，并用因果 Transformer 自回归生成直播作者；针对“内容入码导致行为结构漂移”的失败消融，改为缓存 Top-200 生成候选后做 valid-selected 内容 late fusion。1% 用户 3 seeds 下相较 ID-only generator 实现 **Recall@40 +34.3%、NDCG@40 +46.5%**（paired bootstrap 95% CI 均不跨 0）；进一步在全量 **787 万交互、60.1 万候选作者、2.18 万 test 用户**上实现 **Recall@40 +19.9%、NDCG@40 +19.5%**。
- 为解决逐物品打分难以兼顾准确率与内容生态的问题，将 Top-10 推荐列表建模为长程 Slate 决策，并在 Top-50 候选上实现受约束 **GRPO**：通过同一用户多组列表的相对反馈降低 Reward 尺度波动，通过 PPO Clip 与 Reference KL 限制策略偏离稳定召回模型，同时联合优化相关性、跨域兴趣和长尾曝光；相较 Creator-Bridge 进一步实现 **Recall@10 +11.0%、NDCG@10 +4.7%**，Long-tail Share@10 **0.3093→0.5411（+75%）**，3 个种子主指标均同向提升。
- 完成作者内容画像融合、Reward Ablation、冷启动分桶、实验 Dashboard 与 RTX 5090 可复现训练链路，沉淀配置、逐 epoch checkpoint、日志、逐用户结果和聚合报告；Python 测试 `17 passed`，Dashboard production build / ESLint 通过。
- 针对首版 Agent 的稀疏奖励与动作塌缩，设计 **Counterfactual-GRPO Agentic Router**：将上一条消费内容、Slate 统计和时间差写入因果 Session State，并枚举 5 类工具路由构造组内相对优势；在公开数据 10% 用户 Pilot、3 seeds 中，相较最强固定路由实现离线 Session Return `+0.60%`、Recall@10 `+0.73%`，平均动作切换率 `16.2%`，no-memory 消融退化回固定基线。结果为方向性 Pilot（Bootstrap 95% CI 跨 0），不包装为显著或线上收益；10,000 请求 CPU 基准中策略 P95 `0.85ms`，故障注入后无效曝光降至 `0`、fallback 成功率 `100%`。

## 一页简历压缩版（只能放 3 条时）

- 针对直播域行为稀疏、短视频兴趣迁移困难与长尾曝光不足，基于公开 KuaiLive-M3 从 0 到 1 搭建跨域推荐系统，完成多 GB 流式处理、逐用户时间门禁及 5 类基线的统一 Full-sort 评测。
- 设计共享 Creator 表征的跨域召回；在 **1% 用户、3 seeds 可复现离线验证**中，相较单域最强基线 LightGCN，实现 **Recall@10 +162%、NDCG@10 +168%**。
- 实现跨模态 InfoNCE + Residual Semantic ID + Transformer 自回归生成召回，并以缓存内容 late fusion 修复内容入码漂移；1% 三种子相较 ID-only 实现 **Recall@40 +34.3%、NDCG@40 +46.5%**，全量 60.1 万候选上仍实现 **+19.9%/+19.5%**。
- 为兼顾推荐准确率与内容生态，将 Top-10 列表建模为长程决策并实现受约束 GRPO；相较跨域召回进一步实现 **Recall@10 +11.0%、NDCG@10 +4.7%**，Long-tail Share@10 **+75%**，3 seeds 均同向提升。

## 面试 30 秒介绍

这是一个短视频到直播的跨域推荐项目，解决直播行为稀疏、短视频兴趣难迁移和长尾曝光不足三个问题。召回层通过共享 Creator 表征迁移用户的短视频作者兴趣；重排层把 Top-10 列表作为整体，用受约束 GRPO 平衡相关性、跨域兴趣和长尾生态。在 1% 用户、3 个种子的离线验证中，跨域召回相较单域最强基线的 Recall@10 提升 162%，GRPO 又在此基础上提升 11%，并将长尾曝光占比提升 75%。

## 高频追问回答

**为什么用 GRPO？** 逐物品打分容易得到“每个物品看起来都不错，但整张列表同质化”的结果。GRPO 对同一用户采样多组列表并比较相对收益，能够直接优化列表级目标；组内相对反馈不需要额外训练价值网络，PPO Clip 和 Reference KL 则防止策略为了长尾奖励过度牺牲相关性。

**Agent 为什么没有直接进直播热链路？** V1 的随机轨迹 GRPO 没超过固定路由；V2 用消费内容反馈和五动作 Counterfactual-GRPO 修复后，10% 三种子 Return/Recall 均值小幅提升，但 Bootstrap CI 仍跨 0。因此只让 Agent 异步刷新可缓存的路由决策，请求内执行状态过滤与快速 Ranker；这既保留工具编排和故障降级，也避免训练或复杂规划阻塞实时直播推荐。

**怎么防止数据泄漏？** 对每个用户以直播域训练截止时间作为门禁，只允许该时间之前的短视频行为进入跨域特征；测试集不参与召回模型或 GRPO 策略训练。

**为什么不把内容直接做 Semantic ID？** 失败消融表明内容入码虽然把碰撞率从 55.8% 降到 36.4%，却让 valid Recall@40 下降 16.9%，因为更唯一的内容码破坏了行为可预测性。最终方案固定行为 Semantic ID，只在生成候选上用缓存内容向量 late fusion，并在 valid 集选择融合权重；这样保留生成目标稳定性，也能在 α=0 时安全退化为 ID-only。

**这是线上收益吗？** 不是。结果来自 KuaiLive-M3 公开数据集离线实验；生成式分支完成全量 seed42，GRPO 的三种子结论仍来自 1% 协议。相关性奖励是 logged-positive proxy，不是带 propensity 的无偏 OPE，也不代表线上 CTR、观看时长或 GMV 提升。

**你的个人工作是什么？** 独立完成问题定义、数据审计、时间安全预处理、基线复现、Creator-Bridge、GRPO 重排、消融实验、结果分析及 Dashboard 工程化。

## 表述红线

- 可以写“离线 Recall/NDCG 提升”，不能写“线上 CTR/GMV 提升”。
- 可以写“生成式分支完成公开数据全量 seed42”，不能写“全量 SOTA”或“全量统计显著”。
- 可以写“logged-positive reward proxy”，不能写“无偏反事实评估”。
- `author_profile.csv` 是静态作者画像，不能包装成实时直播视频理解或多模态大模型。
- 生成式内容分支使用数据集预计算 128D/64D embedding，不能写成“训练了原始视频 MLLM”；10% 和全量均只有 seed42，不能包装成多种子显著性。
- Agentic Router 可以写“10% 三种子方向性提升”并附 `+0.60%/+0.73%`，不能写“统计显著”“全量数据提升”或线上收益。
