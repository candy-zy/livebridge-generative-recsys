# 强基线与冷启动实验 Runbook

## 实验矩阵

每个随机种子 `42/43/44` 在对应的时间安全 1% 数据切分上运行：

| 类型 | 模型 | 作用 |
|---|---|---|
| 非个性化下界 | Popularity | 检查热门主播偏置 |
| 目标域 | BPR | 现有基础 MF 对照 |
| 目标域 | LightGCN | 图协同过滤强基线 |
| 目标域 | SASRec | 时序推荐强基线 |
| 跨域 | EMCDR | 经典 embedding mapping 对照 |
| 跨域 | Creator-Bridge BPR | 本项目方法 |

所有模型共用目标候选作者集合、已见作者屏蔽、Full-sort Recall/NDCG@10/20/40，以及按目标域训练历史长度划分的 `5-10`、`11-30`、`31+` 分桶。

SASRec 采用修正后的 `N(0, 0.02)` embedding 初始化；在 seed 42 验证集的 `{5e-4, 1e-3, 2e-3}` 学习率网格中选择 `5e-4`，固定后以 100 epoch 运行三个 seed。选参和复现实验分别执行：

```bash
bash scripts/run_sasrec_tune.sh
bash scripts/run_sasrec_tuned_seeds.sh
```

## 服务器启动命令

代码同步并安装完成后：

```bash
cd /path/to/livebridge-generative-recsys
source .venv/bin/activate
python -m pytest -q
screen -dmS strong_suite bash -lc \
  'cd /path/to/livebridge-generative-recsys && bash scripts/run_strong_baseline_suite.sh > runs/strong_suite.log 2>&1'
```

监控：

```bash
screen -ls
tail -f /path/to/livebridge-generative-recsys/runs/strong_suite.log
nvidia-smi
```

## 完成条件

- `python -m pytest -q` 全部通过；
- 18 个模型运行目录均存在 `metrics.json` 和 `per_user_metrics.csv`；
- `runs/strong_suite_summary.json` 包含六个模型的三随机种子均值/标准差；
- `runs/sasrec_tuned/seed{42,43,44}` 均包含修复后的 SASRec 指标与逐用户结果；
- 三个冷启动分桶均有用户，或在报告中明确说明空桶；
- Creator-Bridge 同时与 LightGCN、SASRec、EMCDR 比较，不能只对比 BPR；
- 简历只采用时间安全、多随机种子且通过上述门禁的数字。
