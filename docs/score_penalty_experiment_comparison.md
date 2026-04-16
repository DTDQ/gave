# Score Penalty 实验对比

## 实验设置

- 训练数据：`data/train/split/train.csv` + `data/train/split/val.csv`
- 测试数据：`data/test/pt_d=20260207/test_500_rlData.csv`
- 环境数据：`data/test/pt_d=20260207/env_500.csv`
- 训练参数：`step_num=20000`, `save_step=2000`, `eval_step=500`, `batch_size=64`, `learning_rate=1e-4`, `seed=42`
- 测试模型：各分支训练得到的 `best_model.pt`

## 分支定义

| 分支 | `getScore` 形式 | 关键参数 |
| --- | --- | --- |
| `main` | 原始分数 | `beta=2`, `penalty=min((target_CPA / CPA)^2, target_CPA / CPA)` |
| `exp/score-exp-penalty` | 指数惩罚 | `alpha=5` |
| `exp/score-power-penalty` | 幂次惩罚 | `gamma=4` |
| `exp/score-cliff-penalty` | 悬崖惩罚 | `k=10` |

## 结果总表

| 方案 | 最优验证损失 | 最优步数 | avg_conversion | avg_CPA | avg_CPA_violation | avg_budget_used | avg_win_rate | avg_spend | total_win_num |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始 `main` | 6350.5770 | 19500 | 39.4864 | 166.6955 | 4.1126 | 0.8759 | 0.8603 | 507.6644 | 129753 |
| 指数惩罚 `exp` | 11284.6984 | 17000 | 51.0133 | 22.0458 | 1.4425 | 0.4756 | 0.4510 | 207.7496 | 50710 |
| 幂次惩罚 `power` | 5687.8748 | 13000 | 43.4561 | 102.6431 | 2.5815 | 0.7651 | 0.5659 | 375.0113 | 109390 |
| 悬崖惩罚 `cliff` | 7468.2508 | 11000 | 43.4997 | 57.4208 | 2.1056 | 0.6562 | 0.5784 | 297.8252 | 90370 |
| 真实轨迹 `real` | - | - | 42.3976 | 18.1337 | -0.1100 | 0.2992 | 0.2491 | 108.5214 | 100922075 |

## 结论

- 如果只看验证损失：`power > main > cliff > exp`
- 如果看 500 条轨迹离线测试的业务结果：`exp` 最优
- `exp` 的平均转化最高，同时平均 CPA 和 CPA 超约束程度也最低
- `main` 的原始 score 在验证集上不差，但离线测试时明显过度花费，CPA 和 CPA 超约束最差
- `power` 虽然验证损失最好，但测试时花费、预算消耗和 CPA 都明显偏高
- `cliff` 比 `power` 收敛更稳，测试指标也更好，但仍弱于 `exp`

## 建议

- 下一轮如果优先看离线测试业务指标，建议以 `exp/score-exp-penalty` 为主线继续调参
- 如果要继续研究训练稳定性，可以基于 `power` 和 `cliff` 方案继续约束预算使用和 CPA 超约束

## 结果文件

| 方案 | 模型目录 | 测试汇总 |
| --- | --- | --- |
| `main` | `experiments/exp_score_original/models/DT_20260416_064004` | `experiments/exp_score_original/results/test_results_500/test_summary_20260416_075101.csv` |
| `exp` | `experiments/exp_score_exp_penalty/models/DT_20260415_231141` | `experiments/exp_score_exp_penalty/results/test_results_500/test_summary_20260416_002229.csv` |
| `power` | `experiments/exp_score_power_penalty/models/DT_20260416_002523` | `experiments/exp_score_power_penalty/results/test_results_500/test_summary_20260416_013637.csv` |
| `cliff` | `experiments/exp_score_cliff_penalty/models/DT_20260416_013826` | `experiments/exp_score_cliff_penalty/results/test_results_500/test_summary_20260416_024942.csv` |
