# GAVE 数据流处理文档

## 1. 概述

完整流程：**Raw TXT → RL CSV → Train/Val Split → 模型训练 → Test Data生成 → 离线回放测试**

```
data/raw_data/
├── train_pt_d=20260206/train_data.txt   (1.55GB, 训练用)
├── train_pt_d=20260207/train_data.txt   (1.53GB, 测试用)
└── env_pt_d=20260207/env_data.txt       (13.7GB, 测试环境)

                    ┌─────────────────────────────────────────────────┐
                    │             Step 1: 训练数据生成                  │
                    │   train_data.txt(20260206) → RL CSV             │
                    └──────────────────────┬──────────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────────┐
                    │             Step 2: 数据分割                      │
                    │   RL CSV → train.csv + val.csv                  │
                    └──────────────────────┬──────────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────────┐
                    │             Step 3: 模型训练                      │
                    │   train.csv + val.csv → best_model.pt           │
                    └──────────────────────┬──────────────────────────┘
                                           │
     ┌─────────────────────────────────────┼─────────────────────────────────┐
     │                                     │                                 │
     ▼                                     │                                 ▼
┌─────────────────────────┐                │          ┌──────────────────────────────┐
│  Step 4a: 测试RL数据生成  │                │          │  Step 4b: 环境数据过滤          │
│  train_data.txt(20260207)│                │          │  env_data.txt → env_data.csv  │
│  → test_rlData.csv       │                │          │  (仅保留有效adgroup_id的行)     │
└───────────┬─────────────┘                │          └──────────────┬───────────────┘
            │                              │                         │
            └──────────────┬───────────────┘─────────────────────────┘
                           │
            ┌──────────────▼──────────────────────────┐
            │             Step 5: 离线回放测试           │
            │   model + test_rlData + env_data → 指标  │
            └─────────────────────────────────────────┘
```

---

## 2. 输入数据格式

### 2.1 训练/测试日志 (train_data.txt) — 15列，无表头，逗号分隔

| 列序 | 字段名         | 类型    | 说明                                      |
|------|---------------|---------|------------------------------------------|
| 0    | adgroup_id    | int     | 广告组ID                                   |
| 1    | effect_type   | int     | 效果类型                                   |
| 2    | budget        | float   | 当日预算                                   |
| 3    | tcpa          | float   | 目标CPA                                   |
| 4    | timestep      | int     | 时间步编号 (1-72, 每步20分钟)                |
| 5    | remain_budget | float   | 当前步开始时的剩余预算                       |
| 6    | pvalue_mean   | float   | 当前步pCTCVR均值                            |
| 7    | pacer         | float   | 当前步出价系数 alpha (action = pacer * tcpa) |
| 8    | win_num       | int     | 当前步竞得曝光数                             |
| 9    | win_rate      | float   | 当前步竞胜率                                |
| 10   | cost          | float   | 当前步花费                                  |
| 11   | pvalue_sum    | float   | 当前步转化总量 (reward)                      |
| 12   | ci_mean       | float   | 当前步实际ecpm均值 (**需除以1000**)           |
| 13   | pv_num        | int     | 当前步参与竞拍的总曝光数                      |
| 14   | pt_d          | int     | 日期分区 (如20260206)                       |

### 2.2 环境数据 (env_data.txt) — 11列，无表头，逗号分隔

| 列序 | 字段名           | 类型   | 说明                                   |
|------|-----------------|--------|---------------------------------------|
| 0    | adgroup_id      | int    | 广告组ID                                |
| 1    | min_win_ecpm    | float  | 竞胜最低ecpm (**测试时除以1000**)         |
| 2    | p_ctr           | float  | 预估点击率                              |
| 3    | p_cvr           | float  | 预估转化率                              |
| 4    | pltv            | float  | 预估LTV (-1表示无)                      |
| 5    | ocpx_deep_pcvr  | string | OCPX深度转化率 (可为null)                |
| 6    | log_time        | long   | 日志时间戳 (毫秒级)                      |
| 7    | ab_tag          | string | AB实验标签                              |
| 8    | real_ecpm       | float  | 实际ecpm (**测试时除以1000**)            |
| 9    | bid_tag         | string | 竞价标签 (如gsp)                        |
| 10   | pt_d            | int    | 日期分区                                |

**注意**: env_data每行是一条独立曝光记录，单个广告组可能有数万条。总量约1.43亿行。

---

## 3. 数据过滤逻辑 (关键！需人工Check)

### 3.1 过滤条件一：win_num全天累计 >= 1000

**位置**: TrainDataGenerator._load_file() / TestDataGenerator._generate_test_data()

```python
adid_total_win = df.groupby('adgroup_id')['win_num'].sum()
valid_adids = adid_total_win[adid_total_win >= 1000].index
```

**逻辑**: 将同一 adgroup_id 下所有 timestep 的 win_num 求和，只保留全天总竞得数 >= 1000 的广告组。

**目的**: 排除曝光过少的广告组，这些广告组的策略学习信号太弱。

**影响 (20260206训练数据)**: 约 33696 → 13401 个广告组 (移除约60%)

### 3.2 过滤条件二：轨迹完整性检查

**位置**: _is_trajectory_complete()

```python
timesteps = group_df['timestep'].values
if timesteps[0] != 1:
    return False
expected = np.arange(1, len(timesteps) + 1)
return np.array_equal(timesteps, expected)
```

**逻辑**:
- timestep 必须从 1 开始
- timestep 必须连续递增 (1, 2, 3, ..., N)
- 不允许跳步或重复

**目的**: 确保训练/测试用的轨迹是完整的投放过程，避免截断或缺失的序列影响策略学习。

### 3.3 过滤条件三 (仅测试)：adgroup_id 必须在 env_data 中存在

**位置**: TestDataGenerator._generate_test_data()

```python
env_adgroup_ids = set(env_df['adgroup_id'].astype(int).unique())
if adgroup_id not in env_adgroup_ids:
    continue
```

**逻辑**: 测试时需要用 env_data 进行离线回放竞价模拟，如果 env_data 中没有该广告组的曝光记录，则无法测试。

### 3.4 过滤条件四 (仅测试)：总转化 > 0

**位置**: TestDataGenerator._generate_test_data()

```python
realAllConversion = group['pvalue_sum'].sum()
if realAllConversion <= 0:
    skipped += 1
    continue
```

**逻辑**: 跳过全天没有任何转化的广告组，因为它们无法提供有意义的策略评价。

### 3.5 过滤条件五 (数据分割阶段)：轨迹长度限制

**位置**: split_data.py → check_trajectory_complete()

```python
if len(timesteps) < min_len or len(timesteps) > max_len:
    return False
```

**默认**: min_traj_length=3, max_traj_length=100

**逻辑**: 排除过短（不足以学习策略）或过长（可能是数据异常）的轨迹。

---

## 4. 状态向量构造 (19维)

每个 timestep 对应一个 19维 状态向量。**状态表示的是该步开始前的信息**（即 shift(1) 取前一步的累积）。

| 维度 | 特征名                        | 计算方式                                              | 说明                          |
|------|------------------------------|------------------------------------------------------|-------------------------------|
| 0    | time_left_ratio              | (max_timestep - (t-1)) / max_timestep                | 剩余时间比例                    |
| 1    | budget_left_ratio            | remain_budget(t-1) / budget                           | 剩余预算比例 (t=1时为1.0)       |
| 2    | avg_pacer_all                | expanding().mean().shift(1)                           | 历史所有步的平均出价系数         |
| 3    | avg_pacer_last3              | rolling(3).mean().shift(1)                            | 最近3步的平均出价系数            |
| 4    | avg_ci_mean_all              | expanding().mean().shift(1)                           | 历史平均ecpm (已/1000)          |
| 5    | avg_pvalue_mean_all          | expanding().mean().shift(1)                           | 历史平均pCTCVR                  |
| 6    | avg_conversion_per_pv_all    | expanding().mean().shift(1)                           | 历史平均每PV转化                |
| 7    | avg_win_prob_all             | expanding().mean().shift(1)                           | 历史平均竞胜率                  |
| 8    | avg_ci_mean_last3            | rolling(3).mean().shift(1)                            | 近3步平均ecpm                   |
| 9    | avg_pvalue_mean_last3        | rolling(3).mean().shift(1)                            | 近3步平均pCTCVR                 |
| 10   | avg_conversion_per_pv_last3  | rolling(3).mean().shift(1)                            | 近3步平均每PV转化               |
| 11   | avg_win_prob_last3           | rolling(3).mean().shift(1)                            | 近3步平均竞胜率                 |
| 12   | pvalue_mean_agg              | 当前步的 pvalue_mean 原始值                            | 当前步pCTCVR (未shift)          |
| 13   | pv_num_agg                   | 当前步的 pv_num 原始值                                 | 当前步竞拍数 (未shift)          |
| 14   | last_3_volume                | rolling(3).sum().shift(1) on pv_num                    | 近3步累计PV                    |
| 15   | historical_volume            | cumsum().shift(1) on pv_num                            | 历史累计PV                     |
| 16   | avg_cost_all                 | cost.expanding().mean().shift(1)                       | 历史平均花费                    |
| 17   | avg_cost_last3               | cost.rolling(3).mean().shift(1)                        | 近3步平均花费                   |
| 18   | cumulative_cost_ratio        | cost.cumsum().shift(1) / budget                        | 累计花费占预算比例               |

**注意事项**:
- 维度 0-1 是时间/预算全局特征
- 维度 2-11 是历史聚合特征 (shift(1), 不含当前步)
- 维度 12-13 是当前步的瞬时值 (**没有shift，包含当前步信息——这是否合理需要确认**)
- 维度 14-18 是PV量/花费相关特征
- `ci_mean` 在读入时已除以1000
- `t=1` 时所有 shift(1) 的特征都填充为 0

### 4.1 Action 计算

```python
action = pacer * tcpa
```

模型的输出是 alpha (出价系数)，实际出价 = alpha * pCTCVR。

### 4.2 Reward 计算

```python
reward = pvalue_sum  # 当前步转化量
```

### 4.3 next_state 构造

- 中间步: next_state = 下一步的 state (通过 shift(-1) 获取)
- 最后一步 (done=1): 用 "_now" 版本的特征构造终止状态 (timeleft=0, 包含当前步统计)

---

## 5. 训练/验证数据分割

**脚本**: split_data.py

**输入**: 合并的 RL CSV 文件

**逻辑**:
1. 按 (advertiserNumber, timeStepIndex) 排序
2. 按 (deliveryPeriodIndex, advertiserNumber) 分组
3. 对每组检查轨迹完整性 (timestep从1开始、连续、长度在[3, 100])
4. 随机打乱合法轨迹 (seed=0)
5. 按比例分割 (默认 train:val = 99:1)

**输出**:
- `train.csv`: 训练集轨迹
- `val.csv`: 验证集轨迹
- `stats.json`: 统计信息
- `config.json`: 分割配置

---

## 6. 测试环境数据处理 (env_data)

### 6.1 内存问题

env_data.txt 有 13.7GB / 1.43亿行，pandas 直接读取会 OOM。

### 6.2 优化策略

分两步处理：
1. **提取 adgroup_ids**: 用 shell `cut` 命令从 env_data.txt 提取第1列唯一值 (1968个)
2. **生成测试RL数据**: 只加载 train_data.txt (1.5GB)，用 env adgroup_ids 做交集过滤
3. **过滤 env_data**: 用 `awk` 按最终有效的 adgroup_ids 从 env_data.txt 中逐行过滤，只保留需要的行

过滤后 env_data.csv 从 13.7GB 降至约 3.3GB。

### 6.3 OfflineEnv 竞价模拟

测试时，OfflineEnv 按如下逻辑模拟竞价：
```python
pCTCVR = p_ctr * p_cvr
bid_price = alpha * pCTCVR
win = (bid_price >= min_win_ecpm) and (remaining_budget >= real_ecpm)
cost = real_ecpm * 100 if win else 0
conversion = pCTCVR * 100 if win else 0
```

注意 `min_win_ecpm` 和 `real_ecpm` 在 OfflineEnv.reset() 中已除以 1000。

---

## 7. RL CSV 输出格式

### 7.1 训练数据列

| 列名                   | 类型   | 说明                           |
|------------------------|--------|-------------------------------|
| deliveryPeriodIndex    | int    | 投放周期（从目录名提取的日期）    |
| advertiserNumber       | int    | 广告组ID                        |
| advertiserCategoryIndex| int    | 广告主类别 (固定为0)             |
| budget                 | float  | 预算                            |
| CPAConstraint          | float  | 目标CPA                         |
| realAllCost            | float  | 该轨迹全天总花费                 |
| realAllConversion      | float  | 该轨迹全天总转化                 |
| timeStepIndex          | int    | 时间步编号                       |
| state                  | string | 19维状态元组的字符串表示          |
| action                 | float  | pacer * tcpa                    |
| reward                 | float  | 当前步转化量                     |
| reward_continuous      | float  | 同reward                        |
| done                   | int    | 是否为最后一步 (0/1)             |
| next_state             | string | 下一步状态元组的字符串表示        |

### 7.2 测试数据额外列

| 列名     | 类型  | 说明              |
|----------|-------|------------------|
| cost     | float | 当前步实际花费     |
| win_num  | int   | 当前步实际竞得数   |
| pv_num   | int   | 当前步参竞数       |

---

## 8. 需要人工确认的潜在问题

### 8.1 状态维度12-13未做shift
`pvalue_mean_agg` (维度12) 和 `pv_num_agg` (维度13) 直接使用当前步的原始值，**没有 shift(1)**。这意味着模型在决策时已经知道当前步的 pCTCVR 和 PV 数。对于训练数据这是"观察后记录"所以可接受，但在线上使用时需确认这些信息在出价前是否真的可用。

### 8.2 ci_mean 除以 1000
原始数据中的 `ci_mean` 在读入时立即除以 1000。而 env_data 中的 `min_win_ecpm` 和 `real_ecpm` 是在 OfflineEnv.reset() 时除以 1000。需确认两边的量纲是否一致。

### 8.3 action 的定义
训练数据中 `action = pacer * tcpa`，但模型输出的 alpha 在 OfflineEnv 中使用时应为 `bid = alpha * pCTCVR`。

### 8.4 reward 的单位
训练数据中 `reward = pvalue_sum` (原始转化量)，但 OfflineEnv 中 `conversion = pCTCVR * 100`。两者量纲是否一致？

### 8.5 win_num 过滤阈值
当前阈值为全天总 win_num >= 1000。训练和测试都使用相同阈值。该阈值是否合适需根据业务情况判断。

### 8.6 训练/测试数据日期一致性
训练用20260206，测试用20260207。需确认这两天的数据分布是否接近（如流量模式、广告主构成等）。
