#!/usr/bin/env python3
"""
内存高效版测试数据生成器。
- 从预先提取的文件读取env adgroup_ids（避免加载13.7GB env_data）
- 处理train_data.txt（1.5GB）生成RL数据
- 输出有效adgroup_ids供后续过滤env_data
"""
import os
import sys
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

# === 配置 ===
RAW_TEST = "./data/raw_data/train_pt_d=20260207/train_data.txt"
ENV_ADIDS_FILE = "/tmp/env_adgroup_ids.txt"  # 预提取的env adgroup_ids
OUTPUT_DIR = "./data/test/pt_d=20260207"

TEST_COLUMNS = [
    'adgroup_id', 'effect_type', 'budget', 'tcpa', 'timestep',
    'remain_budget', 'pvalue_mean', 'pacer', 'win_num', 'win_rate',
    'cost', 'pvalue_sum', 'ci_mean', 'pv_num', 'pt_d'
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 读取env adgroup_ids
print("读取环境文件 adgroup_ids...")
env_adgroup_ids = set()
with open(ENV_ADIDS_FILE, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            env_adgroup_ids.add(int(line))
print(f"环境文件中包含 {len(env_adgroup_ids)} 个广告组ID")

# 2. 读取train_data.txt
print(f"读取测试文件: {RAW_TEST}")
df = pd.read_csv(RAW_TEST, header=None, names=TEST_COLUMNS, dtype=str, on_bad_lines='skip')

# 清洗
for col in df.columns:
    if col in ['adgroup_id', 'timestep']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    else:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

df = df.sort_values(['adgroup_id', 'timestep']).reset_index(drop=True)
df['ci_mean'] = df['ci_mean'] / 1000.0

# 3. 过滤 win_num < 1000
adid_total_win = df.groupby('adgroup_id')['win_num'].sum()
valid_adids = adid_total_win[adid_total_win >= 1000].index
before = df['adgroup_id'].nunique()
df = df[df['adgroup_id'].isin(valid_adids)].reset_index(drop=True)
after = df['adgroup_id'].nunique()
print(f"过滤 win_num<1000: {before} -> {after} (移除 {before - after})")

# 4. 提取period
dir_name = os.path.basename(os.path.dirname(RAW_TEST))
numbers = re.findall(r'\d+', dir_name)
deliveryPeriodIndex = int(numbers[0]) if numbers else 1

df['deliveryPeriodIndex'] = deliveryPeriodIndex
df['advertiserCategoryIndex'] = 0
df['conversion_per_pv'] = df['pvalue_sum'] / df['pv_num'].replace(0, 1)
df['win_prob'] = df['win_num'] / df['pv_num'].replace(0, 1)

# 5. 生成RL数据
training_rows = []
skipped = 0
group_keys = ['deliveryPeriodIndex', 'adgroup_id', 'advertiserCategoryIndex', 'budget', 'tcpa']
groups = list(df.groupby(group_keys))
print(f"总共 {len(groups)} 个广告组轨迹")

final_valid_adids = set()

for _, group in tqdm(groups, desc="处理广告组"):
    group = group.sort_values('timestep').reset_index(drop=True)
    if len(group) == 0:
        continue

    adgroup_id = group['adgroup_id'].iloc[0]
    if adgroup_id not in env_adgroup_ids:
        continue

    # 轨迹完整性检查
    timesteps = group['timestep'].values
    if timesteps[0] != 1 or not np.array_equal(timesteps, np.arange(1, len(timesteps) + 1)):
        skipped += 1
        continue

    max_timestep = group['timestep'].max()
    group['isEnd'] = (group['timestep'] == max_timestep).astype(int)

    pv_by_step = group.groupby('timestep')['pv_num'].first()
    group['historical_volume'] = group['timestep'].map(
        pv_by_step.cumsum().shift(1).fillna(0).astype(int))
    group['last_3_volume'] = group['timestep'].map(
        pv_by_step.rolling(3, min_periods=1).sum().shift(1).fillna(0).astype(int))

    for col in ['pacer', 'ci_mean', 'conversion_per_pv', 'win_prob', 'pvalue_mean']:
        group[f'avg_{col}_all'] = group[col].expanding().mean().shift(1).fillna(0)
        group[f'avg_{col}_last3'] = group[col].rolling(3, min_periods=1).mean().shift(1).fillna(0)

    group['avg_cost_all'] = group['cost'].expanding().mean().shift(1).fillna(0)
    group['avg_cost_last3'] = group['cost'].rolling(3, min_periods=1).mean().shift(1).fillna(0)
    group['cumulative_cost_ratio'] = group['cost'].cumsum().shift(1).fillna(0) / group['budget'].replace(0, 1)

    group['avg_pacer_all_now'] = group['pacer'].expanding().mean()
    group['avg_pacer_last3_now'] = group['pacer'].rolling(3, min_periods=1).mean()
    group['avg_ci_mean_all_now'] = group['ci_mean'].expanding().mean()
    group['avg_ci_mean_last3_now'] = group['ci_mean'].rolling(3, min_periods=1).mean()
    group['avg_pvalue_mean_all_now'] = group['pvalue_mean'].expanding().mean()
    group['avg_pvalue_mean_last3_now'] = group['pvalue_mean'].rolling(3, min_periods=1).mean()
    group['avg_conversion_per_pv_all_now'] = group['conversion_per_pv'].expanding().mean()
    group['avg_conversion_per_pv_last3_now'] = group['conversion_per_pv'].rolling(3, min_periods=1).mean()
    group['avg_win_prob_all_now'] = group['win_prob'].expanding().mean()
    group['avg_win_prob_last3_now'] = group['win_prob'].rolling(3, min_periods=1).mean()
    group['avg_cost_all_now'] = group['cost'].expanding().mean()
    group['avg_cost_last3_now'] = group['cost'].rolling(3, min_periods=1).mean()
    group['cumulative_cost_ratio_now'] = group['cost'].cumsum() / group['budget'].replace(0, 1)
    group['historical_volume_now'] = group['timestep'].map(pv_by_step.cumsum().fillna(0).astype(int))
    group['last_3_volume_now'] = group['timestep'].map(
        pv_by_step.rolling(3, min_periods=1).sum().fillna(0).astype(int))

    group['pvalue_mean_agg'] = group['pvalue_mean']
    group['pv_num_agg'] = group['pv_num']

    realAllCost = group['cost'].sum()
    realAllConversion = group['pvalue_sum'].sum()
    if realAllConversion <= 0:
        skipped += 1
        continue

    group['prev_remain_budget'] = group['remain_budget'].shift(1)

    states_start = {}
    for _, row in group.iterrows():
        t = int(row['timestep'])
        timeleft_start = (max_timestep - (t - 1)) / max_timestep if max_timestep > 0 else 0.0
        if t == 1:
            bgtleft_start = 1.0
        else:
            bgtleft_start = row['prev_remain_budget'] / row['budget'] if row['budget'] > 0 else 0.0
        state = (
            float(timeleft_start), float(bgtleft_start),
            float(row.get('avg_pacer_all', 0.0)), float(row.get('avg_pacer_last3', 0.0)),
            float(row.get('avg_ci_mean_all', 0.0)), float(row.get('avg_pvalue_mean_all', 0.0)),
            float(row.get('avg_conversion_per_pv_all', 0.0)), float(row.get('avg_win_prob_all', 0.0)),
            float(row.get('avg_ci_mean_last3', 0.0)), float(row.get('avg_pvalue_mean_last3', 0.0)),
            float(row.get('avg_conversion_per_pv_last3', 0.0)), float(row.get('avg_win_prob_last3', 0.0)),
            float(row.get('pvalue_mean_agg', 0.0)), float(row.get('pv_num_agg', 0.0)),
            float(row.get('last_3_volume', 0.0)), float(row.get('historical_volume', 0.0)),
            float(row.get('avg_cost_all', 0.0)), float(row.get('avg_cost_last3', 0.0)),
            float(row.get('cumulative_cost_ratio', 0.0))
        )
        states_start[t] = state

    for i, row in group.iterrows():
        t = int(row['timestep'])
        record = {
            'deliveryPeriodIndex': deliveryPeriodIndex,
            'advertiserNumber': row['adgroup_id'],
            'advertiserCategoryIndex': 0,
            'budget': row['budget'],
            'CPAConstraint': row['tcpa'],
            'realAllCost': realAllCost,
            'realAllConversion': realAllConversion,
            'timeStepIndex': t,
            'state': states_start[t],
            'action': float(row['pacer'] * row['tcpa']),
            'reward': float(row['pvalue_sum']),
            'reward_continuous': float(row['pvalue_sum']),
            'done': 1 if t == max_timestep else 0,
            'remain_budget': row['remain_budget'],
            'avg_pacer_all_now': row['avg_pacer_all_now'],
            'avg_pacer_last3_now': row['avg_pacer_last3_now'],
            'avg_ci_mean_all_now': row['avg_ci_mean_all_now'],
            'avg_ci_mean_last3_now': row['avg_ci_mean_last3_now'],
            'avg_pvalue_mean_all_now': row['avg_pvalue_mean_all_now'],
            'avg_pvalue_mean_last3_now': row['avg_pvalue_mean_last3_now'],
            'avg_conversion_per_pv_all_now': row['avg_conversion_per_pv_all_now'],
            'avg_conversion_per_pv_last3_now': row['avg_conversion_per_pv_last3_now'],
            'avg_win_prob_all_now': row['avg_win_prob_all_now'],
            'avg_win_prob_last3_now': row['avg_win_prob_last3_now'],
            'pvalue_mean_agg': row['pvalue_mean_agg'],
            'pv_num_agg': row['pv_num_agg'],
            'last_3_volume_now': row['last_3_volume_now'],
            'historical_volume_now': row['historical_volume_now'],
            'avg_cost_all_now': row['avg_cost_all_now'],
            'avg_cost_last3_now': row['avg_cost_last3_now'],
            'cumulative_cost_ratio_now': row['cumulative_cost_ratio_now'],
            'cost': float(row['cost']),
            'win_num': int(row['win_num']),
            'pv_num': int(row['pv_num']),
        }
        training_rows.append(record)

    final_valid_adids.add(adgroup_id)

if skipped > 0:
    print(f"共跳过 {skipped} 条不完整/无转化轨迹")

print(f"有效广告组数: {len(final_valid_adids)}")

if not training_rows:
    print("错误：没有生成任何有效轨迹！")
    sys.exit(1)

training_df = pd.DataFrame(training_rows)
training_df = training_df.sort_values(
    ['deliveryPeriodIndex', 'advertiserNumber', 'timeStepIndex']).reset_index(drop=True)

training_df['next_state'] = training_df.groupby(
    ['deliveryPeriodIndex', 'advertiserNumber'])['state'].shift(-1)

mask_last = training_df['done'] == 1
for idx in training_df[mask_last].index:
    row = training_df.loc[idx]
    end_state = (
        0.0,
        float(row['remain_budget'] / row['budget'] if row['budget'] > 0 else 0.0),
        float(row['avg_pacer_all_now']), float(row['avg_pacer_last3_now']),
        float(row['avg_ci_mean_all_now']), float(row['avg_pvalue_mean_all_now']),
        float(row['avg_conversion_per_pv_all_now']), float(row['avg_win_prob_all_now']),
        float(row['avg_ci_mean_last3_now']), float(row['avg_pvalue_mean_last3_now']),
        float(row['avg_conversion_per_pv_last3_now']), float(row['avg_win_prob_last3_now']),
        float(row['pvalue_mean_agg']), float(row['pv_num_agg']),
        float(row['last_3_volume_now']), float(row['historical_volume_now']),
        float(row['avg_cost_all_now']), float(row['avg_cost_last3_now']),
        float(row['cumulative_cost_ratio_now'])
    )
    training_df.at[idx, 'next_state'] = end_state

training_df['state'] = training_df['state'].apply(str)
training_df['next_state'] = training_df['next_state'].apply(str)

cols_to_drop = ['remain_budget', 'avg_pacer_all_now', 'avg_pacer_last3_now',
                'avg_ci_mean_all_now', 'avg_ci_mean_last3_now',
                'avg_pvalue_mean_all_now', 'avg_pvalue_mean_last3_now',
                'avg_conversion_per_pv_all_now', 'avg_conversion_per_pv_last3_now',
                'avg_win_prob_all_now', 'avg_win_prob_last3_now',
                'pvalue_mean_agg', 'pv_num_agg', 'last_3_volume_now', 'historical_volume_now',
                'avg_cost_all_now', 'avg_cost_last3_now', 'cumulative_cost_ratio_now']
training_df.drop(columns=[c for c in cols_to_drop if c in training_df.columns], inplace=True)

output_path = os.path.join(OUTPUT_DIR, "train_data_rlData.csv")
training_df.to_csv(output_path, index=False)
print(f"\nRL数据已保存: {output_path}")
print(f"总行数: {len(training_df)}")

# 保存有效adgroup_ids供env_data过滤
adids_file = os.path.join(OUTPUT_DIR, "valid_adgroup_ids.txt")
with open(adids_file, 'w') as f:
    for aid in sorted(final_valid_adids):
        f.write(f"{aid}\n")
print(f"有效广告组ID已保存: {adids_file} ({len(final_valid_adids)} 个)")
