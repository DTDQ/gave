#!/usr/bin/env python3
"""
split_data.py - 分割训练集和验证集，直接输出 CSV 格式（优化版）
- 输入：原始CSV数据（包含完整轨迹）
- 输出：train.csv 和 val.csv（以及可选的 npy 轨迹文件）
"""

import numpy as np
import pandas as pd
import os
import sys
import json
import argparse
import shutil


def parse_arguments():
    parser = argparse.ArgumentParser(description='RL数据分割（仅训练/验证）')
    parser.add_argument('--input_csv', type=str, required=True,
                        help='输入数据路径（包含训练+验证的所有轨迹）')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录，将创建 train.csv 和 val.csv')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='验证集比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--min_traj_length', type=int, default=3,
                        help='最小轨迹长度')
    parser.add_argument('--max_traj_length', type=int, default=100,
                        help='最大轨迹长度')
    parser.add_argument('--save_npy', action='store_true',
                        help='同时保存 npy 格式轨迹（兼容旧版，较慢）')
    return parser.parse_args()


def check_trajectory_complete(timesteps, min_len=3, max_len=100):
    """检查轨迹完整性：长度在范围内，从1开始，连续递增"""
    if len(timesteps) < min_len or len(timesteps) > max_len:
        return False
    if timesteps[0] != 1:
        return False
    # 检查连续递增
    return np.all(np.diff(timesteps) == 1)


def save_csv_from_indices(df, indices_list, output_path):
    """根据索引区间列表从 DataFrame 中提取行并保存为 CSV"""
    if not indices_list:
        return
    # 合并所有区间（假设区间不重叠）
    rows = []
    for start, end in indices_list:
        rows.append(df.iloc[start:end])
    result_df = pd.concat(rows, ignore_index=True)
    result_df.to_csv(output_path, index=False)
    print(f"已保存: {output_path} ({len(result_df)} 行)")


def save_npy_trajectories(df, indices_list, output_dir, prefix):
    """
    根据索引区间列表，将每个轨迹保存为 npy 文件（兼容旧版 EpisodeReplayBuffer）
    注意：此函数会解析 state/next_state，速度较慢，仅在 --save_npy 时调用
    """
    os.makedirs(output_dir, exist_ok=True)
    state_dim = 16

    # 快速解析 state 的函数（向量化）
    def parse_state_series(series):
        # 去除括号，分割成多列，转换为 float
        stripped = series.str.strip('()')
        split = stripped.str.split(',', expand=True).astype(float)
        return split.values

    # 快速解析 next_state（处理空值）
    def parse_next_series(series):
        next_states = np.zeros((len(series), state_dim), dtype=np.float32)
        valid_mask = series.notna() & (series != 'None') & (series != '')
        if valid_mask.any():
            valid_series = series[valid_mask].str.strip('()')
            valid_split = valid_series.str.split(',', expand=True).astype(float)
            next_states[valid_mask] = valid_split.values
        return next_states

    for idx, (start, end) in enumerate(indices_list):
        group = df.iloc[start:end].copy()
        # 解析状态
        states = parse_state_series(group['state'])
        next_states = parse_next_series(group['next_state'])
        # 处理最后一个 next_state 可能为全零（原为 None）的情况
        if np.all(next_states[-1] == 0) and len(next_states) >= 2:
            next_states[-1] = next_states[-2].copy()

        episode = {
            'states': states,
            'actions': group['action'].values.astype(np.float32).reshape(-1, 1),
            'rewards': group['reward'].values.astype(np.float32).reshape(-1, 1),
            'dones': group['done'].values.astype(np.float32),
            'next_states': next_states,
            'timesteps': group['timeStepIndex'].values.astype(np.int32),
            'metadata': {
                'deliveryPeriodIndex': int(group['deliveryPeriodIndex'].iloc[0]),
                'advertiserNumber': int(group['advertiserNumber'].iloc[0]),
                'budget': float(group['budget'].iloc[0]),
                'CPAConstraint': float(group['CPAConstraint'].iloc[0]),
                'realAllCost': float(group['realAllCost'].iloc[0]),
                'realAllConversion': float(group['realAllConversion'].iloc[0]),
                'length': len(group)
            }
        }
        np.save(os.path.join(output_dir, f'{prefix}_episode_{idx:04d}.npy'), episode)
    print(f"已保存 {len(indices_list)} 个 npy 轨迹到 {output_dir}")


def main():
    args = parse_arguments()

    if abs(args.train_ratio + args.val_ratio - 1.0) > 1e-6:
        print(f"错误: train_ratio + val_ratio = {args.train_ratio + args.val_ratio} ≠ 1")
        return 1

    print("开始数据分割（优化版）...")
    # 读取 CSV（指定类型以加速）
    dtype = {
        'deliveryPeriodIndex': int,
        'advertiserNumber': int,
        'budget': float,
        'CPAConstraint': float,
        'timeStepIndex': int,
        'state': str,
        'action': float,
        'reward': float,
        'done': int,
        'next_state': str,
    }
    df = pd.read_csv(args.input_csv, dtype=dtype)
    print(f"原始数据形状: {df.shape}")

    # 按广告组和时间排序，确保组内行连续
    df = df.sort_values(['advertiserNumber', 'timeStepIndex']).reset_index(drop=True)

    # 获取所有合法轨迹的索引区间
    group_indices = []  # 每个元素为 (period, adid, start, end)
    for (period, adid), group in df.groupby(['deliveryPeriodIndex', 'advertiserNumber'], sort=False):
        timesteps = group['timeStepIndex'].values
        if check_trajectory_complete(timesteps, args.min_traj_length, args.max_traj_length):
            start = group.index[0]
            end = group.index[-1] + 1
            group_indices.append((period, adid, start, end))

    print(f"找到 {len(group_indices)} 条合法轨迹")
    if not group_indices:
        print("错误: 没有可用的合法轨迹!")
        return 1

    # 随机打乱轨迹
    np.random.seed(args.seed)
    np.random.shuffle(group_indices)

    # 分割
    n_total = len(group_indices)
    n_train = int(n_total * args.train_ratio)
    train_indices = group_indices[:n_train]
    val_indices = group_indices[n_train:]

    print(f"分割结果: 训练集 {len(train_indices)} 条轨迹, 验证集 {len(val_indices)} 条轨迹")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存 CSV（只需起始、结束位置）
    train_csv_path = os.path.join(args.output_dir, 'train.csv')
    val_csv_path = os.path.join(args.output_dir, 'val.csv')
    save_csv_from_indices(df, [(s, e) for _, _, s, e in train_indices], train_csv_path)
    save_csv_from_indices(df, [(s, e) for _, _, s, e in val_indices], val_csv_path)

    # 可选：保存 npy 轨迹
    if args.save_npy:
        npy_train_dir = os.path.join(args.output_dir, 'train')
        npy_val_dir = os.path.join(args.output_dir, 'val')
        save_npy_trajectories(df, [(s, e) for _, _, s, e in train_indices], npy_train_dir, 'train')
        save_npy_trajectories(df, [(s, e) for _, _, s, e in val_indices], npy_val_dir, 'val')

    # 保存配置和统计
    config = {
        'input_file': args.input_csv,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'seed': args.seed,
        'min_traj_length': args.min_traj_length,
        'max_traj_length': args.max_traj_length,
        'state_dim': 16,
        'act_dim': 1,
    }
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    stats = {
        'total_episodes': len(group_indices),
        'train_episodes': len(train_indices),
        'val_episodes': len(val_indices),
        'train_steps': int(sum(e - s for _, _, s, e in train_indices)),
        'val_steps': int(sum(e - s for _, _, s, e in val_indices)),
        'state_dim': 16,
        'act_dim': 1,
    }
    stats_path = os.path.join(args.output_dir, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"统计信息: {stats}")

    print(f"\n所有数据已保存至: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
