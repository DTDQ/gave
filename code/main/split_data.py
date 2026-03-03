#!/usr/bin/env python3
"""
split_data.py - 仅分割训练集和验证集
- 输入：原始CSV数据（包含完整轨迹）
- 输出：train/ 和 val/ 目录下的npy轨迹文件
- 不处理测试集（测试集已单独提供）
"""

import numpy as np
import pandas as pd
import os
import sys
import json
import argparse
import ast
import shutil


def parse_arguments():
    """解析命令行参数（仅保留训练/验证相关）"""
    parser = argparse.ArgumentParser(description='RL数据分割（仅训练/验证）')
    
    parser.add_argument('--input_csv', type=str, required=True,
                       help='输入数据路径（包含训练+验证的所有轨迹）')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录，将创建train/和val/子目录')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例，默认0.8')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='验证集比例，默认0.2（需与train_ratio之和为1）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--min_traj_length', type=int, default=3,
                       help='最小轨迹长度')
    parser.add_argument('--max_traj_length', type=int, default=100,
                       help='最大轨迹长度')
    
    return parser.parse_args()


def parse_state_string(state_str):
    """解析state字符串为numpy数组"""
    if isinstance(state_str, str):
        try:
            if state_str.startswith('(') and state_str.endswith(')'):
                state_str = state_str[1:-1]
                values = [float(x.strip()) for x in state_str.split(',')]
                return np.array(values, dtype=np.float32)
            else:
                parsed = ast.literal_eval(state_str)
                if isinstance(parsed, (tuple, list)):
                    return np.array(parsed, dtype=np.float32)
        except:
            pass
    return np.zeros(16, dtype=np.float32)


def parse_next_state(next_state_str):
    """解析next_state字符串，若无效则返回None"""
    if pd.isna(next_state_str) or next_state_str is None:
        return None
    if isinstance(next_state_str, str):
        if next_state_str.lower() == 'none' or next_state_str == '':
            return None
        try:
            return parse_state_string(next_state_str)
        except:
            return None
    return None


def create_episode_replay_buffer_format(trajectories):
    """将轨迹列表转换为EpisodeReplayBuffer兼容的npy字典格式"""
    buffer_data = []
    for traj in trajectories:
        states_array = np.array(traj['states'], dtype=np.float32)
        if len(states_array.shape) == 1:
            state_dim = states_array.shape[0]
            states = states_array.reshape(1, -1)
        else:
            state_dim = states_array.shape[1]
            states = states_array
        
        next_states_processed = []
        for ns in traj['next_states']:
            if ns is None:
                next_states_processed.append(np.zeros(state_dim, dtype=np.float32))
            else:
                if len(ns.shape) == 0 or ns.shape[0] != state_dim:
                    next_states_processed.append(np.zeros(state_dim, dtype=np.float32))
                else:
                    next_states_processed.append(ns)
        
        next_states = np.array(next_states_processed, dtype=np.float32)
        if len(next_states.shape) == 1:
            next_states = next_states.reshape(-1, state_dim)
        
        episode = {
            'states': states,
            'actions': np.array(traj['actions'], dtype=np.float32).reshape(-1, 1),
            'rewards': np.array(traj['rewards'], dtype=np.float32).reshape(-1, 1),
            'dones': np.array(traj['dones'], dtype=np.float32),
            'next_states': next_states,
            'timesteps': np.array(traj['timesteps'], dtype=np.int32),
            'metadata': traj['metadata']
        }
        buffer_data.append(episode)
    return buffer_data


def create_trajectories_from_df(df, min_length=3, max_length=100):
    """从DataFrame构建轨迹列表"""
    trajectories = []
    grouped = df.groupby(['deliveryPeriodIndex', 'advertiserNumber'])
    
    for (period_idx, adgroup_id), group in grouped:
        group = group.sort_values('timeStepIndex')
        states = []
        next_states = []
        for _, row in group.iterrows():
            state = parse_state_string(row['state'])
            next_state = parse_next_state(row['next_state'])
            states.append(state)
            next_states.append(next_state)
        
        traj_length = len(states)
        if traj_length < min_length or traj_length > max_length:
            continue
        
        trajectory = {
            'states': states,
            'actions': group['action'].values.astype(np.float32).tolist(),
            'rewards': group['reward'].values.astype(np.float32).tolist(),
            'next_states': next_states,
            'dones': group['done'].values.astype(np.int32).tolist(),
            'timesteps': group['timeStepIndex'].values.astype(np.int32).tolist(),
            'metadata': {
                'deliveryPeriodIndex': int(period_idx),
                'advertiserNumber': int(adgroup_id),
                'budget': float(group['budget'].iloc[0]),
                'CPAConstraint': float(group['CPAConstraint'].iloc[0]),
                'realAllCost': float(group['realAllCost'].iloc[0]),
                'realAllConversion': float(group['realAllConversion'].iloc[0]),
                'length': traj_length
            }
        }
        trajectories.append(trajectory)
    
    print(f"共创建 {len(trajectories)} 条轨迹")
    return trajectories


def split_train_val(trajectories, train_ratio, val_ratio, seed):
    """仅分割训练集和验证集，无测试集"""
    np.random.seed(seed)
    indices = np.random.permutation(len(trajectories))
    
    n_total = len(trajectories)
    n_train = int(n_total * train_ratio)
    
    # 确保比例总和为1（如果因浮点误差导致索引越界，微调）
    if n_train + int(n_total * val_ratio) != n_total:
        n_val = n_total - n_train
    else:
        n_val = int(n_total * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    
    train_trajs = [trajectories[i] for i in train_idx]
    val_trajs = [trajectories[i] for i in val_idx]
    
    print(f"分割结果: 训练集 {len(train_trajs)} 条轨迹, 验证集 {len(val_trajs)} 条轨迹")
    return train_trajs, val_trajs


def save_episode_data(output_dir, train_episodes, val_episodes, config):
    """保存训练集和验证集数据，不创建test目录"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 清理并重新创建子目录
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    for dir_path in [train_dir, val_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)
    
    # 保存训练集
    for i, episode in enumerate(train_episodes):
        file_path = os.path.join(train_dir, f'episode_{i:04d}.npy')
        np.save(file_path, episode)
    
    # 保存验证集
    for i, episode in enumerate(val_episodes):
        file_path = os.path.join(val_dir, f'episode_{i:04d}.npy')
        np.save(file_path, episode)
    
    # 保存配置信息
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # 统计信息
    stats = {
        'total_episodes': len(train_episodes) + len(val_episodes),
        'train_episodes': len(train_episodes),
        'val_episodes': len(val_episodes),
        'train_steps': sum(ep['states'].shape[0] for ep in train_episodes) if train_episodes else 0,
        'val_steps': sum(ep['states'].shape[0] for ep in val_episodes) if val_episodes else 0,
        'state_dim': train_episodes[0]['states'].shape[1] if train_episodes else 0,
        'act_dim': 1,
    }
    
    stats_path = os.path.join(output_dir, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def main():
    args = parse_arguments()
    
    # 检查比例和是否为1
    if abs(args.train_ratio + args.val_ratio - 1.0) > 1e-6:
        print(f"错误: train_ratio + val_ratio = {args.train_ratio + args.val_ratio} ≠ 1")
        print("请确保两个比例之和为1（例如 0.8 + 0.2）")
        return 1
    
    print("开始数据分割（仅训练/验证）...")
    
    # 读取数据
    df = pd.read_csv(args.input_csv)
    print(f"数据形状: {df.shape}")
    
    # 构建轨迹
    trajectories = create_trajectories_from_df(
        df, args.min_traj_length, args.max_traj_length
    )
    if not trajectories:
        print("错误: 没有可用的轨迹!")
        return 1
    
    # 分割训练/验证
    train_trajs, val_trajs = split_train_val(
        trajectories, args.train_ratio, args.val_ratio, args.seed
    )
    
    # 转换为EpisodeReplayBuffer格式
    train_episodes = create_episode_replay_buffer_format(train_trajs)
    val_episodes = create_episode_replay_buffer_format(val_trajs)
    
    # 保存配置
    config = {
        'input_file': args.input_csv,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'seed': args.seed,
        'min_traj_length': args.min_traj_length,
        'max_traj_length': args.max_traj_length,
        'state_dim': train_episodes[0]['states'].shape[1] if train_episodes else 16,
        'act_dim': 1,
    }
    
    # 保存数据
    stats = save_episode_data(args.output_dir, train_episodes, val_episodes, config)
    
    print(f"数据分割完成，保存到 {args.output_dir}")
    print(f"总轨迹数: {stats['total_episodes']}")
    print(f"训练集: {stats['train_episodes']} 条轨迹, {stats['train_steps']} 个时间步")
    print(f"验证集: {stats['val_episodes']} 条轨迹, {stats['val_steps']} 个时间步")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())