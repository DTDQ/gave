import torch
from torch.utils.data import Dataset
import pandas as pd
import ast
import numpy as np
import pickle
import random


def getScore(budget, cpa_cons, states, all_reward):
    beta = 2
    curr_cost = budget * (1 - states[:, 1]).reshape(-1,1)
    curr_all_reward = all_reward.reshape(-1,1)
    curr_cpa = curr_cost / (curr_all_reward + 1e-10)
    curr_coef = cpa_cons / (curr_cpa + 1e-10)
    curr_penalty = pow(curr_coef, beta)
    curr_penalty = np.where(curr_penalty > 1.0, 1.0, curr_coef)
    curr_score = curr_penalty * curr_all_reward

    return curr_score

class EpisodeReplayBuffer(Dataset):
    def __init__(self, device, state_dim, act_dim, data_path, scale=2000, K=20, if_test=False):
        self.device = device
        super(EpisodeReplayBuffer, self).__init__()
        self.scale = scale

        self.state_dim = state_dim
        self.act_dim = act_dim
        training_data = pd.read_csv(data_path)

        def safe_literal_eval(val):
            if pd.isna(val):
                return val
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                print(ValueError)
                return val

        training_data["state"] = training_data["state"].apply(safe_literal_eval)
        training_data["next_state"] = training_data["next_state"].apply(safe_literal_eval)
        self.trajectories = training_data

        (self.period, self.adid, self.states, self.rewards, self.actions, self.returns, self.traj_lens, self.dones,
         self.next_states, self.budget, self.cpacons) = [], [], [], [], [], [], [], [], [], [], []
        state = []
        reward = []
        action = []
        dones = []
        next_state = []
        budget = []
        cpacons = []
        period = []
        adid = []
        for index, row in self.trajectories.iterrows():
            period.append(row['deliveryPeriodIndex'])
            adid.append(row['advertiserNumber'])
            state.append(row["state"])
            reward.append(row['reward'])
            action.append(row["action"])
            dones.append(row["done"])
            next_state.append(row["next_state"])
            budget.append(row["budget"])
            cpacons.append(row["CPAConstraint"])
            if row["done"]:
                if len(state) != 1:
                    self.period.append(period[0])
                    self.adid.append(adid[0])
                    self.states.append(np.array(state))
                    self.rewards.append(np.expand_dims(np.array(reward), axis=1))
                    self.actions.append(np.expand_dims(np.array(action), axis=1))
                    self.returns.append(sum(reward))
                    self.traj_lens.append(len(state))
                    self.dones.append(np.array(dones))
                    if next_state[-1] is None:          # 新增条件判断
                        next_state[-1] = next_state[-2] # 仅当 None 时替换
                    self.next_states.append(np.array(next_state))
                    self.budget.append(np.expand_dims(np.array(budget), axis=1))
                    self.cpacons.append(np.expand_dims(np.array(cpacons), axis=1))
                period = []
                adid = []
                state = []
                reward = []
                action = []
                dones = []
                next_state = []
                budget = []
                cpacons = []
        self.traj_lens, self.returns = np.array(self.traj_lens), np.array(self.returns)

        tmp_states = np.concatenate(self.states, axis=0)
        self.state_mean, self.state_std = np.mean(tmp_states, axis=0), np.std(tmp_states, axis=0) + 1e-6

        self.trajectories = []
        for i in range(len(self.states)):
            all_reward = np.zeros(1 + len(self.rewards[i]))
            all_reward[0] = 0
            for ind in range(1, len(all_reward)):
                all_reward[ind] = all_reward[ind - 1] + self.rewards[i][ind - 1]
            s_rtg = np.concatenate((self.states[i], self.next_states[i][-1].reshape((1,-1))),axis=0)
            curr_score = getScore(self.budget[i][0], self.cpacons[i][0], s_rtg, all_reward)
            # curr_score[t] equals final_score - score_if_stop_at_t, which means it would get how manny scores from step t to the end.
            curr_score = curr_score[-1]-curr_score
            self.trajectories.append(
                {"observations": self.states[i], "actions": self.actions[i], "rewards": self.rewards[i],
                 "dones": self.dones[i], "next_states": self.next_states[i], "budget": self.budget[i],
                 "cpacons": self.cpacons[i], "all_reward": all_reward, "curr_score": curr_score,
                 "final_score": curr_score[0], "ad_id": self.adid[i], "period": self.period[i]})

        self.K = K
        self.pct_traj = 1.

        num_timesteps = sum(self.traj_lens)
        num_timesteps = max(int(self.pct_traj * num_timesteps), 1)
        sorted_inds = np.argsort(self.returns)  # lowest to highest for training

        num_trajectories = 1
        timesteps = self.traj_lens[sorted_inds[-1]]
        ind = len(self.trajectories) - 2
        while ind >= 0 and timesteps + self.traj_lens[sorted_inds[ind]] <= num_timesteps:
            timesteps += self.traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1

        if if_test:
            self.sorted_inds = np.arange(len(self.trajectories))
        else:
            self.sorted_inds = sorted_inds[-num_trajectories:]

        self.p_sample = self.traj_lens[self.sorted_inds] / sum(self.traj_lens[self.sorted_inds])

    def __getitem__(self, index, if_test=False):
        if if_test:
            return 0

        traj = self.trajectories[int(self.sorted_inds[index])]
        start_t = random.randint(0, max(traj['rewards'].shape[0] -self.K, 0))

        s = traj['observations'][start_t: start_t + self.K]
        a = traj['actions'][start_t: start_t + self.K]
        r = traj['rewards'][start_t: start_t + self.K].reshape(-1, 1)
        sn = traj['next_states'][start_t: start_t + self.K]
        all_reward = traj['all_reward'][start_t: start_t + self.K+1].reshape(-1, 1)
        # curr_score[t] is the return-to-go at state t, so we need K+1 of them
        # to align with K states/actions in Decision Transformer.
        curr_score = traj['curr_score'][start_t: start_t + self.K+1].reshape(-1, 1)
        if 'terminals' in traj:
            d = traj['terminals'][start_t: start_t + self.K]
        else:
            d = traj['dones'][start_t: start_t + self.K]
        timesteps = np.arange(start_t, start_t + s.shape[0])

        tlen = s.shape[0]

        s = np.concatenate([np.zeros((self.K - tlen, self.state_dim)), s], axis=0)
        a = np.concatenate([np.ones((self.K - tlen, self.act_dim)) * -10., a], axis=0)
        r = np.concatenate([np.zeros((self.K - tlen, 1)), r], axis=0)
        sn = np.concatenate([np.zeros((self.K - tlen, self.state_dim)), sn], axis=0)
        d = np.concatenate([np.ones((self.K - tlen)) * 2, d], axis=0)
        all_reward = np.concatenate([np.zeros((self.K - tlen, 1)), all_reward], axis=0)
        curr_score = np.concatenate([np.zeros((self.K - tlen, 1)), curr_score], axis=0)
        timesteps = np.concatenate([np.zeros((self.K - tlen)), timesteps], axis=0)
        mask = np.concatenate([np.zeros((self.K - tlen)), np.ones((tlen))], axis=0)
        s = (s - self.state_mean) / self.state_std
        r = r / self.scale
        sn = (sn - self.state_mean) / self.state_std
        all_reward = all_reward / self.scale
        curr_score = curr_score / self.scale

        s = torch.from_numpy(s).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(a).to(dtype=torch.float32, device=self.device)
        r = torch.from_numpy(r).to(dtype=torch.float32, device=self.device)
        sn = torch.from_numpy(sn).to(dtype=torch.float32, device=self.device)
        d = torch.from_numpy(d).to(dtype=torch.long, device=self.device)
        all_reward = torch.from_numpy(all_reward).to(dtype=torch.float32, device=self.device)
        curr_score = torch.from_numpy(curr_score).to(dtype=torch.float32, device=self.device)
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.long, device=self.device)
        mask = torch.from_numpy(mask).to(device=self.device)
        return s, a, r, d, all_reward, curr_score, timesteps, mask, sn

    def discount_cumsum(self, x, gamma=1.):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum
class ValidationDataset(Dataset):
    """
    验证数据集 - 严格对齐 EpisodeReplayBuffer 的处理逻辑
    关键修复：
      1. 使用训练集的 state_mean/std 归一化
      2. 正确计算 all_reward 和 curr_score（与训练集完全一致）
      3. 复现 next_state 的特殊处理（done 时用倒数第二状态）
      4. 保持相同的填充规则和掩码逻辑
    """
    def __init__(self, 
                 device, 
                 state_dim, 
                 data_path, 
                 n_ctx=20, 
                 scale=2000,
                 state_mean=None, 
                 state_std=None):
        self.device = device
        self.state_dim = state_dim
        self.n_ctx = n_ctx
        self.scale = scale
        
        # 强制要求传入训练集归一化参数
        if state_mean is None or state_std is None:
            raise ValueError(
                "ValidationDataset 必须传入训练集的 state_mean 和 state_std！\n"
                "请从 EpisodeReplayBuffer 获取: replay_buffer.state_mean/std"
            )
        self.state_mean = state_mean
        self.state_std = state_std
        
        # 加载CSV
        self.df = pd.read_csv(data_path)
        
        # 按轨迹分组（按 deliveryPeriodIndex + advertiserNumber）
        self.trajectories = []
        grouped = self.df.groupby(['deliveryPeriodIndex', 'advertiserNumber'])
        
        for (period_idx, adgroup_id), group in grouped:
            group = group.sort_values('timeStepIndex').reset_index(drop=True)
            
            # 解析状态
            states = []
            for state_str in group['state']:
                state = self._parse_state(state_str)
                states.append(state)
            
            # 解析 next_state（处理 None 值）
            next_states = []
            for ns_str in group['next_state']:
                if pd.isna(ns_str) or ns_str == '' or ns_str == 'None':
                    next_states.append(None)
                else:
                    next_states.append(self._parse_state(ns_str))
            
            # 特殊处理：最后一个 next_state 为 None 时，用倒数第二个状态代替（与训练集一致）
            if len(next_states) > 0 and next_states[-1] is None:
                if len(next_states) >= 2:
                    next_states[-1] = next_states[-2].copy()
                else:
                    next_states[-1] = np.zeros(self.state_dim, dtype=np.float32)
            
            # 构建轨迹
            traj_length = len(states)
            if traj_length < 2:  # 至少需要2步（避免除零错误）
                continue
                
            traj = {
                'states': np.array(states, dtype=np.float32),
                'actions': group['action'].values.astype(np.float32),
                'rewards': group['reward'].values.astype(np.float32),
                'dones': group['done'].values.astype(np.float32),
                'next_states': np.array(next_states, dtype=np.float32),
                'length': traj_length,
                'metadata': {
                    'deliveryPeriodIndex': int(period_idx),
                    'advertiserNumber': int(adgroup_id),
                    'budget': float(group['budget'].iloc[0]),
                    'CPAConstraint': float(group['CPAConstraint'].iloc[0]),
                    'realAllCost': float(group['realAllCost'].iloc[0]),
                    'realAllConversion': float(group['realAllConversion'].iloc[0]),
                }
            }
            
            # 计算 all_reward 和 curr_score（与训练集完全一致）
            traj = self._compute_trajectory_scores(traj)
            self.trajectories.append(traj)
        
        print(f"✅ 验证集加载完成: {len(self.trajectories)} 条轨迹 | "
              f"归一化参数来源: 训练集统计量")
    
    def _parse_state(self, state_str):
        """安全解析状态字符串"""
        if pd.isna(state_str) or state_str == '' or state_str == 'None':
            return np.zeros(self.state_dim, dtype=np.float32)
        
        try:
            if isinstance(state_str, str):
                # 处理括号 (1.0, 2.0, ...) 或 [1.0, 2.0, ...]
                state_str = state_str.strip().replace('(', '[').replace(')', ']')
                values = ast.literal_eval(state_str)
                if isinstance(values, (list, tuple)) and len(values) == self.state_dim:
                    return np.array(values, dtype=np.float32)
            elif isinstance(state_str, np.ndarray):
                return state_str.astype(np.float32)
        except Exception as e:
            warnings.warn(f"状态解析失败: {state_str} | 错误: {e}")
        
        return np.zeros(self.state_dim, dtype=np.float32)
    
    def _compute_trajectory_scores(self, traj):
        """
        复现 EpisodeReplayBuffer 中的分数计算逻辑
        返回: 增强后的轨迹字典（含 all_reward, curr_score, final_score）
        """
        rewards = traj['rewards']
        budget = traj['metadata']['budget']
        cpa_cons = traj['metadata']['CPAConstraint']
        states = traj['states']
        next_states = traj['next_states']
        
        # 1. 计算 all_reward: [0, r0, r0+r1, ..., sum(rewards)]
        all_reward = np.zeros(len(rewards) + 1)
        for i in range(1, len(all_reward)):
            all_reward[i] = all_reward[i-1] + rewards[i-1]
        
        # 2. 构造 s_rtg: states + last next_state (长度 = len(states) + 1)
        s_rtg = np.concatenate([states, next_states[-1].reshape(1, -1)], axis=0)
        
        # 3. 计算 curr_score (基于预算/CPA约束的动态分数)
        curr_score = getScore(budget, cpa_cons, s_rtg, all_reward)
        
        # 4. 转换为 return-to-go: final_score - score_if_stop_at_t
        curr_score = curr_score[-1] - curr_score
        
        # 5. 添加到轨迹
        traj['all_reward'] = all_reward.reshape(-1, 1)  # 形状: (T+1, 1)
        traj['curr_score'] = curr_score.reshape(-1, 1)  # 形状: (T+1, 1)
        traj['final_score'] = float(curr_score[0])
        
        return traj
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        """
        返回单条轨迹的验证样本（从轨迹开头开始，固定窗口）
        输出格式与 EpisodeReplayBuffer.__getitem__ 完全一致:
          (states, actions, rewards, dones, all_reward, curr_score, timesteps, mask, next_states)
        """
        traj = self.trajectories[idx]
        traj_len = traj['length']
        K = self.n_ctx
        
        # 固定从轨迹开头采样（验证集标准做法）
        start_t = 0
        actual_len = min(traj_len, K)  # 实际有效长度
        
        # 计算需要填充的长度（前面填充）
        pad_len = K - actual_len
        
        # ========== 提取原始数据 ==========
        # 状态/动作/奖励/终止信号（长度 = actual_len）
        s = traj['states'][start_t:start_t + actual_len]
        a = traj['actions'][start_t:start_t + actual_len]
        r = traj['rewards'][start_t:start_t + actual_len]
        d = traj['dones'][start_t:start_t + actual_len]
        sn = traj['next_states'][start_t:start_t + actual_len]
        
        # all_reward 和 curr_score 长度 = actual_len + 1（需要 K+1 个）
        all_r = traj['all_reward'][start_t:start_t + actual_len + 1].flatten()
        curr_s = traj['curr_score'][start_t:start_t + actual_len + 1].flatten()
        
        # 时间步
        timesteps = np.arange(start_t, start_t + actual_len)
        
        # ========== 填充到 K 长度（前面填充）==========
        # 状态: 前面填充0向量
        if pad_len > 0:
            s = np.concatenate([np.zeros((pad_len, self.state_dim)), s], axis=0)
            sn = np.concatenate([np.zeros((pad_len, self.state_dim)), sn], axis=0)
            a = np.concatenate([np.ones(pad_len) * -10.0, a], axis=0)  # 动作用-10填充
            r = np.concatenate([np.zeros(pad_len), r], axis=0)
            d = np.concatenate([np.ones(pad_len) * 2, d], axis=0)  # done用2填充（表示padding）
            timesteps = np.concatenate([np.zeros(pad_len), timesteps], axis=0)
        
        # all_reward/curr_score 填充到 K+1 长度
        if pad_len > 0:
            all_r = np.concatenate([np.zeros(pad_len), all_r], axis=0)
            curr_s = np.concatenate([np.zeros(pad_len), curr_s], axis=0)
        # 确保长度为 K+1
        if len(all_r) < K + 1:
            all_r = np.concatenate([all_r, np.zeros(K + 1 - len(all_r))], axis=0)
            curr_s = np.concatenate([curr_s, np.zeros(K + 1 - len(curr_s))], axis=0)
        all_r = all_r[:K + 1]
        curr_s = curr_s[:K + 1]
        
        # 掩码: 前面 pad_len 个为0，后面 actual_len 个为1
        mask = np.zeros(K)
        mask[pad_len:] = 1.0
        
        # ========== 归一化（使用训练集统计量）==========
        s = (s - self.state_mean) / self.state_std
        sn = (sn - self.state_mean) / self.state_std
        r = r / self.scale
        all_r = all_r / self.scale
        curr_s = curr_s / self.scale
        
        # ========== 转换为 Tensor ==========
        s = torch.from_numpy(s).float()
        a = torch.from_numpy(a).float().unsqueeze(-1)  # (K, 1)
        r = torch.from_numpy(r).float().unsqueeze(-1)  # (K, 1)
        d = torch.from_numpy(d).long()
        sn = torch.from_numpy(sn).float()
        all_r = torch.from_numpy(all_r).float().unsqueeze(-1)  # (K+1, 1)
        curr_s = torch.from_numpy(curr_s).float().unsqueeze(-1)  # (K+1, 1)
        timesteps = torch.from_numpy(timesteps).long()
        mask = torch.from_numpy(mask).float()
        
        return s, a, r, d, all_r, curr_s, timesteps, mask, sn

# ==================== 调试用例 ====================
if __name__ == "__main__":
    # 示例：如何正确初始化 ValidationDataset
    from bidding_train_env.baseline.dt.utils import EpisodeReplayBuffer
    
    # 1. 先创建训练集 ReplayBuffer 获取归一化参数
    train_buffer = EpisodeReplayBuffer(
        device="cpu",
        state_dim=16,
        act_dim=1,
        data_path="data/trajectory/train_data.csv",  # 替换为实际路径
        K=20
    )
    
    # 2. 用训练集参数初始化验证集
    val_dataset = ValidationDataset(
        device="cpu",
        state_dim=16,
        data_path="data/trajectory/val_data.csv",  # 替换为实际路径
        n_ctx=20,
        state_mean=train_buffer.state_mean,
        state_std=train_buffer.state_std
    )
    
    # 3. 测试取样
    print(f"\n✅ 验证集样本数: {len(val_dataset)}")
    sample = val_dataset[0]
    print(f"✅ 样本格式验证:")
    print(f"   states shape: {sample[0].shape} | dtype: {sample[0].dtype}")
    print(f"   actions shape: {sample[1].shape}")
    print(f"   all_reward shape: {sample[4].shape} (应为 [21, 1])")
    print(f"   curr_score shape: {sample[5].shape} (应为 [21, 1])")
    print(f"   mask sum: {sample[7].sum().item()} (有效步数)")