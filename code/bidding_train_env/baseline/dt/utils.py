import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import warnings
import os


def getScore(budget, cpa_cons, states, all_reward):
    beta = 2
    curr_cost = budget * (1 - states[:, 1]).reshape(-1, 1)
    curr_all_reward = all_reward.reshape(-1, 1)
    curr_cpa = curr_cost / (curr_all_reward + 1e-10)
    curr_coef = cpa_cons / (curr_cpa + 1e-10)
    curr_penalty = pow(curr_coef, beta)
    curr_penalty = np.where(curr_penalty > 1.0, 1.0, curr_coef)
    curr_score = curr_penalty * curr_all_reward
    return curr_score


class EpisodeReplayBuffer(Dataset):
    """
    训练数据缓冲区，直接从 CSV 加载，使用向量化解析 state/next_state。
    """
    def __init__(self, device, state_dim, act_dim, data_path, scale=2000, K=20, if_test=False):
        self.device = device
        super(EpisodeReplayBuffer, self).__init__()
        self.scale = scale
        self.state_dim = state_dim
        self.act_dim = act_dim

        # 读取 CSV（指定 dtype 加速）
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
        df = pd.read_csv(data_path, dtype=dtype)

        # ========== 向量化解析 state ==========
        # 去除括号，按逗号分割，转换为 float
        state_series = df['state'].str.strip('()')
        state_split = state_series.str.split(',', expand=True).astype(float)
        states_np = state_split.values  # shape: (n_rows, state_dim)

        # ========== 向量化解析 next_state ==========
        # 先创建全零矩阵
        next_states_np = np.zeros((len(df), state_dim), dtype=np.float32)
        # 筛选出非空且有效的 next_state
        valid_mask = df['next_state'].notna() & (df['next_state'] != 'None') & (df['next_state'] != '')
        if valid_mask.any():
            valid_series = df.loc[valid_mask, 'next_state'].str.strip('()')
            valid_split = valid_series.str.split(',', expand=True).astype(float)
            next_states_np[valid_mask] = valid_split.values

        # 提取其他列
        periods = df['deliveryPeriodIndex'].values
        adids = df['advertiserNumber'].values
        actions = df['action'].values.astype(np.float32)
        rewards = df['reward'].values.astype(np.float32)
        dones = df['done'].values.astype(np.float32)
        budgets = df['budget'].values.astype(np.float32)
        cpacons = df['CPAConstraint'].values.astype(np.float32)
        timesteps = df['timeStepIndex'].values.astype(np.int32)

        # ========== 按 done 切分轨迹 ==========
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.budgets = []
        self.cpacons = []
        self.periods = []
        self.adids = []
        self.timesteps = []

        current_state = []
        current_action = []
        current_reward = []
        current_done = []
        current_next = []
        current_budget = []
        current_cpacon = []
        current_period = []
        current_adid = []
        current_timestep = []

        for i in range(len(df)):
            current_state.append(states_np[i])
            current_action.append(actions[i])
            current_reward.append(rewards[i])
            current_done.append(dones[i])
            current_next.append(next_states_np[i])
            current_budget.append(budgets[i])
            current_cpacon.append(cpacons[i])
            current_period.append(periods[i])
            current_adid.append(adids[i])
            current_timestep.append(timesteps[i])

            # 遇到 done 或最后一行时结束轨迹
            if dones[i] == 1 or i == len(df) - 1:
                if len(current_state) > 1:  # 至少 2 步
                    # 处理最后一个 next_state 可能为全零（原为 None）的情况
                    if np.all(current_next[-1] == 0) and len(current_next) >= 2:
                        current_next[-1] = current_next[-2].copy()

                    self.states.append(np.array(current_state))
                    self.actions.append(np.expand_dims(np.array(current_action), axis=1))
                    self.rewards.append(np.expand_dims(np.array(current_reward), axis=1))
                    self.dones.append(np.array(current_done))
                    self.next_states.append(np.array(current_next))
                    self.budgets.append(np.expand_dims(np.array(current_budget), axis=1))
                    self.cpacons.append(np.expand_dims(np.array(current_cpacon), axis=1))
                    self.periods.append(current_period[0])
                    self.adids.append(current_adid[0])

                # 重置
                current_state = []
                current_action = []
                current_reward = []
                current_done = []
                current_next = []
                current_budget = []
                current_cpacon = []
                current_period = []
                current_adid = []
                current_timestep = []

        self.traj_lens = np.array([len(s) for s in self.states])
        self.returns = np.array([np.sum(r) for r in self.rewards])

        # 计算归一化参数
        tmp_states = np.concatenate(self.states, axis=0)
        self.state_mean = np.mean(tmp_states, axis=0)
        self.state_std = np.std(tmp_states, axis=0) + 1e-6

        # 构建 trajectories 列表（包含 all_reward, curr_score 等）
        self.trajectories = []
        for i in range(len(self.states)):
            all_reward = np.zeros(1 + len(self.rewards[i]))
            all_reward[0] = 0
            for ind in range(1, len(all_reward)):
                all_reward[ind] = all_reward[ind - 1] + self.rewards[i][ind - 1]

            # s_rtg 包含 states 和最后一个 next_state
            s_rtg = np.concatenate(
                (self.states[i], self.next_states[i][-1].reshape((1, -1))),
                axis=0
            )
            curr_score = getScore(self.budgets[i][0], self.cpacons[i][0], s_rtg, all_reward)
            curr_score = curr_score[-1] - curr_score

            self.trajectories.append({
                "observations": self.states[i],
                "actions": self.actions[i],
                "rewards": self.rewards[i],
                "dones": self.dones[i],
                "next_states": self.next_states[i],
                "budget": self.budgets[i],
                "cpacons": self.cpacons[i],
                "all_reward": all_reward,
                "curr_score": curr_score,
                "final_score": curr_score[0],
                "ad_id": self.adids[i],
                "period": self.periods[i],
            })

        self.K = K
        self.pct_traj = 1.0

        # 选择用于训练的轨迹（基于 return 排序）
        num_timesteps = sum(self.traj_lens)
        num_timesteps = max(int(self.pct_traj * num_timesteps), 1)
        sorted_inds = np.argsort(self.returns)  # 升序

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

    def __len__(self):
        return len(self.sorted_inds)

    def __getitem__(self, index):
        traj = self.trajectories[int(self.sorted_inds[index])]
        start_t = np.random.randint(0, max(traj['rewards'].shape[0] - self.K, 1))

        s = traj['observations'][start_t: start_t + self.K]
        a = traj['actions'][start_t: start_t + self.K]
        r = traj['rewards'][start_t: start_t + self.K].reshape(-1, 1)
        sn = traj['next_states'][start_t: start_t + self.K]
        all_reward = traj['all_reward'][start_t: start_t + self.K + 1].reshape(-1, 1)
        curr_score = traj['curr_score'][start_t: start_t + self.K + 1].reshape(-1, 1)
        d = traj['dones'][start_t: start_t + self.K]
        timesteps = np.arange(start_t, start_t + s.shape[0])

        tlen = s.shape[0]

        # 填充到 K 长度
        s = np.concatenate([np.zeros((self.K - tlen, self.state_dim)), s], axis=0)
        a = np.concatenate([np.ones((self.K - tlen, self.act_dim)) * -10., a], axis=0)
        r = np.concatenate([np.zeros((self.K - tlen, 1)), r], axis=0)
        sn = np.concatenate([np.zeros((self.K - tlen, self.state_dim)), sn], axis=0)
        d = np.concatenate([np.ones((self.K - tlen)) * 2, d], axis=0)
        all_reward = np.concatenate([np.zeros((self.K - tlen, 1)), all_reward], axis=0)
        curr_score = np.concatenate([np.zeros((self.K - tlen, 1)), curr_score], axis=0)
        timesteps = np.concatenate([np.zeros((self.K - tlen)), timesteps], axis=0)
        mask = np.concatenate([np.zeros((self.K - tlen)), np.ones(tlen)], axis=0)

        # 归一化
        s = (s - self.state_mean) / self.state_std
        r = r / self.scale
        sn = (sn - self.state_mean) / self.state_std
        all_reward = all_reward / self.scale
        curr_score = curr_score / self.scale

        # 转为 Tensor
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
    验证数据集，使用训练集的归一化参数，严格对齐 EpisodeReplayBuffer 的处理逻辑。
    """
    def __init__(self, device, state_dim, data_path, n_ctx=20, scale=2000,
                 state_mean=None, state_std=None):
        self.device = device
        self.state_dim = state_dim
        self.n_ctx = n_ctx
        self.scale = scale

        if state_mean is None or state_std is None:
            raise ValueError(
                "ValidationDataset 必须传入训练集的 state_mean 和 state_std！"
            )
        self.state_mean = state_mean
        self.state_std = state_std

        df = pd.read_csv(data_path)

        # 按轨迹分组
        self.trajectories = []
        grouped = df.groupby(['deliveryPeriodIndex', 'advertiserNumber'])

        for (period_idx, adgroup_id), group in grouped:
            group = group.sort_values('timeStepIndex').reset_index(drop=True)

            # 解析状态（使用快速解析函数）
            states = []
            for state_str in group['state']:
                states.append(self._parse_state_fast(state_str))

            # 解析 next_state
            next_states = []
            for ns_str in group['next_state']:
                if pd.isna(ns_str) or ns_str == '' or ns_str == 'None':
                    next_states.append(None)
                else:
                    next_states.append(self._parse_state_fast(ns_str))

            # 处理最后一个 next_state
            if len(next_states) > 0 and next_states[-1] is None:
                if len(next_states) >= 2:
                    next_states[-1] = next_states[-2].copy()
                else:
                    next_states[-1] = np.zeros(self.state_dim, dtype=np.float32)

            traj_length = len(states)
            if traj_length < 2:
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

            # 计算 all_reward 和 curr_score
            traj = self._compute_trajectory_scores(traj)
            self.trajectories.append(traj)

        print(f"✅ 验证集加载完成: {len(self.trajectories)} 条轨迹")

    def _parse_state_fast(self, state_str):
        """快速解析状态字符串（不使用 literal_eval）"""
        if pd.isna(state_str) or state_str == '' or state_str == 'None':
            return np.zeros(self.state_dim, dtype=np.float32)

        # 去除括号，按逗号分割
        state_str = state_str.strip()
        if state_str.startswith('(') and state_str.endswith(')'):
            state_str = state_str[1:-1]
        elif state_str.startswith('[') and state_str.endswith(']'):
            state_str = state_str[1:-1]

        parts = state_str.split(',')
        if len(parts) != self.state_dim:
            return np.zeros(self.state_dim, dtype=np.float32)

        try:
            values = [float(p.strip()) for p in parts]
            return np.array(values, dtype=np.float32)
        except:
            return np.zeros(self.state_dim, dtype=np.float32)

    def _compute_trajectory_scores(self, traj):
        """计算轨迹的 all_reward 和 curr_score"""
        rewards = traj['rewards']
        budget = traj['metadata']['budget']
        cpa_cons = traj['metadata']['CPAConstraint']
        states = traj['states']
        next_states = traj['next_states']

        all_reward = np.zeros(len(rewards) + 1)
        for i in range(1, len(all_reward)):
            all_reward[i] = all_reward[i-1] + rewards[i-1]

        s_rtg = np.concatenate([states, next_states[-1].reshape(1, -1)], axis=0)
        curr_score = getScore(budget, cpa_cons, s_rtg, all_reward)
        curr_score = curr_score[-1] - curr_score

        traj['all_reward'] = all_reward.reshape(-1, 1)
        traj['curr_score'] = curr_score.reshape(-1, 1)
        traj['final_score'] = float(curr_score[0])
        return traj

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['length']
        K = self.n_ctx

        # 固定从开头采样
        start_t = 0
        actual_len = min(traj_len, K)
        pad_len = K - actual_len

        s = traj['states'][start_t:start_t + actual_len]
        a = traj['actions'][start_t:start_t + actual_len]
        r = traj['rewards'][start_t:start_t + actual_len]
        d = traj['dones'][start_t:start_t + actual_len]
        sn = traj['next_states'][start_t:start_t + actual_len]

        all_r = traj['all_reward'][start_t:start_t + actual_len + 1].flatten()
        curr_s = traj['curr_score'][start_t:start_t + actual_len + 1].flatten()

        timesteps = np.arange(start_t, start_t + actual_len)

        # 前面填充
        if pad_len > 0:
            s = np.concatenate([np.zeros((pad_len, self.state_dim)), s], axis=0)
            sn = np.concatenate([np.zeros((pad_len, self.state_dim)), sn], axis=0)
            a = np.concatenate([np.ones(pad_len) * -10.0, a], axis=0)
            r = np.concatenate([np.zeros(pad_len), r], axis=0)
            d = np.concatenate([np.ones(pad_len) * 2, d], axis=0)
            timesteps = np.concatenate([np.zeros(pad_len), timesteps], axis=0)

        if pad_len > 0:
            all_r = np.concatenate([np.zeros(pad_len), all_r], axis=0)
            curr_s = np.concatenate([np.zeros(pad_len), curr_s], axis=0)

        # 确保长度为 K+1
        if len(all_r) < K + 1:
            all_r = np.concatenate([all_r, np.zeros(K + 1 - len(all_r))], axis=0)
            curr_s = np.concatenate([curr_s, np.zeros(K + 1 - len(curr_s))], axis=0)
        all_r = all_r[:K + 1]
        curr_s = curr_s[:K + 1]

        mask = np.zeros(K)
        mask[pad_len:] = 1.0

        # 归一化
        s = (s - self.state_mean) / self.state_std
        sn = (sn - self.state_mean) / self.state_std
        r = r / self.scale
        all_r = all_r / self.scale
        curr_s = curr_s / self.scale

        # 转 Tensor
        s = torch.from_numpy(s).float()
        a = torch.from_numpy(a).float().unsqueeze(-1)
        r = torch.from_numpy(r).float().unsqueeze(-1)
        d = torch.from_numpy(d).long()
        sn = torch.from_numpy(sn).float()
        all_r = torch.from_numpy(all_r).float().unsqueeze(-1)
        curr_s = torch.from_numpy(curr_s).float().unsqueeze(-1)
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
