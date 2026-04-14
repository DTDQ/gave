"""
offline_env.py - 离线竞价环境（按时间步聚合，16维状态）

该环境根据实际曝光数据，按20分钟一个时间步聚合曝光，实时生成与训练数据完全一致的16维状态向量。
特征顺序与 TrainDataGenerator 中的定义保持一致。

依赖环境数据列：adgroup_id, min_win_ecpm, p_ctr, p_cvr, real_ecpm, log_time
"""

import numpy as np
import pandas as pd
from collections import deque


class OfflineEnv:
    """
    离线竞价环境，每个时间步处理该20分钟内所有曝光，返回16维状态。
    """

    def __init__(self, traffic_df):
        """
        参数:
            traffic_df: pd.DataFrame，包含该广告组在某投放周期内的所有曝光记录
                        必须包含列: adgroup_id, min_win_ecpm, p_ctr, p_cvr, real_ecpm, log_time
        """
        # 预处理：一次性完成所有计算，按 adgroup_id 缓存 numpy 数组
        df = traffic_df.copy()
        df['min_win_ecpm'] = df['min_win_ecpm'] / 1000.0
        df['real_ecpm'] = df['real_ecpm'] / 1000.0
        df['pCTCVR'] = df['p_ctr'] * df['p_cvr']

        # 向量化计算 timestep（避免逐行 apply）
        dt = pd.to_datetime(df['log_time'], unit='ms')
        df['timestep'] = (dt.dt.hour * 60 + dt.dt.minute) // 20 + 1

        # 按 adgroup_id 预分组，每个 ad 再按 timestep 分组提取 numpy 数组
        self._ad_cache = {}
        for ad_id, ad_df in df.groupby('adgroup_id'):
            ad_df = ad_df.sort_values('log_time')
            step_groups = ad_df.groupby('timestep')
            time_steps = sorted(step_groups.groups.keys())
            step_arrays = {}
            for ts in time_steps:
                g = step_groups.get_group(ts)
                step_arrays[ts] = {
                    'pCTCVR': g['pCTCVR'].values,
                    'min_win_ecpm': g['min_win_ecpm'].values,
                    'real_ecpm': g['real_ecpm'].values,
                }
            self._ad_cache[ad_id] = {
                'time_steps': time_steps,
                'step_arrays': step_arrays,
            }

    def reset(self, ad_id, period, initial_budget, cpacons):
        """
        重置环境到指定广告组、投放周期、预算和 tCPA。

        参数:
            ad_id: 广告组ID，对应环境中的 adgroup_id
            period: 投放周期索引（仅用于日志，筛选已在外部完成）
            initial_budget: 初始预算
            cpacons: 目标 CPA（tCPA）

        返回:
            state: 16维初始状态向量（时间步1开始前的状态）
        """
        assert ad_id in self._ad_cache, f"广告组 {ad_id} 在环境数据中无曝光记录"

        cached = self._ad_cache[ad_id]
        self.time_steps = cached['time_steps']
        self.step_arrays = cached['step_arrays']
        self.num_steps = len(self.time_steps)
        self.current_idx = 0
        self.last_time_step = 0

        self.initial_budget = initial_budget
        self.remaining_budget = self.initial_budget
        self.tcpa = cpacons

        self.total_cost = 0.0
        self.total_conversion = 0.0

        # ---------- 历史记录（用于实时计算状态）----------
        self.history = []

        # 最近3步的滚动窗口
        self.last3_pacer = deque(maxlen=3)
        self.last3_ci_mean = deque(maxlen=3)
        self.last3_pvalue_mean = deque(maxlen=3)
        self.last3_conversion_per_pv = deque(maxlen=3)
        self.last3_win_prob = deque(maxlen=3)
        self.last3_pv_num = deque(maxlen=3)
        # 返回初始状态（时间步1开始前的状态）
        return self._get_initial_state()

    def _get_initial_state(self):
        """返回初始状态：timeleft=1.0, bgtleft=1.0, 其余为0（16维）"""
        return np.array([
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)

    def step(self, alpha):
        """
        执行一个时间步的竞价决策（处理该步内的所有曝光），并返回下一状态。

        参数:
            alpha: 出价系数，模型输出（该步内所有曝光共用）

        返回:
            next_state: 16维下一状态向量（该步结束后的新状态）
            done: 是否终止（所有步处理完毕或预算耗尽）
            info: dict，包含 step_cost, step_conversion, step_win, step_pv
        """
        if self._is_done():
            return self._get_current_state(), True, {
                "step_cost": 0.0,
                "step_conversion": 0.0,
                "step_win": 0,
                "step_pv": 0
            }

        # 获取当前时间步的预计算 numpy 数组
        current_timestep = self.time_steps[self.current_idx]
        arrays = self.step_arrays[current_timestep]
        pCTCVR = arrays['pCTCVR']
        min_win_ecpm = arrays['min_win_ecpm']
        real_ecpm = arrays['real_ecpm']
        n = len(pCTCVR)

        # 向量化计算出价
        bid_price = alpha * pCTCVR
        bid_eligible = bid_price >= min_win_ecpm

        # 预算约束循环（直接操作 numpy 数组，比 iterrows 快 ~1000x）
        # 数据为1%采样，实际花费 = real_ecpm * 100；预算不足时有多少花多少
        win_mask = np.zeros(n, dtype=np.bool_)
        remaining = self.remaining_budget
        processed_count = n
        step_cost = 0.0
        step_conversion = 0.0

        for i in range(n):
            if remaining <= 0:
                processed_count = i
                break
            if bid_eligible[i]:
                actual_cost = real_ecpm[i] * 100
                if remaining >= actual_cost:
                    # 预算充足：完整扣费，100条曝光的conversion
                    win_mask[i] = True
                    remaining -= actual_cost
                    step_cost += actual_cost
                    step_conversion += pCTCVR[i] * 100
                else:
                    # 预算不足：有多少花多少，conversion按比例折算
                    ratio = remaining / actual_cost
                    win_mask[i] = True
                    step_cost += remaining
                    step_conversion += pCTCVR[i] * ratio * 100
                    remaining = 0.0

        step_win = int(win_mask.sum())
        step_pv = processed_count

        self.remaining_budget = remaining
        self.total_cost += step_cost
        self.total_conversion += step_conversion

        # 统计量（基于所有已处理的行，非仅 win）
        if step_pv > 0:
            avg_ci_mean = float(real_ecpm[:processed_count].mean())
            avg_pvalue_mean = float(pCTCVR[:processed_count].mean())
            avg_conversion_per_pv = avg_pvalue_mean
            avg_win_prob = step_win / step_pv
        else:
            avg_ci_mean = 0.0
            avg_pvalue_mean = 0.0
            avg_conversion_per_pv = 0.0
            avg_win_prob = 0.0

        step_stats = {
            'pacer': alpha,
            'ci_mean': avg_ci_mean,
            'pvalue_mean': avg_pvalue_mean,
            'conversion_per_pv': avg_conversion_per_pv,
            'win_prob': avg_win_prob,
            'pv_num': step_pv,
            'timestep': current_timestep
        }

        self.history.append(step_stats)
        self.last3_pacer.append(alpha)
        self.last3_ci_mean.append(avg_ci_mean)
        self.last3_pvalue_mean.append(avg_pvalue_mean)
        self.last3_conversion_per_pv.append(avg_conversion_per_pv)
        self.last3_win_prob.append(avg_win_prob)
        self.last3_pv_num.append(step_pv)
        self.current_idx += 1
        self.last_time_step = current_timestep

        done = self._is_done()

        info = {
            "step_cost": step_cost,
            "step_conversion": step_conversion,
            "step_win": step_win,
            "step_pv": step_pv
        }

        return self._get_current_state(), done, info

    def _is_done(self):
        return (self.current_idx >= self.num_steps) or (self.remaining_budget <= 0)

    def _get_current_state(self):
        """
        基于当前历史（已完成的所有步）计算当前状态（即下一步开始时的状态）。
        """
        eps = 1e-6
        timeleft = (72 - self.last_time_step) / 72.0
        bgtleft = self.remaining_budget / (self.initial_budget + eps)

        if len(self.history) == 0:
            return self._get_initial_state()

        avg_pacer_all = np.mean([s['pacer'] for s in self.history])
        avg_pacer_last3 = np.mean(self.last3_pacer) if self.last3_pacer else 0.0

        avg_ci_mean_all = np.mean([s['ci_mean'] for s in self.history])
        avg_pvalue_mean_all = np.mean([s['pvalue_mean'] for s in self.history])
        avg_conversion_per_pv_all = np.mean([s['conversion_per_pv'] for s in self.history])
        avg_win_prob_all = np.mean([s['win_prob'] for s in self.history])

        avg_ci_mean_last3 = np.mean(self.last3_ci_mean) if self.last3_ci_mean else 0.0
        avg_pvalue_mean_last3 = np.mean(self.last3_pvalue_mean) if self.last3_pvalue_mean else 0.0
        avg_conversion_per_pv_last3 = np.mean(self.last3_conversion_per_pv) if self.last3_conversion_per_pv else 0.0
        avg_win_prob_last3 = np.mean(self.last3_win_prob) if self.last3_win_prob else 0.0

        pvalue_mean_agg = self.history[-1]['pvalue_mean']
        pv_num_agg = self.history[-1]['pv_num']
        last_3_volume = sum(self.last3_pv_num) if self.last3_pv_num else 0.0
        historical_volume = sum([s['pv_num'] for s in self.history])

        return np.array([
            timeleft,
            bgtleft,
            avg_pacer_all,
            avg_pacer_last3,
            avg_ci_mean_all,
            avg_pvalue_mean_all,
            avg_conversion_per_pv_all,
            avg_win_prob_all,
            avg_ci_mean_last3,
            avg_pvalue_mean_last3,
            avg_conversion_per_pv_last3,
            avg_win_prob_last3,
            pvalue_mean_agg,
            pv_num_agg,
            last_3_volume,
            historical_volume
        ], dtype=np.float32)

    def get_metrics(self):
        eps = 1e-6
        return {
            "spend": self.total_cost,
            "conversion": self.total_conversion,
            "CPA": self.total_cost / (self.total_conversion + eps),
            "budget_usage": self.total_cost / (self.initial_budget + eps),
            "win_rate": np.mean([s['win_prob'] for s in self.history]) if self.history else 0.0
        }
