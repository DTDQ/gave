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
        self.df = traffic_df

    def _logtime_to_timestep(self, log_time):
        """
        将 log_time（毫秒级时间戳）转换为时间步编号（1-72）。
        假设一天24小时，每20分钟为一个步。
        """
        dt = pd.to_datetime(log_time, unit='ms')
        minute_of_day = dt.hour * 60 + dt.minute
        return minute_of_day // 20 + 1

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
        # 筛选该广告组的数据
        self.episode_df = self.df[self.df.adgroup_id == ad_id].copy()
        self.episode_df.sort_values('log_time', inplace=True)
        self.episode_df.reset_index(drop=True, inplace=True)

        assert len(self.episode_df) > 0, f"广告组 {ad_id} 在环境数据中无曝光记录"

        # ===== 关键修改：将 min_win_ecpm 和 real_ecpm 除以1000 =====
        self.episode_df['min_win_ecpm'] = self.episode_df['min_win_ecpm'] / 1000.0
        self.episode_df['real_ecpm'] = self.episode_df['real_ecpm'] / 1000.0

        # 为每条曝光计算时间步编号（1-72）
        self.episode_df['timestep'] = self.episode_df['log_time'].apply(self._logtime_to_timestep)

        # 按时间步分组，得到每个时间步内的曝光记录
        self.step_groups = self.episode_df.groupby('timestep')
        # 获取有曝光的时间步列表（已排序）
        self.time_steps = sorted(self.step_groups.groups.keys())
        self.num_steps = len(self.time_steps)          # 实际有曝光的步数
        self.current_idx = 0                            # 当前处理的时间步在列表中的索引
        self.last_time_step = 0                          # 刚完成的时间步编号（0表示未开始）

        self.initial_budget = initial_budget
        self.remaining_budget = self.initial_budget
        self.tcpa = cpacons

        self.total_cost = 0.0
        self.total_conversion = 0.0

        # ---------- 历史记录（用于实时计算16维状态）----------
        # 每个元素是一个字典，包含该步的统计量
        self.history = []          # 存储每一步的统计量，按时间顺序

        # 最近3步的滚动窗口（用于快速计算）
        self.last3_pacer = deque(maxlen=3)
        self.last3_ci_mean = deque(maxlen=3)
        self.last3_pvalue_mean = deque(maxlen=3)
        self.last3_conversion_per_pv = deque(maxlen=3)
        self.last3_win_prob = deque(maxlen=3)
        self.last3_pv_num = deque(maxlen=3)

        # 返回初始状态（时间步1开始前的状态）
        return self._get_initial_state()

    def _get_initial_state(self):
        """返回初始状态：timeleft=1.0, bgtleft=1.0, 其余为0"""
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

        # 获取当前时间步的曝光数据
        current_timestep = self.time_steps[self.current_idx]
        step_df = self.step_groups.get_group(current_timestep).copy()

        # 初始化该步的统计量
        step_cost = 0.0
        step_conversion = 0.0
        step_win = 0
        step_pv = 0

        step_ci_sum = 0.0          # 用于计算平均 ci_mean
        step_pvalue_sum = 0.0       # 用于计算平均 pvalue_mean
        step_conversion_per_pv_sum = 0.0
        step_win_sum = 0.0

        for _, row in step_df.iterrows():
            if self.remaining_budget <= 0:
                break

            pCTCVR = row.p_ctr * row.p_cvr
            bid_price = alpha * self.tcpa * pCTCVR

            win = (bid_price >= row.min_win_ecpm) and (self.remaining_budget >= row.real_ecpm)

            cost = row.real_ecpm if win else 0.0
            conv = pCTCVR if win else 0.0

            if win:
                self.remaining_budget -= cost
                self.total_cost += cost
                self.total_conversion += conv

            step_cost += cost
            step_conversion += conv
            step_win += 1 if win else 0
            step_pv += 1

            step_ci_sum += row.real_ecpm
            step_pvalue_sum += pCTCVR
            step_conversion_per_pv_sum += pCTCVR
            step_win_sum += (1 if win else 0)

        if step_pv > 0:
            avg_ci_mean = step_ci_sum / step_pv
            avg_pvalue_mean = step_pvalue_sum / step_pv
            avg_conversion_per_pv = step_conversion_per_pv_sum / step_pv
            avg_win_prob = step_win_sum / step_pv
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