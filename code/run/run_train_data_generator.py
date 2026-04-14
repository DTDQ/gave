import os
import pandas as pd
import warnings
import re
import random
import numpy as np

# 尝试导入 tqdm
try:
    from tqdm import tqdm
except ImportError:
    # 如果没有 tqdm，定义一个简单的 fallback
    class tqdm:
        def __init__(self, total=None, desc='', **kwargs):
            self.total = total
            self.desc = desc
            self.n = 0
        def update(self, n=1):
            self.n += n
            if self.total:
                print(f"\r{self.desc}: {self.n}/{self.total} ({self.n/self.total*100:.1f}%)", end='')
        def close(self):
            print()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()

warnings.filterwarnings('ignore')


class TrainDataGenerator:
    def __init__(self, file_paths, output_path, num_trajectories=None):
        """
        :param file_paths: 待处理的 TXT 文件路径列表
        :param output_path: 处理后合并文件的保存路径（包括文件名）
        :param num_trajectories: 要采样的完整轨迹数量，若为 None 则处理所有完整轨迹
        """
        self.file_paths = file_paths
        self.output_path = output_path
        self.num_trajectories = num_trajectories
        self.numeric_cols = [
            'budget', 'tcpa', 'remain_budget', 'pvalue_mean', 'pacer',
            'win_num', 'win_rate', 'cost', 'pvalue_sum', 'ci_mean', 'pv_num'
        ]
        # TXT 文件列名（无表头，按顺序）
        self.txt_columns = [
            'adgroup_id', 'effect_type', 'budget', 'tcpa', 'timestep',
            'remain_budget', 'pvalue_mean', 'pacer', 'win_num', 'win_rate',
            'cost', 'pvalue_sum', 'ci_mean', 'pv_num', 'pt_d'
        ]
        # 文件缓存，避免重复读取
        self.file_cache = {}

    def _load_file(self, file_path):
        """加载文件并缓存"""
        if file_path not in self.file_cache:
            df = pd.read_csv(file_path, header=None, names=self.txt_columns, dtype=str, on_bad_lines='skip')
            df = self._clean_and_prepare_data(df)
            # 过滤全天总 win_num < 1000 的广告组
            adid_total_win = df.groupby('adgroup_id')['win_num'].sum()
            valid_adids = adid_total_win[adid_total_win >= 1000].index
            before = df['adgroup_id'].nunique()
            df = df[df['adgroup_id'].isin(valid_adids)].reset_index(drop=True)
            after = df['adgroup_id'].nunique()
            print(f"过滤 win_num<1000 的广告组: {before} -> {after} (移除 {before - after})")
            self.file_cache[file_path] = df
        return self.file_cache[file_path]

    def batch_generate_train_data(self):
        """批量处理传入的 TXT 文件列表，合并保存为一个文件（支持采样）"""
        if not self.file_paths:
            print("⚠️ 未提供任何文件")
            return []

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # 构建所有可能的轨迹标识列表 (file_path, period_index, adgroup_id)
        print("正在构建轨迹索引...")
        trajectory_ids = []
        for file_path in self.file_paths:
            df = self._load_file(file_path)
            # 从文件所在目录名提取投放周期索引
            period_index = self._extract_period_from_path(file_path)
            # 获取该文件中的所有广告组
            adgroups = df['adgroup_id'].unique()
            for adid in adgroups:
                trajectory_ids.append((file_path, period_index, adid))
        total_trajectories = len(trajectory_ids)
        print(f"共发现 {total_trajectories} 条潜在轨迹")

        # 确定要采样的数量
        if self.num_trajectories is None:
            target_num = total_trajectories
            print(f"将处理所有轨迹（共 {target_num} 条）")
        else:
            target_num = self.num_trajectories
            print(f"目标采样 {target_num} 条完整轨迹")

        # 随机打乱轨迹列表，以便随机采样
        random.shuffle(trajectory_ids)

        data_frames = []
        processed = 0
        skipped = 0

        with tqdm(total=target_num, desc="采样轨迹") as pbar:
            idx = 0
            while processed < target_num and idx < len(trajectory_ids):
                file_path, period_index, adid = trajectory_ids[idx]
                idx += 1

                # 从缓存中加载该文件的数据
                df_file = self.file_cache[file_path]
                # 筛选该广告组的数据
                group_df = df_file[df_file['adgroup_id'] == adid].copy()
                group_df = group_df.sort_values('timestep').reset_index(drop=True)

                # 检查轨迹完整性
                if not self._is_trajectory_complete(group_df):
                    skipped += 1
                    continue

                try:
                    df_processed = self._generate_train_data_for_group(group_df, period_index, adid)
                    if not df_processed.empty:
                        data_frames.append(df_processed)
                        processed += 1
                        pbar.update(1)
                except Exception as e:
                    import traceback
                    print(f"\n处理广告组 {adid} 时出错: {e}")
                    traceback.print_exc()
                    skipped += 1

        if processed == 0:
            print("\n⚠️ 未生成任何有效数据")
            return []

        print(f"\n采样完成: 成功 {processed} 条, 跳过 {skipped} 条不完整轨迹")

        # 合并所有采样的轨迹数据
        combined = pd.concat(data_frames, ignore_index=True)
        combined.to_csv(self.output_path, index=False)
        print(f"\n✓ 所有数据合并完成! 共 {len(combined)} 条记录")
        print(f"  保存至: {self.output_path}")
        return data_frames

    def _is_trajectory_complete(self, group_df):
        """检查轨迹是否完整：timestep从1开始且连续"""
        if len(group_df) == 0:
            return False
        timesteps = group_df['timestep'].values
        if timesteps[0] != 1:
            return False
        # 检查是否连续递增
        expected = np.arange(1, len(timesteps) + 1)
        return np.array_equal(timesteps, expected)

    def _generate_train_data_for_group(self, group, deliveryPeriodIndex, adgroup_id):
        """为单个广告组生成训练数据（复用原 _generate_train_data 的逻辑）"""
        # 该方法与原有 _generate_train_data 类似，但只处理单个组
        # 可以直接调用 _generate_train_data，但需要构造一个包含该组的 DataFrame
        # 为简化，我们复用 _generate_train_data 的逻辑，但仅传入单个组的 DataFrame
        # 注意：原有 _generate_train_data 会按广告组分组，这里我们只传入单个组，所以直接应用内部逻辑
        df = group.copy()
        df['deliveryPeriodIndex'] = deliveryPeriodIndex
        df['advertiserCategoryIndex'] = 0

        # 计算衍生特征
        df['conversion_per_pv'] = df['pvalue_sum'] / df['pv_num'].replace(0, 1)
        df['win_prob'] = df['win_num'] / df['pv_num'].replace(0, 1)

        training_rows = []

        # 单组处理，不需要再分组
        group_data = df.sort_values('timestep').reset_index(drop=True)
        max_timestep = group_data['timestep'].max()
        group_data['isEnd'] = (group_data['timestep'] == max_timestep).astype(int)

        # PV相关特征（开始前的累计值）
        pv_by_step = group_data.groupby('timestep')['pv_num'].first()
        group_data['historical_volume'] = group_data['timestep'].map(
            pv_by_step.cumsum().shift(1).fillna(0).astype(int)
        )
        group_data['last_3_volume'] = group_data['timestep'].map(
            pv_by_step.rolling(3, min_periods=1).sum().shift(1).fillna(0).astype(int)
        )

        # 历史平均特征（开始前）
        for col in ['pacer', 'ci_mean', 'conversion_per_pv', 'win_prob', 'pvalue_mean']:
            group_data[f'avg_{col}_all'] = group_data[col].expanding().mean().shift(1).fillna(0)
            group_data[f'avg_{col}_last3'] = group_data[col].rolling(3, min_periods=1).mean().shift(1).fillna(0)

        # 为结束状态准备”当前步”统计量
        group_data['avg_pacer_all_now'] = group_data['pacer'].expanding().mean()
        group_data['avg_pacer_last3_now'] = group_data['pacer'].rolling(3, min_periods=1).mean()
        group_data['avg_ci_mean_all_now'] = group_data['ci_mean'].expanding().mean()
        group_data['avg_ci_mean_last3_now'] = group_data['ci_mean'].rolling(3, min_periods=1).mean()
        group_data['avg_pvalue_mean_all_now'] = group_data['pvalue_mean'].expanding().mean()
        group_data['avg_pvalue_mean_last3_now'] = group_data['pvalue_mean'].rolling(3, min_periods=1).mean()
        group_data['avg_conversion_per_pv_all_now'] = group_data['conversion_per_pv'].expanding().mean()
        group_data['avg_conversion_per_pv_last3_now'] = group_data['conversion_per_pv'].rolling(3, min_periods=1).mean()
        group_data['avg_win_prob_all_now'] = group_data['win_prob'].expanding().mean()
        group_data['avg_win_prob_last3_now'] = group_data['win_prob'].rolling(3, min_periods=1).mean()

        # PV累计（包含当前步）
        group_data['historical_volume_now'] = group_data['timestep'].map(
            pv_by_step.cumsum().fillna(0).astype(int)
        )
        group_data['last_3_volume_now'] = group_data['timestep'].map(
            pv_by_step.rolling(3, min_periods=1).sum().fillna(0).astype(int)
        )

        group_data['pvalue_mean_agg'] = group_data['pvalue_mean']
        group_data['pv_num_agg'] = group_data['pv_num']

        # 全局统计
        realAllCost = group_data['cost'].sum()
        realAllConversion = group_data['pvalue_sum'].sum()

        # 上一时间步剩余预算
        group_data['prev_remain_budget'] = group_data['remain_budget'].shift(1)

        # 构建开始前状态的字典
        states_start = {}
        for _, row in group_data.iterrows():
            t = int(row['timestep'])
            timeleft_start = (max_timestep - (t - 1)) / max_timestep if max_timestep > 0 else 0.0
            if t == 1:
                bgtleft_start = 1.0
            else:
                prev_remain = row['prev_remain_budget']
                bgtleft_start = prev_remain / row['budget'] if row['budget'] > 0 else 0.0

            state = (
                float(timeleft_start),
                float(bgtleft_start),
                float(row.get('avg_pacer_all', 0.0)),
                float(row.get('avg_pacer_last3', 0.0)),
                float(row.get('avg_ci_mean_all', 0.0)),
                float(row.get('avg_pvalue_mean_all', 0.0)),
                float(row.get('avg_conversion_per_pv_all', 0.0)),
                float(row.get('avg_win_prob_all', 0.0)),
                float(row.get('avg_ci_mean_last3', 0.0)),
                float(row.get('avg_pvalue_mean_last3', 0.0)),
                float(row.get('avg_conversion_per_pv_last3', 0.0)),
                float(row.get('avg_win_prob_last3', 0.0)),
                float(row.get('pvalue_mean_agg', 0.0)),
                float(row.get('pv_num_agg', 0.0)),
                float(row.get('last_3_volume', 0.0)),
                float(row.get('historical_volume', 0.0))
            )
            states_start[t] = state

        # 生成转移样本
        for i, row in group_data.iterrows():
            t = int(row['timestep'])
            state_prev = states_start[t]
            action = row['pacer'] * row['tcpa']
            reward = float(row['pvalue_sum'])
            done = 1 if t == max_timestep else 0

            record = {
                'deliveryPeriodIndex': deliveryPeriodIndex,
                'advertiserNumber': row['adgroup_id'],
                'advertiserCategoryIndex': 0,
                'budget': row['budget'],
                'CPAConstraint': row['tcpa'],
                'realAllCost': realAllCost,
                'realAllConversion': realAllConversion,
                'timeStepIndex': t,
                'state': state_prev,
                'action': float(action),
                'reward': reward,
                'reward_continuous': reward,
                'done': done,
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
                'historical_volume_now': row['historical_volume_now']
            }
            training_rows.append(record)

        if not training_rows:
            return pd.DataFrame()

        training_df = pd.DataFrame(training_rows)
        training_df = training_df.sort_values(
            ['deliveryPeriodIndex', 'advertiserNumber', 'timeStepIndex']
        ).reset_index(drop=True)

        # 用 shift(-1) 生成 next_state
        training_df['next_state'] = training_df.groupby(
            ['deliveryPeriodIndex', 'advertiserNumber']
        )['state'].shift(-1)

        # 最后一行（done == 1）构造结束状态覆盖 next_state
        mask_last = training_df['done'] == 1
        for idx in training_df[mask_last].index:
            row = training_df.loc[idx]
            end_state = (
                0.0,
                float(row['remain_budget'] / row['budget'] if row['budget'] > 0 else 0.0),
                float(row['avg_pacer_all_now']),
                float(row['avg_pacer_last3_now']),
                float(row['avg_ci_mean_all_now']),
                float(row['avg_pvalue_mean_all_now']),
                float(row['avg_conversion_per_pv_all_now']),
                float(row['avg_win_prob_all_now']),
                float(row['avg_ci_mean_last3_now']),
                float(row['avg_pvalue_mean_last3_now']),
                float(row['avg_conversion_per_pv_last3_now']),
                float(row['avg_win_prob_last3_now']),
                float(row['pvalue_mean_agg']),
                float(row['pv_num_agg']),
                float(row['last_3_volume_now']),
                float(row['historical_volume_now'])
            )
            training_df.at[idx, 'next_state'] = end_state

        # 将 state/next_state 转为字符串
        training_df['state'] = training_df['state'].apply(str)
        training_df['next_state'] = training_df['next_state'].apply(str)

        # 删除临时列
        cols_to_drop = ['remain_budget', 'avg_pacer_all_now', 'avg_pacer_last3_now',
                        'avg_ci_mean_all_now', 'avg_ci_mean_last3_now',
                        'avg_pvalue_mean_all_now', 'avg_pvalue_mean_last3_now',
                        'avg_conversion_per_pv_all_now', 'avg_conversion_per_pv_last3_now',
                        'avg_win_prob_all_now', 'avg_win_prob_last3_now',
                        'pvalue_mean_agg', 'pv_num_agg', 'last_3_volume_now', 'historical_volume_now']
        training_df.drop(columns=[c for c in cols_to_drop if c in training_df.columns], inplace=True)

        return training_df

    def _clean_and_prepare_data(self, df):
        """清洗并转换数据类型，直接使用原始列（TXT 已包含所需字段）"""
        # 将各列转换为正确的数值类型
        for col in df.columns:
            if col in ['adgroup_id', 'timestep']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # 确保 timestep 按广告组内排序（文件可能已排序，但保险起见）
        df = df.sort_values(['adgroup_id', 'timestep']).reset_index(drop=True)

        # 检查必需列
        required = ['adgroup_id', 'timestep', 'budget', 'tcpa', 'remain_budget',
                    'pvalue_mean', 'pacer', 'win_num', 'cost', 'pvalue_sum', 'ci_mean', 'pv_num']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"缺失必需列: {missing}")

        # 确保数值列类型正确（已在上面转换）
        # ci_mean 除以 1000（与原始脚本保持一致）
        df['ci_mean'] = df['ci_mean'] / 1000.0

        return df

    def _extract_period_from_path(self, filepath):
        """从文件所在目录名中提取日期作为投放周期索引（如 pt_d=20260206 -> 20260206）"""
        dir_name = os.path.basename(os.path.dirname(filepath))
        numbers = re.findall(r'\d+', dir_name)
        if numbers:
            return int(numbers[0])
        return 1

    # 保留原有的 _generate_train_data 方法以兼容旧调用（但本脚本中不再使用）
    def _generate_train_data(self, df, deliveryPeriodIndex):
        """原有方法，此处留空或调用新的方法，但为了完整性保留"""
        # 此方法不再使用，改用 _generate_train_data_for_group
        pass


def main():
    # ==================== 用户配置区域 ====================
    # 请直接修改下面的文件路径列表，填入需要处理的所有 TXT 文件（支持绝对或相对路径）
    train_files = [
        # 示例：
        "./data/raw_data/train_pt_d=20260206/train_data.txt"
    ]
    output_file = "./data/train/train_data_all-rlData.csv"  # 合并后输出文件路径（确保是文件，不是目录）
    num_trajectories = 50000  # 要采样的完整轨迹数量，设为 None 则处理所有轨迹
    # =====================================================

    if train_files:
        print(f"找到 {len(train_files)} 个训练文件")
        generator = TrainDataGenerator(
            file_paths=train_files,
            output_path=output_file,
            num_trajectories=num_trajectories
        )
        generator.batch_generate_train_data()
    else:
        print("未指定训练文件，程序退出。")


if __name__ == '__main__':
    main()
