import os
import pandas as pd
import warnings
import re
import numpy as np

# 尝试导入 tqdm（若无则降级为简单打印）
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, desc='', total=None, **kwargs):
            self.iterable = iterable
            self.desc = desc
            self.total = total
        def __iter__(self):
            if self.iterable:
                for item in self.iterable:
                    yield item
        def set_description(self, desc):
            pass
        def update(self, n=1):
            pass
        def close(self):
            pass

warnings.filterwarnings('ignore')


class TestDataGenerator:
    def __init__(self, test_file, env_file, output_dir, dataset_type='test'):
        """
        :param test_file: 测试日志文件路径（无表头，15列，与训练数据格式相同）
        :param env_file: 环境参数文件路径（无表头，11列）
        :param output_dir: 输出文件夹路径
        :param dataset_type: 数据集类型，默认 'test' 以保留评估列
        """
        self.test_file = test_file
        self.env_file = env_file
        self.output_dir = output_dir
        self.dataset_type = dataset_type
        os.makedirs(output_dir, exist_ok=True)

        self.numeric_cols = [
            'budget', 'tcpa', 'remain_budget', 'pvalue_mean', 'pacer',
            'win_num', 'win_rate', 'cost', 'pvalue_sum', 'ci_mean', 'pv_num'
        ]

        # 测试文件列名（15列，与训练数据一致）
        self.test_columns = [
            'adgroup_id', 'effect_type', 'budget', 'tcpa', 'timestep',
            'remain_budget', 'pvalue_mean', 'pacer', 'win_num', 'win_rate',
            'cost', 'pvalue_sum', 'ci_mean', 'pv_num', 'pt_d'
        ]

        # 环境文件列名（11列）
        self.env_columns = [
            'adgroup_id', 'min_win_ecpm', 'p_ctr', 'p_cvr', 'pltv',
            'ocpx_deep_pcvr', 'log_time', 'ab_tag', 'real_ecpm', 'bid_tag', 'pt_d'
        ]

    def process(self):
        """处理文件对，生成两个输出文件"""
        print(f"处理测试文件: {self.test_file}")
        print(f"处理环境文件: {self.env_file}")

        # 1. 处理环境文件：添加表头保存为CSV
        env_output = self._save_env_as_csv()

        # 2. 处理测试文件，生成RL数据
        rl_output = self._process_test_file()

        print(f"\n✓ 生成完成:")
        print(f"  RL 数据: {rl_output}")
        print(f"  环境映射: {env_output}")

    def _save_env_as_csv(self):
        """读取环境文件，添加表头，保存为CSV"""
        env_df = pd.read_csv(self.env_file, header=None, names=self.env_columns, dtype=str)
        output_path = os.path.join(self.output_dir, f"{os.path.splitext(os.path.basename(self.env_file))[0]}.csv")
        env_df.to_csv(output_path, index=False)
        return output_path

    def _process_test_file(self):
        """处理测试日志文件，生成RL数据"""
        # 读取测试文件
        df = pd.read_csv(self.test_file, header=None, names=self.test_columns, dtype=str, on_bad_lines='skip')
        df = self._clean_and_prepare_data(df)

        # 从文件所在目录名提取投放周期索引
        period_index = self._extract_period_from_path(self.test_file)

        # 生成RL数据（过滤+完整性检查）
        rl_df = self._generate_test_data(df, period_index)

        # 保存到输出目录
        base_name = os.path.splitext(os.path.basename(self.test_file))[0]
        output_path = os.path.join(self.output_dir, f"{base_name}_rlData.csv")
        rl_df.to_csv(output_path, index=False)
        return output_path

    def _clean_and_prepare_data(self, df):
        """清洗测试数据，直接使用文件中的列"""
        # 转换数值类型
        for col in df.columns:
            if col in ['adgroup_id', 'timestep']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # 确保按广告组和时间排序
        df = df.sort_values(['adgroup_id', 'timestep']).reset_index(drop=True)

        # ci_mean 除以1000（与训练脚本保持一致）
        df['ci_mean'] = df['ci_mean'] / 1000.0

        # 检查必需列
        required = ['adgroup_id', 'timestep', 'budget', 'tcpa', 'remain_budget',
                    'pvalue_mean', 'pacer', 'win_num', 'cost', 'pvalue_sum', 'ci_mean', 'pv_num']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"测试文件缺失必需列: {missing}")

        return df

    def _extract_period_from_path(self, filepath):
        """从文件所在目录名中提取日期作为投放周期索引"""
        dir_name = os.path.basename(os.path.dirname(filepath))
        numbers = re.findall(r'\d+', dir_name)
        if numbers:
            return int(numbers[0])
        return 1

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

    def _generate_test_data(self, df, deliveryPeriodIndex):
        """生成RL格式数据（只处理环境文件中存在的广告组，并检查轨迹完整性）"""
        # 读取环境文件中的广告组ID集合
        env_df = pd.read_csv(self.env_file, header=None, names=self.env_columns, dtype=str)
        env_adgroup_ids = set(env_df['adgroup_id'].astype(int).unique())
        print(f"环境文件中包含 {len(env_adgroup_ids)} 个广告组ID")

        # 过滤全天总 win_num < 1000 的广告组
        adid_total_win = df.groupby('adgroup_id')['win_num'].sum()
        valid_adids = adid_total_win[adid_total_win >= 1000].index
        before = df['adgroup_id'].nunique()
        df = df[df['adgroup_id'].isin(valid_adids)].reset_index(drop=True)
        after = df['adgroup_id'].nunique()
        print(f"过滤 win_num<1000 的广告组: {before} -> {after} (移除 {before - after})")

        df = df.copy()
        df['deliveryPeriodIndex'] = deliveryPeriodIndex
        df['advertiserCategoryIndex'] = 0

        # 计算衍生特征
        df['conversion_per_pv'] = df['pvalue_sum'] / df['pv_num'].replace(0, 1)
        df['win_prob'] = df['win_num'] / df['pv_num'].replace(0, 1)

        training_rows = []
        skipped = 0

        # 按广告组分组
        group_keys = ['deliveryPeriodIndex', 'adgroup_id', 'advertiserCategoryIndex', 'budget', 'tcpa']
        groups = list(df.groupby(group_keys))
        print(f"测试数据中总共有 {len(groups)} 个广告组")

        # 使用 tqdm 添加进度条
        for _, group in tqdm(groups, desc="处理广告组"):
            group = group.sort_values('timestep').reset_index(drop=True)
            if len(group) == 0:
                continue

            adgroup_id = group['adgroup_id'].iloc[0]
            # 过滤：只处理环境文件中存在的广告组
            if adgroup_id not in env_adgroup_ids:
                continue

            # 检查轨迹完整性
            if not self._is_trajectory_complete(group):
                print(f"⚠️ 广告组 {adgroup_id} 的轨迹不完整（timestep不为1或不连续），跳过")
                skipped += 1
                continue

            max_timestep = group['timestep'].max()
            group['isEnd'] = (group['timestep'] == max_timestep).astype(int)

            # PV相关特征（开始前的累计值）
            pv_by_step = group.groupby('timestep')['pv_num'].first()
            group['historical_volume'] = group['timestep'].map(
                pv_by_step.cumsum().shift(1).fillna(0).astype(int)
            )
            group['last_3_volume'] = group['timestep'].map(
                pv_by_step.rolling(3, min_periods=1).sum().shift(1).fillna(0).astype(int)
            )

            # 历史平均特征（开始前）
            for col in ['pacer', 'ci_mean', 'conversion_per_pv', 'win_prob', 'pvalue_mean']:
                group[f'avg_{col}_all'] = group[col].expanding().mean().shift(1).fillna(0)
                group[f'avg_{col}_last3'] = group[col].rolling(3, min_periods=1).mean().shift(1).fillna(0)

            # 为结束状态准备”当前步”统计量
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

            # PV累计（包含当前步）
            group['historical_volume_now'] = group['timestep'].map(
                pv_by_step.cumsum().fillna(0).astype(int)
            )
            group['last_3_volume_now'] = group['timestep'].map(
                pv_by_step.rolling(3, min_periods=1).sum().fillna(0).astype(int)
            )

            group['pvalue_mean_agg'] = group['pvalue_mean']
            group['pv_num_agg'] = group['pv_num']

            # 全局统计
            realAllCost = group['cost'].sum()
            realAllConversion = group['pvalue_sum'].sum()

            # 剔除没有真实转化的任务
            if realAllConversion <= 0:
                skipped += 1
                continue

            # 上一时间步剩余预算
            group['prev_remain_budget'] = group['remain_budget'].shift(1)

            # 构建开始前状态的字典
            states_start = {}
            for _, row in group.iterrows():
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
            for i, row in group.iterrows():
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

                # 如果是测试集，保留原始列以便评估
                if self.dataset_type == 'test':
                    record['cost'] = float(row['cost'])
                    record['win_num'] = int(row['win_num'])
                    record['pv_num'] = int(row['pv_num'])

                training_rows.append(record)

        if skipped > 0:
            print(f"共跳过 {skipped} 条不完整轨迹")

        if not training_rows:
            print("警告：没有生成任何有效轨迹，返回空DataFrame")
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


def main():
    # ==================== 用户配置区域 ====================
    test_file = "./data/raw_data/train_pt_d=20260207/train_data.txt"          # 测试日志文件（15列）
    env_file = "./data/raw_data/env_pt_d=20260207/env_data.txt"               # 环境文件（11列）
    output_dir = "./data/test/pt_d=20260207"                                   # 输出文件夹
    # =====================================================

    if not os.path.exists(test_file):
        print(f"错误: 测试文件不存在: {test_file}")
        return
    if not os.path.exists(env_file):
        print(f"错误: 环境文件不存在: {env_file}")
        return

    generator = TestDataGenerator(
        test_file=test_file,
        env_file=env_file,
        output_dir=output_dir,
        dataset_type='test'  # 固定为 test，以保留评估列
    )
    generator.process()


if __name__ == '__main__':
    main()
