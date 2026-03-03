import os
import pandas as pd
import warnings
import glob
import re

warnings.filterwarnings('ignore')


class TrainDataGenerator:
    def __init__(self, file_folder_path="./data/raw_data", 
                 training_data_path="./data/traffic",
                 dataset_type=None):   # 参数，用于指定数据集类型（'train' 或 'test'）
        self.file_folder_path = file_folder_path
        self.training_data_path = training_data_path
        self.dataset_type = dataset_type
        self.numeric_cols = [
            'budget', 'tcpa', 'remain_budget', 'pvalue_mean', 'pacer',
            'win_num', 'win_rate', 'cost', 'pvalue_sum', 'ci_mean', 'pv_num'
        ]

    def batch_generate_train_data(self):
        # 确保输出目录存在
        os.makedirs(self.training_data_path, exist_ok=True)
        
        excel_files = glob.glob(os.path.join(self.file_folder_path, '*.xlsx'))
        csv_files = glob.glob(os.path.join(self.file_folder_path, '*.csv'))
        all_files = excel_files + csv_files
        
        if not all_files:
            print("⚠️ 未找到任何可处理的文件")
            return []
        
        print(f"✓ 找到 {len(all_files)} 个待处理文件")
        training_data_list = []
        
        for file_path in all_files:
            filename = os.path.basename(file_path)
            print(f"  → 处理: {filename}")
            
            df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
            df = self._clean_and_prepare_data(df)
            period_index = self._extract_period_from_filename(filename)
            
            try:
                df_processed = self._generate_train_data(df, period_index)
                if df_processed.empty:
                    print(f"    ✗ 跳过空数据文件: {filename}")
                    continue
                
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(self.training_data_path, f"{base_name}-rlData.csv")
                df_processed.to_csv(output_path, index=False)
                training_data_list.append(df_processed)
                print(f"    ✓ 完成: {filename} → {len(df_processed)} 条记录")
            except Exception as e:
                import traceback
                print(f"    ✗ 处理失败 {filename}: {str(e)}")
                print(f"异常类型: {type(e).__name__}")
                print(traceback.format_exc())
                continue
        
        if training_data_list:
            combined = pd.concat(training_data_list, ignore_index=True)
            
            # 合并文件命名规则：优先使用外部传入的 dataset_type
            if self.dataset_type is not None:
                prefix = self.dataset_type
            else:
                # 向后兼容：从输出目录名推断
                dir_name = os.path.basename(os.path.normpath(self.training_data_path))
                if dir_name.lower() in ['train', 'training']:
                    prefix = 'train'
                elif dir_name.lower() in ['test', 'testing']:
                    prefix = 'test'
                else:
                    prefix = 'training'
            
            combined_filename = f"{prefix}_data_all-rlData.csv"
            combined_path = os.path.join(self.training_data_path, combined_filename)
            combined.to_csv(combined_path, index=False)
            print(f"\n✓ 全部处理完成! 共 {len(combined)} 条记录")
            print(f"  保存至: {combined_path}")
            return training_data_list
        else:
            print("\n⚠️ 未生成任何有效训练数据")
            return []

    def _clean_and_prepare_data(self, df):
        """清洗列名并转换数值列"""
        df.columns = [re.sub(r'\s+', '', str(col).strip().replace('\n', '').replace('\r', '')) 
                     for col in df.columns]
        required = ['adgroup_id', 'timestep', 'budget', 'tcpa', 'remain_budget',
                   'pvalue_mean', 'pacer', 'win_num', 'cost', 'pvalue_sum', 'ci_mean', 'pv_num']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"缺失必需列: {missing}. 可用列: {list(df.columns)}")
        for col in self.numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        return df

    def _extract_period_from_filename(self, filename):
        """从文件名提取日期作为投放周期索引 (YYYYMMDD)"""
        try:
            name = os.path.splitext(filename)[0]
            parts = name.split('_')
            if len(parts) >= 3:
                return int(f"{parts[0]}{parts[1].zfill(2)}{parts[2].zfill(2)}")
        except:
            pass
        return 1

    def _generate_train_data(self, df, deliveryPeriodIndex):
        """核心数据生成逻辑（使用 shift(-1) + 最后一行特殊处理）
        若 dataset_type 为 'test'，则额外保留 cost、win_num、pv_num 列以便计算基准指标。
        """
        df = df.copy()
        df['deliveryPeriodIndex'] = deliveryPeriodIndex
        df['advertiserCategoryIndex'] = 0
        df['ci_mean'] = df['ci_mean'] / 1000.0

        # 计算衍生特征
        df['conversion_per_pv'] = df['pvalue_sum'] / df['pv_num'].replace(0, 1)
        df['win_prob'] = df['win_num'] / df['pv_num'].replace(0, 1)

        training_rows = []

        # 按广告组分组处理
        group_keys = ['deliveryPeriodIndex', 'adgroup_id', 'advertiserCategoryIndex', 'budget', 'tcpa']
        for _, group in df.groupby(group_keys):
            group = group.sort_values('timestep').reset_index(drop=True)
            if len(group) == 0:
                continue

            total_steps = group['timestep'].max()
            max_timestep = total_steps

            group['isEnd'] = (group['timestep'] == max_timestep).astype(int)

            # 计算PV相关特征（开始前的累计值）
            pv_by_step = group.groupby('timestep')['pv_num'].first()
            group['historical_volume'] = group['timestep'].map(
                pv_by_step.cumsum().shift(1).fillna(0).astype(int)
            )
            group['last_3_volume'] = group['timestep'].map(
                pv_by_step.rolling(3, min_periods=1).sum().shift(1).fillna(0).astype(int)
            )

            # 计算历史平均特征（开始前的平均值）
            for col in ['pacer', 'ci_mean', 'conversion_per_pv', 'win_prob', 'pvalue_mean']:
                group[f'avg_{col}_all'] = group[col].expanding().mean().shift(1).fillna(0)
                group[f'avg_{col}_last3'] = group[col].rolling(3, min_periods=1).mean().shift(1).fillna(0)

            # ========== 为结束状态准备“当前步”统计量 ==========
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

            # 预估列
            group['pvalue_mean_agg'] = group['pvalue_mean']
            group['pv_num_agg'] = group['pv_num']

            # 全局统计
            realAllCost = group['cost'].sum()
            realAllConversion = group['pvalue_sum'].sum()

            # 预计算上一时间步剩余预算（用于开始前状态）
            group['prev_remain_budget'] = group['remain_budget'].shift(1)

            # ---------- 构建开始前状态的字典 ----------
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

            # ---------- 生成转移样本（暂不设置 next_state）----------
            for i, row in group.iterrows():
                t = int(row['timestep'])
                state_prev = states_start[t]
                action = row['pacer'] * row['tcpa']
                reward = float(row['pvalue_sum'])
                done = 1 if t == max_timestep else 0

                # 基础字典（所有数据都包含的字段）
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
                    # 构造结束状态所需的临时列（用于最后一行）
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

                # 如果是测试集，额外保留原始成本、获胜数和曝光数列，用于后续基准计算
                if self.dataset_type == 'test':
                    record['cost'] = float(row['cost'])
                    record['win_num'] = int(row['win_num'])
                    record['pv_num'] = int(row['pv_num'])

                training_rows.append(record)

        if not training_rows:
            return pd.DataFrame()

        training_df = pd.DataFrame(training_rows)

        # 按组排序（确保顺序正确，为 shift 做准备）
        training_df = training_df.sort_values(
            ['deliveryPeriodIndex', 'advertiserNumber', 'timeStepIndex']
        ).reset_index(drop=True)

        # 使用 shift(-1) 生成 next_state（基于元组的 state 列）
        training_df['next_state'] = training_df.groupby(
            ['deliveryPeriodIndex', 'advertiserNumber']
        )['state'].shift(-1)

        # 针对最后一行（done == 1）构造结束状态并覆盖
        mask_last = training_df['done'] == 1
        for idx in training_df[mask_last].index:
            row = training_df.loc[idx]
            # 构造结束状态元组
            end_state = (
                0.0,  # timeleft_end
                float(row['remain_budget'] / row['budget'] if row['budget'] > 0 else 0.0),  # bgtleft_end
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

        # 将 state 和 next_state 转为字符串
        training_df['state'] = training_df['state'].apply(str)
        training_df['next_state'] = training_df['next_state'].apply(str)

        # 删除临时列（不影响测试集新增的 cost/win_num/pv_num）
        cols_to_drop = ['remain_budget', 'avg_pacer_all_now', 'avg_pacer_last3_now',
                        'avg_ci_mean_all_now', 'avg_ci_mean_last3_now',
                        'avg_pvalue_mean_all_now', 'avg_pvalue_mean_last3_now',
                        'avg_conversion_per_pv_all_now', 'avg_conversion_per_pv_last3_now',
                        'avg_win_prob_all_now', 'avg_win_prob_last3_now',
                        'pvalue_mean_agg', 'pv_num_agg', 'last_3_volume_now', 'historical_volume_now']
        training_df.drop(columns=[c for c in cols_to_drop if c in training_df.columns], inplace=True)

        return training_df


def run_generate_train_data():
    """分别处理训练集和测试集，显式传入 dataset_type"""
    # 训练数据
    generator_train = TrainDataGenerator(
        file_folder_path="./data/raw_data/train",
        training_data_path="./data/train",
        dataset_type='train'
    )
    generator_train.batch_generate_train_data()
    
    # 测试数据
    generator_test = TrainDataGenerator(
        file_folder_path="./data/raw_data/test",
        training_data_path="./data/test/trajectory",  # 测试输出路径
        dataset_type='test'
    )
    generator_test.batch_generate_train_data()


if __name__ == '__main__':
    run_generate_train_data()