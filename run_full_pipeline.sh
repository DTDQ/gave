#!/bin/bash
# run_full_pipeline.sh - 完整训练+测试管道（带详细数据流日志）
#
# 用法:
#   ./run_full_pipeline.sh [实验名称]
#   例如: ./run_full_pipeline.sh exp_20260324
#
# 流程:
#   Step 1: 从 20260206 raw data 生成训练 RL CSV
#   Step 2: 分割 train/val
#   Step 3: 模型训练
#   Step 4: 从 20260207 raw data + env_data 生成测试数据 (内存优化)
#   Step 5: 离线回放测试

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# ==================== 配置 ====================
CONDA_ROOT="${CONDA_ROOT:-/home/deborah/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-gave_env}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ROOT/envs/$CONDA_ENV_NAME/bin/python}"
DEVICE="${DEVICE:-cuda}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

# 数据路径
TRAIN_RAW="./data/raw_data/train_pt_d=20260206/train_data.txt"
TEST_RAW="./data/raw_data/train_pt_d=20260207/train_data.txt"
ENV_RAW="./data/raw_data/env_pt_d=20260207/env_data.txt"

# 实验配置
EXP_NAME="${1:-exp_$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="./experiments"
EXP_PATH="$EXP_ROOT/$EXP_NAME"

# 训练参数
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
SAVE_STEP="${SAVE_STEP:-2000}"
EVAL_STEP="${EVAL_STEP:-500}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-50000}"
TRAIN_RATIO="${TRAIN_RATIO:-0.99}"
VAL_RATIO="${VAL_RATIO:-0.01}"

if [[ "$DEVICE" == "cuda" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
fi

# ==================== 辅助函数 ====================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_section() {
    echo ""
    echo "================================================================"
    echo "  $*"
    echo "================================================================"
}

check_file() {
    local path="$1"
    local desc="$2"
    if [[ -f "$path" ]]; then
        local size=$(ls -lh "$path" | awk '{print $5}')
        local lines=$(wc -l < "$path")
        log "  [OK] $desc: $path ($size, $lines 行)"
    else
        log "  [MISSING] $desc: $path"
        return 1
    fi
}

# ==================== 环境检查 ====================
log_section "环境检查"

if [[ ! -x "$PYTHON_BIN" ]]; then
    log "错误: Python 解释器不存在: $PYTHON_BIN"
    exit 1
fi
log "Python: $PYTHON_BIN"
log "设备: $DEVICE (GPU: $CUDA_DEVICE)"
log "实验目录: $EXP_PATH"
echo ""

log "检查原始数据文件..."
check_file "$TRAIN_RAW" "训练原始数据 (20260206)"
check_file "$TEST_RAW"  "测试原始数据 (20260207)"
check_file "$ENV_RAW"   "环境数据 (20260207)"

# 创建实验目录结构
mkdir -p "$EXP_PATH"/{data,models,logs,results}

# ==================== Step 1: 训练数据生成 ====================
log_section "Step 1: 训练数据生成 (20260206 raw → RL CSV)"

TRAIN_RL_CSV="$EXP_PATH/data/train_data_d20260206-rlData.csv"

log "输入: $TRAIN_RAW"
log "输出: $TRAIN_RL_CSV"
log "采样轨迹数: $NUM_TRAJECTORIES"
log ""
log "数据处理详情:"
log "  1. 读取 TXT (15列, 无表头)"
log "  2. 数据清洗: 列类型转换, ci_mean /= 1000"
log "  3. 过滤: 全天总 win_num < 1000 的广告组被移除"
log "  4. 轨迹完整性: timestep 必须从1开始且连续递增"
log "  5. 随机采样 $NUM_TRAJECTORIES 条完整轨迹"
log "  6. 构造19维状态向量 + action/reward/done/next_state"
log ""

export TRAIN_RAW TRAIN_RL_CSV NUM_TRAJECTORIES
"$PYTHON_BIN" - <<'PYSTEP1'
import os, sys, time
sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from run.run_train_data_generator import TrainDataGenerator
import pandas as pd

t0 = time.time()

raw_file = os.environ["TRAIN_RAW"]
output_path = os.environ["TRAIN_RL_CSV"]
num_traj = int(os.environ.get("NUM_TRAJECTORIES", "50000"))

print(f"\n--- 读取原始文件 ---")
print(f"文件: {raw_file}")

# 先统计原始数据规模
df_raw = pd.read_csv(raw_file, header=None, usecols=[0], names=['adgroup_id'])
total_rows = len(df_raw)
unique_adids = df_raw['adgroup_id'].nunique()
print(f"原始数据: {total_rows} 行, {unique_adids} 个唯一广告组ID")
del df_raw

t1 = time.time()
print(f"(预扫描耗时: {t1-t0:.1f}s)")

print(f"\n--- 开始 RL 数据生成 ---")
generator = TrainDataGenerator(
    file_paths=[raw_file],
    output_path=output_path,
    num_trajectories=num_traj
)
generator.batch_generate_train_data()

t2 = time.time()
print(f"\n--- 训练数据生成完成 ---")
print(f"总耗时: {t2-t0:.1f}s")

# 输出数据质量统计
if os.path.exists(output_path):
    df = pd.read_csv(output_path, usecols=['advertiserNumber', 'timeStepIndex', 'action', 'reward', 'done'])
    n_rows = len(df)
    n_trajs = df[df['done'] == 1].shape[0]
    avg_len = n_rows / max(n_trajs, 1)
    print(f"\n--- 输出数据统计 ---")
    print(f"总行数: {n_rows}")
    print(f"轨迹数: {n_trajs}")
    print(f"平均轨迹长度: {avg_len:.1f} 步")
    print(f"Action 范围: [{df['action'].min():.4f}, {df['action'].max():.4f}]")
    print(f"Reward 范围: [{df['reward'].min():.4f}, {df['reward'].max():.4f}]")
    print(f"Reward > 0 的行占比: {(df['reward'] > 0).mean()*100:.1f}%")
PYSTEP1

log ""
check_file "$TRAIN_RL_CSV" "训练 RL 数据"

# ==================== Step 2: 数据分割 ====================
log_section "Step 2: 数据分割 (RL CSV → train.csv + val.csv)"

log "输入: $TRAIN_RL_CSV"
log "输出目录: $EXP_PATH/data/"
log "分割比例: train=$TRAIN_RATIO, val=$VAL_RATIO"
log "轨迹长度限制: [3, 100]"
log ""
log "分割逻辑:"
log "  1. 按 (advertiserNumber, timeStepIndex) 排序"
log "  2. 按 (deliveryPeriodIndex, advertiserNumber) 分组"
log "  3. 检查轨迹完整性 (从1开始, 连续递增, 长度3-100)"
log "  4. 随机打乱 (seed=0)"
log "  5. 按比例分割"
log ""

"$PYTHON_BIN" ./code/main/split_data.py \
    --input_csv "$TRAIN_RL_CSV" \
    --output_dir "$EXP_PATH/data" \
    --train_ratio "$TRAIN_RATIO" \
    --val_ratio "$VAL_RATIO" \
    --seed 0 \
    --min_traj_length 3 \
    --max_traj_length 100

log ""
check_file "$EXP_PATH/data/train.csv" "训练集"
check_file "$EXP_PATH/data/val.csv"   "验证集"
check_file "$EXP_PATH/data/stats.json" "统计信息"

log ""
log "Stats.json 内容:"
cat "$EXP_PATH/data/stats.json"
echo ""

# ==================== Step 3: 模型训练 ====================
log_section "Step 3: 模型训练"

log "数据目录: $EXP_PATH/data/"
log "模型保存: $EXP_PATH/models/"
log "训练步数: $TRAIN_STEPS"
log "批次大小: $BATCH_SIZE"
log "学习率: $LEARNING_RATE"
log "保存间隔: $SAVE_STEP 步"
log "评估间隔: $EVAL_STEP 步"
log ""

TRAIN_LOG="$EXP_PATH/logs/train_$(date +%Y%m%d_%H%M%S).log"

"$PYTHON_BIN" ./code/main/train_dt.py \
    --data_dir "$EXP_PATH/data" \
    --save_dir "$EXP_PATH/models" \
    --step_num "$TRAIN_STEPS" \
    --save_step "$SAVE_STEP" \
    --eval_step "$EVAL_STEP" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --device "$DEVICE" \
    --seed 42 2>&1 | tee "$TRAIN_LOG"

# 获取最新模型目录
LATEST_MODEL=$(ls -td "$EXP_PATH"/models/DT_* 2>/dev/null | head -1)
log ""
log "最新模型目录: $LATEST_MODEL"
log "模型文件:"
ls -lh "$LATEST_MODEL"/*.pt 2>/dev/null | while read line; do log "  $line"; done

# ==================== Step 4: 测试数据生成 (内存优化) ====================
log_section "Step 4: 测试数据生成 (20260207 raw + env_data → test CSV)"

TEST_OUTPUT_DIR="./data/test/pt_d=20260207"
mkdir -p "$TEST_OUTPUT_DIR"

log "输入:"
log "  测试日志: $TEST_RAW"
log "  环境数据: $ENV_RAW ($(ls -lh "$ENV_RAW" | awk '{print $5}'))"
log "输出目录: $TEST_OUTPUT_DIR"
log ""
log "内存优化策略:"
log "  1. 用 shell cut 提取 env_data 中的唯一 adgroup_ids (避免加载13.7GB)"
log "  2. 只加载 train_data.txt (1.5GB) 生成 RL 数据"
log "  3. 用 awk 按有效 adgroup_ids 过滤 env_data (逐行处理, 不占内存)"
log ""

# Step 4a: 提取 env adgroup_ids
log "Step 4a: 提取环境数据 adgroup_ids..."
ENV_ADIDS_FILE="$TEST_OUTPUT_DIR/env_adgroup_ids.txt"
cut -d',' -f1 "$ENV_RAW" | sort -u > "$ENV_ADIDS_FILE"
ENV_ADID_COUNT=$(wc -l < "$ENV_ADIDS_FILE")
log "  环境数据中唯一广告组数: $ENV_ADID_COUNT"

# Step 4b: 生成测试 RL 数据
log ""
log "Step 4b: 生成测试 RL 数据..."
log "  过滤逻辑:"
log "    1. 全天 win_num >= 1000"
log "    2. adgroup_id 在 env_data 中存在"
log "    3. 轨迹完整性 (timestep从1开始, 连续)"
log "    4. 总转化 > 0"
log ""

export TEST_RAW ENV_ADIDS_FILE TEST_OUTPUT_DIR
"$PYTHON_BIN" - <<'PYSTEP4'
import os, sys, re, time
import numpy as np
import pandas as pd
from tqdm import tqdm

t0 = time.time()

RAW_TEST = os.environ["TEST_RAW"]
ENV_ADIDS_FILE = os.environ["ENV_ADIDS_FILE"]
OUTPUT_DIR = os.environ["TEST_OUTPUT_DIR"]

TEST_COLUMNS = [
    'adgroup_id', 'effect_type', 'budget', 'tcpa', 'timestep',
    'remain_budget', 'pvalue_mean', 'pacer', 'win_num', 'win_rate',
    'cost', 'pvalue_sum', 'ci_mean', 'pv_num', 'pt_d'
]

# 1. 读取 env adgroup_ids
print("--- 读取环境 adgroup_ids ---")
env_adgroup_ids = set()
with open(ENV_ADIDS_FILE, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            env_adgroup_ids.add(int(line))
print(f"环境文件中包含 {len(env_adgroup_ids)} 个广告组ID")

# 2. 读取测试原始数据
print(f"\n--- 读取测试原始数据 ---")
print(f"文件: {RAW_TEST}")
df = pd.read_csv(RAW_TEST, header=None, names=TEST_COLUMNS, dtype=str, on_bad_lines='skip')
print(f"原始行数: {len(df)}")
print(f"原始唯一广告组: {df['adgroup_id'].nunique()}")

# 清洗
for col in df.columns:
    if col in ['adgroup_id', 'timestep']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    else:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
df = df.sort_values(['adgroup_id', 'timestep']).reset_index(drop=True)
df['ci_mean'] = df['ci_mean'] / 1000.0

# 3. 过滤 win_num < 1000
print(f"\n--- 过滤: win_num < 1000 ---")
adid_total_win = df.groupby('adgroup_id')['win_num'].sum()
valid_adids = adid_total_win[adid_total_win >= 1000].index
before = df['adgroup_id'].nunique()
df = df[df['adgroup_id'].isin(valid_adids)].reset_index(drop=True)
after = df['adgroup_id'].nunique()
print(f"过滤前: {before} 个广告组")
print(f"过滤后: {after} 个广告组 (移除 {before - after})")

# 4. 与 env adgroup_ids 交集
adids_in_both = set(df['adgroup_id'].unique()) & env_adgroup_ids
print(f"\n--- 交集: 测试数据 ∩ 环境数据 ---")
print(f"测试数据广告组: {after}")
print(f"环境数据广告组: {len(env_adgroup_ids)}")
print(f"交集: {len(adids_in_both)}")

# 准备衍生字段
dir_name = os.path.basename(os.path.dirname(RAW_TEST))
numbers = re.findall(r'\d+', dir_name)
deliveryPeriodIndex = int(numbers[0]) if numbers else 1

df['deliveryPeriodIndex'] = deliveryPeriodIndex
df['advertiserCategoryIndex'] = 0
df['conversion_per_pv'] = df['pvalue_sum'] / df['pv_num'].replace(0, 1)
df['win_prob'] = df['win_num'] / df['pv_num'].replace(0, 1)

# 5. 逐组生成 RL 数据
print(f"\n--- 生成 RL 数据 ---")
training_rows = []
skipped_incomplete = 0
skipped_no_env = 0
skipped_no_conversion = 0
final_valid_adids = set()

group_keys = ['deliveryPeriodIndex', 'adgroup_id', 'advertiserCategoryIndex', 'budget', 'tcpa']
groups = list(df.groupby(group_keys))
print(f"总组数: {len(groups)}")

for _, group in tqdm(groups, desc="处理广告组"):
    group = group.sort_values('timestep').reset_index(drop=True)
    if len(group) == 0:
        continue

    adgroup_id = group['adgroup_id'].iloc[0]
    if adgroup_id not in env_adgroup_ids:
        skipped_no_env += 1
        continue

    timesteps = group['timestep'].values
    if timesteps[0] != 1 or not np.array_equal(timesteps, np.arange(1, len(timesteps) + 1)):
        skipped_incomplete += 1
        continue

    max_timestep = group['timestep'].max()
    group['isEnd'] = (group['timestep'] == max_timestep).astype(int)

    pv_by_step = group.groupby('timestep')['pv_num'].first()
    group['historical_volume'] = group['timestep'].map(pv_by_step.cumsum().shift(1).fillna(0).astype(int))
    group['last_3_volume'] = group['timestep'].map(pv_by_step.rolling(3, min_periods=1).sum().shift(1).fillna(0).astype(int))

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
    group['last_3_volume_now'] = group['timestep'].map(pv_by_step.rolling(3, min_periods=1).sum().fillna(0).astype(int))
    group['pvalue_mean_agg'] = group['pvalue_mean']
    group['pv_num_agg'] = group['pv_num']

    realAllCost = group['cost'].sum()
    realAllConversion = group['pvalue_sum'].sum()
    if realAllConversion <= 0:
        skipped_no_conversion += 1
        continue

    group['prev_remain_budget'] = group['remain_budget'].shift(1)

    states_start = {}
    for _, row in group.iterrows():
        t = int(row['timestep'])
        timeleft_start = (max_timestep - (t - 1)) / max_timestep if max_timestep > 0 else 0.0
        bgtleft_start = 1.0 if t == 1 else (row['prev_remain_budget'] / row['budget'] if row['budget'] > 0 else 0.0)
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
            'deliveryPeriodIndex': deliveryPeriodIndex, 'advertiserNumber': row['adgroup_id'],
            'advertiserCategoryIndex': 0, 'budget': row['budget'], 'CPAConstraint': row['tcpa'],
            'realAllCost': realAllCost, 'realAllConversion': realAllConversion,
            'timeStepIndex': t, 'state': states_start[t],
            'action': float(row['pacer'] * row['tcpa']),
            'reward': float(row['pvalue_sum']), 'reward_continuous': float(row['pvalue_sum']),
            'done': 1 if t == max_timestep else 0,
            'remain_budget': row['remain_budget'],
            'avg_pacer_all_now': row['avg_pacer_all_now'], 'avg_pacer_last3_now': row['avg_pacer_last3_now'],
            'avg_ci_mean_all_now': row['avg_ci_mean_all_now'], 'avg_ci_mean_last3_now': row['avg_ci_mean_last3_now'],
            'avg_pvalue_mean_all_now': row['avg_pvalue_mean_all_now'], 'avg_pvalue_mean_last3_now': row['avg_pvalue_mean_last3_now'],
            'avg_conversion_per_pv_all_now': row['avg_conversion_per_pv_all_now'], 'avg_conversion_per_pv_last3_now': row['avg_conversion_per_pv_last3_now'],
            'avg_win_prob_all_now': row['avg_win_prob_all_now'], 'avg_win_prob_last3_now': row['avg_win_prob_last3_now'],
            'pvalue_mean_agg': row['pvalue_mean_agg'], 'pv_num_agg': row['pv_num_agg'],
            'last_3_volume_now': row['last_3_volume_now'], 'historical_volume_now': row['historical_volume_now'],
            'avg_cost_all_now': row['avg_cost_all_now'], 'avg_cost_last3_now': row['avg_cost_last3_now'],
            'cumulative_cost_ratio_now': row['cumulative_cost_ratio_now'],
            'cost': float(row['cost']), 'win_num': int(row['win_num']), 'pv_num': int(row['pv_num']),
        }
        training_rows.append(record)
    final_valid_adids.add(adgroup_id)

# 统计
print(f"\n--- 过滤统计 ---")
print(f"跳过 (不在env中): {skipped_no_env}")
print(f"跳过 (轨迹不完整): {skipped_incomplete}")
print(f"跳过 (无转化): {skipped_no_conversion}")
print(f"有效广告组: {len(final_valid_adids)}")

if not training_rows:
    print("错误: 没有生成任何有效轨迹！")
    sys.exit(1)

training_df = pd.DataFrame(training_rows)
training_df = training_df.sort_values(['deliveryPeriodIndex', 'advertiserNumber', 'timeStepIndex']).reset_index(drop=True)

training_df['next_state'] = training_df.groupby(['deliveryPeriodIndex', 'advertiserNumber'])['state'].shift(-1)
mask_last = training_df['done'] == 1
for idx in training_df[mask_last].index:
    row = training_df.loc[idx]
    end_state = (
        0.0, float(row['remain_budget'] / row['budget'] if row['budget'] > 0 else 0.0),
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

rl_output = os.path.join(OUTPUT_DIR, "train_data_rlData.csv")
training_df.to_csv(rl_output, index=False)

adids_file = os.path.join(OUTPUT_DIR, "valid_adgroup_ids.txt")
with open(adids_file, 'w') as f:
    for aid in sorted(final_valid_adids):
        f.write(f"{aid}\n")

t1 = time.time()
print(f"\n--- 测试 RL 数据生成完成 ---")
print(f"输出: {rl_output} ({len(training_df)} 行, {len(final_valid_adids)} 条轨迹)")
print(f"有效ID: {adids_file}")
print(f"耗时: {t1-t0:.1f}s")
PYSTEP4

log ""
check_file "$TEST_OUTPUT_DIR/train_data_rlData.csv" "测试 RL 数据"
check_file "$TEST_OUTPUT_DIR/valid_adgroup_ids.txt"  "有效广告组 ID 列表"

# Step 4c: 过滤 env_data
log ""
log "Step 4c: 过滤环境数据 (awk 按有效 adgroup_ids 逐行过滤)..."
VALID_ADIDS="$TEST_OUTPUT_DIR/valid_adgroup_ids.txt"
ENV_CSV="$TEST_OUTPUT_DIR/env_data.csv"

VALID_COUNT=$(wc -l < "$VALID_ADIDS")
log "  有效广告组数: $VALID_COUNT"
log "  输入: $ENV_RAW ($(ls -lh "$ENV_RAW" | awk '{print $5}'))"
log "  输出: $ENV_CSV"

echo "adgroup_id,min_win_ecpm,p_ctr,p_cvr,pltv,ocpx_deep_pcvr,log_time,ab_tag,real_ecpm,bid_tag,pt_d" > "$ENV_CSV"
awk -F',' 'NR==FNR{ids[$1]=1; next} ($1 in ids)' "$VALID_ADIDS" "$ENV_RAW" >> "$ENV_CSV"

log ""
check_file "$ENV_CSV" "过滤后环境数据"

ENV_LINES=$(wc -l < "$ENV_CSV")
log "  原始 env_data: $(wc -l < "$ENV_RAW") 行"
log "  过滤后: $ENV_LINES 行 (含表头)"

# ==================== Step 5: 离线回放测试 ====================
log_section "Step 5: 离线回放测试"

if [[ -z "$LATEST_MODEL" || ! -d "$LATEST_MODEL" ]]; then
    log "错误: 未找到训练好的模型目录"
    exit 1
fi

TEST_CSV="$TEST_OUTPUT_DIR/train_data_rlData.csv"
TEST_RESULT_DIR="$EXP_PATH/results/test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_RESULT_DIR"

log "模型: $LATEST_MODEL"
log "模型文件: best_model.pt"
log "测试数据: $TEST_CSV"
log "环境数据: $ENV_CSV"
log "结果输出: $TEST_RESULT_DIR"
log ""

"$PYTHON_BIN" ./code/main/test_dt.py \
    --model_dir "$LATEST_MODEL" \
    --model_name best_model.pt \
    --test_csv "$TEST_CSV" \
    --env_csv "$ENV_CSV" \
    --output_dir "$TEST_RESULT_DIR" \
    --device "$DEVICE" 2>&1

# ==================== 完成 ====================
log_section "全部完成!"

log "实验目录: $EXP_PATH"
log ""
log "输出文件:"
log "  训练 RL 数据:    $TRAIN_RL_CSV"
log "  训练集:          $EXP_PATH/data/train.csv"
log "  验证集:          $EXP_PATH/data/val.csv"
log "  模型目录:        $LATEST_MODEL"
log "  测试 RL 数据:    $TEST_CSV"
log "  环境数据 (过滤): $ENV_CSV"
log "  测试结果:        $TEST_RESULT_DIR"
log ""
log "测试汇总:"
cat "$TEST_RESULT_DIR"/test_summary_*.csv 2>/dev/null || log "  (汇总文件未找到)"
