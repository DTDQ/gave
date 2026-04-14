#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

CONDA_ROOT="${CONDA_ROOT:-/home/deborah/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-gave_env}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ROOT/envs/$CONDA_ENV_NAME/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "错误: 未找到 Python 解释器: $PYTHON_BIN"
    exit 1
fi

# 配置
# 训练数据：仅使用 20260206
TRAIN_RAW="${TRAIN_RAW:-./data/raw_data/train_pt_d=20260206/train_data.txt}"
INPUT_DATA="${INPUT_DATA:-./data/train/train_data_d20260206-rlData.csv}"
# 测试数据：使用 20260207
TEST_RAW="${TEST_RAW:-./data/raw_data/train_pt_d=20260207/train_data.txt}"
ENV_RAW="${ENV_RAW:-./data/raw_data/env_pt_d=20260207/env_data.txt}"
TEST_CSV="${TEST_CSV:-./data/test/pt_d=20260207/train_data_rlData.csv}"
ENV_CSV="${ENV_CSV:-./data/test/pt_d=20260207/env_data.csv}"
EXP_NAME="${1:-exp_$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-./experiments}"
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
DEVICE="${DEVICE:-cuda}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

if [[ "$DEVICE" == "cuda" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
fi

# 创建实验目录
mkdir -p "$EXP_PATH"/{data,models,logs,results}

echo "实验目录: $EXP_PATH"

# 如果缺少训练 RL 数据，则从 raw_data 自动生成（仅使用 20260206）
if [[ ! -f "$INPUT_DATA" ]]; then
    echo "训练数据不存在，开始从 raw_data 生成: $INPUT_DATA"
    echo "训练原始文件: $TRAIN_RAW"
    mkdir -p "$(dirname "$INPUT_DATA")"

    if [[ ! -f "$TRAIN_RAW" ]]; then
        echo "错误: 训练原始文件不存在: $TRAIN_RAW"
        exit 1
    fi

    export INPUT_DATA NUM_TRAJECTORIES TRAIN_RAW
    "$PYTHON_BIN" - <<'PY'
import os
import sys

project_root = os.getcwd()
sys.path.insert(0, os.path.join(project_root, "code"))

from run.run_train_data_generator import TrainDataGenerator

raw_files = [os.environ["TRAIN_RAW"]]

output_path = os.environ.get("INPUT_DATA", "./data/train/train_data_d20260206-rlData.csv")
num_trajectories = os.environ.get("NUM_TRAJECTORIES", "50000")
num_trajectories = None if num_trajectories.lower() == "none" else int(num_trajectories)

print("使用训练原始文件（仅20260206）:")
for path in raw_files:
    print(" -", path)
print("输出路径:", output_path)
print("轨迹采样数:", num_trajectories)

generator = TrainDataGenerator(
    file_paths=raw_files,
    output_path=output_path,
    num_trajectories=num_trajectories,
)
generator.batch_generate_train_data()
PY
fi

if [[ ! -f "$INPUT_DATA" ]]; then
    echo "错误: 训练 RL 数据仍不存在: $INPUT_DATA"
    exit 1
fi

# 1. 数据分割
echo "1. 数据分割..."
"$PYTHON_BIN" ./code/main/split_data.py \
    --input_csv "$INPUT_DATA" \
    --output_dir "$EXP_PATH/data" \
    --train_ratio "$TRAIN_RATIO" \
    --val_ratio "$VAL_RATIO" \
    --seed 0 \
    --min_traj_length 3 \
    --max_traj_length 100

# 2. 模型训练
echo "2. 模型训练..."
LOG_FILE="$EXP_PATH/logs/train_$(date +%H%M%S).log"
"$PYTHON_BIN" ./code/main/train_dt.py \
    --data_dir "$EXP_PATH/data" \
    --save_dir "$EXP_PATH/models" \
    --step_num "$TRAIN_STEPS" \
    --save_step "$SAVE_STEP" \
    --eval_step "$EVAL_STEP" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --device "$DEVICE" \
    --seed 42 2>&1 | tee "$LOG_FILE"

# 获取最新模型目录
LATEST_MODEL=$(ls -td "$EXP_PATH"/models/DT_* 2>/dev/null | head -1)

# 3. 模型测试（适配新数据格式）
if [[ -n "$LATEST_MODEL" && -d "$LATEST_MODEL" ]]; then
    echo "3. 模型测试..."

    # 如果测试CSV不存在，从 20260207 原始数据自动生成
    if [[ ! -f "$TEST_CSV" || ! -f "$ENV_CSV" ]]; then
        echo "测试CSV不存在，从原始数据生成..."
        if [[ ! -f "$TEST_RAW" ]]; then
            echo "错误: 测试原始文件不存在: $TEST_RAW"
            exit 1
        fi
        if [[ ! -f "$ENV_RAW" ]]; then
            echo "错误: 环境原始文件不存在: $ENV_RAW"
            exit 1
        fi

        TEST_OUT_DIR="$(dirname "$TEST_CSV")"
        mkdir -p "$TEST_OUT_DIR"

        export TEST_RAW ENV_RAW TEST_OUT_DIR
        "$PYTHON_BIN" - <<'PY'
import os
import sys

project_root = os.getcwd()
sys.path.insert(0, os.path.join(project_root, "code"))

from run.run_test_data_generator import TestDataGenerator

generator = TestDataGenerator(
    test_file=os.environ["TEST_RAW"],
    env_file=os.environ["ENV_RAW"],
    output_dir=os.environ["TEST_OUT_DIR"],
    dataset_type='test'
)
generator.process()
PY
        echo "测试数据生成完成: $TEST_OUT_DIR"
    fi

    if [[ ! -f "$TEST_CSV" ]]; then
        echo "错误: 测试文件不存在: $TEST_CSV"
        exit 1
    fi
    if [[ ! -f "$ENV_CSV" ]]; then
        echo "错误: 环境文件不存在: $ENV_CSV"
        exit 1
    fi

    TEST_OUTPUT="$EXP_PATH/results/test_$(date +%H%M%S)"
    mkdir -p "$TEST_OUTPUT"

    "$PYTHON_BIN" ./code/main/test_dt.py \
        --model_dir "$LATEST_MODEL" \
        --test_csv "$TEST_CSV" \
        --env_csv "$ENV_CSV" \
        --output_dir "$TEST_OUTPUT" \
        --device "$DEVICE"

    echo "测试结果: $TEST_OUTPUT"
else
    echo "警告: 未找到模型目录，请检查训练是否成功"
fi
