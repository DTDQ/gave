#!/bin/bash

set -e

# 配置
INPUT_DATA="./data/train/train_data_all-rlData.csv"
EXP_NAME="${1:-exp_$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="./experiments"
EXP_PATH="$EXP_ROOT/$EXP_NAME"

# 训练参数
TRAIN_STEPS=5000
BATCH_SIZE=16
LEARNING_RATE=0.00005

# 选择 GPU（例如 GPU 0）
export CUDA_VISIBLE_DEVICES=1   # 使用 GPU 0

# 创建实验目录
mkdir -p "$EXP_PATH"/{data,models,logs,results}

echo "实验目录: $EXP_PATH"

# 检查输入数据
if [[ ! -f "$INPUT_DATA" ]]; then
    echo "错误: 输入数据不存在: $INPUT_DATA"
    exit 1
fi

# 1. 数据分割
echo "1. 数据分割..."
python ./code/main/split_data.py \
    --input_csv "$INPUT_DATA" \
    --output_dir "$EXP_PATH/data" \
    --train_ratio 0.8 \
    --val_ratio 0.2 \
    --seed 0 \
    --min_traj_length 3 \
    --max_traj_length 100

# 2. 模型训练
echo "2. 模型训练..."
LOG_FILE="$EXP_PATH/logs/train_$(date +%H%M%S).log"
python ./code/main/train_dt.py \
    --data_dir "$EXP_PATH/data" \
    --save_dir "$EXP_PATH/models" \
    --step_num "$TRAIN_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --device cuda \
    --seed 42 2>&1 | tee "$LOG_FILE"

# 获取最新模型目录
LATEST_MODEL=$(ls -td "$EXP_PATH"/models/DT_* 2>/dev/null | head -1)

# 3. 模型测试（修正后的调用）
if [[ -n "$LATEST_MODEL" && -d "$LATEST_MODEL" ]]; then
    echo "3. 模型测试..."
    TEST_OUTPUT="$EXP_PATH/results/test_$(date +%H%M%S)"
    mkdir -p "$TEST_OUTPUT"
    
    # ========== 关键修改：移除 --data_dir，添加 --test_csv 和 --env_csv ==========
    python ./code/main/test_dt.py \
        --model_dir "$LATEST_MODEL" \
        --test_csv "data/test/trajectory/test_data_all-rlData.csv" \
        --env_csv "data/test/env/64160896_env.xlsx" \
        --output_dir "$TEST_OUTPUT" \
        --device cuda
    
    echo "测试结果: $TEST_OUTPUT"
else
    echo "警告: 未找到模型目录，请检查训练是否成功"
fi