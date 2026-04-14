#!/bin/bash
# run_test.sh - 灵活的离线测试脚本，支持不同测试集、环境数据和已训练模型
#
# 用法示例:
#   # 1. 指定实验目录 + 测试数据（最常用）
#   ./run_test.sh --exp_dir experiments/exp_winnum_filter \
#                 --test_csv ./data/test/pt_d=20260207/train_data_rlData.csv \
#                 --env_csv ./data/test/pt_d=20260207/env_data.csv
#
#   # 2. 从原始TXT自动生成测试数据再测试
#   ./run_test.sh --exp_dir experiments/exp_winnum_filter \
#                 --test_raw ./data/raw_data/train_pt_d=20260207/train_data.txt \
#                 --env_raw ./data/raw_data/env_pt_d=20260207/env_data.txt
#
#   # 3. 用日期分区自动查找数据（在 data/test/ 和 data/raw_data/ 下查找）
#   ./run_test.sh --exp_dir experiments/exp_winnum_filter --test_date 20260207
#
#   # 4. 直接指定模型目录（跳过自动查找）
#   ./run_test.sh --model_dir experiments/exp_winnum_filter/models/DT_20260319_210240 \
#                 --test_csv ./data/test/pt_d=20260207/train_data_rlData.csv \
#                 --env_csv ./data/test/pt_d=20260207/env_data.csv
#
#   # 5. 只测试特定checkpoint（如 best_model.pt）
#   ./run_test.sh --exp_dir experiments/exp_winnum_filter \
#                 --model_name best_model.pt \
#                 --test_date 20260207
#
#   # 6. 测试多个日期的数据（逗号分隔）
#   ./run_test.sh --exp_dir experiments/exp_winnum_filter \
#                 --test_date 20260206,20260207

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# ==================== 默认配置 ====================
CONDA_ROOT="${CONDA_ROOT:-/home/deborah/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-gave_env}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ROOT/envs/$CONDA_ENV_NAME/bin/python}"
DEVICE="${DEVICE:-cuda}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
TEST_SCRIPT="./code/main/test_dt.py"
TEST_GEN_SCRIPT="./code/run/run_test_data_generator.py"

# 命令行参数默认值
EXP_DIR=""
MODEL_DIR=""
MODEL_NAME=""
TEST_CSV=""
ENV_CSV=""
TEST_RAW=""
ENV_RAW=""
TEST_DATE=""
OUTPUT_DIR=""

# ==================== 参数解析 ====================
usage() {
    cat <<'USAGE'
用法: ./run_test.sh [选项]

必需（三选一指定测试数据）:
  --test_csv PATH       测试RL数据CSV路径
  --test_raw PATH       原始测试TXT路径（自动生成CSV）
  --test_date DATE      日期分区（如 20260207），自动查找数据
                        支持逗号分隔多个日期: 20260206,20260207

必需（与 --test_csv/--test_raw 配合）:
  --env_csv PATH        环境数据CSV路径
  --env_raw PATH        原始环境TXT路径（自动生成CSV）

必需（二选一指定模型）:
  --exp_dir PATH        实验目录（自动查找最新模型）
  --model_dir PATH      模型目录路径（DT_*目录）

可选:
  --model_name NAME     指定模型文件名（如 best_model.pt, 20000.pt）
  --output_dir PATH     输出目录（默认: <exp_dir>/test_results/<timestamp>）
  --device DEVICE       设备: cuda 或 cpu（默认: cuda）
  --cuda_device ID      GPU设备ID（默认: 0）
  -h, --help            显示帮助
USAGE
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp_dir)      EXP_DIR="$2"; shift 2 ;;
        --model_dir)    MODEL_DIR="$2"; shift 2 ;;
        --model_name)   MODEL_NAME="$2"; shift 2 ;;
        --test_csv)     TEST_CSV="$2"; shift 2 ;;
        --env_csv)      ENV_CSV="$2"; shift 2 ;;
        --test_raw)     TEST_RAW="$2"; shift 2 ;;
        --env_raw)      ENV_RAW="$2"; shift 2 ;;
        --test_date)    TEST_DATE="$2"; shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --device)       DEVICE="$2"; shift 2 ;;
        --cuda_device)  CUDA_DEVICE="$2"; shift 2 ;;
        -h|--help)      usage ;;
        *)              echo "未知参数: $1"; usage ;;
    esac
done

# ==================== 环境检查 ====================
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "错误: 未找到 Python 解释器: $PYTHON_BIN"
    exit 1
fi

if [[ "$DEVICE" == "cuda" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
fi

# ==================== 模型目录解析 ====================
if [[ -z "$MODEL_DIR" ]]; then
    if [[ -z "$EXP_DIR" ]]; then
        echo "错误: 必须指定 --exp_dir 或 --model_dir"
        exit 1
    fi
    # 自动查找最新模型目录
    MODEL_DIR=$(ls -td "$EXP_DIR"/models/DT_* 2>/dev/null | head -1)
    if [[ -z "$MODEL_DIR" ]]; then
        echo "错误: 在 $EXP_DIR/models/ 下未找到任何 DT_* 模型目录"
        exit 1
    fi
fi

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "错误: 模型目录不存在: $MODEL_DIR"
    exit 1
fi

# 如果没有指定 EXP_DIR，从 MODEL_DIR 推断
if [[ -z "$EXP_DIR" ]]; then
    EXP_DIR=$(dirname "$(dirname "$MODEL_DIR")")
fi

echo "============================================"
echo "  GAVE 离线测试"
echo "============================================"
echo "模型目录: $MODEL_DIR"
if [[ -n "$MODEL_NAME" ]]; then
    echo "指定模型: $MODEL_NAME"
fi

# ==================== 辅助函数: 从原始数据生成测试CSV ====================
generate_test_data() {
    local raw_test="$1"
    local raw_env="$2"
    local out_dir="$3"

    echo "从原始数据生成测试CSV..."
    echo "  测试文件: $raw_test"
    echo "  环境文件: $raw_env"
    echo "  输出目录: $out_dir"

    "$PYTHON_BIN" - "$raw_test" "$raw_env" "$out_dir" <<'PYGEN'
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))) if len(sys.argv) > 0 else os.getcwd()
# 确保在项目根目录
os.chdir(os.environ.get('PROJECT_ROOT', '.'))
sys.path.insert(0, os.path.join(os.getcwd(), "code"))

from run.run_test_data_generator import TestDataGenerator

test_file = sys.argv[1]
env_file = sys.argv[2]
output_dir = sys.argv[3]

generator = TestDataGenerator(
    test_file=test_file,
    env_file=env_file,
    output_dir=output_dir,
    dataset_type='test'
)
generator.process()
PYGEN

    echo "测试数据生成完成: $out_dir"
}

# ==================== 运行单次测试 ====================
run_single_test() {
    local test_csv="$1"
    local env_csv="$2"
    local output_dir="$3"
    local label="$4"

    echo ""
    echo "---------- 测试: $label ----------"
    echo "  测试数据: $test_csv"
    echo "  环境数据: $env_csv"
    echo "  输出目录: $output_dir"

    # 检查文件存在
    if [[ ! -f "$test_csv" ]]; then
        echo "错误: 测试数据文件不存在: $test_csv"
        return 1
    fi
    if [[ ! -f "$env_csv" ]]; then
        echo "错误: 环境数据文件不存在: $env_csv"
        return 1
    fi

    mkdir -p "$output_dir"

    # 构建命令
    local cmd=("$PYTHON_BIN" "$TEST_SCRIPT"
        --model_dir "$MODEL_DIR"
        --test_csv "$test_csv"
        --env_csv "$env_csv"
        --output_dir "$output_dir"
        --device "$DEVICE"
    )

    if [[ -n "$MODEL_NAME" ]]; then
        cmd+=(--model_name "$MODEL_NAME")
    fi

    "${cmd[@]}"

    echo "测试完成: $output_dir"
}

# ==================== 数据解析与测试执行 ====================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 情况1: 通过 --test_date 指定日期分区
if [[ -n "$TEST_DATE" ]]; then
    IFS=',' read -ra DATES <<< "$TEST_DATE"

    for date in "${DATES[@]}"; do
        date=$(echo "$date" | tr -d ' ')
        echo ""
        echo "========== 处理日期分区: $date =========="

        # 查找已有的测试CSV
        found_test_csv=""
        found_env_csv=""

        # 优先查找已处理好的CSV
        for pattern in \
            "./data/test/pt_d=${date}/train_data_rlData.csv" \
            "./data/test/pt_d=${date}/*rlData*.csv" \
            "./data/test/${date}/*rlData*.csv"; do
            matches=$(ls $pattern 2>/dev/null | head -1)
            if [[ -n "$matches" ]]; then
                found_test_csv="$matches"
                break
            fi
        done

        for pattern in \
            "./data/test/pt_d=${date}/env_data.csv" \
            "./data/test/pt_d=${date}/env*.csv" \
            "./data/test/${date}/env*.csv"; do
            matches=$(ls $pattern 2>/dev/null | head -1)
            if [[ -n "$matches" ]]; then
                found_env_csv="$matches"
                break
            fi
        done

        # 如果CSV不存在，尝试从原始数据生成
        if [[ -z "$found_test_csv" || -z "$found_env_csv" ]]; then
            echo "未找到预处理的测试CSV，尝试从原始数据生成..."

            raw_test=""
            raw_env=""

            for pattern in \
                "./data/raw_data/train_pt_d=${date}/train_data.txt" \
                "./data/raw_data/test_pt_d=${date}/train_data.txt"; do
                if [[ -f "$pattern" ]]; then
                    raw_test="$pattern"
                    break
                fi
            done

            for pattern in \
                "./data/raw_data/env_pt_d=${date}/env_data.txt" \
                "./data/raw_data/test_env_pt_d=${date}/env_data.txt"; do
                if [[ -f "$pattern" ]]; then
                    raw_env="$pattern"
                    break
                fi
            done

            if [[ -n "$raw_test" && -n "$raw_env" ]]; then
                gen_out="./data/test/pt_d=${date}"
                export PROJECT_ROOT
                generate_test_data "$raw_test" "$raw_env" "$gen_out"

                # 重新查找生成的文件
                found_test_csv=$(ls "${gen_out}"/*rlData*.csv 2>/dev/null | head -1)
                found_env_csv=$(ls "${gen_out}"/env*.csv 2>/dev/null | head -1)
            else
                echo "错误: 未找到日期 $date 的原始数据"
                [[ -z "$raw_test" ]] && echo "  缺少测试文件: data/raw_data/train_pt_d=${date}/train_data.txt"
                [[ -z "$raw_env" ]] && echo "  缺少环境文件: data/raw_data/env_pt_d=${date}/env_data.txt"
                continue
            fi
        fi

        if [[ -z "$found_test_csv" || -z "$found_env_csv" ]]; then
            echo "错误: 日期 $date 的测试数据不完整"
            continue
        fi

        # 设置输出目录
        if [[ -n "$OUTPUT_DIR" ]]; then
            test_output="$OUTPUT_DIR/date_${date}"
        else
            test_output="$EXP_DIR/test_results/${TIMESTAMP}/date_${date}"
        fi

        run_single_test "$found_test_csv" "$found_env_csv" "$test_output" "date=$date"
    done

# 情况2: 通过 --test_raw / --env_raw 指定原始文件
elif [[ -n "$TEST_RAW" && -n "$ENV_RAW" ]]; then
    if [[ ! -f "$TEST_RAW" ]]; then
        echo "错误: 原始测试文件不存在: $TEST_RAW"
        exit 1
    fi
    if [[ ! -f "$ENV_RAW" ]]; then
        echo "错误: 原始环境文件不存在: $ENV_RAW"
        exit 1
    fi

    # 生成到临时目录
    gen_out="./data/test/_generated_${TIMESTAMP}"
    export PROJECT_ROOT
    generate_test_data "$TEST_RAW" "$ENV_RAW" "$gen_out"

    # 查找生成的文件
    gen_test_csv=$(ls "${gen_out}"/*rlData*.csv 2>/dev/null | head -1)
    gen_env_csv=$(ls "${gen_out}"/env*.csv 2>/dev/null | head -1)

    if [[ -z "$gen_test_csv" || -z "$gen_env_csv" ]]; then
        echo "错误: 测试数据生成失败"
        exit 1
    fi

    if [[ -n "$OUTPUT_DIR" ]]; then
        test_output="$OUTPUT_DIR"
    else
        test_output="$EXP_DIR/test_results/$TIMESTAMP"
    fi

    run_single_test "$gen_test_csv" "$gen_env_csv" "$test_output" "raw_data"

# 情况3: 直接指定 --test_csv / --env_csv
elif [[ -n "$TEST_CSV" && -n "$ENV_CSV" ]]; then
    if [[ -n "$OUTPUT_DIR" ]]; then
        test_output="$OUTPUT_DIR"
    else
        test_output="$EXP_DIR/test_results/$TIMESTAMP"
    fi

    run_single_test "$TEST_CSV" "$ENV_CSV" "$test_output" "custom"

else
    echo "错误: 必须指定测试数据来源，支持以下方式:"
    echo "  1. --test_csv + --env_csv     (已有CSV文件)"
    echo "  2. --test_raw + --env_raw     (原始TXT文件，自动生成CSV)"
    echo "  3. --test_date DATE           (按日期分区自动查找)"
    echo ""
    echo "运行 ./run_test.sh --help 查看完整用法"
    exit 1
fi

echo ""
echo "============================================"
echo "  所有测试完成!"
echo "============================================"
