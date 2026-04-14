#!/bin/bash
# 网格搜索：CPA阈值 x lambda_cpa 系数
# 阈值: 0, 20, 50, 100
# 系数: 25, 50
# 共 8 组实验

set -e

PROJECT_ROOT="/home/deborah/Desktop/gave"
DATA_DIR="$PROJECT_ROOT/experiments/exp_20260325/data"
BASE_SAVE_DIR="$PROJECT_ROOT/experiments/grid_search_cpa/models"
TEST_CSV="$PROJECT_ROOT/data/test/pt_d=20260207/test_500_rlData.csv"
ENV_CSV="$PROJECT_ROOT/data/test/pt_d=20260207/env_500.csv"
RESULTS_DIR="$PROJECT_ROOT/experiments/grid_search_cpa/results"

CONDA="/home/deborah/miniconda3/bin/conda"

THRESHOLDS=(0 20 50 100)
LAMBDAS=(25 50)

mkdir -p "$RESULTS_DIR"

cd "$PROJECT_ROOT"

for threshold in "${THRESHOLDS[@]}"; do
    for lambda in "${LAMBDAS[@]}"; do
        EXP_NAME="thresh${threshold}_lambda${lambda}"
        SAVE_DIR="$BASE_SAVE_DIR/$EXP_NAME"
        TEST_OUT="$RESULTS_DIR/$EXP_NAME"

        echo ""
        echo "========================================"
        echo "训练: threshold=${threshold}%, lambda_cpa=${lambda}"
        echo "========================================"

        $CONDA run -n gave_env python code/main/train_dt.py \
            --data_dir "$DATA_DIR" \
            --save_dir "$SAVE_DIR" \
            --step_num 20000 \
            --save_step 2000 \
            --eval_step 500 \
            --batch_size 64 \
            --learning_rate 0.0001 \
            --loss_report 500 \
            --lambda_cpa "$lambda" \
            --cpa_threshold "$threshold" \
            --device cuda 2>&1 | tail -5

        # 找到训练产生的模型目录（DT_开头的最新目录）
        MODEL_DIR=$(ls -td "$SAVE_DIR"/DT_* 2>/dev/null | head -1)
        if [ -z "$MODEL_DIR" ]; then
            echo "错误: 未找到模型目录"
            continue
        fi

        echo ""
        echo "测试: $EXP_NAME"
        $CONDA run -n gave_env python code/main/test_dt.py \
            --model_dir "$MODEL_DIR" \
            --model_name best_model.pt \
            --test_csv "$TEST_CSV" \
            --env_csv "$ENV_CSV" \
            --output_dir "$TEST_OUT" \
            --device cuda 2>&1 | tail -5

        echo ""
    done
done

echo ""
echo "========================================"
echo "所有实验完成！汇总结果："
echo "========================================"

# 汇总所有实验结果
$CONDA run -n gave_env python3 -c "
import pandas as pd
import os
import glob

results_dir = '$RESULTS_DIR'
all_rows = []
for d in sorted(os.listdir(results_dir)):
    summary_files = glob.glob(os.path.join(results_dir, d, 'test_summary_*.csv'))
    if not summary_files:
        continue
    df = pd.read_csv(summary_files[0])
    model_row = df[df['type'] == 'model'].iloc[0].to_dict()
    model_row['experiment'] = d
    all_rows.append(model_row)

if all_rows:
    result_df = pd.DataFrame(all_rows)
    cols = ['experiment', 'avg_conversion', 'avg_CPA', 'avg_CPA_violation', 'avg_budget_used', 'avg_win_rate', 'avg_spend']
    result_df = result_df[[c for c in cols if c in result_df.columns]]
    print(result_df.to_string(index=False))
    result_df.to_csv(os.path.join(results_dir, 'grid_summary.csv'), index=False)
    print(f'\n汇总已保存至: {results_dir}/grid_summary.csv')
"
