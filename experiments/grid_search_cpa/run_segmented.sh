#!/bin/bash
# 分段训练实验：按 tCPA 拆分为 3 个子模型
# 段: low(0-10), mid(10-50), high(50+)
# 无 loss4，纯 loss1+loss2+loss3

set -e

PROJECT_ROOT="/home/deborah/Desktop/gave"
TRAIN_CSV="$PROJECT_ROOT/experiments/exp_20260325/data/train.csv"
VAL_CSV="$PROJECT_ROOT/experiments/exp_20260325/data/val.csv"
TEST_CSV="$PROJECT_ROOT/data/test/pt_d=20260207/train_data_rlData.csv"
ENV_CSV="$PROJECT_ROOT/data/test/pt_d=20260207/env_data.csv"

EXP_DIR="$PROJECT_ROOT/experiments/grid_search_cpa/segmented"
DATA_DIR="$EXP_DIR/data"
MODELS_DIR="$EXP_DIR/models"
RESULTS_DIR="$EXP_DIR/results"

CONDA="/home/deborah/miniconda3/bin/conda"

mkdir -p "$DATA_DIR" "$MODELS_DIR" "$RESULTS_DIR"

cd "$PROJECT_ROOT"

# ========== Step 1: 按 tCPA 拆分数据 ==========
echo "=========================================="
echo "Step 1: 按 tCPA 拆分训练/验证/测试数据"
echo "=========================================="

$CONDA run -n gave_env python3 -c "
import pandas as pd
import os

data_dir = '$DATA_DIR'

segments = {
    'low':  (0, 10),
    'mid':  (10, 50),
    'high': (50, float('inf')),
}

# 拆分训练集
print('拆分训练集...')
train = pd.read_csv('$TRAIN_CSV')
for name, (lo, hi) in segments.items():
    mask = (train['CPAConstraint'] >= lo) & (train['CPAConstraint'] < hi)
    sub = train[mask]
    n_ads = sub['advertiserNumber'].nunique()
    out = os.path.join(data_dir, f'train_{name}.csv')
    sub.to_csv(out, index=False)
    print(f'  {name} (tCPA {lo}-{hi}): {len(sub)} rows, {n_ads} ads -> {out}')

# 拆分验证集
print('拆分验证集...')
val = pd.read_csv('$VAL_CSV')
for name, (lo, hi) in segments.items():
    mask = (val['CPAConstraint'] >= lo) & (val['CPAConstraint'] < hi)
    sub = val[mask]
    n_ads = sub['advertiserNumber'].nunique()
    out = os.path.join(data_dir, f'val_{name}.csv')
    sub.to_csv(out, index=False)
    print(f'  {name} (tCPA {lo}-{hi}): {len(sub)} rows, {n_ads} ads -> {out}')

# 拆分测试集
print('拆分测试集...')
test = pd.read_csv('$TEST_CSV')
for name, (lo, hi) in segments.items():
    mask = (test['CPAConstraint'] >= lo) & (test['CPAConstraint'] < hi)
    sub = test[mask]
    n_ads = sub['advertiserNumber'].nunique()
    out = os.path.join(data_dir, f'test_{name}.csv')
    sub.to_csv(out, index=False)
    print(f'  {name} (tCPA {lo}-{hi}): {len(sub)} rows, {n_ads} ads -> {out}')

# 拆分 env 数据：需要根据测试集的 advertiserNumber 过滤
print('拆分环境数据...')
env = pd.read_csv('$ENV_CSV')
for name, (lo, hi) in segments.items():
    test_sub = pd.read_csv(os.path.join(data_dir, f'test_{name}.csv'))
    ad_ids = test_sub['advertiserNumber'].unique()
    env_sub = env[env['adgroup_id'].isin(ad_ids)]
    out = os.path.join(data_dir, f'env_{name}.csv')
    env_sub.to_csv(out, index=False)
    print(f'  {name}: {len(env_sub)} env rows -> {out}')

print('数据拆分完成!')
"

# ========== Step 2: 分别训练 3 个模型 ==========
for SEG in low mid high; do
    echo ""
    echo "=========================================="
    echo "Step 2: 训练 $SEG 模型"
    echo "=========================================="

    SEG_DATA_DIR="$DATA_DIR"
    SAVE_DIR="$MODELS_DIR/$SEG"
    mkdir -p "$SAVE_DIR"

    # 构造 data_dir 结构：train_dt.py 需要 data_dir 下有 train.csv 和 val.csv
    # 创建临时目录，软链接拆分后的文件
    TMP_DATA="$EXP_DIR/tmp_data_$SEG"
    mkdir -p "$TMP_DATA"
    ln -sf "$SEG_DATA_DIR/train_${SEG}.csv" "$TMP_DATA/train.csv"
    ln -sf "$SEG_DATA_DIR/val_${SEG}.csv" "$TMP_DATA/val.csv"

    $CONDA run -n gave_env python code/main/train_dt.py \
        --data_dir "$TMP_DATA" \
        --save_dir "$SAVE_DIR" \
        --step_num 20000 \
        --save_step 2000 \
        --eval_step 500 \
        --batch_size 64 \
        --learning_rate 0.0001 \
        --loss_report 500 \
        --lambda_cpa 0 \
        --cpa_threshold 0 \
        --device cuda 2>&1 | tail -5

    rm -rf "$TMP_DATA"
done

# ========== Step 3: 分别测试 3 个模型 ==========
for SEG in low mid high; do
    echo ""
    echo "=========================================="
    echo "Step 3: 测试 $SEG 模型"
    echo "=========================================="

    MODEL_DIR=$(ls -td "$MODELS_DIR/$SEG"/DT_* 2>/dev/null | head -1)
    if [ -z "$MODEL_DIR" ]; then
        echo "错误: 未找到 $SEG 模型目录"
        continue
    fi

    TEST_OUT="$RESULTS_DIR/$SEG"
    mkdir -p "$TEST_OUT"

    $CONDA run -n gave_env python code/main/test_dt.py \
        --model_dir "$MODEL_DIR" \
        --model_name best_model.pt \
        --test_csv "$DATA_DIR/test_${SEG}.csv" \
        --env_csv "$DATA_DIR/env_${SEG}.csv" \
        --output_dir "$TEST_OUT" \
        --device cuda 2>&1 | tail -5
done

# ========== Step 4: 汇总结果 ==========
echo ""
echo "=========================================="
echo "所有分段实验完成！汇总结果："
echo "=========================================="

$CONDA run -n gave_env python3 -c "
import pandas as pd
import os, glob

results_dir = '$RESULTS_DIR'
all_model = []
all_real = []

for seg in ['low', 'mid', 'high']:
    d = os.path.join(results_dir, seg)
    sf = glob.glob(os.path.join(d, 'test_summary_*.csv'))
    if not sf:
        print(f'{seg}: 无结果')
        continue
    df = pd.read_csv(sf[0])
    m = df[df['type']=='model'].iloc[0].to_dict()
    r = df[df['type']=='real'].iloc[0].to_dict()
    m['segment'] = seg
    r['segment'] = seg
    all_model.append(m)
    all_real.append(r)

if all_model:
    mdf = pd.DataFrame(all_model)
    rdf = pd.DataFrame(all_real)
    cols = ['segment','num_trajectories','avg_conversion','avg_CPA','avg_CPA_violation','avg_budget_used','avg_win_rate','avg_spend']
    print('Model:')
    print(mdf[[c for c in cols if c in mdf.columns]].to_string(index=False))
    print()
    print('Real:')
    print(rdf[[c for c in cols if c in rdf.columns]].to_string(index=False))

    # 保存
    mdf.to_csv(os.path.join(results_dir, 'segmented_summary.csv'), index=False)
    print(f'\n汇总已保存至: {results_dir}/segmented_summary.csv')
"
