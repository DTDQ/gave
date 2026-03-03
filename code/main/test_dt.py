#!/usr/bin/env python3
"""
test_dt.py - 离线回放测试脚本（使用16维环境，适配实际数据）

用法示例：
  python test_dt.py --model_dir ./saved_model/DT_20250213_143022/ \
                    --test_csv /data/test/trajectory/test_data_all-rlData.csv \
                    --env_csv /data/test/env/64160896_env.xlsx \
                    --output_dir ./test_results \
                    --device cuda
"""

import os
import sys
import glob
import json
import argparse
import datetime
import pandas as pd
import torch
import numpy as np
from collections import deque

# ---------- 添加项目根目录到系统路径 ----------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from bidding_train_env.baseline.dt.utils import EpisodeReplayBuffer
from bidding_train_env.strategy import PlayerBiddingStrategy
# 导入16维的OfflineEnv（请确保文件位于 bidding_train_env/environment/offline_env.py）
from bidding_train_env.environment.offline_env import OfflineEnv


# ---------- 辅助函数 ----------
def load_env_data(env_path):
    """根据文件扩展名自动读取环境数据"""
    if env_path.endswith('.csv'):
        return pd.read_csv(env_path)
    elif env_path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(env_path, engine='openpyxl')
    else:
        raise ValueError(f"不支持的环境文件格式: {env_path}")


def load_training_config(model_dir):
    """从模型目录加载训练配置和归一化参数"""
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
        print(f"警告: {config_path} 不存在，使用默认参数")

    norm_path = os.path.join(model_dir, "normalize_dict.json")
    if os.path.exists(norm_path):
        with open(norm_path, 'r') as f:
            norm_dict = json.load(f)
        state_mean = torch.tensor(norm_dict.get('state_mean', [0.0]*config.get('state_dim', 16)))
        state_std = torch.tensor(norm_dict.get('state_std', [1.0]*config.get('state_dim', 16)))
    else:
        state_dim = config.get('state_dim', 16)
        state_mean = torch.zeros(state_dim)
        state_std = torch.ones(state_dim)
        print(f"警告: {norm_path} 不存在，使用默认归一化参数")

    return config, state_mean, state_std


def run_single_model_test(model_path, model_param, test_csv, env_csv, device):
    """对单个模型文件执行测试，返回详细结果DataFrame"""
    # 更新模型参数中的路径和设备
    model_param['test_csv'] = test_csv
    model_param['env_csv'] = env_csv
    model_param['device'] = device
    model_param['save_dir'] = os.path.dirname(model_path)

    # 加载测试轨迹缓存（if_test=True 返回完整轨迹）
    test_buffer = EpisodeReplayBuffer(
        device=device,
        state_dim=model_param['state_dim'],
        act_dim=1,
        data_path=test_csv,
        if_test=True
    )

    # 加载环境数据
    traffic_df = load_env_data(env_csv)
    env = OfflineEnv(traffic_df)   # 使用16维环境

    results = []
    for traj_idx, traj in enumerate(test_buffer.trajectories):
        adid = traj["ad_id"]                 # 对应 advertiserNumber
        period = traj["period"]              # 对应 deliveryPeriodIndex
        budget = traj["budget"][0][0]        # 初始预算
        cpacons = traj["cpacons"][0][0]      # 目标 CPA
        target_score = traj["final_score"]   # 目标转化数（realAllConversion）

        # 初始化策略
        agent = PlayerBiddingStrategy(
            model_name=os.path.basename(model_path),
            model_param=model_param,
            budget=budget,
            cpa=cpacons
        )
        agent.reset(target_return=target_score)

        state = env.reset(adid, period, budget, cpacons)
        last_step_reward = 0.0
        done = False
        t = 0

        # 回放循环（环境内部控制步数）
        while not done:
            if agent.remaining_budget <= 0 or env.remaining_budget <= 0:
                alpha = 0.0
            else:
                if not isinstance(state, torch.Tensor):
                    state = torch.tensor(state, dtype=torch.float32)
                if str(state.device) != device:
                    state = state.to(device)
                alpha = agent.bidding(state, timeStepIndex=t, last_step_conversion=last_step_reward)

            next_state, done, info = env.step(alpha)

            if not isinstance(next_state, torch.Tensor):
                next_state = torch.tensor(next_state, dtype=torch.float32)
            next_state = next_state.to(device)

            last_step_reward = info["step_conversion"]
            state = next_state
            t += 1

        # 获取轨迹汇总指标
        metrics = env.get_metrics()
        results.append({
            "model": os.path.basename(model_path),
            "ad_id": adid,
            "period": period,
            "budget": budget,
            "target_CPA": cpacons,
            "target_conversion": target_score,
            "achieved_conversion": metrics["conversion"],
            "achieved_CPA": metrics["CPA"],
            "budget_used": metrics["budget_usage"],
            "win_rate": metrics["win_rate"],
            "spend": metrics["spend"],
            "trajectory_length": env.current_idx  # 实际执行的曝光步数
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="离线回放测试 Decision Transformer 模型（适配实际数据）")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="训练好的模型目录（包含 config.json, normalize_dict.json, *.pt）")
    parser.add_argument("--model_name", type=str, default=None,
                        help="指定单个模型文件名（如 'best_model.pt'），不指定则测试目录下所有 .pt 文件")
    parser.add_argument("--test_csv", type=str, required=True,
                        help="测试轨迹数据 CSV 路径，例如 /data/test/trajectory/test_data_all-rlData.csv")
    parser.add_argument("--env_csv", type=str, required=True,
                        help="环境数据 Excel/CSV 路径，例如 /data/test/env/64160896_env.xlsx")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="推理设备")
    parser.add_argument("--output_dir", type=str, default="./test_results",
                        help="测试结果保存目录")
    args = parser.parse_args()

    # ---------- 1. 加载训练配置 ----------
    train_config, state_mean, state_std = load_training_config(args.model_dir)
    state_dim = train_config.get('state_dim', 16)
    print(f"加载训练配置: 状态维度 = {state_dim}, 设备 = {args.device}")

    # ---------- 2. 构建模型参数 ----------
    block_config = train_config.get('block_config', {
        "n_ctx": 20,
        "n_embd": 512,
        "n_layer": 8,
        "n_head": 16,
        "n_inner": 1024,
        "activation_function": "relu",
        "n_position": 1024,
        "resid_pdrop": 0.1,
        "attn_pdrop": 0.1
    })

    model_param = {
        "state_dim": state_dim,
        "hidden_size": train_config.get('hidden_size', 512),
        "time_dim": train_config.get('time_dim', 8),
        "device": args.device,
        "expectile": train_config.get('expectile', 0.99),
        "budget_rate": 1.0,
        "block_config": block_config,
        "state_mean": state_mean.tolist() if isinstance(state_mean, torch.Tensor) else state_mean,
        "state_std": state_std.tolist() if isinstance(state_std, torch.Tensor) else state_std,
        "learning_rate": train_config.get('learning_rate', 0.0001),  # 模型初始化需要
        # 以下字段由 run_single_model_test 动态填充
        "test_csv": args.test_csv,
        "env_csv": args.env_csv,
        "save_dir": args.model_dir,
    }

    # ---------- 3. 确定要测试的模型文件 ----------
    if args.model_name:
        model_files = [os.path.join(args.model_dir, args.model_name)]
    else:
        pattern = os.path.join(args.model_dir, "*.pt")
        model_files = glob.glob(pattern)
        # 按文件名中的数字排序（如果文件名是数字）
        model_files = sorted(model_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
                             if os.path.splitext(os.path.basename(x))[0].isdigit() else 0)

    if not model_files:
        print(f"错误: 在 {args.model_dir} 中未找到任何 .pt 文件")
        sys.exit(1)

    print(f"找到 {len(model_files)} 个模型文件：{[os.path.basename(f) for f in model_files]}")

    # ---------- 4. 创建输出目录 ----------
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(args.output_dir, f"test_summary_{timestamp}.csv")
    detail_dir = os.path.join(args.output_dir, f"details_{timestamp}")
    os.makedirs(detail_dir, exist_ok=True)

    # ---------- 5. 逐个模型测试 ----------
    all_summaries = []
    for model_path in model_files:
        model_name = os.path.basename(model_path)
        print(f"\n========== 测试模型: {model_name} ==========")

        try:
            df_result = run_single_model_test(
                model_path=model_path,
                model_param=model_param.copy(),
                test_csv=args.test_csv,
                env_csv=args.env_csv,
                device=args.device
            )

            # 保存详细结果
            detail_csv = os.path.join(detail_dir, f"{model_name}_details.csv")
            df_result.to_csv(detail_csv, index=False)
            print(f"详细结果已保存至: {detail_csv}")

            # 汇总指标（按模型平均）
            summary = {
                "model": model_name,
                "avg_conversion": df_result["achieved_conversion"].mean(),
                "avg_CPA": df_result["achieved_CPA"].mean(),
                "avg_budget_used": df_result["budget_used"].mean(),
                "avg_win_rate": df_result["win_rate"].mean(),
                "avg_spend": df_result["spend"].mean(),
                "num_trajectories": len(df_result),
            }
            all_summaries.append(summary)
            print(f"汇总: 平均转化={summary['avg_conversion']:.4f}, "
                  f"平均CPA={summary['avg_CPA']:.4f}, "
                  f"预算使用率={summary['avg_budget_used']:.4f}")

        except Exception as e:
            print(f"测试模型 {model_name} 时出错: {e}")
            import traceback
            traceback.print_exc()

    # ---------- 6. 保存汇总结果 ----------
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(summary_file, index=False)
        print(f"\n所有模型汇总结果已保存至: {summary_file}")
        print("\n各模型平均指标:")
        print(summary_df.to_string(index=False))
    else:
        print("未生成任何有效结果。")

    print("\n测试完成！")


if __name__ == "__main__":
    main()