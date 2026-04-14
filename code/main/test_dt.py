#!/usr/bin/env python3
"""
test_dt.py - 离线回放测试脚本（真实指标独立文件，不含模型数据）
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

# ---------- 添加项目根目录 ----------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from bidding_train_env.baseline.dt.utils import EpisodeReplayBuffer
from bidding_train_env.strategy import PlayerBiddingStrategy
from bidding_train_env.environment.offline_env import OfflineEnv


def load_env_data(env_path):
    if env_path.endswith('.csv'):
        return pd.read_csv(env_path)
    elif env_path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(env_path, engine='openpyxl')
    else:
        raise ValueError(f"不支持的环境文件格式: {env_path}")


def load_training_config(model_dir):
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


def build_block_config(train_config):
    """兼容两种训练配置格式：嵌套 block_config 或 train_dt.py 保存的扁平参数。"""
    block_config = train_config.get('block_config')
    if isinstance(block_config, dict):
        return block_config

    return {
        "n_ctx": train_config.get("n_ctx", 20),
        "n_embd": train_config.get("n_embd", 512),
        "n_layer": train_config.get("n_layer", 8),
        "n_head": train_config.get("n_head", 16),
        "n_inner": train_config.get("n_inner", 1024),
        "activation_function": train_config.get("activation_function", "relu"),
        "n_position": train_config.get("n_position", 1024),
        "resid_pdrop": train_config.get("resid_pdrop", 0.1),
        "attn_pdrop": train_config.get("attn_pdrop", 0.1),
    }


def run_single_model_test(model_path, model_param, test_csv, env_csv, device):
    model_param['test_csv'] = test_csv
    model_param['env_csv'] = env_csv
    model_param['device'] = device
    model_param['save_dir'] = os.path.dirname(model_path)

    # 加载测试轨迹（模型模拟用）
    test_buffer = EpisodeReplayBuffer(
        device=device,
        state_dim=model_param['state_dim'],
        act_dim=1,
        data_path=test_csv,
        if_test=True
    )

    # 加载环境
    traffic_df = load_env_data(env_csv)
    env = OfflineEnv(traffic_df)

    results = []
    agent = None
    num_traj = len(test_buffer.trajectories)

    with torch.no_grad():
        for traj_idx, traj in enumerate(test_buffer.trajectories):
            adid = traj["ad_id"]
            period = traj["period"]
            budget = traj["budget"][0][0]
            cpacons = traj["cpacons"][0][0]
            target_score = traj["final_score"]

            if agent is None:
                agent = PlayerBiddingStrategy(
                    model_name=os.path.basename(model_path),
                    model_param=model_param,
                    budget=budget,
                    cpa=cpacons
                )
            else:
                agent.budget = budget
                agent.cpa = cpacons

            agent.reset(target_return=target_score)

            state = env.reset(adid, period, budget, cpacons)
            state = torch.tensor(state, dtype=torch.float32, device=device)
            last_step_reward = 0.0
            done = False
            t = 0
            total_win = 0

            while not done:
                if agent.remaining_budget <= 0 or env.remaining_budget <= 0:
                    alpha = 0.0
                else:
                    alpha = agent.bidding(state, timeStepIndex=t, last_step_conversion=last_step_reward)

                next_state, done, info = env.step(alpha)
                state = torch.tensor(next_state, dtype=torch.float32, device=device)

                last_step_reward = info["step_conversion"]
                total_win += info["step_win"]
                t += 1

            metrics = env.get_metrics()
            results.append({
                "model": os.path.basename(model_path),
                "ad_id": adid,
                "period": period,
                "budget": budget,
                "target_CPA": cpacons,
                "target_conversion": target_score,
                "conversion": metrics["conversion"],
                "CPA": metrics["CPA"],
                "budget_used": metrics["budget_usage"],
                "win_rate": metrics["win_rate"],
                "spend": metrics["spend"],
                "total_win": total_win,
                "trajectory_length": env.current_idx,
                "CPA_violation": (metrics["CPA"] - cpacons) / cpacons if cpacons > 0 else float('nan'),
            })

            if (traj_idx + 1) % 10 == 0 or traj_idx == num_traj - 1:
                print(f"  进度: {traj_idx + 1}/{num_traj}")

    return pd.DataFrame(results)


def compute_real_metrics(test_csv):
    """读取测试数据，计算每个广告组的真实指标，返回 DataFrame 和汇总行"""
    df = pd.read_csv(test_csv)
    df['advertiserNumber'] = df['advertiserNumber'].astype(int)

    rows = []
    for (period, adid), group in df.groupby(['deliveryPeriodIndex', 'advertiserNumber']):
        group = group.sort_values('timeStepIndex')
        total_conversion = group['reward'].sum()
        total_cost = group['cost'].sum()
        total_win = group['win_num'].sum()
        total_pv = group['pv_num'].sum()
        budget = group['budget'].iloc[0]
        target_cpa = group['CPAConstraint'].iloc[0]

        # 跳过无转化的任务
        if total_conversion <= 0:
            continue

        real_cpa = total_cost / total_conversion
        real_violation = (real_cpa - target_cpa) / target_cpa if target_cpa > 0 else float('nan')

        rows.append({
            "ad_id": adid,
            "period": period,
            "budget": budget,
            "target_CPA": target_cpa,
            "conversion": total_conversion,
            "CPA": real_cpa,
            "budget_used": total_cost / budget if budget > 0 else 0.0,
            "win_rate": total_win / total_pv if total_pv > 0 else 0.0,
            "spend": total_cost,
            "total_win": total_win,
            "trajectory_length": len(group),
            "CPA_violation": real_violation,
        })

    real_df = pd.DataFrame(rows)
    # 汇总行
    if not real_df.empty:
        total_win_sum = real_df["total_win"].sum()
        real_summary = {
            "type": "real",
            "num_trajectories": len(real_df),
            "avg_tCPA": real_df["target_CPA"].mean(),
            "avg_conversion": real_df["conversion"].mean(),
            "avg_CPA": real_df["CPA"].mean(),
            "avg_CPA_violation": (real_df["CPA_violation"] * real_df["spend"]).sum() / real_df["spend"].sum() if real_df["spend"].sum() > 0 else 0.0,
            "avg_budget_used": real_df["budget_used"].mean(),
            "avg_win_rate": (real_df["win_rate"] * real_df["total_win"]).sum() / total_win_sum if total_win_sum > 0 else 0.0,
            "avg_spend": real_df["spend"].mean(),
            "total_win_num": int(total_win_sum),
            "avg_total_win": real_df["total_win"].mean(),
        }
    else:
        real_summary = None

    return real_df, real_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--env_csv", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="./test_results")
    args = parser.parse_args()

    # 加载配置
    train_config, state_mean, state_std = load_training_config(args.model_dir)
    state_dim = train_config.get('state_dim', 16)
    print(f"状态维度 = {state_dim}, 设备 = {args.device}")

    # 构建模型参数
    block_config = build_block_config(train_config)
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
        "learning_rate": train_config.get('learning_rate', 0.0001),
        "test_csv": args.test_csv,
        "env_csv": args.env_csv,
        "save_dir": args.model_dir,
    }

    # 模型文件列表
    if args.model_name:
        model_files = [os.path.join(args.model_dir, args.model_name)]
    else:
        pattern = os.path.join(args.model_dir, "*.pt")
        model_files = glob.glob(pattern)
        model_files = sorted(model_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
                             if os.path.splitext(os.path.basename(x))[0].isdigit() else 0)

    if not model_files:
        print(f"错误: 未找到 .pt 文件")
        sys.exit(1)

    print(f"找到 {len(model_files)} 个模型文件")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(args.output_dir, f"test_summary_{timestamp}.csv")
    detail_dir = os.path.join(args.output_dir, f"details_{timestamp}")
    os.makedirs(detail_dir, exist_ok=True)

    # ---------- 计算真实指标并保存 ----------
    real_df, real_summary = compute_real_metrics(args.test_csv)
    if real_df is not None:
        real_detail_path = os.path.join(args.output_dir, "real_details.csv")
        real_df.to_csv(real_detail_path, index=False)
        print(f"真实详细数据已保存至: {real_detail_path}")

        # 真实汇总行（后续加入汇总表）
        real_row = {
            "model": "N/A",
            "type": "real",
            "num_trajectories": real_summary["num_trajectories"],
            "avg_tCPA": real_summary["avg_tCPA"],
            "avg_conversion": real_summary["avg_conversion"],
            "avg_CPA": real_summary["avg_CPA"],
            "avg_CPA_violation": real_summary["avg_CPA_violation"],
            "avg_budget_used": real_summary["avg_budget_used"],
            "avg_win_rate": real_summary["avg_win_rate"],
            "avg_spend": real_summary["avg_spend"],
            "total_win_num": real_summary["total_win_num"],
            "avg_total_win": real_summary["avg_total_win"],
        }
    else:
        real_row = None
        print("警告：未找到任何原始轨迹！")

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

            if df_result.empty:
                print(f"跳过空结果")
                continue

            # 保存模型详细结果（仅模型指标）
            detail_csv = os.path.join(detail_dir, f"{model_name}_details.csv")
            df_result.to_csv(detail_csv, index=False)
            print(f"详细结果已保存至: {detail_csv}")

            # 模型行汇总
            total_win_sum = df_result["total_win"].sum()
            model_summary = {
                "model": model_name,
                "type": "model",
                "num_trajectories": len(df_result),
                "avg_tCPA": df_result["target_CPA"].mean(),
                "avg_conversion": df_result["conversion"].mean(),
                "avg_CPA": df_result["CPA"].mean(),
                "avg_CPA_violation": (df_result["CPA_violation"] * df_result["spend"]).sum() / df_result["spend"].sum() if df_result["spend"].sum() > 0 else 0.0,
                "avg_budget_used": df_result["budget_used"].mean(),
                "avg_win_rate": (df_result["win_rate"] * df_result["total_win"]).sum() / total_win_sum if total_win_sum > 0 else 0.0,
                "avg_spend": df_result["spend"].mean(),
                "total_win_num": int(total_win_sum),
                "avg_total_win": df_result["total_win"].mean(),
            }
            all_summaries.append(model_summary)

        except Exception as e:
            print(f"测试模型 {model_name} 时出错: {e}")
            import traceback
            traceback.print_exc()

    # 添加真实行
    if real_row:
        all_summaries.append(real_row)

    # 保存汇总结果
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(summary_file, index=False)
        print(f"\n所有模型汇总结果已保存至: {summary_file}")
        print("\n各模型平均指标:")
        print(summary_df.to_string(index=False))
    else:
        print("未生成任何有效结果。")

    # ---------- 生成逐任务对比 CSV ----------
    if real_df is not None and not real_df.empty and all_summaries:
        # 取第一个模型的详细结果做对比
        first_model_detail = os.path.join(detail_dir,
            f"{os.path.basename(model_files[0])}_details.csv")
        if os.path.exists(first_model_detail):
            model_df = pd.read_csv(first_model_detail)
            # 合并：按 ad_id + period 做对比
            comparison = real_df.merge(
                model_df,
                on=["ad_id", "period"],
                suffixes=("_real", "_model"),
                how="inner"
            )
            # 选取关键对比列
            cols = []
            for c in ["ad_id", "period", "budget_real", "target_CPA_real"]:
                if c in comparison.columns:
                    cols.append(c)
            for metric in ["conversion", "CPA", "spend", "total_win", "win_rate", "budget_used", "CPA_violation"]:
                for suffix in ["_real", "_model"]:
                    col = metric + suffix
                    if col in comparison.columns:
                        cols.append(col)
            comparison_out = comparison[[c for c in cols if c in comparison.columns]]
            comparison_path = os.path.join(args.output_dir, f"task_comparison_{timestamp}.csv")
            comparison_out.to_csv(comparison_path, index=False)
            print(f"\n逐任务对比结果已保存至: {comparison_path}")

    print("\n测试完成！")


if __name__ == "__main__":
    main()
