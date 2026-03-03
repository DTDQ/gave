#!/usr/bin/env python3
"""
train_dt.py - Decision Transformer训练脚本（GAVE loss 精准复现版）
关键修复：
  1. evaluate_model 精准复现 GAVE.step 的三项复合 loss 计算
  2. 验证后恢复 train 模式（确保训练连续性）
  3. 保持单值返回接口（与原代码完全兼容）
"""

import numpy as np
import torch
import os
import sys
import json
import datetime
import argparse
import glob
import pandas as pd
import logging
from torch.utils.data import DataLoader, WeightedRandomSampler

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bidding_train_env.baseline.dt.utils import EpisodeReplayBuffer, ValidationDataset
from bidding_train_env.baseline.dt.dt import GAVE


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Decision Transformer 训练')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True,
                       help='预处理数据目录')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, required=True,
                       help='模型保存目录')
    
    # 训练参数
    parser.add_argument('--step_num', type=int, default=5000,
                       help='训练步数')
    parser.add_argument('--save_step', type=int, default=1000,
                       help='保存模型的步数间隔')
    parser.add_argument('--eval_step', type=int, default=500,
                       help='评估模型的步数间隔')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='学习率')
    parser.add_argument('--loss_report', type=int, default=100,
                       help='报告loss的步数间隔')
    
    # 模型参数
    parser.add_argument('--state_dim', type=int, default=16,
                       help='状态维度')
    parser.add_argument('--hidden_size', type=int, default=512,
                       help='隐藏层大小')
    parser.add_argument('--time_dim', type=int, default=8,
                       help='时间维度')
    parser.add_argument('--expectile', type=float, default=0.99,
                       help='期望分位数')
    
    # Transformer参数
    parser.add_argument('--n_ctx', type=int, default=20,
                       help='上下文长度')
    parser.add_argument('--n_embd', type=int, default=512,
                       help='嵌入维度')
    parser.add_argument('--n_layer', type=int, default=8,
                       help='层数')
    parser.add_argument('--n_head', type=int, default=16,
                       help='注意力头数')
    parser.add_argument('--n_inner', type=int, default=1024,
                       help='内部维度')
    parser.add_argument('--activation_function', type=str, default="relu",
                       help='激活函数')
    parser.add_argument('--n_position', type=int, default=1024,
                       help='位置编码长度')
    parser.add_argument('--resid_pdrop', type=float, default=0.1,
                       help='残差dropout率')
    parser.add_argument('--attn_pdrop', type=float, default=0.1,
                       help='注意力dropout率')
    
    # 其他参数
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help='设备 (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    return parser.parse_args()


def create_csv_from_npy(data_dir, split_type, output_csv_path):
    """从npy文件创建CSV"""
    split_dir = os.path.join(data_dir, split_type)
    episode_files = glob.glob(os.path.join(split_dir, "*.npy"))
    
    all_rows = []
    
    for ep_file in episode_files:
        episode = np.load(ep_file, allow_pickle=True).item()
        
        states = episode['states']
        actions = episode['actions']
        rewards = episode['rewards']
        dones = episode['dones']
        next_states = episode['next_states']
        timesteps = episode['timesteps']
        metadata = episode['metadata']
        
        for i in range(len(states)):
            state_str = "(" + ", ".join([str(x) for x in states[i]]) + ")"
            
            if next_states[i] is not None:
                next_state_str = "(" + ", ".join([str(x) for x in next_states[i]]) + ")"
            else:
                next_state_str = ""
            
            row = {
                'deliveryPeriodIndex': metadata['deliveryPeriodIndex'],
                'advertiserNumber': metadata['advertiserNumber'],
                'advertiserCategoryIndex': 0,
                'budget': metadata['budget'],
                'CPAConstraint': metadata['CPAConstraint'],
                'realAllCost': metadata['realAllCost'],
                'realAllConversion': metadata['realAllConversion'],
                'timeStepIndex': int(timesteps[i]),
                'state': state_str,
                'action': float(actions[i][0]),
                'reward': float(rewards[i][0]),
                'reward_continuous': float(rewards[i][0]),
                'done': int(dones[i]),
                'next_state': next_state_str
            }
            all_rows.append(row)
    
    df = pd.DataFrame(all_rows)
    df = df.sort_values(['deliveryPeriodIndex', 'advertiserNumber', 'timeStepIndex'])
    df.to_csv(output_csv_path, index=False)
    
    return df


def evaluate_model(model, val_csv_path, device, state_dim, n_ctx, state_mean, state_std):

    # 使用验证数据集（强制使用训练集归一化参数）
    val_dataset = ValidationDataset(
        device=device,
        state_dim=state_dim,
        data_path=val_csv_path,
        n_ctx=n_ctx,
        state_mean=state_mean,
        state_std=state_std
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    try:
        model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # 解包数据
                states, actions, rewards, dones, all_reward, curr_score, timesteps, mask, next_states = batch
                
                # 移动到设备
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                dones = dones.to(device)
                all_reward = all_reward.to(device)
                curr_score = curr_score.to(device)
                timesteps = timesteps.to(device)
                mask = mask.to(device)
                next_states = next_states.to(device)
                
                # ✅ 关键：与 GAVE.step 完全一致的 curr_score 截断
                # 输入模型: curr_score[:, :-1] (K 个 return-to-go)
                # 目标值:   curr_score[:, 1:]  (K 个 target return-to-go)
                curr_score_input = curr_score[:, :-1, :]  # [B, K, 1]
                curr_score_target = curr_score[:, 1:, :]   # [B, K, 1]
                
                # 前向传播（train 模式下输出完整中间结果）
                state_preds, action_preds, curr_score_preds, _, curr_score_preds_1, action_1, value_preds = model.forward(
                    states, actions, rewards, curr_score_input, timesteps, attention_mask=mask
                )
                
                # ========== 精准复现 GAVE.step 的 loss 计算 ==========
                act_dim = action_preds.shape[2]
                
                # 1. 动作 loss (loss1) - 加权 MSE
                action_preds_flat = action_preds.reshape(-1, act_dim)[mask.reshape(-1) > 0]
                action_target_flat = actions.reshape(-1, act_dim)[mask.reshape(-1) > 0]
                action_1_flat = action_1.reshape(-1, act_dim)[mask.reshape(-1) > 0]
                action_1_frozen = action_1_flat.clone().detach()
                
                # 计算 wo 权重（与训练完全一致）
                wo = torch.sigmoid(1.0 * (curr_score_preds_1 - curr_score_preds.clone().detach()))
                wo_flat = wo.reshape(-1, 1)[mask.reshape(-1) > 0]
                wo_frozen = wo_flat.clone().detach()
                
                loss1 = torch.mean(
                    (1 - wo_frozen) * ((action_preds_flat - action_target_flat) ** 2) +
                    wo_frozen * ((action_preds_flat - action_1_frozen) ** 2)
                )
                
                # 2. Return-to-go loss (loss2)
                curr_score_dim = curr_score_preds.shape[2]
                curr_score_preds_flat = curr_score_preds.reshape(-1, curr_score_dim)[mask.reshape(-1) > 0]
                curr_score_target_flat = curr_score_target.reshape(-1, curr_score_dim)[mask.reshape(-1) > 0]
                loss2 = torch.mean((curr_score_preds_flat - curr_score_target_flat) ** 2) * 2000
                
                # 3. Exploration loss (loss3)
                loss3 = torch.mean(1 - wo_flat) * 100.0
                
                # 总 loss（与 GAVE.step 完全一致）
                total_batch_loss = loss1 + loss2 + loss3
                
                # 累加
                batch_samples = mask.sum().item()
                total_loss += total_batch_loss.item() * batch_samples
                total_samples += batch_samples
        
        # 计算平均 loss
        if total_samples > 0:
            avg_loss = total_loss / total_samples
        else:
            avg_loss = float('inf')
        
        return avg_loss
        
    finally:
        model.train()


def train_dt_with_eval(device, step_num, state_dim, train_dir, val_dir, 
                       save_step, eval_step, model_param, batch_size, 
                       save_dir, loss_report):

    logger = logging.getLogger(__name__)
    
    n_ctx = model_param.get('block_config', {}).get('n_ctx', 20)
    
    # 加载训练数据
    replay_buffer = EpisodeReplayBuffer(
        device, 
        state_dim, 
        1, 
        data_path=train_dir, 
        K=n_ctx
    )
    
        # ===== 直接保存 JSON 格式的归一化参数 =====
    try:
        # 确保 state_mean/state_std 是列表格式
        state_mean = replay_buffer.state_mean
        state_std = replay_buffer.state_std
        if isinstance(state_mean, np.ndarray):
            state_mean = state_mean.tolist()
        if isinstance(state_std, np.ndarray):
            state_std = state_std.tolist()
        
        norm_dict = {
            "state_mean": state_mean,
            "state_std": state_std
        }
        norm_json_path = os.path.join(save_dir, "normalize_dict.json")
        with open(norm_json_path, 'w') as f:
            json.dump(norm_dict, f, indent=2)
        logger.info(f"归一化参数已保存到: {norm_json_path}")
    except Exception as e:
        logger.warning(f"保存归一化参数失败: {e}")

    model_param['state_mean'] = replay_buffer.state_mean
    model_param['state_std'] = replay_buffer.state_std
    model = GAVE(state_dim=state_dim, act_dim=1,
                 hidden_size=model_param['hidden_size'], 
                 state_mean=model_param['state_mean'],
                 state_std=model_param['state_std'], 
                 device=model_param['device'],
                 learning_rate=model_param["learning_rate"], 
                 time_dim=model_param['time_dim'],
                 block_config=model_param['block_config'], 
                 expectile=model_param['expectile']).to(device)
    
    sampler = WeightedRandomSampler(replay_buffer.p_sample, num_samples=step_num * batch_size, replacement=True)
    dataloader = DataLoader(replay_buffer, sampler=sampler, batch_size=batch_size)

    total_steps = len(dataloader)
    
    if loss_report > total_steps:
        loss_report = max(1, total_steps // 10)
        logger.info(f"调整 loss_report 为 {loss_report} (总步数: {total_steps})")
    
    running_losses = [0.0] * 8
    print_interval = loss_report
    
    best_val_loss = float('inf')
    best_step = 0
    
    logger.info("=" * 70)
    logger.info("开始训练")
    logger.info(f"总步数: {total_steps} | 批次大小: {batch_size}")
    logger.info(f"上下文长度 K: {n_ctx}")
    logger.info("=" * 70)
    
    model.train()
    for step_idx, (states, actions, rewards, dones, all_reward, 
                   curr_score, timesteps, attention_mask, next_states) in enumerate(dataloader, 1):
        
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        all_reward = all_reward.to(device)
        curr_score = curr_score.to(device)
        timesteps = timesteps.to(device)
        attention_mask = attention_mask.to(device)
        next_states = next_states.to(device)
        
        # ✅ 关键：训练时也使用与验证完全一致的 curr_score 截断
        # GAVE.step 内部会自动处理: curr_score[:, :-1] 作为输入, curr_score[:, 1:] 作为目标
        train_loss = model.step(states, actions, rewards, dones, all_reward, 
                               curr_score, timesteps, attention_mask, next_states)
        
        for i in range(len(train_loss)):
            running_losses[i] += train_loss[i]
        
        if step_idx % print_interval == 0 or step_idx == 1 or step_idx == total_steps:
            steps_for_avg = print_interval if step_idx % print_interval == 0 else step_idx
            avg_losses = [loss / steps_for_avg for loss in running_losses]
            
            if step_idx == 1:
                log_msg = f"[Step {step_idx:6d}/{total_steps}] 初始损失: {avg_losses[0]:.6f} "
            else:
                log_msg = f"[Step {step_idx:6d}/{total_steps}] 训练损失: {avg_losses[0]:.6f} "
            
            log_msg += f"(L1: {avg_losses[1]:.6f}, L2: {avg_losses[2]:.6f}, L3: {avg_losses[3]:.6f})"
            logger.info(log_msg)
            
            if step_idx == 1 or step_idx == total_steps or (step_idx // print_interval) % 5 == 0:
                logger.info(f"  指标 - W: {avg_losses[4]:.6f}, Target: {avg_losses[5]:.6f}, "
                          f"Pred: {avg_losses[6]:.6f}, Pred1: {avg_losses[7]:.6f}")
            
            running_losses = [0.0] * 8
        
        # ========== 验证评估 ==========
        if val_dir and step_idx % eval_step == 0:
            # 传入训练集归一化参数
            val_loss = evaluate_model(
                model, 
                val_dir, 
                device, 
                state_dim, 
                n_ctx,
                replay_buffer.state_mean,
                replay_buffer.state_std
            )
            
            if step_idx == eval_step:
                logger.info("=" * 50)
                logger.info("✅ 验证评估")
                logger.info(f"[Step {step_idx:6d}] 验证损失: {val_loss:.4f}")
            else:
                logger.info(f"[Step {step_idx:6d}] 验证损失: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step_idx
                model.save_net(save_dir, "best_model.pt")
                logger.info(f"  🏆 新的最佳模型! (验证损失: {val_loss:.4f})")
        
        model.scheduler.step()
        
        if step_idx % save_step == 0:
            model.save_net(save_dir, f"{step_idx}.pt")
            logger.info(f"💾 检查点保存于步数 {step_idx}")
    
    logger.info("=" * 70)
    logger.info("✅ 训练完成!")
    logger.info(f"最佳验证损失: {best_val_loss:.4f} (步数: {best_step})")
    
    test_state = np.ones(state_dim, dtype=np.float32)
    logger.info(f"测试动作: {model.take_actions(test_state)}")

def main():
    """主函数"""
    args = parse_arguments()
    
    # 配置日志（使用与 train_dt_with_eval 相同的格式）
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Decision Transformer 训练")
    logger.info("=" * 80)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = os.path.join(args.save_dir, f"DT_{timestamp}")
    os.makedirs(model_save_dir, exist_ok=True)
    
    logger.info(f"模型将保存到: {model_save_dir}")
    
    # 获取状态维度
    data_config_path = os.path.join(args.data_dir, 'config.json')
    if os.path.exists(data_config_path):
        with open(data_config_path, 'r') as f:
            data_config = json.load(f)
        state_dim = data_config.get('state_dim', args.state_dim)
        logger.info(f"状态维度: {state_dim} (从数据配置加载)")
    else:
        state_dim = args.state_dim
        logger.info(f"状态维度: {state_dim} (使用默认值)")
    
    # 创建CSV
    train_csv_path = os.path.join(args.data_dir, "temp_train_data.csv")
    val_csv_path = os.path.join(args.data_dir, "temp_val_data.csv")
    
    logger.info("正在创建训练集CSV...")
    train_df = create_csv_from_npy(args.data_dir, 'train', train_csv_path)
    
    logger.info("正在创建验证集CSV...")
    val_df = create_csv_from_npy(args.data_dir, 'val', val_csv_path)
    
    logger.info(f"训练集: {train_df.shape[0]} 行数据")
    logger.info(f"验证集: {val_df.shape[0]} 行数据")
    
    # 构建模型参数
    model_param = {
        "step_num": args.step_num,
        "save_step": args.save_step,
        "eval_step": args.eval_step,
        "state_dim": state_dim,
        "dir": train_csv_path,
        "val_dir": val_csv_path,
        "test_csv": None,
        "env_csv": None,
        "hidden_size": args.hidden_size,
        "learning_rate": args.learning_rate,
        "time_dim": args.time_dim,
        "batch_size": args.batch_size,
        "device": args.device,
        "expectile": args.expectile,
        "loss_report": args.loss_report,
        "block_config": {
            "n_ctx": args.n_ctx,
            "n_embd": args.n_embd,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "n_inner": args.n_inner,
            "activation_function": args.activation_function,
            "n_position": args.n_position,
            "resid_pdrop": args.resid_pdrop,
            "attn_pdrop": args.attn_pdrop,
        }
    }
    
    # 保存配置
    config = vars(args)
    config['timestamp'] = timestamp
    config['model_save_dir'] = model_save_dir
    config['state_dim'] = state_dim
    
    config_path = os.path.join(model_save_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"配置已保存: {config_path}")
    
    # 开始训练
    logger.info("\n🚀 开始训练...")
    try:
        train_dt_with_eval(
            device=model_param["device"],
            step_num=model_param["step_num"],
            state_dim=model_param["state_dim"],
            train_dir=model_param["dir"],
            val_dir=model_param["val_dir"],
            save_step=model_param["save_step"],
            eval_step=model_param["eval_step"],
            model_param=model_param,
            batch_size=model_param["batch_size"],
            save_dir=model_save_dir,
            loss_report=model_param['loss_report']
        )
        
        logger.info(f"\n✅ 训练完成! 模型保存在: {model_save_dir}")
        
    except Exception as e:
        logger.error(f"\n❌ 训练失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # 删除临时文件
    if os.path.exists(train_csv_path):
        os.remove(train_csv_path)
        logger.info(f"🗑️ 已删除临时文件: {train_csv_path}")
    if os.path.exists(val_csv_path):
        os.remove(val_csv_path)
        logger.info(f"🗑️ 已删除临时文件: {val_csv_path}")

if __name__ == "__main__":
    main()