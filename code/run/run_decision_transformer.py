import numpy as np
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.dt.utils import EpisodeReplayBuffer
from bidding_train_env.baseline.dt.dt import GAVE
from torch.utils.data import DataLoader, WeightedRandomSampler
import logging
import pickle

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def run_dt(device="cpu", step_num=10000, state_dim=0, dir=None, save_step=5000, model_param={},
           batch_size=32, save_dir="saved_model/DTtest", loss_report=2):
    train_model(device, step_num, state_dim, dir=dir, save_step=save_step, model_param=model_param, batch_size=batch_size,
                save_dir=save_dir, loss_report=loss_report)


def train_model(device="cpu", step_num=10000, state_dim=0, dir=None, save_step=5000, model_param={},
                batch_size=32, save_dir="saved_model/DTtest", loss_report=2):

    # 从 model_param 中获取 n_ctx（上下文长度）
    n_ctx = model_param.get('block_config', {}).get('n_ctx', 20)
    
    # 将 n_ctx 传递给 EpisodeReplayBuffer
    replay_buffer = EpisodeReplayBuffer(
        device, 
        state_dim, 
        1, 
        data_path=dir, 
        K=n_ctx  # 传入模型的实际上下文长度
    )
    
    save_normalize_dict({"state_mean": replay_buffer.state_mean, "state_std": replay_buffer.state_std},
                        save_dir)
    logger.info(f"Replay buffer size: {len(replay_buffer.trajectories)}")

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
    
    step_num = step_num
    batch_size = batch_size
    sampler = WeightedRandomSampler(replay_buffer.p_sample, num_samples=step_num * batch_size, replacement=True)
    dataloader = DataLoader(replay_buffer, sampler=sampler, batch_size=batch_size)

    # ========== 训练统计 ==========
    total_steps = len(dataloader)
    
    # 确保 loss_report 不超过总步数
    if loss_report > total_steps:
        loss_report = max(1, total_steps // 10)
        logger.info(f"调整 loss_report 为 {loss_report} (总步数: {total_steps})")
    
    running_losses = [0.0] * 8
    print_interval = loss_report
    
    # 训练开始
    logger.info("=" * 70)
    logger.info("Starting Training...")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Loss report interval: {loss_report}")
    logger.info("=" * 70)
    
    model.train()
    for step_idx, (states, actions, rewards, dones, all_reward, 
                   curr_score, timesteps, attention_mask, next_states) in enumerate(dataloader, 1):
        
        # 移动到设备
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        all_reward = all_reward.to(device)
        curr_score = curr_score.to(device)
        timesteps = timesteps.to(device)
        attention_mask = attention_mask.to(device)
        next_states = next_states.to(device)

        # 前向传播和损失计算
        train_loss = model.step(states, actions, rewards, dones, all_reward, 
                               curr_score, timesteps, attention_mask, next_states)
        
        # 更新运行损失
        for i in range(len(train_loss)):
            running_losses[i] += train_loss[i]
        
        # ========== 标准 Loss 打印 ==========
        if step_idx % print_interval == 0 or step_idx == 1 or step_idx == total_steps:
            # 计算平均损失
            steps_for_avg = print_interval if step_idx % print_interval == 0 else step_idx
            avg_losses = [loss / steps_for_avg for loss in running_losses]
            
            # 构建完整的日志消息
            if step_idx == 1:
                log_msg = f"[Step {step_idx:6d}/{total_steps}] Initial Loss: {avg_losses[0]:.6f} "
            else:
                log_msg = f"[Step {step_idx:6d}/{total_steps}] Loss: {avg_losses[0]:.6f} "
            
            log_msg += f"(L1: {avg_losses[1]:.6f}, L2: {avg_losses[2]:.6f}, L3: {avg_losses[3]:.6f})"
            logger.info(log_msg)
            
            # 每隔一定周期显示详细指标
            if step_idx == 1 or step_idx == total_steps or (step_idx // print_interval) % 5 == 0:
                logger.info(f"  Metrics - W: {avg_losses[4]:.6f}, Target: {avg_losses[5]:.6f}, "
                          f"Pred: {avg_losses[6]:.6f}, Pred1: {avg_losses[7]:.6f}")
            
            # 重置运行损失
            running_losses = [0.0] * 8
        
        # 学习率调度
        model.scheduler.step()
        
        # 保存检查点
        if step_idx % save_step == 0:
            model.save_net(save_dir, f"{step_idx}.pt")
            logger.info(f"Checkpoint saved at step {step_idx}")
    
    # 训练结束
    logger.info("=" * 70)
    logger.info("Training completed!")
    
    # 最终测试
    test_state = np.ones(state_dim, dtype=np.float32)
    logger.info(f"Test action: {model.take_actions(test_state)}")

if __name__ == "__main__":
    run_dt()
