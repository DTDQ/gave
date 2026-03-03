import numpy as np
import torch
import math
import logging
from bidding_train_env.strategy import PlayerBiddingStrategy
from bidding_train_env.dataloader.test_dataloader import TestDataLoader
from bidding_train_env.environment.offline_env import OfflineEnv
import pandas as pd
import datatable as dt
from bidding_train_env.baseline.dt.utils import EpisodeReplayBuffer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def run_test(model_name="dt.pt", model_param=None):
    test_buffer = EpisodeReplayBuffer(
        device=model_param['device'],
        state_dim=model_param['state_dim'],
        act_dim=1,
        data_path=model_param['test_csv'],
        if_test=True 
    )

    traffic_df = pd.read_csv(model_param['env_csv'])
    env = OfflineEnv(traffic_df)
    
    # 获取设备信息
    device = model_param['device']
    print(f"使用设备: {device}")

    results = []
    for traj_idx, traj in enumerate(test_buffer.trajectories):
        print(f"处理轨迹 {traj_idx + 1}/{len(test_buffer.trajectories)}")
        
        adid = traj["ad_id"]
        period = traj["period"]
        budget = traj["budget"][0][0]
        cpacons = traj["cpacons"][0][0]
        target_score = traj["final_score"]

        agent = PlayerBiddingStrategy(model_name=model_name, model_param=model_param, budget=budget, cpa=cpacons)
        agent.reset(target_return=target_score)

        state = env.reset(adid, period, budget, cpacons)
        last_step_reward = 0.0

        done = False
        all_rewards = []
        all_costs = []
        t = 0

        while not done and t < len(traj['observations']):
            if agent.remaining_budget <= 0 or env.remaining_budget <= 0:
                alpha = 0
            else:
                # 关键修复：确保状态转移到正确的设备
                if not isinstance(state, torch.Tensor):
                    state = torch.tensor(state, dtype=torch.float32)
                
                # 确保状态在正确的设备上
                if str(state.device) != device:
                    state = state.to(device)
                
                alpha = agent.bidding(state, timeStepIndex=t, last_step_conversion=last_step_reward)

            next_state, done, info = env.step(alpha)
            
            # 确保 next_state 也是正确的设备
            if not isinstance(next_state, torch.Tensor):
                next_state = torch.tensor(next_state, dtype=torch.float32)
            next_state = next_state.to(device)

            last_step_reward = info["step_conversion"]
            all_rewards.append(last_step_reward)
            all_costs.append(info["step_cost"])

            state = next_state
            t += 1

        # env metrics
        metrics = env.get_metrics()

        results.append({
            "ad_id": adid,
            "period": period,
            "final_score": target_score,
            "achieved_conversion": metrics["conversion"],
            "achieved_CPA": metrics["CPA"],
            "budget_used": metrics["budget_usage"],
            "win_rate": metrics["win_rate"]
        })

    results_df = pd.DataFrame(results)
    return results_df

    #     logger.info(f'Total Reward: {all_reward}')
    #     logger.info(f'Total Cost: {all_cost}')
    #     logger.info(f'CPA-real: {cpa_real}')
    #     logger.info(f'CPA-constraint: {cpa_constraint}')
    #     logger.info(f'Score: {score}')
    # overall_score = overall_score/len(keys)
    # overall_score1 = overall_score1 / len(keys)
    # overall_conversion = overall_conversion / len(keys)
    # logger.info(f'Period Score: {overall_score}')
    # logger.info(f'Excced rate: {excced_rate / len(keys)}')
    # return overall_score, overall_score1, overall_conversion, excced_rate / len(keys)


if __name__ == '__main__':
    model_param = {
        "step_num": 2,
        "save_step": 1,
        "state_dim": 13,
        "dir": "",
        "test_csv": "data/traffic/period-x.csv",
        "env_csv": "data/traffic/env-x.csv",
        "hidden_size": 512,
        "learning_rate": 0.0001,
        "time_dim":8,
        "batch_size": 128,
        "device": "cuda",
        "expectile": 0.99,
        "loss_report": 100,
        "budget_rate": 1.0,
        "block_config": {
            "n_ctx": 1024,
            "n_embd": 512,
            "n_layer": 8,
            "n_head": 16,
            "n_inner": 1024,
            "activation_function": "relu",
            "n_position": 1024,
            "resid_pdrop": 0.1,
            "attn_pdrop": 0.1,
        },
        "save_dir": "saved_model/DTtest_20251223170905/"
    }

    run_test(model_name="1.pt", model_param=model_param)
