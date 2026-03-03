import os
import json  # ✅ 改用 json 模块
import torch
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
from bidding_train_env.baseline.dt.dt import GAVE

class DtBiddingStrategy(BaseBiddingStrategy):

    def __init__(
        self,
        budget=0,
        cpa=0,
        model_name=None,
        model_param=None,
    ):
        super().__init__(budget, cpa)

        if model_param is None:
            raise ValueError("model_param must be provided")

        # =========================
        # Load model & normalization (JSON format)
        # =========================
        model_path = os.path.join(model_param["save_dir"], model_name)
        json_path = os.path.join(model_param["save_dir"], "normalize_dict.json")  # ✅ 改为 .json

        with open(json_path, "r") as f:          # ✅ 使用 json.load
            normalize_dict = json.load(f)

        self.device = model_param["device"]

        self.model = GAVE(
            state_dim=model_param["state_dim"],
            act_dim=1,
            hidden_size=model_param["hidden_size"],
            state_mean=normalize_dict["state_mean"],   # ✅ 从 JSON 读取（list）
            state_std=normalize_dict["state_std"],
            device=self.device,
            learning_rate=model_param["learning_rate"],
            time_dim=model_param["time_dim"],
            block_config=model_param["block_config"],
            expectile=model_param["expectile"],
        )

        self.model.load_net(model_path)
        self.model.eval()

        # =========================
        # Evaluation buffers
        # =========================
        self.target_return = None
        self.last_step_reward = None
        self.current_step = 0

    # =========================
    # Reset at episode start
    # =========================
    def reset(self, target_return):
        """
        Called once per (ad_id, period) episode
        """
        self.remaining_budget = self.budget

        self.current_step = 0
        self.last_step_reward = None

        # reset DT internal buffers
        self.model.init_eval()

        self.target_return = torch.tensor(
            [[target_return]], dtype=torch.float32, device=self.device
        )

    # =========================
    # Bidding decision per timeStepIndex
    # =========================
    def bidding(self, state, timeStepIndex, last_step_conversion=None):
        """
        Args:
            state: np.ndarray, current env state (step-level)
            timeStepIndex: int
            last_step_conversion: float, conversion obtained in previous step
        Returns:
            alpha: float
        """

        # -------------------------
        # First step: provide target_return
        # -------------------------
        if timeStepIndex == 0:
            assert self.target_return is not None, "target_return must be set before first bidding"

            alpha = self.model.take_actions(
                state=state,
                target_return=self.target_return,
                budget=self.budget,
                cpa=self.cpa,
            )

        # -------------------------
        # Subsequent steps: provide pre_reward
        # -------------------------
        else:
            assert last_step_conversion is not None, "pre_reward (last_step_conversion) is required"

            alpha = self.model.take_actions(
                state=state,
                pre_reward=last_step_conversion,
                budget=self.budget,
                cpa=self.cpa,
            )

        self.current_step += 1
        return float(alpha)