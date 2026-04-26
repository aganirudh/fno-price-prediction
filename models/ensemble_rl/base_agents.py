"""
Base agents - StockTradingEnv + PPO/A2C/DDPG wrappers via stable-baselines3.
"""
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


class StockTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: "pd.DataFrame",
        stock_dim: int = 10,
        initial_amount: float = 1_000_000,
        transaction_cost_pct: float = 0.001,
        max_shares_per_trade: int = 100,
        tech_indicator_list: List[str] = None,
        fundamental_list: List[str] = None,
        turbulence_threshold: float = 200,
    ):
        super().__init__()
        import pandas as pd

        self.df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.stt_exercise_pct = 0.00125  # 0.125% STT on exercise/heavy liquidations
        self.max_shares = max_shares_per_trade
        self.turbulence_threshold = turbulence_threshold

        self.tech_indicators = tech_indicator_list or [
            "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30",
            "dx_30", "close_30_sma", "close_60_sma",
        ]
        self.fundamentals = fundamental_list or ["pe_ratio", "eps_momentum", "book_to_market"]

        state_dim = stock_dim + stock_dim * len(self.tech_indicators) + stock_dim * len(self.fundamentals) + stock_dim + 2

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(stock_dim,), dtype=np.float32
        )

        self.dates = sorted(self.df["date"].unique())
        self.day = 0
        self.terminal = False

        self.cash = initial_amount
        self.holdings = np.zeros(stock_dim)
        self.portfolio_value_history = [initial_amount]
        self.cost_history = []
        self.trade_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day = 0
        self.cash = self.initial_amount
        self.holdings = np.zeros(self.stock_dim)
        self.portfolio_value_history = [self.initial_amount]
        self.cost_history = []
        self.trade_count = 0
        self.terminal = False
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        if self.day >= len(self.dates) - 1:
            self.terminal = True
            return self._get_obs(), 0.0, True, False, self._get_info()

        prices = self._get_prices(self.day)
        prev_value = self._portfolio_value(prices)

        turbulence = self._get_turbulence(self.day)
        if turbulence > self.turbulence_threshold:
            # Crisis/Exercise regime: apply higher STT tax
            for i in range(self.stock_dim):
                if self.holdings[i] > 0 and prices[i] > 0:
                    proceeds = self.holdings[i] * prices[i]
                    cost = proceeds * self.stt_exercise_pct
                    self.cash += proceeds - cost
                    self.cost_history.append(cost)
                    self.trade_count += 1
                    self.holdings[i] = 0
        else:
            action = np.clip(action, -1, 1)
            for i in range(self.stock_dim):
                if action[i] < -0.1 and prices[i] > 0 and self.holdings[i] > 0:
                    shares = min(int(abs(action[i]) * self.max_shares), int(self.holdings[i]))
                    if shares > 0:
                        proceeds = shares * prices[i]
                        # Standard trade cost
                        cost = proceeds * self.transaction_cost_pct
                        self.cash += proceeds - cost
                        self.holdings[i] -= shares
                        self.cost_history.append(cost)
                        self.trade_count += 1

                elif action[i] > 0.1 and prices[i] > 0:
                    shares = min(int(action[i] * self.max_shares), int(self.cash / (prices[i] * (1 + self.transaction_cost_pct))))
                    if shares > 0:
                        cost_basis = shares * prices[i]
                        txn_cost = cost_basis * self.transaction_cost_pct
                        self.cash -= cost_basis + txn_cost
                        self.holdings[i] += shares
                        self.cost_history.append(txn_cost)
                        self.trade_count += 1

        self.day += 1
        new_prices = self._get_prices(self.day)
        new_value = self._portfolio_value(new_prices)
        self.portfolio_value_history.append(new_value)
        reward = (new_value - prev_value) / prev_value if prev_value > 0 else 0.0
        done = self.day >= len(self.dates) - 1
        return self._get_obs(), float(reward), done, False, self._get_info()

    def _get_obs(self) -> np.ndarray:
        day_data = self._get_day_data(self.day)
        prices = self._get_prices(self.day)
        tech = []
        for ind in self.tech_indicators:
            tech.extend(day_data[ind].values[:self.stock_dim].tolist() if ind in day_data.columns else [0.0]*self.stock_dim)
        fund = []
        for f in self.fundamentals:
            fund.extend(day_data[f].values[:self.stock_dim].tolist() if f in day_data.columns else [0.0]*self.stock_dim)
        total_val = self._portfolio_value(prices)
        weights = (self.holdings * prices) / max(total_val, 1e-8)
        turb = self._get_turbulence(self.day)
        obs = np.concatenate([prices, np.array(tech), np.array(fund), weights, [self.cash / max(total_val, 1e-8)], [turb]]).astype(np.float32)
        return np.nan_to_num(obs)

    def _get_day_data(self, day: int):
        target_date = self.dates[min(day, len(self.dates)-1)]
        data = self.df[self.df["date"] == target_date]
        if len(data) < self.stock_dim:
            import pandas as pd
            pad = pd.DataFrame(0.0, index=range(self.stock_dim - len(data)), columns=data.columns)
            data = pd.concat([data, pad], ignore_index=True)
        return data.head(self.stock_dim)

    def _get_prices(self, day: int) -> np.ndarray:
        return self._get_day_data(day)["close"].values[:self.stock_dim].astype(float)

    def _get_turbulence(self, day: int) -> float:
        data = self._get_day_data(day)
        return float(data["turbulence"].iloc[0]) if "turbulence" in data.columns else 0.0

    def _portfolio_value(self, prices: np.ndarray) -> float:
        return self.cash + float(np.sum(self.holdings * prices))

    def _get_info(self) -> dict:
        return {"portfolio_value": self.portfolio_value_history[-1], "cash": self.cash, "trade_count": self.trade_count}


class BaseAgentWrapper:
    def __init__(self, name: str):
        self.name = name
        self.model = None

    def predict(self, obs: np.ndarray) -> np.ndarray:
        if self.model is None: return np.zeros(50)
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def get_sharpe(self, env, n_episodes: int = 2) -> float:
        rets = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done, values = False, [env.initial_amount]
            while not done:
                action = self.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                values.append(info.get("portfolio_value", values[-1]))
                done = done or truncated
            rets.extend((np.diff(values) / np.array(values[:-1])).tolist())
        return float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if np.std(rets) > 0 else 0.0

    def save(self, path: str):
        if self.model: self.model.save(path)

    def load(self, path: str):
        raise NotImplementedError


class PPOAgent(BaseAgentWrapper):
    def __init__(self): super().__init__("PPO")
    def train(self, env, total_timesteps=50000):
        from stable_baselines3 import PPO
        from config.settings import get_settings
        device = get_settings().training.ensemble_device
        # Optimized for A10G: Larger batch size and n_steps
        self.model = PPO("MlpPolicy", env, verbose=0, n_steps=2048, batch_size=256, device=device)
        self.model.learn(total_timesteps=total_timesteps)
    def load(self, path):
        from stable_baselines3 import PPO
        self.model = PPO.load(path)


class A2CAgent(BaseAgentWrapper):
    def __init__(self): super().__init__("A2C")
    def train(self, env, total_timesteps=50000):
        from stable_baselines3 import A2C
        from config.settings import get_settings
        device = get_settings().training.ensemble_device
        # Optimized for A10G
        self.model = A2C("MlpPolicy", env, verbose=0, n_steps=128, device=device)
        self.model.learn(total_timesteps=total_timesteps)
    def load(self, path):
        from stable_baselines3 import A2C
        self.model = A2C.load(path)


class DDPGAgent(BaseAgentWrapper):
    def __init__(self): super().__init__("DDPG")
    def train(self, env, total_timesteps=50000):
        from stable_baselines3 import DDPG
        from config.settings import get_settings
        device = get_settings().training.ensemble_device
        # Optimized for A10G: Larger buffer and batch size
        self.model = DDPG("MlpPolicy", env, verbose=0, buffer_size=100000, batch_size=256, device=device)
        self.model.learn(total_timesteps=total_timesteps)
    def load(self, path):
        from stable_baselines3 import DDPG
        self.model = DDPG.load(path)
