"""
EnsembleSelector - rolling-window Sharpe-based agent selection.
"""
from __future__ import annotations
import json
import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class EnsembleSelector:
    def __init__(self, agents: list, rebalance_window_days: int = 63, validation_window_days: int = 63, turbulence_threshold: float = 200):
        self.agents = agents
        self.rebalance_window = rebalance_window_days
        self.validation_window = validation_window_days
        self.turbulence_threshold = turbulence_threshold
        
        # Optimized storage: Separate lists are 10-100x faster for DataFrame creation than list-of-dicts
        self._h_period = []
        self._h_selected_agent = []
        self._h_turbulence = []
        self._h_sharpes = {a.name: [] for a in agents}
        
        # Cache last state for fast prediction access
        self._last_sharpes = {a.name: 1.0 for a in agents}
        self._last_best_name = agents[0].name if agents else None

    def train_all(self, train_df: pd.DataFrame, val_df: pd.DataFrame, timesteps_per_agent: int = 25000):
        from models.ensemble_rl.base_agents import StockTradingEnv
        train_dates = sorted(train_df["date"].unique())
        stock_dim = min(50, train_df["tic"].nunique())
        
        n_periods = max(1, len(train_dates) // self.rebalance_window)
        for i in range(n_periods):
            end_idx = min((i + 1) * self.rebalance_window, len(train_dates))
            train_end = train_dates[end_idx - 1]
            period_train = train_df[train_df["date"] <= train_end]
            
            val_start_idx = end_idx
            if val_start_idx >= len(train_dates):
                period_val = val_df.head(self.validation_window * stock_dim)
            else:
                val_end_idx = min(val_start_idx + self.validation_window, len(train_dates))
                period_val = train_df[(train_df["date"] >= train_dates[val_start_idx]) & (train_df["date"] <= train_dates[val_end_idx-1])]
            
            if len(period_train) < self.rebalance_window or len(period_val) == 0: continue
            
            t_env, v_env = StockTradingEnv(period_train, stock_dim=stock_dim), StockTradingEnv(period_val, stock_dim=stock_dim)
            sharpes = {}
            for agent in self.agents:
                agent.train(t_env, total_timesteps=timesteps_per_agent)
                sharpes[agent.name] = agent.get_sharpe(v_env)
            
            avg_turb = period_val["turbulence"].mean() if "turbulence" in period_val.columns else 0
            if avg_turb > self.turbulence_threshold:
                if "PPO" in sharpes: sharpes["PPO"] *= 0.5
                if "DDPG" in sharpes: sharpes["DDPG"] *= 0.3
            
            best = max(sharpes, key=sharpes.get)
            
            # Optimized history recording
            self._h_period.append(i)
            self._h_selected_agent.append(best)
            self._h_turbulence.append(float(avg_turb))
            for a_name in self._h_sharpes:
                self._h_sharpes[a_name].append(sharpes.get(a_name, 0))
                
            self._last_sharpes = sharpes
            self._last_best_name = best

    def select_agent(self, current_date: date):
        name = self._last_best_name
        if name is None: return self.agents[0]
        for a in self.agents:
            if a.name == name: return a
        return self.agents[0]

    def get_ensemble_action(self, obs: np.ndarray, current_date: date, turbulence: float) -> np.ndarray:
        if turbulence > self.turbulence_threshold:
            return self.select_agent(current_date).predict(obs)
        
        sharpes = self._last_sharpes
        vals = np.array([max(sharpes.get(a.name, 0), 0.01) for a in self.agents])
        weights = np.exp(vals) / np.sum(np.exp(vals))
        actions = [a.predict(obs) for a in self.agents]
        return sum(w * a for w, a in zip(weights, actions))

    def get_selection_report(self) -> pd.DataFrame:
        """
        Fast creation of selection report DataFrame.
        Building from separate lists is O(N) where N is number of rows, 
        whereas building from list-of-dicts is much slower due to dict parsing.
        """
        if not self._h_period:
            return pd.DataFrame()
            
        data = {
            "period": self._h_period,
            "selected_agent": self._h_selected_agent,
            "turbulence": self._h_turbulence
        }
        # Flatten sharpes into columns for better reporting
        for a_name, vals in self._h_sharpes.items():
            data[f"sharpe_{a_name}"] = vals
            
        return pd.DataFrame(data)
