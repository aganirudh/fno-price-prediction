"""
FundamentalsProcessor - extracts market signals from fundamentals.
Computes PE Z-scores, HHI concentration, and Arb Propensity Score.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class FundamentalsSnapshot:
    """Market-wide fundamental context for a specific date."""
    date: date
    nifty50_pe_zscore: float
    earnings_momentum: float
    market_cap_concentration: float  # HHI index
    fundamental_volatility_signal: float
    nifty50_median_pe: float

class FundamentalsProcessor:
    """Processes raw stock-level fundamentals into market-wide state signals."""

    def __init__(self, rolling_window: int = 252):
        self.window = rolling_window

    def compute_rolling_fundamentals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute market-wide metrics over time:
        - Median PE of NIFTY50
        - PE Z-score (relative to history)
        - HHI (concentration)
        - Earnings momentum (y/y change in median EPS)
        """
        # 1. Group by Date to get market-wide median PE
        market_stats = df.groupby("Date").agg({
            "PE_Ratio": ["median", "mean", "std"],
            "EPS": "median",
            "Market_Cap": "sum"
        })
        market_stats.columns = ["median_pe", "mean_pe", "std_pe", "median_eps", "total_mcap"]
        
        # 2. PE Z-Score
        market_stats["pe_zscore"] = (
            market_stats["median_pe"] - market_stats["median_pe"].rolling(self.window).mean()
        ) / market_stats["median_pe"].rolling(self.window).std()

        # 3. Earnings Momentum (6-month change in median EPS)
        market_stats["eps_momentum"] = market_stats["median_eps"].pct_change(periods=126) * 100

        # 4. HHI Index (Market Concentration)
        def calc_hhi(group):
            if "Market_Cap" not in group.columns or group["Market_Cap"].sum() == 0:
                return 0.0
            weights = group["Market_Cap"] / group["Market_Cap"].sum()
            return (weights ** 2).sum()
        
        hhi_series = df.groupby("Date").apply(calc_hhi)
        market_stats["hhi"] = hhi_series

        # 5. Fundamental Volatility (Std of PE changes)
        market_stats["fund_vol"] = market_stats["median_pe"].pct_change().rolling(21).std()

        return market_stats.fillna(0)

    def get_arb_propensity_score(self, snapshot: FundamentalsSnapshot) -> float:
        """
        Heuristic: High PE Z-score + high volatility = higher probability
         of PCP violations due to rapid regime shifts and mispricing.
        Returns score between 0.0 and 1.0.
        """
        score = 0.5 # Baseline
        
        # Overvaluation signal (Z-score > 1.5)
        if snapshot.nifty50_pe_zscore > 1.5:
            score += 0.2
        elif snapshot.nifty50_pe_zscore < -1.5:
            score -= 0.1 # Undervalued markets are often more 'orderly' for PCP
            
        # Volatility signal
        if snapshot.fundamental_volatility_signal > 0.02: # 2% daily PE vol
            score += 0.2
            
        # Concentration signal (High HHI means few stocks drive index)
        if snapshot.market_cap_concentration > 0.08:
            score += 0.1

        return float(np.clip(score, 0.0, 1.0))

    def get_snapshot(self, processed_df: pd.DataFrame, target_date: date) -> Optional[FundamentalsSnapshot]:
        """Retrieve snapshot for a specific date."""
        target_ts = pd.Timestamp(target_date)
        if target_ts not in processed_df.index:
            # Try to find the closest previous date
            available_dates = processed_df.index[processed_df.index <= target_ts]
            if available_dates.empty:
                return None
            target_ts = available_dates[-1]

        row = processed_df.loc[target_ts]
        return FundamentalsSnapshot(
            date=target_ts.date(),
            nifty50_pe_zscore=row["pe_zscore"],
            earnings_momentum=row["eps_momentum"],
            market_cap_concentration=row["hhi"],
            fundamental_volatility_signal=row["fund_vol"],
            nifty50_median_pe=row["median_pe"]
        )