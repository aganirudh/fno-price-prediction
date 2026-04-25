"""
EnsembleDataPrep - converts NIFTY50 OHLCV + fundamentals into FinRL-style format
for ensemble RL training (PPO + A2C + DDPG).
"""
from __future__ import annotations
import logging
from typing import List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_TECH_INDICATORS = [
    "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30",
    "dx_30", "close_30_sma", "close_60_sma",
]


class EnsembleDataPrep:
    """Prepares data in FinRL format for ensemble training."""

    def prepare_finrl_format(
        self, stock_df: pd.DataFrame, tech_indicators: List[str] = None
    ) -> pd.DataFrame:
        """
        Add technical indicators via stockstats and fundamental columns.
        Returns DataFrame in FinRL format: date, tic, open, high, low, close,
        volume, [tech_indicators], [fundamentals].
        """
        if tech_indicators is None:
            tech_indicators = DEFAULT_TECH_INDICATORS

        df = stock_df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values(["Date", "Symbol"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Rename to FinRL convention
        rename_map = {
            "Date": "date", "Symbol": "tic", "Open": "open",
            "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
        }
        df.rename(columns=rename_map, inplace=True)

        # Add technical indicators per stock using stockstats
        try:
            from stockstats import StockDataFrame
            dfs = []
            for tic, grp in df.groupby("tic"):
                sdf = grp.copy().reset_index(drop=True)
                stock = StockDataFrame.retype(sdf[["date","open","high","low","close","volume"]].copy())
                for ind in tech_indicators:
                    try:
                        sdf[ind] = stock[ind].values
                    except Exception:
                        sdf[ind] = 0.0
                dfs.append(sdf)
            df = pd.concat(dfs, ignore_index=True)
        except ImportError:
            logger.warning("stockstats not installed - computing basic indicators manually")
            df = self._compute_basic_indicators(df, tech_indicators)

        # Add fundamental columns
        if "PE_Ratio" in df.columns:
            df["pe_ratio"] = df["PE_Ratio"].fillna(0)
        else:
            df["pe_ratio"] = 0.0
        if "EPS" in df.columns:
            df["eps_momentum"] = df.groupby("tic")["EPS"].pct_change(periods=63).fillna(0) * 100
        else:
            df["eps_momentum"] = 0.0
        if "Price_to_Book" in df.columns and "close" in df.columns:
            df["book_to_market"] = np.where(
                df["Price_to_Book"] > 0,
                1.0 / df["Price_to_Book"].fillna(1),
                0.0,
            )
        elif "Book_Value" in df.columns and "close" in df.columns:
            df["book_to_market"] = np.where(
                df["close"] > 0,
                df["Book_Value"].fillna(0) / df["close"],
                0.0,
            )
        else:
            df["book_to_market"] = 0.0


        # Clean up
        df.sort_values(["date", "tic"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Fill NaN in tech indicators
        for ind in tech_indicators:
            if ind in df.columns:
                df[ind] = df[ind].fillna(0)

        logger.info("Prepared FinRL format: %d rows, %d columns", len(df), len(df.columns))
        return df

    def _compute_basic_indicators(self, df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
        """Fallback: compute basic technical indicators without stockstats."""
        for tic, grp in df.groupby("tic"):
            idx = grp.index
            close = grp["close"]
            # SMA
            if "close_30_sma" in indicators:
                df.loc[idx, "close_30_sma"] = close.rolling(30, min_periods=1).mean()
            if "close_60_sma" in indicators:
                df.loc[idx, "close_60_sma"] = close.rolling(60, min_periods=1).mean()
            # RSI
            if "rsi_30" in indicators:
                delta = close.diff()
                gain = delta.where(delta > 0, 0).rolling(30, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(30, min_periods=1).mean()
                rs = gain / loss.replace(0, np.nan)
                df.loc[idx, "rsi_30"] = 100 - (100 / (1 + rs))
            # Bollinger Bands
            if "boll_ub" in indicators or "boll_lb" in indicators:
                sma20 = close.rolling(20, min_periods=1).mean()
                std20 = close.rolling(20, min_periods=1).std().fillna(0)
                df.loc[idx, "boll_ub"] = sma20 + 2 * std20
                df.loc[idx, "boll_lb"] = sma20 - 2 * std20
            # MACD
            if "macd" in indicators:
                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                df.loc[idx, "macd"] = ema12 - ema26
            # CCI
            if "cci_30" in indicators:
                tp = (grp["high"] + grp["low"] + close) / 3
                sma_tp = tp.rolling(30, min_periods=1).mean()
                mad = tp.rolling(30, min_periods=1).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
                df.loc[idx, "cci_30"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
            # DX
            if "dx_30" in indicators:
                df.loc[idx, "dx_30"] = 0.0  # Simplified placeholder

        for ind in indicators:
            if ind in df.columns:
                df[ind] = df[ind].fillna(0)
            else:
                df[ind] = 0.0
        return df

    def split_data(
        self, df: pd.DataFrame, train_end: str = "2018-12-31",
        val_end: str = "2021-12-31",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split into train/validation/test strictly by date.
        Train: 1999-2018, Validation: 2019-2021, Test: 2022-2026.
        Never shuffle.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        train = df[df["date"] <= train_end].copy()
        val = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
        test = df[df["date"] > val_end].copy()
        logger.info(
            "Split: train=%d (%s->%s), val=%d, test=%d",
            len(train),
            train["date"].min().date() if len(train) > 0 else "N/A",
            train["date"].max().date() if len(train) > 0 else "N/A",
            len(val), len(test),
        )
        return train, val, test

    def compute_turbulence_index(self, df: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
        """
        Compute turbulence index (Mahalanobis distance of current returns
        from the historical covariance matrix).

        High turbulence = ensemble should prefer A2C (defensive).
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Pivot returns: (date, tic) -> close
        pivot = df.pivot_table(index="date", columns="tic", values="close")
        pivot.sort_index(inplace=True)
        returns = pivot.pct_change().dropna(how="all")

        turbulence_values = {}
        dates = returns.index.tolist()

        for i in range(lookback, len(dates)):
            current = returns.iloc[i:i+1].values.flatten()
            hist_returns = returns.iloc[i-lookback:i]

            # Drop columns with all NaN
            valid_cols = hist_returns.columns[hist_returns.notna().all()]
            if len(valid_cols) < 2:
                turbulence_values[dates[i]] = 0.0
                continue

            hist_clean = hist_returns[valid_cols].fillna(0)
            current_clean = returns.iloc[i:i+1][valid_cols].fillna(0).values.flatten()

            mean_returns = hist_clean.mean().values
            diff = current_clean - mean_returns

            try:
                cov = hist_clean.cov().values
                cov_inv = np.linalg.pinv(cov)
                turb = float(diff @ cov_inv @ diff.T)
                turbulence_values[dates[i]] = turb
            except Exception:
                turbulence_values[dates[i]] = 0.0

        # Map turbulence back to original dataframe
        turb_series = pd.Series(turbulence_values)
        turb_df = pd.DataFrame({"date": turb_series.index, "turbulence": turb_series.values})
        turb_df["date"] = pd.to_datetime(turb_df["date"])

        df = df.merge(turb_df, on="date", how="left")
        df["turbulence"] = df["turbulence"].fillna(0)

        logger.info(
            "Turbulence index: mean=%.1f, max=%.1f, >200 days=%d",
            df["turbulence"].mean(), df["turbulence"].max(),
            (df["turbulence"] > 200).sum(),
        )
        return df