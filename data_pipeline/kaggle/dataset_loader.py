"""
KaggleDatasetLoader - loads and processes the kalyan197 NIFTY50 dataset.
Handles 25 years of daily OHLCV + fundamentals (PE, EPS, BV, Div Yield).
"""
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class KaggleDatasetLoader:
    """Loader for the kalyan197 NIFTY50 fundamentals dataset."""

    def __init__(self, dataset_name: str = "kalyan197/nifty50-stocks1999-2026-daily-ohlcv-and-fundamentals"):
        self.dataset_name = dataset_name

    def download_dataset(self, target_dir: Path):
        """Download dataset from Kaggle if not present."""
        if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
            logger.warning("Kaggle credentials not set. Skipping download.")
            return
        try:
            import kaggle
            target_dir.mkdir(parents=True, exist_ok=True)
            kaggle.api.authenticate()
            logger.info("Downloading %s to %s", self.dataset_name, target_dir)
            kaggle.api.dataset_download_files(self.dataset_name, path=str(target_dir), unzip=True)
        except Exception as e:
            logger.error("Kaggle download failed: %s", e)

    def load_all_stocks(self, data_dir: Path) -> pd.DataFrame:
        """
        Load all stock CSVs from the directory and concatenate into a single DF.
        Aligns columns to a standard format: [Date, Symbol, Open, High, Low, Close, Volume, PE, EPS, BV, DivYield].
        """
        data_dir = Path(data_dir)
        csv_files = list(data_dir.rglob("*.csv"))
        
        if not csv_files:
            logger.warning("No CSV files found in %s. Attempting download...", data_dir)
            self.download_dataset(data_dir)
            csv_files = list(data_dir.rglob("*.csv"))

        all_dfs = []
        for f in csv_files:
            # Skip metadata or index files if they don't match stock format
            if f.stem.lower() in ["nifty50_index", "metadata"]:
                continue
            
            try:
                df = pd.read_csv(f)
                df["Symbol"] = f.stem
                all_dfs.append(df)
            except Exception as e:
                logger.warning("Failed to load %s: %s", f, e)

        if not all_dfs:
            raise FileNotFoundError(f"No valid stock data found in {data_dir}")

        master_df = pd.concat(all_dfs, ignore_index=True)
        master_df["Date"] = pd.to_datetime(master_df["Date"])
        master_df.sort_values(["Date", "Symbol"], inplace=True)
        
        # Forward fill fundamentals per stock
        for sym in master_df["Symbol"].unique():
            mask = master_df["Symbol"] == sym
            master_df.loc[mask] = master_df.loc[mask].ffill()

        return master_df

    def load_nifty_index(self, data_dir: Path) -> pd.DataFrame:
        """Load or construct the market-cap weighted NIFTY50 index."""
        data_dir = Path(data_dir)
        index_file = data_dir / "nifty50_index.csv"
        
        if index_file.exists():
            df = pd.read_csv(index_file)
            df["Date"] = pd.to_datetime(df["Date"])
            return df

        # Fallback: construct from constituents
        logger.info("Constructing synthetic NIFTY50 index from constituent data...")
        stocks = self.load_all_stocks(data_dir)
        
        # We use Market_Cap if available, otherwise equal weight
        if "Market_Cap" in stocks.columns:
            index_df = stocks.groupby("Date").apply(
                lambda x: np.average(x["Close"], weights=x["Market_Cap"])
            ).reset_index()
        else:
            index_df = stocks.groupby("Date")["Close"].mean().reset_index()
        
        index_df.columns = ["Date", "Close"]
        return index_df

    def get_date_range_stats(self, df: pd.DataFrame) -> Dict:
        """Compute basic stats about the dataset coverage."""
        return {
            "start_date": df["Date"].min().date(),
            "end_date": df["Date"].max().date(),
            "total_symbols": df["Symbol"].nunique(),
            "total_trading_days": df["Date"].nunique(),
            "avg_pe": df["PE_Ratio"].mean() if "PE_Ratio" in df.columns else None,
        }