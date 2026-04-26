"""
KaggleDatasetLoader - loads and processes the kalyan197 NIFTY50 dataset.
The dataset is a SINGLE CSV (nifty50_historical_data.csv) with a Ticker column,
NOT 50 separate CSVs. Handles 25 years of daily OHLCV + fundamentals.
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

    def download_dataset(self, target_dir):
        """Download dataset from Kaggle if not present."""
        target_dir = Path(target_dir)
        if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
            logger.warning("Kaggle credentials not set. Skipping download.")
            return
        try:
            import kaggle
            target_dir.mkdir(parents=True, exist_ok=True)
            kaggle.api.authenticate()
            print(f"Dataset URL: https://www.kaggle.com/datasets/{self.dataset_name}")
            kaggle.api.dataset_download_files(self.dataset_name, path=str(target_dir), unzip=True)
            # List what was downloaded
            all_files = list(target_dir.rglob("*"))
            print(f"Downloaded {len(all_files)} files to {target_dir}")
            for f in all_files:
                if f.is_file():
                    size_mb = f.stat().st_size / (1024*1024)
                    print(f"  {f.name} ({size_mb:.1f} MB)")
        except Exception as e:
            logger.error("Kaggle download failed: %s", e)

    def _find_main_csv(self, data_dir: Path) -> Optional[Path]:
        """Find the main historical data CSV (the largest one, or by name)."""
        data_dir = Path(data_dir)
        # Try known names first
        known_names = [
            "nifty50_historical_data.csv",
            "nifty50_all_stocks.csv",
            "nifty_50_historical_data.csv",
        ]
        for name in known_names:
            for f in data_dir.rglob(name):
                return f

        # Fallback: find the largest CSV
        csv_files = list(data_dir.rglob("*.csv"))
        if not csv_files:
            return None
        # Return the largest CSV (the main data file)
        return max(csv_files, key=lambda f: f.stat().st_size)

    def load_all_stocks(self, data_dir) -> pd.DataFrame:
        """
        Load the NIFTY50 dataset. The kalyan197 dataset is a SINGLE CSV
        with a Ticker column containing all 50 stocks.
        """
        data_dir = Path(data_dir)

        main_csv = self._find_main_csv(data_dir)
        if main_csv is None:
            logger.warning("No CSV files found in %s. Attempting download...", data_dir)
            self.download_dataset(data_dir)
            main_csv = self._find_main_csv(data_dir)

        if main_csv is None:
            raise FileNotFoundError(f"No valid stock data found in {data_dir}")

        print(f"[DataLoader] Loading main CSV: {main_csv.name} ({main_csv.stat().st_size / (1024*1024):.1f} MB)")
        # Optimization: only load needed columns and use float32
        needed_cols = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume", "PE_Ratio", "Price_to_Book"]
        df = pd.read_csv(main_csv, usecols=lambda x: any(c in x for c in needed_cols))
        print(f"[DataLoader] Raw shape: {df.shape}")

        # Standardize column names
        col_map = {}
        for col in df.columns:
            cl = col.lower().strip()
            if cl == "ticker":
                col_map[col] = "Symbol"
            elif cl == "company_name":
                col_map[col] = "Company"
            elif cl == "date":
                col_map[col] = "Date"
            elif cl == "open":
                col_map[col] = "Open"
            elif cl == "high":
                col_map[col] = "High"
            elif cl == "low":
                col_map[col] = "Low"
            elif cl == "close":
                col_map[col] = "Close"
            elif cl == "volume":
                col_map[col] = "Volume"
            elif cl in ("pe_ratio", "pe ratio"):
                col_map[col] = "PE_Ratio"
            elif cl == "eps":
                col_map[col] = "EPS"
            elif cl == "beta":
                col_map[col] = "Beta"
            elif cl in ("market_cap", "marketcap"):
                col_map[col] = "Market_Cap"
            elif cl in ("price_to_book", "pb_ratio"):
                col_map[col] = "Price_to_Book"
            elif cl in ("dividend_yield", "div_yield"):
                col_map[col] = "Dividend_Yield"
            elif cl == "sector":
                col_map[col] = "Sector"
            elif cl in ("daily_return", "daily return"):
                col_map[col] = "Daily_Return"
            elif cl in ("volatility_20d", "volatility"):
                col_map[col] = "Volatility_20D"
            elif cl == "ma_50":
                col_map[col] = "MA_50"
            elif cl == "ma_200":
                col_map[col] = "MA_200"

        df.rename(columns=col_map, inplace=True)

        # Parse dates
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.dropna(subset=["Date"], inplace=True)
        df.sort_values(["Date", "Symbol"], inplace=True)

        # Clean the ticker symbols (remove .NS suffix if present)
        if "Symbol" in df.columns:
            df["Symbol"] = df["Symbol"].str.replace(".NS", "", regex=False)

        # Forward fill fundamentals per stock
        fundamental_cols = [c for c in ["PE_Ratio", "EPS", "Beta", "Market_Cap",
                                        "Price_to_Book", "Dividend_Yield"] if c in df.columns]
        if fundamental_cols:
            for col in fundamental_cols:
                df[col] = df.groupby("Symbol")[col].transform(lambda x: x.ffill())

        n_symbols = df["Symbol"].nunique()
        n_days = df["Date"].nunique()
        date_range = f"{df['Date'].min().date()} to {df['Date'].max().date()}"
        print(f"[DataLoader] Loaded: {n_symbols} stocks, {n_days} trading days ({date_range})")
        print(f"[DataLoader] Stocks: {sorted(df['Symbol'].unique())[:10]}... (showing first 10)")

        return df

    def load_nifty_index(self, data_dir) -> pd.DataFrame:
        """Load or construct the market-cap weighted NIFTY50 index."""
        data_dir = Path(data_dir)

        # Try summary file
        for f in data_dir.rglob("*summary*"):
            if f.suffix == ".csv":
                df = pd.read_csv(f)
                if "Date" in df.columns:
                    return df

        # Construct from constituents
        stocks = self.load_all_stocks(data_dir)
        if "Market_Cap" in stocks.columns:
            index_df = stocks.groupby("Date").apply(
                lambda x: np.average(x["Close"], weights=x["Market_Cap"].fillna(1)),
                include_groups=False
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