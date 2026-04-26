"""
Ensemble training pipeline - trains PPO + A2C + DDPG on NIFTY50 data.
Produces equity curve comparison report.
"""
from __future__ import annotations
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def train_ensemble(
    data_dir: Path,
    output_dir: Path,
    timesteps: int = 50000,
    wandb_enabled: bool = True,
):
    """
    Full ensemble training pipeline on the kalyan197 dataset.
    """
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from data_pipeline.kaggle.dataset_loader import KaggleDatasetLoader
    from data_pipeline.kaggle.ensemble_data_prep import EnsembleDataPrep
    from models.ensemble_rl.base_agents import StockTradingEnv, PPOAgent, A2CAgent, DDPGAgent
    from models.ensemble_rl.ensemble_selector import EnsembleSelector

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- WandB init ---
    if wandb_enabled:
        try:
            import wandb
            wandb.init(
                project="pcp-arb-rl",
                name=f"ensemble_train_{datetime.now().strftime('%Y%m%d_%H%M')}",
                tags=["ensemble", "nifty50"],
            )
        except Exception as e:
            logger.warning("WandB init failed: %s", e)
            wandb_enabled = False

    # --- Step 1: Load data ---
    print("[EnsembleTrain] Loading NIFTY50 dataset...")
    loader = KaggleDatasetLoader()
    try:
        stocks = loader.load_all_stocks(data_dir)
    except Exception as e:
        print(f"[EnsembleTrain] Error loading dataset: {e}")
        print("[EnsembleTrain] Generating synthetic NIFTY50 data for training fallback")
        stocks = _generate_synthetic_nifty50()

    # Memory optimization: use only top 10 stocks by data count
    stock_counts = stocks.groupby("Symbol").size().sort_values(ascending=False)
    top_stocks = stock_counts.head(10).index.tolist()
    stocks = stocks[stocks["Symbol"].isin(top_stocks)].copy()
    print(f"[EnsembleTrain] Using top {len(top_stocks)} stocks: {top_stocks}")

    stats = loader.get_date_range_stats(stocks)
    print(f"[EnsembleTrain] Loaded: {stats['total_trading_days']} days, {stats['total_symbols']} symbols")

    # --- Step 2: Prepare FinRL format ---
    print("[EnsembleTrain] Preparing FinRL format...")
    prep = EnsembleDataPrep()
    # Memory optimization: skip heavy indicators for now
    finrl_df = prep.prepare_finrl_format(stocks, tech_indicators=[])

    # Skip turbulence to save memory - set to 0
    finrl_df["turbulence"] = 0.0
    print("[EnsembleTrain] Turbulence index: skipped (memory optimization)")

    # --- Step 3: Split data ---
    train_df, val_df, test_df = prep.split_data(finrl_df)
    print(f"[EnsembleTrain] Train: {len(train_df)} rows, Val: {len(val_df)}, Test: {len(test_df)}")

    stock_dim = min(10, finrl_df["tic"].nunique())


    # --- Step 4: Initialize agents + train ---
    print("[EnsembleTrain] Initializing agents...")
    ppo = PPOAgent()
    a2c = A2CAgent()
    ddpg = DDPGAgent()

    selector = EnsembleSelector(
        agents=[ppo, a2c, ddpg],
        rebalance_window_days=63,
        validation_window_days=63,
    )

    print(f"[EnsembleTrain] Training ensemble ({timesteps} timesteps per agent per period)...")
    selector.train_all(train_df, val_df, timesteps_per_agent=timesteps)

    # Log selection history
    report = selector.get_selection_report()
    if len(report) > 0:
        print("\n[EnsembleTrain] Agent Selection History:")
        print(report.to_string(index=False))

    # --- Step 5: Evaluate on test set ---
    print("\n[EnsembleTrain] Evaluating on test set (2022-2026)...")
    test_env = StockTradingEnv(test_df, stock_dim=stock_dim)

    results = {}
    strategies = {"PPO": ppo, "A2C": a2c, "DDPG": ddpg}
    equity_curves = {}

    # NIFTY50 Baseline (Buy & Hold)
    print("  Calculating NIFTY50 Buy & Hold baseline...")
    index_df = loader.load_nifty_index(data_dir)
    test_dates = sorted(test_df["date"].unique())
    index_test = index_df[index_df["Date"].isin(test_dates)].sort_values("Date")
    if len(index_test) > 0:
        base_prices = index_test["Close"].values
        # Normalize to starting amount
        baseline_values = (base_prices / base_prices[0]) * test_env.initial_amount
        equity_curves["NIFTY50_B&H"] = baseline_values.tolist()
        results["NIFTY50_B&H"] = _compute_metrics(baseline_values)
        print(f"  Baseline: Return={results['NIFTY50_B&H']['total_return']:.1f}%, Sharpe={results['NIFTY50_B&H']['sharpe']:.3f}")

    for name, agent in strategies.items():
        values = _run_agent(agent, test_env)
        equity_curves[name] = values
        metrics = _compute_metrics(values)
        results[name] = metrics
        improvement = ""
        if "NIFTY50_B&H" in results:
            alpha = metrics['total_return'] - results['NIFTY50_B&H']['total_return']
            improvement = f" (Alpha: {alpha:+.1f}%)"
        print(f"  {name}: Sharpe={metrics['sharpe']:.3f}, Return={metrics['total_return']:.1f}%{improvement}")

    # Ensemble strategy
    ensemble_values = _run_ensemble(selector, test_env, test_df)
    equity_curves["Ensemble"] = ensemble_values
    metrics = _compute_metrics(ensemble_values)
    results["Ensemble"] = metrics
    improvement = ""
    if "NIFTY50_B&H" in results:
        alpha = metrics['total_return'] - results['NIFTY50_B&H']['total_return']
        improvement = f" (Alpha: {alpha:+.1f}%)"
    print(f"  Ensemble: Sharpe={metrics['sharpe']:.3f}, Return={metrics['total_return']:.1f}%{improvement}")

    # --- Step 6: Save agents ---
    ckpt_dir = output_dir / "ensemble"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ppo.save(str(ckpt_dir / "ppo_nifty50"))
    a2c.save(str(ckpt_dir / "a2c_nifty50"))
    ddpg.save(str(ckpt_dir / "ddpg_nifty50"))
    print(f"\n[EnsembleTrain] Agents saved to {ckpt_dir}")

    # Save results JSON
    results_path = output_dir / "ensemble_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # --- Generate comparison plot ---
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {
        "PPO": "#4CAF50", 
        "A2C": "#2196F3", 
        "DDPG": "#FF9800", 
        "Ensemble": "#E91E63",
        "NIFTY50_B&H": "#000000" # Black for baseline
    }
    for name, curve in equity_curves.items():
        label = f"{name} (Ret: {results[name]['total_return']:.1f}%)"
        ax.plot(curve, label=label, color=colors.get(name, "#666"), linewidth=2 if name=="Ensemble" else 1.5)
    
    ax.set_title("NIFTY50 Portfolio RL vs Buy & Hold (Original Values)")
    ax.set_ylabel("Portfolio Value (INR)")
    ax.set_xlabel("Test Period (Days)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plot_path = reports_dir / "ensemble_comparison.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"[EnsembleTrain] Comparison plot saved to {plot_path}")

    if wandb_enabled:
        try:
            import wandb
            wandb.log({"ensemble_comparison": wandb.Image(str(plot_path))})
            wandb.finish()
        except Exception:
            pass

    return str(ckpt_dir)


def _run_agent(agent, env) -> list:
    obs, _ = env.reset()
    values = [env.initial_amount]
    done = False
    while not done:
        action = agent.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        values.append(info.get("portfolio_value", values[-1]))
        done = done or truncated
    return values


def _run_ensemble(selector, env, test_df) -> list:
    from datetime import date as dt_date
    obs, _ = env.reset()
    values = [env.initial_amount]
    done = False
    dates = sorted(test_df["date"].unique())
    day_idx = 0
    while not done:
        d = dates[min(day_idx, len(dates) - 1)]
        turb = test_df[test_df["date"] == d]["turbulence"].mean() if "turbulence" in test_df.columns else 0
        action = selector.get_ensemble_action(obs, d, turb)
        obs, reward, done, truncated, info = env.step(action)
        values.append(info.get("portfolio_value", values[-1]))
        done = done or truncated
        day_idx += 1
    return values


def _compute_metrics(values: list) -> dict:
    arr = np.array(values)
    returns = np.diff(arr) / arr[:-1]
    total_return = (arr[-1] - arr[0]) / arr[0] * 100
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
    max_dd = 0
    peak = arr[0]
    for v in arr:
        peak = max(peak, v)
        dd = (peak - v) / peak * 100
        max_dd = max(max_dd, dd)
    return {"total_return": total_return, "sharpe": sharpe, "max_dd": max_dd}


def _generate_synthetic_nifty50():
    import pandas as pd
    dates = pd.bdate_range("2010-01-01", "2026-03-31")
    symbols = [f"STOCK_{i}" for i in range(50)]
    rows = []
    for sym in symbols:
        price = 1000.0
        for d in dates:
            price *= (1 + np.random.normal(0.0005, 0.02))
            rows.append({"Date": d, "Symbol": sym, "Open": price*0.99, "High": price*1.01, "Low": price*0.98, "Close": price, "Volume": 1000000})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/kaggle")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    train_ensemble(Path(args.data_dir), Path(args.output_dir), args.timesteps, not args.no_wandb)
