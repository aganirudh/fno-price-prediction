"""
PCP Arb RL System - Main entry point.
Supports multiple modes: alpha, backtest, train, paper, demo, analyze,
train-ensemble, train-hybrid, compare.
"""
from __future__ import annotations
import argparse
import json
import sys
import os
from datetime import date, datetime
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def cmd_alpha(args):
    """Run alpha analysis to determine if PCP arb opportunity exists."""
    from tools.alpha_analyzer import AlphaAnalyzer
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    analyzer = AlphaAnalyzer()
    report_path = analyzer.generate_alpha_report(args.underlying, start, end)
    print(f"\n[OK] Alpha report generated: {report_path}")
    print("Open this HTML file in a browser to see interactive charts.")


def cmd_backtest(args):
    """Run backtest over a date range."""
    from backtest.engine import BacktestEngine
    from backtest.report import generate_report
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    engine = BacktestEngine(initial_capital=args.capital)

    model, tokenizer = None, None
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"[Main] Loading model from {args.checkpoint}...")
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.checkpoint, max_seq_length=1144,
                dtype=None, load_in_4bit=True)
        except Exception as e:
            print(f"[Main] Could not load model: {e}. Running baseline agent.")

    results = engine.run(model=model, tokenizer=tokenizer,
                         start_date=start, end_date=end,
                         underlying=args.underlying, mode=args.feed_mode)
    report_path = generate_report(
        {"summary": results, "sessions": engine.session_results},
        args.underlying, start, end)
    print(f"\n[OK] Backtest complete!")
    print(f"   Total P&L: INR {results['total_pnl']:,.0f}")
    print(f"   Sharpe: {results['sharpe_ratio']:.3f}")
    print(f"   Win Rate: {results['win_rate_pct']:.1f}%")
    print(f"   Report: {report_path}")

    # Generate step analysis for 3 sample sessions
    if engine.session_results:
        from tools.recorder import StepAnalyzer
        analyzer = StepAnalyzer()
        for session in engine.session_results[:3]:
            log_path = session.get("log_path")
            if log_path and os.path.exists(log_path):
                with open(log_path) as f:
                    for line in f:
                        entry = json.loads(line)
                        analyzer.record_step(
                            step=entry.get("step", 0),
                            observation="",
                            raw_output=json.dumps(entry.get("action", {})),
                            parsed_action=entry.get("action", {}),
                            tool_calls=entry.get("action", {}).get("tool_calls", []),
                            tool_results={},
                            reward_breakdown=entry.get("reward_breakdown", {}),
                            position_delta=0,
                            cumulative_pnl=entry.get("pnl", 0),
                            timestamp=datetime.now())
                report = analyzer.generate_step_report(session.get("date", "unknown"))
                print(f"   Step analysis: {report}")
                analyzer.clear()


def cmd_train(args):
    """Start training - supports grpo, ensemble, or hybrid agents."""
    agent_type = getattr(args, "agent", "grpo")

    if agent_type == "ensemble":
        cmd_train_ensemble(args)
    elif agent_type == "hybrid":
        cmd_train_hybrid(args)
    else:
        # Original GRPO training
        from training.train import train
        print(f"[Main] Starting GRPO training for {args.steps} steps...")
        checkpoint_path = train(
            total_steps=args.steps,
            checkpoint_path=args.checkpoint,
            wandb_enabled=not args.no_wandb)
        print(f"\n[OK] Training complete! Checkpoint: {checkpoint_path}")


def cmd_train_ensemble(args):
    """Train SB3 ensemble (PPO + A2C + DDPG) on NIFTY50 data."""
    from training.ensemble_train import train_ensemble
    data_dir = Path(getattr(args, "data_dir", "data/kaggle"))
    output_dir = Path(getattr(args, "output_dir", "checkpoints"))
    timesteps = getattr(args, "timesteps", 50000)

    print(f"[Main] Starting ensemble training on NIFTY50 data...")
    print(f"   Data dir: {data_dir}")
    print(f"   Timesteps per agent: {timesteps}")

    checkpoint = train_ensemble(
        data_dir=data_dir,
        output_dir=output_dir,
        timesteps=timesteps,
        wandb_enabled=not args.no_wandb,
    )
    print(f"\n[OK] Ensemble training complete! Checkpoint: {checkpoint}")


def cmd_train_hybrid(args):
    """Train hybrid ensemble + GRPO system."""
    from training.hybrid_train import train_hybrid
    ensemble_ckpt = Path(getattr(args, "ensemble_checkpoint", "checkpoints/ensemble"))
    arb_data = Path(getattr(args, "arb_data_dir", "data/historical"))
    output_dir = Path(getattr(args, "output_dir", "checkpoints"))
    steps = getattr(args, "steps", 1000)

    print(f"[Main] Starting hybrid training...")
    print(f"   Ensemble checkpoint: {ensemble_ckpt}")
    print(f"   Arb data: {arb_data}")

    result = train_hybrid(
        ensemble_checkpoint=ensemble_ckpt,
        arb_data_dir=arb_data,
        output_dir=output_dir,
        wandb_enabled=not args.no_wandb,
        grpo_steps=steps,
    )
    print(f"\n[OK] Hybrid training complete! Output: {result}")


def cmd_compare(args):
    """Run full 5-strategy comparison backtest and generate HTML report."""
    from backtest.ensemble_backtest import run_full_comparison
    from models.ensemble_rl.base_agents import PPOAgent, A2CAgent, DDPGAgent
    from models.ensemble_rl.ensemble_selector import EnsembleSelector
    import pandas as pd

    output_dir = Path(getattr(args, "output_dir", "reports"))
    ensemble_ckpt = Path(getattr(args, "ensemble_checkpoint", "checkpoints/ensemble"))

    print("[Main] Running 5-strategy comparison...")

    # Load ensemble if available
    ppo, a2c, ddpg = PPOAgent(), A2CAgent(), DDPGAgent()
    try:
        if (ensemble_ckpt / "ppo_nifty50.zip").exists():
            ppo.load(str(ensemble_ckpt / "ppo_nifty50"))
        if (ensemble_ckpt / "a2c_nifty50.zip").exists():
            a2c.load(str(ensemble_ckpt / "a2c_nifty50"))
        if (ensemble_ckpt / "ddpg_nifty50.zip").exists():
            ddpg.load(str(ensemble_ckpt / "ddpg_nifty50"))
        print("  Loaded pre-trained ensemble agents")
    except Exception as e:
        print(f"  Could not load ensemble: {e} - using untrained agents")

    selector = EnsembleSelector(agents=[ppo, a2c, ddpg])

    # Load test data
    data_dir = Path(getattr(args, "data_dir", "data/kaggle"))
    test_df = None
    try:
        from data_pipeline.kaggle.dataset_loader import KaggleDatasetLoader
        from data_pipeline.kaggle.ensemble_data_prep import EnsembleDataPrep
        loader = KaggleDatasetLoader()
        stocks = loader.load_all_stocks(data_dir)
        prep = EnsembleDataPrep()
        finrl = prep.prepare_finrl_format(stocks)
        _, _, test_df = prep.split_data(finrl)
        print(f"  Loaded test data: {len(test_df)} rows")
    except Exception as e:
        print(f"  Could not load test data: {e} - using synthetic")

    result = run_full_comparison(
        ensemble=selector,
        test_df=test_df,
        output_dir=output_dir,
    )

    print(f"\n[OK] Comparison complete!")
    for name, sr in result.strategies.items():
        print(f"   {name}: Return={sr.total_return_pct:.1f}%, Sharpe={sr.annualized_sharpe:.3f}, MaxDD={sr.max_drawdown_pct:.1f}%")


def cmd_paper(args):
    """Run paper trading with live feed and dashboard."""
    from data.feeds.live_feed import LiveFeed
    from data.feeds.mock_feed import MockFeed
    from mcp_servers.mcp_client import MCPClient
    from pcp_arb_env.environment import PCPArbEnv
    from monitoring.dashboard import Dashboard
    from monitoring.alerts import AlertManager
    from training.rollout import SYSTEM_PROMPT, parse_action

    if args.feed == "live":
        feed = LiveFeed()
        print("[Main] Starting live feed (polling NSE endpoints)...")
    else:
        feed = MockFeed(underlyings=["NIFTY", "BANKNIFTY"],
                        violations_per_session=10,
                        violation_pct_range=(0.3, 1.5))
        print("[Main] Starting mock feed for paper trading...")

    mcp = MCPClient(timeout=5.0)
    env = PCPArbEnv(feed=feed, mcp_client=mcp)
    dashboard = Dashboard()
    alerts = AlertManager()
    dashboard.start_logging()

    model, tokenizer = None, None
    if args.checkpoint and os.path.exists(args.checkpoint):
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.checkpoint, max_seq_length=1144,
                dtype=None, load_in_4bit=True)
            print(f"[Main] Loaded model from {args.checkpoint}")
        except Exception as e:
            print(f"[Main] Model load failed: {e}")

    obs = env.reset()
    dashboard.update(feed_health=mcp.check_health())

    print("[Main] Paper trading started. Press Ctrl+C to stop.")
    try:
        import time
        step = 0
        while not env.done:
            step += 1
            if model and tokenizer:
                import torch
                prompt = f"{SYSTEM_PROMPT}\n\nCurrent state:\n{obs}"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=120,
                                              do_sample=True, temperature=0.5,
                                              pad_token_id=tokenizer.eos_token_id)
                response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                             skip_special_tokens=True)
                action, _ = parse_action(response)
            else:
                action = {"action_type": "hold", "tool_calls": [
                    {"server": "market_data", "tool": "get_option_chain",
                     "params": {"underlying": "NIFTY", "expiry": ""}}
                ], "strike": None, "qty": 1}

            result = env.step(action)
            state = env.state()
            dashboard.update(
                market={"violations": state.get("violations", [])},
                agent={"action": action, "reward_breakdown": result.reward.to_dict(),
                       "tool_calls": action.get("tool_calls", [])},
                positions=state.get("positions", []),
                pnl=state.get("daily_pnl", 0),
                feed_health=mcp.check_health(),
                step=step)
            alerts.check_daily_pnl(state.get("daily_pnl", 0), 50000)
            obs = result.observation
            from rich.console import Console
            console = Console()
            console.print(dashboard.render())
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[Main] Paper trading stopped.")
        print(f"[Main] Final P&L: INR {env._daily_pnl:,.0f}")


def cmd_demo(args):
    """Run interactive CLI demo."""
    from tools.demo import run_demo
    run_demo()


def cmd_analyze(args):
    """Analyze a recorded session log."""
    from tools.recorder import StepAnalyzer
    if not os.path.exists(args.session):
        print(f"[Main] Session file not found: {args.session}")
        return
    analyzer = StepAnalyzer()
    with open(args.session) as f:
        for line in f:
            entry = json.loads(line)
            analyzer.record_step(
                step=entry.get("step", 0), observation="",
                raw_output=json.dumps(entry.get("action", {})),
                parsed_action=entry.get("action", {}),
                tool_calls=entry.get("action", {}).get("tool_calls", []),
                tool_results={}, reward_breakdown=entry.get("reward_breakdown", {}),
                position_delta=0, cumulative_pnl=entry.get("pnl", 0),
                timestamp=datetime.now())
    report = analyzer.generate_step_report(Path(args.session).stem)
    print(f"[OK] Step analysis report: {report}")


def main():
    """Main entry point for the PCP Arb RL system."""
    parser = argparse.ArgumentParser(
        description="PCP Arbitrage RL System - Put-Call Parity Arb with MCP + GRPO + Ensemble",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode alpha --underlying NIFTY --start 2024-01-01 --end 2024-06-30
  python main.py --mode backtest --underlying NIFTY --start 2024-01-01 --end 2024-06-30
  python main.py --mode train --agent grpo --steps 3000
  python main.py --mode train --agent ensemble --data-dir data/kaggle
  python main.py --mode train --agent hybrid --ensemble-checkpoint checkpoints/ensemble
  python main.py --mode train-ensemble --data-dir data/kaggle --timesteps 50000
  python main.py --mode train-hybrid --ensemble-checkpoint checkpoints/ensemble
  python main.py --mode compare --data-dir data/kaggle
  python main.py --mode paper --feed mock
  python main.py --mode demo
  python main.py --mode analyze --session logs/session_20240415.jsonl
        """)
    parser.add_argument("--mode", required=True,
                        choices=["alpha", "backtest", "train", "paper", "demo", "analyze",
                                 "train-ensemble", "train-hybrid", "compare"],
                        help="Operating mode")
    parser.add_argument("--agent", default="grpo", choices=["grpo", "ensemble", "hybrid"],
                        help="Agent type for training (grpo, ensemble, hybrid)")
    parser.add_argument("--underlying", default="NIFTY", help="Underlying symbol")
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-06-30", help="End date (YYYY-MM-DD)")
    parser.add_argument("--steps", type=int, default=3000, help="Training steps")
    parser.add_argument("--timesteps", type=int, default=50000, help="SB3 timesteps per agent")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    parser.add_argument("--ensemble-checkpoint", default="checkpoints/ensemble",
                        help="Ensemble checkpoint directory")
    parser.add_argument("--data-dir", default="data/kaggle", help="NIFTY50 dataset directory")
    parser.add_argument("--arb-data-dir", default="data/historical", help="Arb training data dir")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--feed", default="mock", choices=["mock", "live", "historical"],
                        help="Feed type for paper/backtest")
    parser.add_argument("--feed-mode", default="historical",
                        choices=["historical", "mock"], help="Backtest feed mode")
    parser.add_argument("--capital", type=float, default=1_000_000,
                        help="Initial capital for backtest")
    parser.add_argument("--session", default=None, help="Session log path for analyze mode")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    args = parser.parse_args()

    print(f"PCP Arbitrage RL System - Mode: {args.mode}")
    print(f"{'='*60}")

    mode_map = {
        "alpha": cmd_alpha,
        "backtest": cmd_backtest,
        "train": cmd_train,
        "paper": cmd_paper,
        "demo": cmd_demo,
        "analyze": cmd_analyze,
        "train-ensemble": cmd_train_ensemble,
        "train-hybrid": cmd_train_hybrid,
        "compare": cmd_compare,
    }
    mode_map[args.mode](args)


if __name__ == "__main__":
    main()
