"""
Hybrid training - trains the SB3 ensemble + GRPO LLM agent jointly.
"""
from __future__ import annotations
import json
import logging
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

def train_hybrid(
    ensemble_checkpoint: Path,
    arb_data_dir: Path,
    output_dir: Path,
    wandb_enabled: bool = True,
    grpo_steps: int = 1000,
):
    """
    Hybrid training pipeline.
    """
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from models.ensemble_rl.base_agents import PPOAgent, A2CAgent, DDPGAgent
    from models.ensemble_rl.ensemble_selector import EnsembleSelector
    from models.ensemble_rl.regime_detector import RegimeDetector
    from models.ensemble_rl.hybrid_controller import HybridController
    from data_pipeline.kaggle.fundamentals_processor import FundamentalsProcessor, FundamentalsSnapshot

    ensemble_checkpoint = Path(ensemble_checkpoint)
    arb_data_dir = Path(arb_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[HybridTrain] Loading pre-trained ensemble...")
    ppo, a2c, ddpg = PPOAgent(), A2CAgent(), DDPGAgent()
    
    ckpt = ensemble_checkpoint
    if (ckpt / "ppo_nifty50.zip").exists(): ppo.load(str(ckpt / "ppo_nifty50"))
    if (ckpt / "a2c_nifty50.zip").exists(): a2c.load(str(ckpt / "a2c_nifty50"))
    if (ckpt / "ddpg_nifty50.zip").exists(): ddpg.load(str(ckpt / "ddpg_nifty50"))

    selector = EnsembleSelector(agents=[ppo, a2c, ddpg])
    regime_detector = RegimeDetector()
    fund_processor = FundamentalsProcessor()

    hybrid = HybridController(
        ensemble_selector=selector,
        regime_detector=regime_detector,
        fundamentals_processor=fund_processor,
    )

    # --- Step 3: Proceed to GRPO fine-tuning ---
    print(f"[HybridTrain] Starting GRPO fine-tuning of the Arb Layer ({grpo_steps} steps)...")
    from training.train import train
    
    # We pass the ensemble_checkpoint so the environment or trainer can potentially use it
    # For now, train() handles the LLM fine-tuning part
    try:
        grpo_checkpoint = train(
            total_steps=grpo_steps,
            checkpoint_path=None, # Start fresh or from base
            wandb_enabled=wandb_enabled
        )
        print(f"[HybridTrain] GRPO training complete! Checkpoint: {grpo_checkpoint}")
    except Exception as e:
        print(f"[HybridTrain] GRPO training failed: {e}")
        grpo_checkpoint = "failed"

    final_dir = output_dir / "hybrid_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "grpo_steps": grpo_steps,
        "status": "completed" if grpo_checkpoint != "failed" else "partial",
        "ensemble_checkpoint": str(ensemble_checkpoint),
        "grpo_checkpoint": str(grpo_checkpoint),
        "timestamp": datetime.now().isoformat()
    }
    with open(final_dir / "hybrid_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return str(final_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-checkpoint", type=str, default="checkpoints/ensemble")
    parser.add_argument("--arb-data-dir", type=str, default="data/historical")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    train_hybrid(Path(args.ensemble_checkpoint), Path(args.arb_data_dir), Path(args.output_dir), not args.no_wandb, args.steps)
