#!/bin/bash
# Final Unified Run Script for HF Compute Jobs
set -e

echo "--- INSTALLING BASE DEPENDENCIES ---"
pip install -q stable-baselines3 gymnasium shimmy stockstats scikit-learn kaggle wandb matplotlib

echo "--- STEP 0: DOWNLOADING DATA ---"
export KAGGLE_USERNAME=aganirudh
export KAGGLE_KEY=KGAT_7006cb8c11132de6b9d15252661dc6c1
python tools/download_data.py

echo "--- STEP 1: TRAINING ENSEMBLE (CPU) ---"
python main.py --mode train-ensemble --timesteps 50000 --no-wandb

echo "--- INSTALLING GPU DEPENDENCIES (UNSLOTH) ---"
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

echo "--- STEP 2: TRAINING HYBRID GRPO (GPU) ---"
python main.py --mode train-hybrid --steps 1000 --no-wandb

echo "--- UPLOADING CHECKPOINTS ---"
# Note: Token should be in environment
huggingface-cli upload aganirudh/fno-price-prediction checkpoints/ ./checkpoints --repo-type space
