"""
Submit a training job to Hugging Face Compute.
Uses the Python API directly (avoids Windows CLI PATH issues).
Token is read from HF_TOKEN env var or from huggingface-cli login cache.
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import HfApi

REPO_ID = "aganirudh/fno-price-prediction"

def submit():
    token = os.environ.get("HF_TOKEN")
    if not token:
        # Try to read from cached login
        try:
            from huggingface_hub import login
            token = HfApi().token
        except Exception:
            pass
    
    if not token:
        print("ERROR: Set HF_TOKEN environment variable or run 'huggingface-cli login' first.")
        sys.exit(1)

    # Authenticated as: aganirudh (cached)
    api = HfApi(token=token)
    print(f"Submitting training job for {REPO_ID}...")
    try:
        # Match hf_job.yaml configurations
        job = api.run_job(
            image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
            command=[
                "bash", "-c",
                "apt-get update && apt-get install -y git wget && "
                "git clone https://huggingface.co/spaces/aganirudh/fno-price-prediction /app && "
                "cd /app && "
                "pip install -q stable-baselines3 gymnasium shimmy stockstats scikit-learn kaggle wandb matplotlib && "
                "pip install -q 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git' && "
                "export KAGGLE_USERNAME=aganirudh && "
                "export KAGGLE_KEY=KGAT_7006cb8c11132de6b9d15252661dc6c1 && "
                "python tools/download_data.py && "
                "python main.py --mode train-ensemble --timesteps 50000 --no-wandb && "
                "python main.py --mode train-hybrid --steps 1000 --no-wandb && "
                "huggingface-cli upload aganirudh/fno-price-prediction checkpoints/ ./checkpoints --repo-type space"
            ],
            env={
                "KAGGLE_USERNAME": "aganirudh",
                "KAGGLE_KEY": "KGAT_7006cb8c11132de6b9d15252661dc6c1",
                "PYTHONPATH": "."
            },
            flavor="a10g-small",
        )
        print(f"[OK] Job submitted: {job}")
    except Exception as e:
        print(f"[ERROR] Job submission failed: {e}")
        print("Falling back to local training...")
        fallback_local()

def fallback_local():
    """Run locally - SB3 does NOT need a GPU."""
    import subprocess
    print("\n--- Running Ensemble Training Locally (CPU) ---")
    root = Path(__file__).resolve().parent.parent
    subprocess.run([sys.executable, str(root / "tools/download_data.py")], check=True, cwd=str(root))
    subprocess.run([
        sys.executable, str(root / "main.py"),
        "--mode", "train-ensemble",
        "--timesteps", "50000",
        "--no-wandb"
    ], check=True, cwd=str(root))

if __name__ == "__main__":
    submit()
