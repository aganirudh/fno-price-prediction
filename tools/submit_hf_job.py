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

    api = HfApi(token=token)
    user = api.whoami()
    print(f"Authenticated as: {user['name']}")

    print(f"Submitting training job to {REPO_ID}...")
    try:
        job = api.run_job(
            repo_id=REPO_ID,
            command=[
                "bash", "-c",
                "pip install -q stable-baselines3 gymnasium shimmy stockstats scikit-learn kaggle wandb && "
                "python tools/download_data.py && "
                "python main.py --mode train-ensemble --timesteps 50000 --no-wandb"
            ],
            repo_type="space",
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
    subprocess.run([sys.executable, "tools/download_data.py"], check=True)
    subprocess.run([
        sys.executable, "main.py",
        "--mode", "train-ensemble",
        "--timesteps", "50000",
        "--no-wandb"
    ], check=True)

if __name__ == "__main__":
    submit()
