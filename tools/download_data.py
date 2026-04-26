"""
Helper script to download Kaggle dataset.
Loads credentials from .env file if environment variables are not set.
"""
import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def load_env():
    """Load .env file from project root if it exists."""
    env_file = ROOT / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    if not os.environ.get(key.strip()):
                        os.environ[key.strip()] = val.strip()

def main():
    load_env()

    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    print(f"Kaggle credentials: user={username}, key={'SET' if key else 'MISSING'}")

    from data_pipeline.kaggle.dataset_loader import KaggleDatasetLoader
    
    loader = KaggleDatasetLoader()
    data_dir = Path("data/kaggle")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if data already exists
    existing = list(data_dir.rglob("*.csv"))
    if existing:
        total_size = sum(f.stat().st_size for f in existing) / (1024*1024)
        print(f"Data already exists: {len(existing)} CSV files ({total_size:.1f} MB)")
        return

    print("Initializing Kaggle download...")
    try:
        loader.download_dataset(data_dir)
        downloaded = list(data_dir.rglob("*.csv"))
        if downloaded:
            total_size = sum(f.stat().st_size for f in downloaded) / (1024*1024)
            print(f"Success! {len(downloaded)} CSV files ({total_size:.1f} MB) in {data_dir}")
        else:
            print("WARNING: Download completed but no CSV files found. Check credentials.")
    except Exception as e:
        print(f"Download failed: {e}")

if __name__ == "__main__":
    main()
