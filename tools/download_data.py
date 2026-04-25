"""
Helper script to download Kaggle dataset.
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline.kaggle.dataset_loader import KaggleDatasetLoader

def main():
    print("Initializing Kaggle download...")
    loader = KaggleDatasetLoader()
    data_dir = Path("data/kaggle")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        loader.download_dataset(data_dir)
        print(f"Success! Files downloaded to {data_dir}")
    except Exception as e:
        print(f"Download failed: {e}")

if __name__ == "__main__":
    main()
