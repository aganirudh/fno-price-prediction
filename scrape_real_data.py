
import sys
import time
from datetime import date, timedelta
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.historical.nse_downloader import NSEDownloader

def scrape_last_30_trading_days():
    downloader = NSEDownloader()
    
    # We want at least 30 trading days. 
    # Since weekends and holidays exist, we look back ~45-50 calendar days.
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=50)
    
    print(f"📡 Initializing Real Data Scraper (NSE Bhavcopy)...")
    print(f"Target: ~30 trading days between {start_date} and {end_date}")
    
    trading_days_found = 0
    current = start_date
    
    while current <= end_date:
        if current.weekday() < 5: # Monday to Friday
            print(f"  Attempting {current}...", end=" ", flush=True)
            df = downloader.download_bhavcopy(current)
            if df is not None:
                print(f"✅ Downloaded ({len(df)} rows)")
                trading_days_found += 1
            else:
                print(f"❌ Not found (Holiday?)")
            
            # Respect NSE rate limits
            time.sleep(1.5) 
            
        current += timedelta(days=1)
        if trading_days_found >= 30:
            break

    print(f"\n✨ Task Complete! Successfully scraped {trading_days_found} days of 100% real NSE data.")
    print(f"Data stored in: pcp-arb-rl/data/historical/cache/")

if __name__ == "__main__":
    scrape_last_30_trading_days()
