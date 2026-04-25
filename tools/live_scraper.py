"""
NSE Live Option Chain Scraper — Implementation of Step 2.
Polls NSE API every 60s and saves full snapshots to Parquet for historical reconstruction.
"""
import time
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional
from data.historical.nse_downloader import NSEDownloader
from config.settings import DATA_DIR

class NSELiveScraper:
    """Orchestrates the live scraping of NSE option chains to Parquet."""

    def __init__(self, base_path: Optional[Path] = None):
        self.downloader = NSEDownloader()
        self.base_path = base_path or (DATA_DIR / "historical" / "live_snapshots")
        self.base_path.mkdir(parents=True, exist_ok=True)

    def is_market_open(self) -> bool:
        """Check if current time is within NSE trading hours (9:15 - 15:30)."""
        now = datetime.now()
        if now.weekday() >= 5:  # Saturday/Sunday
            return False
        
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        return market_start <= now <= market_end

    def scrape_to_parquet(self, symbol: str, interval_seconds: int = 60):
        """
        Polls the NSE API at fixed intervals and saves snapshots to Parquet.
        
        Args:
            symbol: Underlying symbol (NIFTY, BANKNIFTY)
            interval_seconds: Polling frequency (Step 2 requirement: 60s)
        """
        symbol_dir = self.base_path / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Initializing Step 2 Scraper for {symbol}...")
        print(f"📁 Data will be saved to: {symbol_dir}")
        print(f"⏱️ Interval: {interval_seconds}s")

        try:
            while True:
                if self.is_market_open():
                    now = datetime.now()
                    print(f"[{now.strftime('%H:%M:%S')}] Fetching snapshot...", end=" ", flush=True)
                    
                    try:
                        chain = self.downloader.download_option_chain_snapshot(symbol)
                        if chain:
                            # Flatten chain strikes into a list of dicts
                            records = []
                            for s in chain.strikes:
                                records.append({
                                    "timestamp": chain.timestamp.isoformat(),
                                    "underlying": symbol,
                                    "spot": chain.spot_price,
                                    "strike": s.strike,
                                    "call_ltp": s.call_ltp,
                                    "call_bid": s.call_bid,
                                    "call_ask": s.call_ask,
                                    "call_oi": s.call_oi,
                                    "call_iv": s.call_iv,
                                    "put_ltp": s.put_ltp,
                                    "put_bid": s.put_bid,
                                    "put_ask": s.put_ask,
                                    "put_oi": s.put_oi,
                                    "put_iv": s.put_iv,
                                })
                            
                            df = pd.DataFrame(records)
                            filename = f"snap_{now.strftime('%Y%m%d_%H%M%S')}.parquet"
                            save_path = symbol_dir / filename
                            df.to_parquet(save_path, compression='snappy')
                            print(f"✅ Success. Saved {len(df)} rows.")
                        else:
                            print("❌ Failed (No data returned).")
                    except Exception as e:
                        print(f"❌ Error during scrape: {e}")
                else:
                    now = datetime.now()
                    print(f"[{now.strftime('%H:%M:%S')}] Market is currently closed. Idling...")
                
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n🛑 Scraper terminated by user.")

if __name__ == "__main__":
    import sys
    symbol_arg = sys.argv[1] if len(sys.argv) > 1 else "NIFTY"
    scraper = NSELiveScraper()
    scraper.scrape_to_parquet(symbol=symbol_arg, interval_seconds=60)
