"""
Scraper Engine — Implements institutional-grade data acquisition.
Handles 60s live polling and Parquet storage for option chains.
"""
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional
from data.historical.nse_downloader import NSEDownloader
from config.settings import CACHE_DIR

class ScraperEngine:
    """Orchestrates live data scraping and Parquet storage."""

    def __init__(self):
        self.downloader = NSEDownloader()
        self.base_path = CACHE_DIR / "live_snapshots"
        self.base_path.mkdir(parents=True, exist_ok=True)

    def scrape_to_parquet(self, symbol: str, interval_seconds: int = 60):
        """
        Polls the NSE live API at fixed intervals and saves snapshots.
        Anchors data for Step 2 of the institutional strategy.
        """
        symbol_dir = self.base_path / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📡 [Step 2] Starting {symbol} Live Scraper...")
        print(f"💾 Saving Parquet snapshots to: {symbol_dir}")

        try:
            while True:
                now = datetime.now()
                # Check if market is open (09:15 to 15:30)
                market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
                market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

                if market_open <= now <= market_close:
                    start_time = time.time()
                    print(f"[{now.strftime('%H:%M:%S')}] Requesting {symbol} snapshot...")
                    
                    chain = self.downloader.download_option_chain_snapshot(symbol)
                    
                    if chain:
                        # Prepare data for Parquet
                        records = []
                        for s in chain.strikes:
                            records.append({
                                "timestamp": chain.timestamp,
                                "strike": s.strike,
                                "spot": chain.spot_price,
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
                                "expiry": chain.expiry
                            })
                        
                        df = pd.DataFrame(records)
                        filename = f"snap_{symbol}_{now.strftime('%Y%m%d_%H%M%S')}.parquet"
                        df.to_parquet(symbol_dir / filename)
                        print(f"  ✅ Snapshot saved: {len(df)} rows")
                    else:
                        print(f"  ⚠️ Failed to fetch {symbol} data (NSE might be rate-limiting).")
                    
                    # Calculate sleep to maintain precise interval
                    elapsed = time.time() - start_time
                    sleep_time = max(0, interval_seconds - elapsed)
                    time.sleep(sleep_time)
                else:
                    print(f"[{now.strftime('%H:%M:%S')}] Market is closed. Checking again in 5 minutes...")
                    time.sleep(300)

        except KeyboardInterrupt:
            print("\n🛑 Scraper halted by user. Data preserved in Parquet.")

if __name__ == "__main__":
    engine = ScraperEngine()
    engine.scrape_to_parquet("NIFTY", interval_seconds=60)
