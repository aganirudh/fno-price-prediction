"""
ICICI Breeze API Downloader — Pulls 3 years of 1-min OHLCV data.
Requires 'breeze_connect' package and ICICI Direct API keys.
"""
import os
import pandas as pd
from datetime import datetime, timedelta
from breeze_connect import BreezeConnect
from config.settings import DATA_DIR

class BreezeDownloader:
    def __init__(self, api_key, api_secret, session_token):
        self.breeze = BreezeConnect(api_key=api_key)
        self.breeze.generate_session(api_secret=api_secret, session_token=session_token)
        self.output_dir = DATA_DIR / "historical" / "breeze_1min"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_historical(self, symbol, from_date, to_date, interval="1minute"):
        """Pull data in chunks to handle API limits."""
        print(f"📥 Downloading {symbol} from {from_date} to {to_date}...")
        
        # Breeze usually limits requests to 30-day chunks
        start = datetime.strptime(from_date, "%Y-%m-%d")
        end = datetime.strptime(to_date, "%Y-%m-%d")
        
        all_data = []
        current = start
        while current < end:
            chunk_end = min(current + timedelta(days=30), end)
            print(f"  Fetching chunk: {current.date()} to {chunk_end.date()}")
            
            resp = self.breeze.get_historical_data_v2(
                interval=interval,
                from_date=current.strftime("%Y-%m-%dT09:15:00.000Z"),
                to_date=chunk_end.strftime("%Y-%m-%dT15:30:00.000Z"),
                stock_code=symbol,
                exchange_code="NFO",
                product_type="options"
            )
            
            if resp['status'] == 200:
                all_data.extend(resp['Success'])
            else:
                print(f"  ⚠️ Error fetching chunk: {resp['message']}")
            
            current = chunk_end + timedelta(days=1)
        
        df = pd.DataFrame(all_data)
        file_path = self.output_dir / f"{symbol}_1min_3years.parquet"
        df.to_parquet(file_path)
        print(f"✅ Saved deep history to {file_path}")
        return df

if __name__ == "__main__":
    # Skeleton usage
    # downloader = BreezeDownloader(api_key="...", api_secret="...", session_token="...")
    # downloader.download_historical("NIFTY", "2022-01-01", "2025-01-01")
    print("Breeze Downloader Ready. Configure your keys to start.")
