import yfinance as yf
import pandas as pd
import os
import json
from datetime import datetime

class YahooFinanceCrawl:
    def __init__(self, base_path="data/raw/yahoo_finance"):
        self.base_path = base_path
        self.meta_file = os.path.join(base_path, "metadata.json")
        os.makedirs(base_path, exist_ok=True)
    def load_metadata(self):
      if os.path.exists(self.meta_file):
          try:
              with open(self.meta_file, "r") as f:
                  content = f.read().strip()
                  if content:  
                      return json.loads(content)
                  else:  
                      return {}
          except (json.JSONDecodeError, FileNotFoundError):
              
              return {}
      return {}  
    
    def save_metadata(self, meta):
        with open(self.meta_file, "w") as f:
            json.dump(meta, f, indent=4)
            
    def make_key(self, ticker, start, end, interval):
        return f"{ticker}_{start}_{end}_{interval}"
    
    def data_exists(self, meta, key):
        return key in meta and os.path.exists(meta[key]["filepath"])
    
    def fetch_data(self, ticker, start, end, interval="1d"):
        meta = self.load_metadata()
        key = self.make_key(ticker, start, end, interval)

        if self.data_exists(meta, key):
            print(f"✅ Yf data exists: {meta[key]['filepath']}")
            return meta[key]["filepath"]
        else:
            print(f"⬇️ Downloading Yf data for {ticker} from {start} to {end}...")
            data_info = self.download_csv(ticker, start, end, interval)
            meta[key] = data_info
            self.save_metadata(meta)
            return data_info["filepath"]
    
    def download_csv(self, ticker, start, end, interval="1d"):
        stock = yf.Ticker(ticker)
        data = stock.history(start=start, end=end, interval=interval)
        
        filename = f"{ticker}_{start}_{end}_{interval}.csv"
        filepath = os.path.join(self.base_path, filename)
        data.to_csv(filepath)
        
        
        return {
            "filepath": filepath,
            "records_count": len(data),
            "actual_start": data.index.min().strftime('%Y-%m-%d') if not data.empty else None,
            "actual_end": data.index.max().strftime('%Y-%m-%d') if not data.empty else None,
            "downloaded_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def load_data(self, ticker, start, end, interval="1d"):
        """Load existing data from CSV file"""
        filepath = self.fetch_data(ticker, start, end, interval)
        return pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    def get_available_data(self):
        """Get list of all available data files"""
        meta = self.load_metadata()
        return list(meta.keys())
