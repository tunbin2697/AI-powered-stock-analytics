import yfinance as yf
import pandas as pd
import os
import json
from datetime import datetime
import requests
from dotenv import load_dotenv

load_dotenv()

class FredCrawl:
    def __init__(self, base_path="data/raw/fred"):
        self.base_path = base_path
        self.apiKey = os.getenv("FRED_API_KEY")
        self.url = "https://api.stlouisfed.org/fred/series/observations"
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
            
    def make_key(self, series_id, start, end):
        return f"{series_id}_{start}_{end}"
    
    def data_exists(self, meta, key):
        return key in meta and os.path.exists(meta[key]["filepath"])
    
    def fetch_data(self, series_id, start, end):
        meta = self.load_metadata()
        key = self.make_key(series_id, start, end)

        if self.data_exists(meta, key):
            print(f"✅ Fred raw data exists: {meta[key]['filepath']}")
            return meta[key]["filepath"]
        else:
            print(f"⬇️ Downloading Fred raw data for {series_id} from {start} to {end}...")
            data_info = self.download_csv(series_id, start, end)
            meta[key] = data_info
            self.save_metadata(meta)
            return data_info["filepath"]
    
    def download_csv(self, series_id, start, end):

        params = {
            "series_id": series_id,
            "api_key": self.apiKey,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end
        }
        response = requests.get(self.url, params=params)
            
        observations = response.json()["observations"]
        

        df = pd.DataFrame(observations)
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.set_index('date')[['value']]  
                
        filename = f"{series_id}_{start}_{end}.csv"
        filepath = os.path.join(self.base_path, filename)
        df.to_csv(filepath)
    
        return {
            "filepath": filepath,
            "records_count": len(df),
            "actual_start": df.index.min().strftime('%Y-%m-%d') if not df.empty else None,
            "actual_end": df.index.max().strftime('%Y-%m-%d') if not df.empty else None,
            "downloaded_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def load_data(self, series_id, start, end):
        """Load existing data from CSV file"""
        filepath = self.fetch_data(series_id, start, end)
        return pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    def get_available_data(self):
        """Get list of all available data files"""
        meta = self.load_metadata()
        return list(meta.keys())
