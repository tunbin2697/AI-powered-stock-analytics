import yfinance as yf
import pandas as pd
import os
import json
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv

load_dotenv()

class NewsapiCrawl:
    def __init__(self, base_path="data/raw/newsapi"):
        self.base_path = base_path
        self.apiKey = os.getenv("NEWS_API_KEY")
        self.url = "https://newsapi.org/v2/everything"
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
            
    def make_key(self, ticker, start, end):
        return f"{ticker}_{start}_{end}"
    
    def data_exists(self, meta, key):
        return key in meta and os.path.exists(meta[key]["filepath"])
    
    def fetch_data(self, ticker, start, end):
        meta = self.load_metadata()
        key = self.make_key(ticker, start, end)

        if self.data_exists(meta, key):
            print(f"✅ Newsapi raw data exists: {meta[key]['filepath']}")
            return meta[key]["filepath"]        
        else:
            print(f"⬇️ Downloading Newsapi raw data for {ticker} from {start} to {end}...")
            data_info = self.download_json(ticker, start, end)
            if data_info is None:
                return None
            meta[key] = data_info
            self.save_metadata(meta)
            return data_info["filepath"]
    def download_json(self, ticker, start, end):
      
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        one_month_ago = today - timedelta(days=30)
        
        actual_start = one_month_ago.strftime('%Y-%m-%d')
        actual_end = yesterday.strftime('%Y-%m-%d')
        
        print(f"Using date range: {actual_start} to {actual_end}")

        params = {
            "q": ticker,
            "apiKey": self.apiKey,
            "from": actual_start,
            "to": actual_end,
            "pageSize": 100,
            "language": "en",
            "sortBy": "publishedAt",
        }
        response = requests.get(self.url, params=params)
        response_data = response.json()
        
        articles = response_data["articles"]
        if not articles:
            print(f"❌ No articles found for {ticker} in date range {actual_start} to {actual_end}")
            return None
                
        filename = f"{ticker}_{actual_start}_{actual_end}.json"
        filepath = os.path.join(self.base_path, filename)
        
        with open(filepath, 'w') as f:
            json.dump(articles, f, indent=2, default=str)
    
        return {
            "filepath": filepath,
            "records_count": len(articles),
            "actual_start": actual_start,
            "actual_end": actual_end,
            "downloaded_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def load_data(self, ticker, start, end):
        """Load existing data from JSON file"""
        filepath = self.fetch_data(ticker, start, end)
        if filepath is None:
            raise Exception(f"Failed to fetch news data for ticker: {ticker}")
        
        with open(filepath, 'r') as f:
            articles = json.load(f)
        
        # Return the raw articles array
        return articles
    
    def get_available_data(self):
        """Get list of all available data files"""
        meta = self.load_metadata()
        return list(meta.keys())
