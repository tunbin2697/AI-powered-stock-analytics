import yfinance as yf
import pandas as pd
import os
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging
from config.settings import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataCache:
    def __init__(self, cache_dir="stock_cache"):
        self.cache_dir = cache_dir
        self.cache_duration = 24 * 60 * 60  # 24 hours in seconds
    
    def _get_cache_path(self, stock_code: str, data_type: str) -> str:
        return os.path.join(self.cache_dir, f"{stock_code}_{data_type}.json")
    
    def _is_cache_valid(self, cache_path: str) -> bool:
        if not os.path.exists(cache_path):
            return False
        
        file_age = time.time() - os.path.getmtime(cache_path)
        return file_age < self.cache_duration
    
    def get_cached_data(self, stock_code: str, data_type: str) -> Optional[Dict]:
        cache_path = self._get_cache_path(stock_code, data_type)
        
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"📂 CACHE HIT: {stock_code} {data_type}")
                return data
            except Exception as e:
                logger.error(f"❌ Cache read error: {e}")
                return None
        
        logger.info(f"⏰ CACHE MISS: {stock_code} {data_type} (expired or not found)")
        return None
    
    def save_to_cache(self, stock_code: str, data_type: str, data: Dict):
        cache_path = self._get_cache_path(stock_code, data_type)
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, default=str)
            logger.info(f"💾 CACHED: {stock_code} {data_type}")
        except Exception as e:
            logger.error(f"❌ Cache write error: {e}")

# Global cache instance
cache = StockDataCache(cache_dir=Config.STOCK_CACHE_DIR)

class StockDataService:
    
    @staticmethod
    def get_stock_price(symbol: str) -> Dict[str, Any]:
        """Get current stock price using yfinance with caching"""
        try:
            # Check cache first
            cached_data = cache.get_cached_data(symbol, "price")
            if cached_data:
                return cached_data
            
            # Fetch from yfinance
            logger.info(f"🌐 FETCHING FROM YFINANCE: {symbol} price")
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                return {"error": "Stock not found"}
            
            result = {
                "symbol": symbol.upper(),
                "currentPrice": info.get('currentPrice') or info.get('regularMarketPrice'),
                "previousClose": info.get('previousClose'),
                "open": info.get('open'),
                "dayHigh": info.get('dayHigh'),
                "dayLow": info.get('dayLow'),
                "currency": info.get('currency', 'USD')
            }
            
            # Cache the result
            cache.save_to_cache(symbol, "price", result)
            return result
            
        except Exception as e:
            logger.error(f"❌ Error fetching price for {symbol}: {e}")
            return {"error": f"Failed to get stock price: {str(e)}"}

    @staticmethod
    def get_historical_data_legacy(symbol: str, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Get historical data with date range using yfinance with caching"""
        try:
            # Create cache key based on parameters
            cache_key = f"historical_{start_date}_{end_date}" if start_date and end_date else "historical_1y"
            
            # Check cache first
            cached_data = cache.get_cached_data(symbol, cache_key)
            if cached_data:
                return cached_data
            
            # Fetch from yfinance
            logger.info(f"🌐 FETCHING FROM YFINANCE: {symbol} historical data")
            ticker = yf.Ticker(symbol)
            
            if start_date and end_date:
                hist_data = ticker.history(start=start_date, end=end_date)
            else:
                hist_data = ticker.history(period="1y")
            
            if hist_data.empty:
                return {"error": "Historical data not found"}
            
            # Convert to JSON format
            hist_data.reset_index(inplace=True)
            data = []
            for _, row in hist_data.iterrows():
                data.append({
                    "Date": row['Date'].strftime('%Y-%m-%d'),
                    "Open": round(float(row['Open']), 2),
                    "High": round(float(row['High']), 2),
                    "Low": round(float(row['Low']), 2),
                    "Close": round(float(row['Close']), 2),
                    "Volume": int(row['Volume']) if pd.notnull(row['Volume']) else 0,
                })
            
            result = {
                "symbol": symbol.upper(),
                "data": data,
                "total_records": len(data)
            }
            
            # Cache the result
            cache.save_to_cache(symbol, cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"❌ Error fetching historical data for {symbol}: {e}")
            return {"error": f"Failed to get historical data: {str(e)}"}

    @staticmethod
    def get_stock_list() -> Dict[str, Any]:
        """Get a sample list of popular stocks"""
        popular_stocks = [
            {"symbol": "AAPL", "name": "Apple Inc."},
            {"symbol": "GOOGL", "name": "Alphabet Inc."},
            {"symbol": "MSFT", "name": "Microsoft Corporation"},
            {"symbol": "AMZN", "name": "Amazon.com Inc."},
            {"symbol": "TSLA", "name": "Tesla Inc."},
            {"symbol": "META", "name": "Meta Platforms Inc."},
            {"symbol": "NVDA", "name": "NVIDIA Corporation"},
            {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
            {"symbol": "V", "name": "Visa Inc."},
            {"symbol": "FPT.VN", "name": "FPT Corporation"},
            {"symbol": "JNJ", "name": "Johnson & Johnson"}
        ]
        
        return {
            "stocks": popular_stocks,
            "total": len(popular_stocks),
            "note": "Sample of popular stocks - you can search for any valid stock symbol"
        }
    
    @staticmethod
    def validate_stock_code(stock_code: str) -> Dict[str, Any]:
        """Validate if a stock code exists with caching"""
        try:
            # Check cache first
            cached_data = cache.get_cached_data(stock_code, "validation")
            if cached_data:
                return cached_data
            
            # Fetch from yfinance
            logger.info(f"🌐 FETCHING FROM YFINANCE: {stock_code} validation")
            ticker = yf.Ticker(stock_code)
            hist = ticker.history(period="5d")
            
            if hist.empty:
                result = {
                    "valid": False,
                    "message": f"No data found for stock code: {stock_code}"
                }
            else:
                info = ticker.info
                result = {
                    "valid": True,
                    "message": "Stock code is valid",
                    "company_name": info.get('longName', 'N/A'),
                    "symbol": stock_code.upper()
                }
            
            # Cache the result
            cache.save_to_cache(stock_code, "validation", result)
            return result
            
        except Exception as e:
            logger.error(f"❌ Error validating {stock_code}: {e}")
            return {
                "valid": False,
                "message": f"Error validating stock: {str(e)}"
            }
    
    @staticmethod
    def get_stock_info(stock_code: str) -> Optional[Dict[str, Any]]:
        """Get basic stock information with caching"""
        try:
            # Check cache first
            cached_data = cache.get_cached_data(stock_code, "info")
            if cached_data:
                return cached_data
            
            # Fetch from yfinance
            logger.info(f"🌐 FETCHING FROM YFINANCE: {stock_code} info")
            ticker = yf.Ticker(stock_code)
            info = ticker.info
            
            if not info:
                return None
                
            result = {
                "symbol": info.get('symbol', stock_code.upper()),
                "longName": info.get('longName'),
                "currency": info.get('currency'),
                "exchange": info.get('exchange'),
                "marketCap": info.get('marketCap'),
                "previousClose": info.get('previousClose'),
                "open": info.get('open'),
                "dayHigh": info.get('dayHigh'),
                "dayLow": info.get('dayLow'),
                "fiftyTwoWeekHigh": info.get('fiftyTwoWeekHigh'),
                "fiftyTwoWeekLow": info.get('fiftyTwoWeekLow'),
            }
            
            # Cache the result
            cache.save_to_cache(stock_code, "info", result)
            return result
            
        except Exception as e:
            logger.error(f"❌ Error getting stock info for {stock_code}: {e}")
            return None
    
    @staticmethod
    def get_historical_data(stock_code: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Get historical stock data as DataFrame with caching"""
        try:
            # Check cache first
            cache_key = f"dataframe_{period}"
            cached_data = cache.get_cached_data(stock_code, cache_key)
            if cached_data:
                # Convert back to DataFrame
                return pd.DataFrame(cached_data['data'])
            
            # Fetch from yfinance
            logger.info(f"🌐 FETCHING FROM YFINANCE: {stock_code} dataframe ({period})")
            ticker = yf.Ticker(stock_code)
            hist_data = ticker.history(period=period)
            
            if hist_data.empty:
                return None
                
            # Reset index to make Date a column
            hist_data.reset_index(inplace=True)
            
            # Cache as JSON
            cache_data = {
                "data": hist_data.to_dict('records'),
                "columns": list(hist_data.columns)
            }
            cache.save_to_cache(stock_code, cache_key, cache_data)
            
            return hist_data
            
        except Exception as e:
            logger.error(f"❌ Error getting historical dataframe for {stock_code}: {e}")
            return None
    
    @staticmethod
    def get_historical_data_json(stock_code: str, period: str = "1y") -> List[Dict]:
        """Get historical data as JSON-serializable list"""
        try:
            df = StockDataService.get_historical_data(stock_code, period)
            if df is None:
                return []
                
            result = []
            for _, row in df.iterrows():
                result.append({
                    "Date": row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date']),
                    "Open": round(float(row['Open']), 2),
                    "High": round(float(row['High']), 2),
                    "Low": round(float(row['Low']), 2),
                    "Close": round(float(row['Close']), 2),
                    "Volume": int(row['Volume']) if pd.notnull(row['Volume']) else 0,
                })
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error converting to JSON for {stock_code}: {e}")
            return []
    
    @staticmethod
    def calculate_technical_indicators(df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Calculate technical indicators (SMA, RSI) - no caching for calculations"""
        try:
            result = {}
            
            # SMA50 and SMA200
            for window in [50, 200]:
                sma_values = df['Close'].rolling(window=window).mean()
                sma_data = []
                
                for idx, value in enumerate(sma_values):
                    if pd.notnull(value):
                        date_val = df.iloc[idx]['Date']
                        if hasattr(date_val, 'strftime'):
                            date_str = date_val.strftime('%Y-%m-%d')
                        else:
                            date_str = str(date_val)[:10]  # Take first 10 chars for date
                        
                        sma_data.append({
                            "Date": date_str,
                            "Value": round(float(value), 2)
                        })
                
                result[f"SMA{window}"] = sma_data
            
            # RSI
            def calculate_rsi(close_prices, window=14):
                delta = close_prices.diff()
                gain = delta.where(delta > 0, 0.0)
                loss = -delta.where(delta < 0, 0.0)
                
                avg_gain = gain.rolling(window=window, min_periods=1).mean()
                avg_loss = loss.rolling(window=window, min_periods=1).mean()
                
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                
                return rsi.fillna(50)
            
            rsi_values = calculate_rsi(df['Close'])
            rsi_data = []
            
            for idx, value in enumerate(rsi_values):
                if pd.notnull(value):
                    date_val = df.iloc[idx]['Date']
                    if hasattr(date_val, 'strftime'):
                        date_str = date_val.strftime('%Y-%m-%d')
                    else:
                        date_str = str(date_val)[:10]
                    
                    rsi_data.append({
                        "Date": date_str,
                        "Value": round(float(value), 2)
                    })
            
            result["RSI"] = rsi_data
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error calculating technical indicators: {e}")
            return {}

# Convenience functions for backward compatibility
def fetch_stock_data(symbol: str) -> Dict[str, Any]:
    """Fetch basic stock data"""
    service = StockDataService()
    
    # Validate stock
    validation = service.validate_stock_code(symbol)
    if not validation["valid"]:
        raise ValueError(validation["message"])
    
    # Get stock info and recent data
    stock_info = service.get_stock_info(symbol)
    historical_data = service.get_historical_data_json(symbol, "1mo")
    
    return {
        "symbol": symbol.upper(),
        "info": stock_info,
        "recent_data": historical_data[-5:] if historical_data else [],
        "total_records": len(historical_data)
    }

def fetch_historical_data(symbol: str, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
    """Fetch historical data with date range"""
    service = StockDataService()
    
    period = "1y"
    historical_data = service.get_historical_data_json(symbol, period)
    
    if not historical_data:
        raise ValueError(f"No historical data found for {symbol}")
    
    return {
        "symbol": symbol.upper(),
        "period": period,
        "data": historical_data,
        "total_records": len(historical_data)
    }