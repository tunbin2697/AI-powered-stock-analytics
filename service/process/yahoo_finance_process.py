import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import os
from datetime import datetime
import ta
import joblib
from service.crawl.yahoo_finance import YahooFinanceCrawl

class YahooFinanceProcessor:
    
    def __init__(self, processed_data_path="data/processed/daily"):
        self.processed_data_path = processed_data_path
        self.yahoo_crawler = YahooFinanceCrawl()
        os.makedirs(processed_data_path, exist_ok=True)
    
    def load_raw_data(self, ticker, start="2020-01-01", end="2025-01-01", interval="1d"):
        """Load raw Yahoo Finance data using the crawler's load_data function"""
        try:
            df = self.yahoo_crawler.load_data(ticker=ticker, start=start, end=end, interval=interval)
            
            df = df.sort_index()
            
            print(f"✅ Loaded data with datetime index: {df.index.min()} to {df.index.max()}")
            return df
        except Exception as e:
            raise FileNotFoundError(f"Failed to load data for ticker {ticker}: {str(e)}")
    
    def calculate_essential_features(self, df):
        
        df = df.copy()
        
        # Essential price features for XAI and model
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Core moving averages (essential for trend analysis)
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        
        # MACD (critical for momentum analysis in TGNN++)
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # RSI (essential momentum indicator for XAI attribution)
        if len(df) >= 14:
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # Bollinger Bands (volatility and price position)
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume analysis (essential for multi-modal TGNN++)
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price position (for DAVOTS feature attribution)
        df['High_Low_Ratio'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        return df
    
    def handle_missing_values(self, df):
        # Separate numeric columns for imputation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # KNN Imputation with k=5 as specified in lab
        imputer = KNNImputer(n_neighbors=5)
        df_imputed = df.copy()
        df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        return df_imputed
    
    def normalize_features(self, df):
        """Apply z-score normalization to Price & Volume as per lab requirements"""
        df_normalized = df.copy()
        
        # Define price and volume columns for z-score normalization
        price_volume_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        
        # Z-score normalization (StandardScaler)
        scaler = StandardScaler()
        
        # Only normalize existing columns
        cols_to_normalize = [col for col in price_volume_cols if col in df.columns]
        
        if cols_to_normalize:
            df_normalized[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
           
        return df_normalized, scaler
    def process_raw_data(self, ticker, start="2020-01-01", end="2025-01-01", interval="1d", save_output=True):
        
        print(f"Processing {ticker} for Step 7 - Creating comprehensive dataset...")
        
        # Load raw data with proper datetime index
        df = self.load_raw_data(ticker, start=start, end=end, interval=interval)
        print(f"Loaded {len(df)} records for {ticker}")
        
        # Calculate all essential features
        df = self.calculate_essential_features(df)
        print(f"Calculated essential features for XAI and TGNN++")
        
        # Handle missing values with KNN imputation
        df = self.handle_missing_values(df)
        print(f"Applied KNN imputation (k=5)")
        
        # Normalize features for model training
        df_normalized, scaler = self.normalize_features(df)
        print(f"Applied z-score normalization")
          # Save the comprehensive dataset
        if save_output:
            self.save_processed_data(ticker, df_normalized, scaler)
        
        return df_normalized, scaler
    
    def save_processed_data(self, ticker, df_normalized, scaler):
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Save main processed dataset with all features
        main_file = f"processed_{ticker}_{today}.csv"
        df_normalized.to_csv(os.path.join(self.processed_data_path, main_file))
        
        # Save scaler using joblib for later reverse transformation (denormalization)
        scaler_file = f"scaler_{ticker}_{today}.pkl"
        scaler_path = os.path.join(self.processed_data_path, scaler_file)
        joblib.dump(scaler, scaler_path)
        
        print(f"✅ Saved processed data to {main_file}")
        print(f"✅ Saved scaler (joblib) to {scaler_file}")
        print(f"📂 Location: {self.processed_data_path}")
        
        return df_normalized
    
    def load_scaler(self, ticker, date_str=None):
        if date_str is None:
            # Find the most recent scaler file for this ticker
            import glob
            pattern = os.path.join(self.processed_data_path, f"scaler_{ticker}_*.pkl")
            scaler_files = glob.glob(pattern)
            if not scaler_files:
                raise FileNotFoundError(f"No scaler files found for ticker {ticker}")
            # Get the most recent file
            scaler_path = max(scaler_files, key=os.path.getctime)
        else:
            scaler_file = f"scaler_{ticker}_{date_str}.pkl"
            scaler_path = os.path.join(self.processed_data_path, scaler_file)
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        try:
            scaler = joblib.load(scaler_path)
            print(f"✅ Loaded scaler from: {os.path.basename(scaler_path)}")
            return scaler
        except Exception as e:
            raise Exception(f"Failed to load scaler: {str(e)}")
    
    def reverse_normalize(self, df, scaler, columns_to_reverse=None):
        df_denorm = df.copy()
        
        if columns_to_reverse is None:
            # Default to price and volume columns that were normalized
            columns_to_reverse = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        
        # Only process columns that exist in the DataFrame
        cols_to_process = [col for col in columns_to_reverse if col in df.columns]
        
        if cols_to_process:
            df_denorm[cols_to_process] = scaler.inverse_transform(df[cols_to_process])
            print(f"✅ Denormalized columns: {cols_to_process}")
        
        return df_denorm
 