import pandas as pd
import numpy as np
from sklearn.preprocessing import  MinMaxScaler
from sklearn.impute import KNNImputer
import os
from datetime import datetime
import json
import joblib
from service.crawl.fred import FredCrawl

class FredProcessor:
    
    def __init__(self, raw_data_path="data/raw/fred", processed_data_path="data/processed/daily"):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.fred_crawler = FredCrawl(base_path=raw_data_path)
        os.makedirs(processed_data_path, exist_ok=True)
        
        # Define essential economic indicators for TGNN++/XAI
        self.indicators = {
            'CPIAUCSL': 'CPI_All_Urban',           # Inflation
            'FEDFUNDS': 'Fed_Funds_Rate',          # Monetary Policy
            'GS10': 'Treasury_10Y',                # Interest Rate Environment
            'UNRATE': 'Unemployment_Rate',         # Labor Market
            'GDP': 'GDP_Growth'                    # Economic Growth
        }
    
    def load_individual_indicator(self, indicator_key, start="2020-01-01", end="2025-01-01"):
        """Load individual FRED indicator using the FRED crawler"""
        try:
            # Use the FRED crawler's load_data method
            df = self.fred_crawler.load_data(series_id=indicator_key, start=start, end=end)
            
            df.columns = [self.indicators.get(indicator_key, indicator_key)]
            
            df = df.sort_index()
            
            print(f"✅ Loaded {indicator_key}: {len(df)} records from {df.index.min()} to {df.index.max()}")
            return df
            
        except Exception as e:
            print(f"❌ Failed to load {indicator_key}: {str(e)}")
            return None
    
    def merge_fred_indicators(self, start="2020-01-01", end="2025-01-01"):
        print("🔄 Merging FRED economic indicators...")
        
        merged_df = None
        successful_indicators = []
        
        for fred_code, column_name in self.indicators.items():
            df = self.load_individual_indicator(fred_code, start, end)
            
            if df is not None:
                if merged_df is None:
                    merged_df = df.copy()
                else:
                    # Outer join to preserve all dates
                    merged_df = merged_df.join(df, how='outer')
                
                successful_indicators.append(column_name)
        
        if merged_df is not None:
            print(f"✅ Successfully merged {len(successful_indicators)} indicators")
            print(f"📊 Final dataset: {len(merged_df)} records, {len(merged_df.columns)} features")
            return merged_df
        else:
            raise ValueError("❌ No FRED indicators could be loaded")
    
    def engineer_macro_features(self, df):
        df_features = df.copy()
        
        # Rate of change features (critical for economic trend analysis)
        for col in df.columns:
            df_features[f'{col}_Change'] = df[col].pct_change()
            df_features[f'{col}_MA3'] = df[col].rolling(window=3).mean()  # 3-month moving average
            
        # Economic spreads and ratios (important for financial markets)
        if 'Treasury_10Y' in df.columns and 'Fed_Funds_Rate' in df.columns:
            df_features['Yield_Spread'] = df['Treasury_10Y'] - df['Fed_Funds_Rate']
            
        # Inflation-adjusted rates
        if 'CPI_All_Urban' in df.columns and 'Fed_Funds_Rate' in df.columns:
            inflation_rate = df['CPI_All_Urban'].pct_change() * 100
            df_features['Real_Fed_Rate'] = df['Fed_Funds_Rate'] - inflation_rate
            
        # Economic momentum indicators
        if 'GDP_Growth' in df.columns:
            df_features['GDP_Momentum'] = df['GDP_Growth'].diff()
              # Labor market strength
        if 'Unemployment_Rate' in df.columns:
            df_features['Unemployment_Change'] = df['Unemployment_Rate'].diff()
            
        print(f"🔧 Engineered {len(df_features.columns)} macro features")
        return df_features
    
    def handle_missing_values(self, df):
        print("🔄 Handling missing values in macro data...")
        
        # Forward fill first (common for economic data released with lags)
        df_filled = df.fillna(method='ffill')
        
        # Then KNN imputation for remaining gaps
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0 and df_filled[numeric_cols].isnull().sum().sum() > 0:
            try:
                imputer = KNNImputer(n_neighbors=5) 
                # Properly handle the imputation
                imputed_array = imputer.fit_transform(df_filled[numeric_cols])
                
                # Create a new dataframe with imputed values
                df_result = df_filled.copy()
                for i, col in enumerate(numeric_cols):
                    df_result[col] = imputed_array[:, i]
                df_filled = df_result
                
            except Exception as e:
                print(f"⚠️ KNN imputation failed: {str(e)}, using backward fill instead")
                df_filled = df_filled.fillna(method='bfill').fillna(0)
        
        missing_after = df_filled.isnull().sum().sum()
        print(f"✅ Missing values after imputation: {missing_after}")
        
        return df_filled
    
    def normalize_features(self, df):
        """Normalize macro features using Min-Max normalization as per lab requirements"""
        df_normalized = df.copy()
        
        # Normalize all numeric columns using Min-Max scaler (FRED requirement)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        scaler = MinMaxScaler()
        df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        print(f"✅ Applied Min-Max normalization to {len(numeric_cols)} macro features")
        return df_normalized, scaler
    
    def process_raw_data(self, start="2020-01-01", end="2025-01-01", save_output=True):
        print(f"🏛️ Processing FRED data for Step 7 (DAVOTS/ICFTS/TGNN++)...")
        
        # Merge all FRED indicators using the crawler
        df_merged = self.merge_fred_indicators(start, end)
        
        # Engineer macro features
        df_features = self.engineer_macro_features(df_merged)
        
        # Handle missing values
        df_clean = self.handle_missing_values(df_features)
        
        # Normalize for model training
        df_normalized, scaler = self.normalize_features(df_clean)
        
        # Save processed data
        if save_output:
            self.save_processed_data(df_normalized, scaler, start, end)
        
        return df_normalized, scaler
    
    def save_processed_data(self, df_normalized, scaler, start, end):
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Save main processed macro dataset
        main_file = f"processed_fred_macro_{today}.csv"
        df_normalized.to_csv(os.path.join(self.processed_data_path, main_file))
        
        # Save scaler for later denormalization using joblib
        scaler_file = f"scaler_fred_macro_{today}.pkl"
        scaler_path = os.path.join(self.processed_data_path, scaler_file)
        joblib.dump(scaler, scaler_path)
        
        print(f"✅ Saved FRED processed data: {main_file}")
        print(f"✅ Saved FRED scaler (joblib): {scaler_file}")
        print(f"📂 Location: {self.processed_data_path}")
        
        return True
    
    def load_saved_scaler(self, date_str=None):
        """Load saved scaler for denormalization"""
        if date_str is None:
            # Find the most recent scaler file
            import glob
            pattern = os.path.join(self.processed_data_path, "scaler_fred_macro_*.pkl")
            scaler_files = glob.glob(pattern)
            if not scaler_files:
                raise FileNotFoundError("No FRED scaler files found")
            scaler_path = max(scaler_files, key=os.path.getctime)
        else:
            scaler_file = f"scaler_fred_macro_{date_str}.pkl"
            scaler_path = os.path.join(self.processed_data_path, scaler_file)
        
        try:
            scaler = joblib.load(scaler_path)
            print(f"✅ Loaded FRED scaler from {os.path.basename(scaler_path)}")
            return scaler
        except Exception as e:
            raise Exception(f"Failed to load FRED scaler: {str(e)}")