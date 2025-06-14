"""
Multi-modal data integration module for TGNN++ model training.
Combines processed price, macro, and news data for unified model input.
"""

import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
import joblib
import os
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiModalDataIntegrator:    
    """
    Integrates multi-modal data (price, macro, news) for TGNN++ model training.
    Handles alignment by date, missing value imputation, and feature engineering.
    """
    def __init__(self, processed_data_dir: str = "data/processed/daily"):
      self.processed_data_dir = processed_data_dir
      self.price_data = None
      self.macro_data = None
      self.news_embeddings = None
      self.combined_data = None
      self.scalers = {}
    def _safe_datetime_conversion(self, date_series, column_name="date"):
        """Safely convert a series to datetime, handling timezone issues."""
        try:
            # Check if data appears to be timezone-aware by looking at sample values
            sample_str = str(date_series.iloc[0]) if len(date_series) > 0 else ""
            
            if any(tz_indicator in sample_str for tz_indicator in ['-05:00', '-04:00', '+00:00', 'UTC']):
                # Handle timezone-aware data
                logger.info(f"Converting {column_name} from timezone-aware to timezone-naive")
                dt_series = pd.to_datetime(date_series, utc=True, errors='coerce')
                # Convert to timezone-naive and normalize to midnight
                dt_series = dt_series.dt.tz_convert('UTC').dt.tz_localize(None).dt.normalize()
            else:
                # Handle timezone-naive data
                dt_series = pd.to_datetime(date_series, errors='coerce')
            
            return dt_series
            
        except Exception as e:
            logger.warning(f"Error converting {column_name} to datetime: {str(e)}")
            # Fallback: try timezone-aware parsing
            try:
                dt_series = pd.to_datetime(date_series, utc=True, errors='coerce')
                if dt_series.dt.tz is not None:
                    dt_series = dt_series.dt.tz_convert('UTC').dt.tz_localize(None).dt.normalize()
                return dt_series
            except Exception as e2:
                logger.error(f"Failed to convert {column_name} to datetime: {str(e2)}, using original values")
                return date_series

    def load_processed_data(self, ticker: str = "AAPL", date_suffix: str = "2025-06-14") -> Dict[str, Any]:
        """Load all processed data files for a given ticker and date."""        # Load price data
        price_file = os.path.join(self.processed_data_dir, f"processed_{ticker}_{date_suffix}.csv")
        if os.path.exists(price_file):
            self.price_data = pd.read_csv(price_file)
            
            # Convert Date column to datetime safely
            self.price_data['Date'] = self._safe_datetime_conversion(self.price_data['Date'], "Date")
            self.price_data.set_index('Date', inplace=True)
            logger.info(f"Loaded price data: {self.price_data.shape}")
        else:
            logger.warning(f"Price data file not found: {price_file}")
            
        # Load macro data
        macro_file = os.path.join(self.processed_data_dir, f"processed_fred_macro_{date_suffix}.csv")
        if os.path.exists(macro_file):
            self.macro_data = pd.read_csv(macro_file)
            
            # Convert date column to datetime safely
            self.macro_data['date'] = self._safe_datetime_conversion(self.macro_data['date'], "date")
            self.macro_data.set_index('date', inplace=True)
            logger.info(f"Loaded macro data: {self.macro_data.shape}")
        else:
            logger.warning(f"Macro data file not found: {macro_file}")
            
        # Load news embeddings
        news_file = os.path.join(self.processed_data_dir, f"processed_news_emb_{ticker}_{date_suffix}.npy")
        if os.path.exists(news_file):
            self.news_embeddings = np.load(news_file)
            logger.info(f"Loaded news embeddings: {self.news_embeddings.shape}")
        else:
            logger.warning(f"News embeddings file not found: {news_file}")
            
        # Load scalers
        price_scaler_file = os.path.join(self.processed_data_dir, f"scaler_{ticker}_{date_suffix}.pkl")
        macro_scaler_file = os.path.join(self.processed_data_dir, f"scaler_fred_macro_{date_suffix}.pkl")
        
        if os.path.exists(price_scaler_file):
            self.scalers['price'] = joblib.load(price_scaler_file)
            
        if os.path.exists(macro_scaler_file):
            self.scalers['macro'] = joblib.load(macro_scaler_file)
        
        return {
            'price_shape': self.price_data.shape if self.price_data is not None else None,
            'macro_shape': self.macro_data.shape if self.macro_data is not None else None,
            'news_shape': self.news_embeddings.shape if self.news_embeddings is not None else None,
            'scalers_loaded': list(self.scalers.keys())
        }
    
    def align_data_by_date(self) -> pd.DataFrame:
        """Align all data sources by date, handling different frequencies."""
        
        if self.price_data is None:
            raise ValueError("Price data must be loaded first")
            
        # Start with price data (daily frequency)
        aligned_data = self.price_data.copy()        # Add macro data (monthly frequency) - forward fill to daily
        if self.macro_data is not None:
            try:
                # Ensure both indices are timezone-naive
                if hasattr(aligned_data.index, 'tz') and aligned_data.index.tz is not None:
                    aligned_data.index = aligned_data.index.tz_localize(None)
                if hasattr(self.macro_data.index, 'tz') and self.macro_data.index.tz is not None:
                    self.macro_data.index = self.macro_data.index.tz_localize(None)
                
                # Resample macro data to daily frequency with forward fill
                macro_daily = self.macro_data.resample('D').ffill()
                
                # Find date range that covers both datasets
                start_date = max(aligned_data.index.min(), macro_daily.index.min())
                end_date = min(aligned_data.index.max(), macro_daily.index.max())
                
                if start_date <= end_date:
                    # Filter both datasets to overlapping period
                    aligned_data = aligned_data.loc[start_date:end_date]
                    macro_aligned = macro_daily.loc[start_date:end_date]
                    
                    # Add macro features with prefix
                    for col in macro_aligned.columns:
                        aligned_data[f'macro_{col}'] = macro_aligned[col]
                        
                    logger.info(f"Added {len(macro_aligned.columns)} macro features for period {start_date} to {end_date}")
                else:
                    logger.warning("No overlapping dates between price and macro data")
            except Exception as e:
                logger.warning(f"Failed to align macro data: {str(e)}")
                # Continue without macro data
          # Add news embeddings with temporal variation
        if self.news_embeddings is not None:
            # Create temporal variation by interpolating between different news embeddings
            # and adding controlled temporal dynamics
            num_news_articles = self.news_embeddings.shape[0]
            num_dates = len(aligned_data)
            embedding_dim = self.news_embeddings.shape[1]
            
            if num_news_articles > 1:
                # Interpolate between different news embeddings over time
                time_indices = np.linspace(0, num_news_articles - 1, num_dates)
                
                for i in range(embedding_dim):
                    # Get embedding values across news articles for dimension i
                    emb_values = self.news_embeddings[:, i]
                    # Interpolate over time
                    interpolated_values = np.interp(time_indices, 
                                                  np.arange(num_news_articles), 
                                                  emb_values)
                    # Add small temporal variation to avoid exact constants
                    # Use a smooth trend based on date index
                    date_trend = np.sin(np.linspace(0, 4*np.pi, num_dates)) * 0.01 * np.std(emb_values)
                    final_values = interpolated_values + date_trend
                    
                    aligned_data[f'news_emb_{i}'] = final_values
            else:
                # Single news article - add temporal variation based on market dynamics
                news_mean = self.news_embeddings[0]  # Use the single news article
                
                for i in range(len(news_mean)):
                    base_value = news_mean[i]
                    # Create temporal variation based on price volatility or returns
                    if 'Returns' in aligned_data.columns:
                        # Scale news embedding influence by recent market activity
                        returns = aligned_data['Returns'].fillna(0)
                        variation = returns.rolling(window=5, min_periods=1).std() * base_value * 0.1
                        temporal_values = base_value + variation.fillna(0)
                    else:
                        # Fallback: add smooth temporal variation
                        time_trend = np.sin(np.linspace(0, 2*np.pi, num_dates)) * abs(base_value) * 0.05
                        temporal_values = base_value + time_trend
                    
                    aligned_data[f'news_emb_{i}'] = temporal_values
                
            logger.info(f"Added {embedding_dim} news embedding features with temporal variation")
            
        self.combined_data = aligned_data
        logger.info(f"Final aligned data shape: {aligned_data.shape}")
        
        return aligned_data
    
    def create_sequences(self, data: pd.DataFrame, sequence_length: int = 30, 
                        target_col: str = 'Close', feature_names: list = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create sequences for TGNN++ training from integrated data.
        
        Args:
            data: Integrated DataFrame
            sequence_length: Length of input sequences
            target_col: Column name for prediction target
            feature_names: List of feature column names (optional)
            
        Returns:
            Tuple of (features, targets) tensors
        """
        
        if data.empty:
            raise ValueError("Data is empty")
            
        # Ensure target column exists
        if target_col not in data.columns:
            # Try common variations
            target_variations = ['close_price', 'Close', 'close', 'CLOSE']
            target_col = None
            for var in target_variations:
                if var in data.columns:
                    target_col = var
                    break
            if target_col is None:
                raise ValueError(f"No suitable target column found. Available: {data.columns.tolist()}")
            
        # Prepare features (exclude target)
        if feature_names is None:
            feature_names = [col for col in data.columns if col != target_col]
        
        features = data[feature_names].values
        targets = data[target_col].values
        
        # Create sequences
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(features[i:i+sequence_length])
            y.append(targets[i+sequence_length])
            
        X = np.array(X)
        y = np.array(y)
        
        # Convert to tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        logger.info(f"Created sequences: {X.shape}, targets: {y.shape}")
        logger.info(f"Features: {len(feature_names)}, Sequence length: {sequence_length}")
        
        return X, y
    
    def get_feature_names(self) -> list:
        """Get list of feature names for XAI analysis."""
        if self.combined_data is None:
            return []
            
        # Exclude target column
        target_col = 'Close'
        feature_names = [col for col in self.combined_data.columns if col != target_col]
        
        return feature_names
    
    def save_integrated_data(self, output_path: str):
        """Save the integrated dataset."""
        if self.combined_data is not None:
            self.combined_data.to_csv(output_path)
            logger.info(f"✅ Saved integrated data to {output_path}")
        else:
            logger.warning("No integrated data to save")
    
    def get_latest_window(self, window_size: int = 30) -> torch.Tensor:
        """Get the latest window of data for prediction."""
        if self.combined_data is None:
            raise ValueError("No integrated data available")
            
        # Get latest window
        latest_data = self.combined_data.tail(window_size)
        
        # Exclude target column
        target_col = 'Close'
        feature_cols = [col for col in latest_data.columns if col != target_col]
        features = latest_data[feature_cols].values
        
        # Convert to tensor and add batch dimension
        tensor = torch.FloatTensor(features).unsqueeze(0)  # Shape: (1, seq_len, features)
        
        return tensor
    
    def integrate_all_data(self, ticker: str = "AAPL", start_date: str = None, end_date: str = None, 
                          date_suffix: str = "2025-06-14") -> pd.DataFrame:
        """
        Main method to integrate all data sources and return aligned DataFrame.
        
        Args:
            ticker: Stock ticker (e.g., 'AAPL')
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            date_suffix: Date suffix for processed files
            
        Returns:
            Integrated DataFrame with all features aligned by date
        """
        try:
            # Load all processed data
            self.load_processed_data(ticker, date_suffix)
            
            # Align data by date
            integrated_data = self.align_data_by_date()            # Filter by date range if provided
            if start_date and end_date:
                try:
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    
                    # Convert to timezone-naive if needed (for Timestamp objects)
                    if hasattr(start_dt, 'tz') and start_dt.tz is not None:
                        start_dt = start_dt.tz_convert('UTC').tz_localize(None)
                    if hasattr(end_dt, 'tz') and end_dt.tz is not None:
                        end_dt = end_dt.tz_convert('UTC').tz_localize(None)
                    
                    # Ensure the DataFrame index is timezone-naive
                    if hasattr(integrated_data.index, 'tz') and integrated_data.index.tz is not None:
                        logger.info("Converting DataFrame index from timezone-aware to timezone-naive")
                        integrated_data.index = integrated_data.index.tz_convert('UTC').tz_localize(None)
                    
                    # Filter data within date range
                    before_filter_shape = integrated_data.shape
                    integrated_data = integrated_data.loc[start_dt:end_dt]
                    logger.info(f"Filtered data from {before_filter_shape} to {integrated_data.shape} for date range {start_date} to {end_date}")
                except Exception as e:
                    logger.warning(f"Date filtering failed: {str(e)}, using all available data")
            
            # Store the combined data
            self.combined_data = integrated_data
            
            logger.info(f"Successfully integrated data for {ticker}: {integrated_data.shape}")
            return integrated_data
            
        except Exception as e:
            logger.error(f"Failed to integrate data for {ticker}: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def prepare_model_data(self, integrated_data: pd.DataFrame, sequence_length: int = 30, 
                          target_col: str = 'Close') -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        Prepare integrated data for TGNN++ model training.
        
        Args:
            integrated_data: Integrated DataFrame from integrate_all_data
            sequence_length: Length of input sequences
            target_col: Target column name for prediction
            
        Returns:
            Tuple of (features, targets, feature_names)
        """
        try:
            if integrated_data.empty:
                raise ValueError("Integrated data is empty")
            
            # Get feature names (excluding target)
            feature_names = [col for col in integrated_data.columns if col != target_col]
            
            # Create sequences
            features, targets = self.create_sequences(
                integrated_data, sequence_length, target_col, feature_names
            )
            
            return features, targets, feature_names
            
        except Exception as e:
            logger.error(f"Failed to prepare model data: {str(e)}")
            return torch.empty(0), torch.empty(0), []

    # Example usage and testing function
def test_integration():
  """Test the data integration functionality."""
  
  integrator = MultiModalDataIntegrator()
  
  # Load data
  data_info = integrator.load_processed_data()
  print("Data loading info:", data_info)
  
  # Align data
  if data_info['price_shape'] is not None:
      aligned_data = integrator.align_data_by_date()
      print(f"Aligned data shape: {aligned_data.shape}")
      print(f"Columns: {list(aligned_data.columns)}")
      
      # Create sequences
      X_train, X_test, y_train, y_test = integrator.create_sequences()
      print(f"Training sequences: {X_train.shape}")
      print(f"Test sequences: {X_test.shape}")
      
      # Get feature names
      feature_names = integrator.get_feature_names()
      print(f"Feature names ({len(feature_names)}): {feature_names[:10]}...")  # Show first 10
      
      # Save integrated data
      integrator.save_integrated_data("data/processed/integrated_data.csv")
      
      return integrator
  else:
      print("No price data available for testing")
      return None

if __name__ == "__main__":
    test_integration()
