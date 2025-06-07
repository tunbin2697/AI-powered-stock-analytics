# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import joblib
# import os
# import pickle
# from typing import Dict, List, Any, Tuple, Optional
# from .stock_data_service import StockDataService
# from config.settings import Config

# # New model imports
# from prophet import Prophet
# from statsmodels.tsa.arima.model import ARIMA
# from keras.models import Sequential, load_model
# from keras.layers import LSTM, Dense, Dropout
# from keras.callbacks import EarlyStopping
# import tensorflow as tf

# class MLService:
#     def __init__(self, models_dir: str = 'models', scalers_dir: str = 'scalers'):
#         self.models_dir = models_dir
#         self.scalers_dir = scalers_dir
        
#         self.supported_models = {
#             'linear_regression': LinearRegression,
#             'random_forest_regressor': RandomForestRegressor,
#             'svm_regressor': SVR,
#             'prophet': Prophet,
#             'arima': ARIMA,
#             'lstm': Sequential
#         }
    
#     # Trong ml_service.py
#     def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Add comprehensive technical indicators"""
#         df_copy = df.copy()
        
#         # Price-based features
#         df_copy['Price_Change'] = df_copy['Close'].pct_change()
#         df_copy['High_Low_Ratio'] = df_copy['High'] / df_copy['Low']
#         df_copy['Close_Open_Ratio'] = df_copy['Close'] / df_copy['Open']
        
#         # Moving averages (multiple periods)
#         for period in [5, 10, 20, 50, 200]:
#             df_copy[f'SMA{period}'] = df_copy['Close'].rolling(window=period).mean()
#             df_copy[f'Close_SMA{period}_Ratio'] = df_copy['Close'] / df_copy[f'SMA{period}']
        
#         # Exponential moving averages
#         for period in [12, 26]:
#             df_copy[f'EMA{period}'] = df_copy['Close'].ewm(span=period).mean()
        
#         # MACD
#         df_copy['MACD'] = df_copy['EMA12'] - df_copy['EMA26']
#         df_copy['MACD_Signal'] = df_copy['MACD'].ewm(span=9).mean()
#         df_copy['MACD_Histogram'] = df_copy['MACD'] - df_copy['MACD_Signal']
        
#         # RSI (multiple periods)
#         for period in [14, 21]:
#             df_copy[f'RSI{period}'] = self._calculate_rsi(df_copy['Close'], period)
        
#         # Bollinger Bands
#         df_copy['BB_Middle'] = df_copy['Close'].rolling(window=20).mean()
#         bb_std = df_copy['Close'].rolling(window=20).std()
#         df_copy['BB_Upper'] = df_copy['BB_Middle'] + (bb_std * 2)
#         df_copy['BB_Lower'] = df_copy['BB_Middle'] - (bb_std * 2)
#         df_copy['BB_Position'] = (df_copy['Close'] - df_copy['BB_Lower']) / (df_copy['BB_Upper'] - df_copy['BB_Lower'])
        
#         # Volume indicators
#         df_copy['Volume_SMA20'] = df_copy['Volume'].rolling(window=20).mean()
#         df_copy['Volume_Ratio'] = df_copy['Volume'] / df_copy['Volume_SMA20']
#         df_copy['Price_Volume'] = df_copy['Close'] * df_copy['Volume']
        
#         # Lag features
#         for lag in [1, 2, 3, 5]:
#             df_copy[f'Close_Lag_{lag}'] = df_copy['Close'].shift(lag)
#             df_copy[f'Volume_Lag_{lag}'] = df_copy['Volume'].shift(lag)
        
#         # Rolling statistics
#         for window in [5, 10, 20]:
#             df_copy[f'Close_Rolling_Mean_{window}'] = df_copy['Close'].rolling(window=window).mean()
#             df_copy[f'Close_Rolling_Std_{window}'] = df_copy['Close'].rolling(window=window).std()
#             df_copy[f'Close_Rolling_Min_{window}'] = df_copy['Close'].rolling(window=window).min()
#             df_copy[f'Close_Rolling_Max_{window}'] = df_copy['Close'].rolling(window=window).max()
        
#         # Time-based features
#         df_copy['days_since_start'] = range(len(df_copy))
#         df_copy['month'] = pd.to_datetime(df_copy.index).month if hasattr(df_copy.index, 'month') else (pd.to_datetime('2006-12-13') + pd.to_timedelta(df_copy.index, unit='D')).month
#         df_copy['quarter'] = df_copy['month'].apply(lambda x: (x-1)//3 + 1)
        
#         # === FIX LỖI RÒ RỈ DỮ LIỆU ===
#         # Thay thế median() bằng ffill() và bfill() để tránh data leakage
#         # ffill: điền giá trị NaN bằng giá trị hợp lệ gần nhất phía trước
#         # bfill: điền các giá trị NaN còn lại ở đầu dataframe
#         df_copy = df_copy.replace([np.inf, -np.inf], np.nan)
#         df_copy.ffill(inplace=True)
#         df_copy.bfill(inplace=True)
        
#         return df_copy
    
#     def _calculate_rsi(self, data, window=14):
#         """Calculate RSI indicator"""
#         delta = data.diff()
#         gain = delta.where(delta > 0, 0)
#         loss = -delta.where(delta < 0, 0)
#         avg_gain = gain.rolling(window=window, min_periods=1).mean()
#         avg_loss = loss.rolling(window=window, min_periods=1).mean()
#         rs = avg_gain / avg_loss
#         rsi = 100 - (100 / (1 + rs))
#         return rsi

#     # ==== MODEL-SPECIFIC DATA PROCESSING FUNCTIONS ====
    
#     def process_linear_regression_data(self, df: pd.DataFrame, stock_code: str, fit_scalers: bool = True) -> Tuple[pd.DataFrame, Dict]:
#         """Process data optimized for Linear Regression - focus on linear relationships"""
#         try:
#             df_processed = self._add_technical_features(df)
            
#             # Select features that work well with linear regression
#             feature_cols = [
#                 'days_since_start', 'Open', 'High', 'Low', 'Volume',
#                 'SMA5', 'SMA10', 'SMA20', 'SMA50', 
#                 'Close_SMA20_Ratio', 'Close_SMA50_Ratio',
#                 'Price_Change', 'High_Low_Ratio', 'Close_Open_Ratio',
#                 'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3',
#                 'Close_Rolling_Mean_5', 'Close_Rolling_Mean_10',
#                 'Volume_Ratio', 'month', 'quarter'
#             ]
            
#             # Fill NaN values
#             for col in feature_cols:
#                 if col in df_processed.columns:
#                     df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            
#             # Use StandardScaler for linear regression
#             scalers = {}
#             if fit_scalers:
#                 scaler_close = StandardScaler()
#                 scaler_features = StandardScaler()
                
#                 df_processed['Close'] = scaler_close.fit_transform(df_processed[['Close']]).flatten()
#                 df_processed[feature_cols] = scaler_features.fit_transform(df_processed[feature_cols])
                
#                 joblib.dump(scaler_close, os.path.join(self.models_dir, f"{stock_code}_linear_regression_scaler_close.pkl"))
#                 joblib.dump(scaler_features, os.path.join(self.models_dir, f"{stock_code}_linear_regression_scaler_features.pkl"))
                
#                 scalers = {'Close': scaler_close, 'features': scaler_features}
#             else:
#                 scaler_close = joblib.load(os.path.join(self.models_dir, f"{stock_code}_linear_regression_scaler_close.pkl"))
#                 scaler_features = joblib.load(os.path.join(self.models_dir, f"{stock_code}_linear_regression_scaler_features.pkl"))
                
#                 df_processed['Close'] = scaler_close.transform(df_processed[['Close']]).flatten()
#                 df_processed[feature_cols] = scaler_features.transform(df_processed[feature_cols])
                
#                 scalers = {'Close': scaler_close, 'features': scaler_features}
            
#             final_cols = feature_cols + ['Close']
#             return df_processed[final_cols].dropna(), scalers
            
#         except Exception as e:
#             print(f"Error in linear regression preprocessing: {e}")
#             return None, None
    
#     def process_random_forest_data(self, df: pd.DataFrame, stock_code: str, fit_scalers: bool = True) -> Tuple[pd.DataFrame, Dict]:
#         """Process data optimized for Random Forest - use raw features, no scaling needed"""
#         try:
#             df_processed = self._add_technical_features(df)
            
#             # Random Forest can handle more features and doesn't need scaling
#             feature_cols = [
#                 'days_since_start', 'Open', 'High', 'Low', 'Volume',
#                 'SMA5', 'SMA10', 'SMA20', 'SMA50', 'SMA200',
#                 'EMA12', 'EMA26', 'MACD', 'MACD_Signal', 'MACD_Histogram',
#                 'RSI14', 'RSI21', 'BB_Position',
#                 'Close_SMA5_Ratio', 'Close_SMA20_Ratio', 'Close_SMA50_Ratio',
#                 'Price_Change', 'High_Low_Ratio', 'Close_Open_Ratio',
#                 'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5',
#                 'Volume_Lag_1', 'Volume_Lag_2',
#                 'Close_Rolling_Mean_5', 'Close_Rolling_Std_5',
#                 'Close_Rolling_Mean_10', 'Close_Rolling_Std_10',
#                 'Close_Rolling_Mean_20', 'Close_Rolling_Std_20',
#                 'Close_Rolling_Min_5', 'Close_Rolling_Max_5',
#                 'Volume_Ratio', 'Price_Volume',
#                 'month', 'quarter'
#             ]
            
#             # Fill NaN values
#             for col in feature_cols:
#                 if col in df_processed.columns:
#                     df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            
#             # Random Forest doesn't need scaling, but we keep target scaling for consistency
#             scalers = {}
#             if fit_scalers:
#                 scaler_close = StandardScaler()
#                 df_processed['Close'] = scaler_close.fit_transform(df_processed[['Close']]).flatten()
#                 joblib.dump(scaler_close, os.path.join(self.models_dir, f"{stock_code}_random_forest_scaler_close.pkl"))
#                 scalers = {'Close': scaler_close}
#             else:
#                 scaler_close = joblib.load(os.path.join(self.models_dir, f"{stock_code}_random_forest_scaler_close.pkl"))
#                 df_processed['Close'] = scaler_close.transform(df_processed[['Close']]).flatten()
#                 scalers = {'Close': scaler_close}
            
#             final_cols = feature_cols + ['Close']
#             return df_processed[final_cols].dropna(), scalers
            
#         except Exception as e:
#             print(f"Error in random forest preprocessing: {e}")
#             return None, None
    
#     def process_svm_data(self, df: pd.DataFrame, stock_code: str, fit_scalers: bool = True) -> Tuple[pd.DataFrame, Dict]:
#         """Process data optimized for SVM - robust scaling and feature selection"""
#         try:
#             df_processed = self._add_technical_features(df)
            
#             # SVM works best with fewer, well-scaled features
#             feature_cols = [
#                 'Open', 'High', 'Low', 'Volume',
#                 'SMA20', 'SMA50', 'RSI14', 'MACD',
#                 'Close_SMA20_Ratio', 'BB_Position',
#                 'Price_Change', 'High_Low_Ratio',
#                 'Close_Lag_1', 'Close_Lag_2',
#                 'Close_Rolling_Mean_10', 'Close_Rolling_Std_10',
#                 'Volume_Ratio'
#             ]
            
#             # Fill NaN values
#             for col in feature_cols:
#                 if col in df_processed.columns:
#                     df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            
#             # Use RobustScaler for SVM (better with outliers)
#             scalers = {}
#             if fit_scalers:
#                 scaler_close = RobustScaler()
#                 scaler_features = RobustScaler()
                
#                 df_processed['Close'] = scaler_close.fit_transform(df_processed[['Close']]).flatten()
#                 df_processed[feature_cols] = scaler_features.fit_transform(df_processed[feature_cols])
                
#                 joblib.dump(scaler_close, os.path.join(self.models_dir, f"{stock_code}_svm_scaler_close.pkl"))
#                 joblib.dump(scaler_features, os.path.join(self.models_dir, f"{stock_code}_svm_scaler_features.pkl"))
                
#                 scalers = {'Close': scaler_close, 'features': scaler_features}
#             else:
#                 scaler_close = joblib.load(os.path.join(self.models_dir, f"{stock_code}_svm_scaler_close.pkl"))
#                 scaler_features = joblib.load(os.path.join(self.models_dir, f"{stock_code}_svm_scaler_features.pkl"))
                
#                 df_processed['Close'] = scaler_close.transform(df_processed[['Close']]).flatten()
#                 df_processed[feature_cols] = scaler_features.transform(df_processed[feature_cols])
                
#                 scalers = {'Close': scaler_close, 'features': scaler_features}
            
#             final_cols = feature_cols + ['Close']
#             return df_processed[final_cols].dropna(), scalers
            
#         except Exception as e:
#             print(f"Error in SVM preprocessing: {e}")
#             return None, None
    
#     def process_prophet_data(self, df: pd.DataFrame, stock_code: str, fit_scalers: bool = True) -> Tuple[pd.DataFrame, Dict]:
#         """Process data for Prophet - simple date-value format with additional regressors"""
#         try:
#             # Prophet expects specific column names: 'ds' (date) and 'y' (value)
#             start_date = pd.to_datetime('2006-12-13')
#             df_prophet = df.copy()
#             df_prophet['ds'] = start_date + pd.to_timedelta(df_prophet.index, unit='D')
#             df_prophet['y'] = df_prophet['Close']
            
#             # Add technical indicators as additional regressors for better R²
#             df_enhanced = self._add_technical_features(df)
            
#             # Select key indicators as regressors
#             regressor_cols = ['SMA20', 'RSI14', 'Volume_Ratio', 'MACD', 'BB_Position']
#             for col in regressor_cols:
#                 if col in df_enhanced.columns:
#                     df_prophet[col] = df_enhanced[col].fillna(df_enhanced[col].median())
            
#             # Prophet doesn't need target scaling but we store scalers for consistency
#             scalers = {}
#             if fit_scalers:
#                 # Save regressor scalers for prediction consistency
#                 for col in regressor_cols:
#                     if col in df_prophet.columns:
#                         scaler = StandardScaler()
#                         df_prophet[col] = scaler.fit_transform(df_prophet[[col]]).flatten()
#                         joblib.dump(scaler, os.path.join(self.models_dir, f"{stock_code}_prophet_scaler_{col}.pkl"))
#                         scalers[col] = scaler
#             else:
#                 for col in regressor_cols:
#                     try:
#                         scaler = joblib.load(os.path.join(self.models_dir, f"{stock_code}_prophet_scaler_{col}.pkl"))
#                         df_prophet[col] = scaler.transform(df_prophet[[col]]).flatten()
#                         scalers[col] = scaler
#                     except:
#                         pass
            
#             return df_prophet[['ds', 'y'] + regressor_cols].dropna(), scalers
            
#         except Exception as e:
#             print(f"Error in Prophet preprocessing: {e}")
#             return None, None
    
#     # Thêm hàm này vào class MLService trong ml_service.py
#     def process_arima_data(self, df: pd.DataFrame) -> pd.Series:
#         """Chuẩn bị dữ liệu cho ARIMA: chỉ lấy giá đóng cửa và làm cho nó bất biến."""
#         close_series = df['Close'].copy()
#         # Dùng log difference để ổn định phương sai và loại bỏ xu hướng
#         return np.log(close_series).diff().dropna()
    
#     def process_lstm_data(self, df: pd.DataFrame, stock_code: str, fit_scalers: bool = True, look_back: int = 60) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict]:
#         """Process data for LSTM - sequences with multiple features"""
#         try:
#             # Use multiple features for better LSTM performance
#             df_enhanced = self._add_technical_features(df)
            
#             # Select key features for LSTM
#             feature_cols = ['Close', 'Volume', 'SMA20', 'RSI14', 'MACD', 'BB_Position']
#             lstm_df = df_enhanced[feature_cols].dropna()
            
#             # Scale data using MinMaxScaler
#             scalers = {}
#             if fit_scalers:
#                 scaler = MinMaxScaler(feature_range=(0, 1))
#                 scaled_data = scaler.fit_transform(lstm_df.values)
#                 joblib.dump(scaler, os.path.join(self.models_dir, f"{stock_code}_lstm_scaler.pkl"))
#                 scalers['features'] = scaler
#             else:
#                 scaler = joblib.load(os.path.join(self.models_dir, f"{stock_code}_lstm_scaler.pkl"))
#                 scaled_data = scaler.transform(lstm_df.values)
#                 scalers['features'] = scaler
            
#             # Create sequences
#             X, y = [], []
#             for i in range(look_back, len(scaled_data)):
#                 X.append(scaled_data[i-look_back:i])  # Use all features for sequence
#                 y.append(scaled_data[i, 0])  # Predict Close price (first column)
            
#             X, y = np.array(X), np.array(y)
            
#             return (X, y), scalers
            
#         except Exception as e:
#             print(f"Error in LSTM preprocessing: {e}")
#             return None, None

#     # ==== UPDATED TRAINING METHOD ====
#     # Thay thế hàm này trong file ml_service.py

#     def _train_single_model(self, model_type: str, original_df: pd.DataFrame, train_size_idx: int,
#                         stock_code: str, epochs: int, retrain: bool) -> Dict[str, Any]:
#         model_name = model_type.replace("_", " ").title()
#         result = {"model_type": model_type, "model_name": model_name, "status": "pending"}

#         try:
#             df_train = original_df.iloc[:train_size_idx].copy()
#             df_test = original_df.iloc[train_size_idx:].copy()

#             df_train_featured = self._add_technical_features(df_train)
#             df_test_featured = self._add_technical_features(df_test)
            
#             if model_type == 'linear_regression':
#                 feature_cols = ['days_since_start', 'Open', 'High', 'Low', 'Volume', 'SMA20', 'SMA50', 'Close_Lag_1', 'Close_Lag_2']
#             else:
#                 feature_cols = ['Open', 'High', 'Low', 'Volume', 'SMA20', 'RSI14', 'MACD', 'BB_Position', 'Close_Lag_1', 'Price_Change']
            
#             feature_cols = [col for col in feature_cols if col in df_train_featured.columns and col in df_test_featured.columns]
            
#             df_train_final = df_train_featured[feature_cols + ['Close']].dropna()
#             df_test_final = df_test_featured[feature_cols + ['Close']].dropna()

#             X_train, y_train_unscaled = df_train_final[feature_cols], df_train_final['Close']
#             X_test, y_test_unscaled = df_test_final[feature_cols], df_test_final['Close']

#             if X_train.empty or X_test.empty:
#                 raise ValueError("Not enough data to train or test after processing.")
                
#             # === THÊM BƯỚC KIỂM TRA PHÒNG NGỪA LỖI ===
#             if X_train.isnull().values.any() or X_test.isnull().values.any():
#                 raise ValueError("NaN values detected in feature sets before scaling. Halting training.")

#             feature_scaler = RobustScaler()
#             target_scaler = RobustScaler()
            
#             X_train_scaled = feature_scaler.fit_transform(X_train)
#             X_test_scaled = feature_scaler.transform(X_test)
#             y_train_scaled = target_scaler.fit_transform(y_train_unscaled.values.reshape(-1, 1)).flatten()

#             joblib.dump(feature_scaler, os.path.join(self.scalers_dir, f"{stock_code}_{model_type}_feature_scaler.pkl"))
#             joblib.dump(target_scaler, os.path.join(self.scalers_dir, f"{stock_code}_{model_type}_target_scaler.pkl"))
#             joblib.dump(feature_cols, os.path.join(self.scalers_dir, f"{stock_code}_{model_type}_feature_cols.pkl"))

#             model_path = os.path.join(self.models_dir, f"{stock_code}_{model_type}_model.pkl")
#             train_func = getattr(self, f"_train_{model_type.replace('_regressor', '')}")
            
#             model_args = [X_train_scaled, y_train_scaled, model_path, retrain]
#             if "random_forest" in model_type:
#                 model_args.append(epochs)
            
#             model = train_func(*model_args)

#             predictions_scaled = model.predict(X_test_scaled)
#             predictions_unscaled = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

#             r2 = r2_score(y_test_unscaled, predictions_unscaled)
            
#             result.update({
#                 "status": "success", "retrained": retrain,
#                 "evaluation": {"metrics": {"R2": round(r2, 4)}, "test_samples": len(y_test_unscaled)}
#             })

#         except Exception as e:
#             result.update({"status": "error", "error_message": str(e)})

#         return result
    
#     # ==== MODEL TRAINING HELPER METHODS ====
    
#     def _train_linear_regression(self, X_train, y_train, model_path, retrain):
#         """Train optimized linear regression"""
#         if not retrain and os.path.exists(model_path):
#             return joblib.load(model_path)
        
#         model = LinearRegression(n_jobs=-1) # Sử dụng tất cả các CPU
#         model.fit(X_train, y_train)
#         joblib.dump(model, model_path)
#         return model
    
#     def _train_random_forest(self, X_train, y_train, model_path, retrain, epochs):
#         """Train optimized random forest with high-performance default parameters."""
#         if not retrain and os.path.exists(model_path):
#             return joblib.load(model_path)
        
#         # Tham số tối ưu thường dùng cho dữ liệu chứng khoán
#         model = RandomForestRegressor(
#             n_estimators=300,        # Tăng số lượng cây để mô hình ổn định hơn
#             max_depth=None,          # Cho phép cây phát triển sâu
#             min_samples_split=5,     # Yêu cầu ít nhất 5 mẫu để tách một node
#             min_samples_leaf=2,      # Mỗi lá phải có ít nhất 2 mẫu
#             max_features='sqrt',     # Giảm tương quan giữa các cây
#             random_state=42,
#             n_jobs=-1                # Sử dụng tất cả các CPU
#         )
#         model.fit(X_train, y_train)
#         joblib.dump(model, model_path)
#         return model
    
#     def _train_svm(self, X_train, y_train, model_path, retrain):
#         """Train optimized SVM with high-performance default parameters."""
#         if not retrain and os.path.exists(model_path):
#             return joblib.load(model_path)
        
#         # Tham số tối ưu thường dùng cho SVR với RBF kernel
#         model = SVR(
#             kernel='rbf',
#             C=100,                   # Tham số điều chuẩn (regularization)
#             gamma='scale',           # Tự động điều chỉnh gamma dựa trên features
#             epsilon=0.01,            # Vùng an toàn (margin of tolerance)
#             cache_size=1000          # Tăng bộ nhớ đệm để tăng tốc training
#         )
#         model.fit(X_train, y_train)
#         joblib.dump(model, model_path)
#         return model
    
#     def _train_prophet(self, train_df, model_path, retrain, regressor_cols):
#         """Train optimized Prophet"""
#         if not retrain and os.path.exists(model_path):
#             with open(model_path, 'rb') as f:
#                 return pickle.load(f)
        
#         # Optimized Prophet with regressors
#         model = Prophet(
#             yearly_seasonality=True,
#             weekly_seasonality=False,
#             daily_seasonality=False,
#             seasonality_mode='multiplicative',
#             changepoint_prior_scale=0.05,
#             seasonality_prior_scale=10
#         )
        
#         # Add regressors for better performance
#         for col in regressor_cols:
#             if col in train_df.columns:
#                 model.add_regressor(col)
        
#         model.fit(train_df)
#         with open(model_path, 'wb') as f:
#             pickle.dump(model, f)
#         return model
    
#     # Thay thế hàm _train_arima trong ml_service.py
#     # Đảm bảo bạn đã import ARIMA ở đầu file: from statsmodels.tsa.arima.model import ARIMA
#     def _train_arima(self, train_series, model_path, retrain):
#         """Huấn luyện mô hình ARIMA với order cố định (5,1,0)."""
#         if not retrain and os.path.exists(model_path):
#             with open(model_path, 'rb') as f:
#                 return pickle.load(f)
        
#         # Sử dụng ARIMA gốc với order (5,1,0) - một lựa chọn phổ biến cho dữ liệu tài chính
#         model = ARIMA(train_series, order=(5, 1, 0))
#         model_fit = model.fit()
        
#         with open(model_path, 'wb') as f:
#             pickle.dump(model_fit, f)
#         return model_fit
    
#     def _train_lstm(self, X_train, y_train, model_path, retrain, epochs, input_shape):
#         """Train optimized LSTM"""
#         if not retrain and os.path.exists(model_path):
#             return load_model(model_path)
        
#         # Optimized LSTM architecture
#         tf.random.set_seed(42)
#         model = Sequential([
#             LSTM(100, return_sequences=True, input_shape=(input_shape[1], input_shape[2])),
#             Dropout(0.2),
#             LSTM(50, return_sequences=False),
#             Dropout(0.2),
#             Dense(25),
#             Dense(1)
#         ])
        
#         model.compile(
#             optimizer='adam',
#             loss='huber',  # More robust to outliers
#             metrics=['mae']
#         )
        
#         es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
#         model.fit(
#             X_train, y_train,
#             epochs=min(epochs, 200),
#             batch_size=32,
#             validation_split=0.2,
#             callbacks=[es],
#             verbose=0
#         )
        
#         model.save(model_path)
#         return model

#     # Update the main training method to not use the old preprocess_data
#     def train_and_evaluate_models(self, stock_code: str, period: str, models_to_train: List[str],
#                               split_ratio: float, epochs: int, retrain: bool = True) -> Dict[str, Any]:
#         """
#         Pipeline chính để huấn luyện và đánh giá.
#         Hàm này xử lý ARIMA như một trường hợp đặc biệt.
#         """
#         try:
#             stock_service = StockDataService()
#             df = stock_service.get_historical_data(stock_code, period)
#             if df is None or len(df) < 50:
#                 raise ValueError(f"Not enough data for {stock_code} (need at least 50 days)")
            
#             # Xác định chỉ số (index) để chia train/test
#             train_size_idx = int(len(df) * split_ratio)

#             results = []
#             for model_type in models_to_train:
#                 # === XỬ LÝ ARIMA RIÊNG BIỆT ===
#                 if model_type == 'arima':
#                     try:
#                         # Gọi hàm xử lý dữ liệu của ARIMA (chỉ nhận 1 tham số `df`)
#                         processed_series = self.process_arima_data(df)
                        
#                         # Chia dữ liệu đã xử lý
#                         train_series = processed_series.iloc[:train_size_idx]
#                         test_series = processed_series.iloc[train_size_idx:]

#                         if train_series.empty or test_series.empty:
#                             raise ValueError("Not enough data for ARIMA after processing.")

#                         model_path = os.path.join(self.models_dir, f"{stock_code}_arima_model.pkl")
#                         model = self._train_arima(train_series, model_path, retrain)
                        
#                         predictions = model.forecast(steps=len(test_series))
#                         r2 = r2_score(test_series, predictions)

#                         results.append({
#                             "model_type": "arima", "model_name": "ARIMA", "status": "success",
#                             "evaluation": {"metrics": {"R2": round(r2, 4)}}
#                         })
#                     except Exception as e:
#                         results.append({"model_type": "arima", "status": "error", "error_message": str(e)})
#                     # Chuyển sang model tiếp theo
#                     continue

#                 # === XỬ LÝ CÁC MODEL SKLEARN KHÁC ===
#                 if model_type in self.supported_models:
#                     # Gọi hàm _train_single_model cho các model còn lại
#                     results.append(self._train_single_model(model_type, df, train_size_idx, stock_code, epochs, retrain))
#                 else:
#                     results.append({"model_type": model_type, "status": "error", "error_message": "Model not supported"})
            
#             status = "success" if all(r['status'] == 'success' for r in results) else "partial_success"
#             return {"status": status, "results": results}
            
#         except Exception as e:
#             return {"status": "error", "message": f"Pipeline failed: {str(e)}"}

#     def get_model_status(self, stock_code: str) -> Dict[str, Any]:
#         """Get available models for a stock"""
#         available_models = []
        
#         for model_type in self.supported_models.keys():
#             model_paths = [
#                 os.path.join(self.models_dir, f"{stock_code}_{model_type}_model.pkl"),
#                 os.path.join(self.models_dir, f"{stock_code}_{model_type}_model.h5")
#             ]
            
#             for model_path in model_paths:
#                 if os.path.exists(model_path):
#                     available_models.append({
#                         "model_type": model_type,
#                         "model_name": model_type.replace("_", " ").title(),
#                         "path": model_path,
#                         "size_kb": round(os.path.getsize(model_path) / 1024, 2)
#                     })
#                     break
        
#         return {
#             "stock_code": stock_code,
#             "available_models": available_models,
#             "supported_models": list(self.supported_models.keys()),
#             "total_available": len(available_models)
#         }

# # Create singleton instance
# ml_service = MLService(
#     models_dir=Config.ML_MODELS_DIR,
#     scalers_dir=Config.SCALERS_DIR
# )

import pandas as pd
import numpy as np
import os
import pickle
import joblib
from typing import Dict, List, Any, Tuple

# ML Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

# Processing
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# App-specific
from .stock_data_service import StockDataService
from config.settings import Config

class MLService:
    def __init__(self, models_dir: str = 'models', scalers_dir: str = 'scalers'):
        self.models_dir = models_dir
        self.scalers_dir = scalers_dir
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.scalers_dir, exist_ok=True)
        
        # Hỗ trợ tất cả các model
        self.supported_models = {
            'linear_regression': LinearRegression,
            'random_forest_regressor': RandomForestRegressor,
            'svm_regressor': SVR,
            'arima': ARIMA,
            'lstm': Sequential
        }
    
    # --- CÁC HÀM HELPER (Không thay đổi) ---
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy['Price_Change'] = df_copy['Close'].pct_change()
        for period in [5, 10, 20, 50]:
            df_copy[f'SMA{period}'] = df_copy['Close'].rolling(window=period).mean()
        df_copy['RSI14'] = self._calculate_rsi(df_copy['Close'], 14)
        df_copy['EMA12'] = df_copy['Close'].ewm(span=12).mean()
        df_copy['EMA26'] = df_copy['Close'].ewm(span=26).mean()
        df_copy['MACD'] = df_copy['EMA12'] - df_copy['EMA26']
        for lag in [1, 2, 3]:
            df_copy[f'Close_Lag_{lag}'] = df_copy['Close'].shift(lag)
        df_copy = df_copy.replace([np.inf, -np.inf], np.nan)
        df_copy.ffill(inplace=True)
        df_copy.bfill(inplace=True)
        return df_copy

    def _calculate_rsi(self, data, window=14):
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).abs().rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    # --- CÁC HÀM XỬ LÝ DỮ LIỆU RIÊNG CHO TỪNG MODEL ---

    def process_arima_data(self, df: pd.DataFrame) -> pd.Series:
        return np.log(df['Close']).diff().dropna()

    def process_lstm_data(self, df: pd.DataFrame, look_back: int = 60) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        close_prices = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)
        
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
            
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y, scaler

    # --- CÁC HÀM HUẤN LUYỆN RIÊNG CHO TỪNG MODEL ---
    def _train_linear_regression(self, X_train, y_train, path, retrain):
        if not retrain and os.path.exists(path): return joblib.load(path)
        model = LinearRegression(n_jobs=-1).fit(X_train, y_train)
        joblib.dump(model, path)
        return model

    def _train_random_forest(self, X_train, y_train, path, retrain, epochs=None):
        if not retrain and os.path.exists(path): return joblib.load(path)
        model = RandomForestRegressor(n_estimators=300, min_samples_split=5, random_state=42, n_jobs=-1).fit(X_train, y_train)
        joblib.dump(model, path)
        return model

    def _train_svm(self, X_train, y_train, path, retrain):
        if not retrain and os.path.exists(path): return joblib.load(path)
        model = SVR(kernel='rbf', C=100, epsilon=0.01).fit(X_train, y_train)
        joblib.dump(model, path)
        return model

    def _train_arima(self, train_series, path, retrain):
        if not retrain and os.path.exists(path):
            with open(path, 'rb') as f: return pickle.load(f)
        model = ARIMA(train_series, order=(5, 1, 0)).fit()
        with open(path, 'wb') as f: pickle.dump(model, f)
        return model
        
    def _train_lstm(self, X_train, y_train, path, retrain, epochs, input_shape):
        if not retrain and os.path.exists(path): return load_model(path)
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape), Dropout(0.2),
            LSTM(50, return_sequences=False), Dropout(0.2),
            Dense(25), Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
        model.save(path)
        return model
    # Dán hàm này vào bên trong class MLService trong file ml_service.py

    def get_model_status(self, stock_code: str) -> Dict[str, Any]:
        """Kiểm tra các model đã được huấn luyện cho một mã cổ phiếu."""
        available_models = []
        
        for model_type in self.supported_models.keys():
            # Xác định các đuôi file có thể có
            model_paths = []
            if model_type == 'lstm':
                model_paths.append(os.path.join(self.models_dir, f"{stock_code}_{model_type}_model.h5"))
            else:
                model_paths.append(os.path.join(self.models_dir, f"{stock_code}_{model_type}_model.pkl"))
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    available_models.append({
                        "model_type": model_type,
                        "model_name": model_type.replace("_", " ").title(),
                        "path": model_path,
                        "size_kb": round(os.path.getsize(model_path) / 1024, 2)
                    })
                    # Nếu đã tìm thấy model thì không cần tìm các đuôi file khác
                    break
        
        return {
            "stock_code": stock_code,
            "available_models": available_models,
            "supported_models": list(self.supported_models.keys()),
            "total_available": len(available_models)
        }    
    # --- PIPELINE HUẤN LUYỆN CHÍNH (ĐÃ SỬA) ---
    def train_and_evaluate_models(self, stock_code: str, period: str, models_to_train: List[str],
                              split_ratio: float, epochs: int, retrain: bool = True) -> Dict[str, Any]:
        try:
            df = StockDataService.get_historical_data(stock_code, period)
            if df is None or len(df) < 60:
                raise ValueError(f"Not enough data for {stock_code}")
            
            # Chuyển Date từ index thành cột để xử lý nhất quán
            df.reset_index(inplace=True)
            
            train_size_idx = int(len(df) * split_ratio)
            results = []

            for model_type in models_to_train:
                result = {"model_type": model_type, "model_name": model_type.replace("_", " ").title()}
                try:
                    y_test_unscaled, predictions_unscaled = None, None

                    # === XỬ LÝ ARIMA ===
                    if model_type == 'arima':
                        series_arima = self.process_arima_data(df)
                        train_series, test_series = series_arima.iloc[:train_size_idx], series_arima.iloc[train_size_idx:]
                        model_path = os.path.join(self.models_dir, f"{stock_code}_arima_model.pkl")
                        model = self._train_arima(train_series, model_path, retrain)
                        predictions_unscaled = model.forecast(steps=len(test_series))
                        y_test_unscaled = test_series

                    # === XỬ LÝ LSTM ===
                    elif model_type == 'lstm':
                        X, y, scaler = self.process_lstm_data(df)
                        joblib.dump(scaler, os.path.join(self.scalers_dir, f"{stock_code}_lstm_scaler.pkl"))
                        train_idx = int(len(X) * split_ratio)
                        X_train, X_test = X[:train_idx], X[train_idx:]
                        y_train, y_test = y[:train_idx], y[train_idx:]
                        model_path = os.path.join(self.models_dir, f"{stock_code}_lstm_model.h5")
                        model = self._train_lstm(X_train, y_train, model_path, retrain, epochs, (X_train.shape[1], 1))
                        predictions_scaled = model.predict(X_test)
                        y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
                        predictions_unscaled = scaler.inverse_transform(predictions_scaled)
                        
                    # # === XỬ LÝ CÁC MODEL SKLEARN ===
                    # elif model_type in ['linear_regression', 'random_forest_regressor', 'svm_regressor']:
                    #     df_featured = self._add_technical_features(df)
                    #     feature_cols = [col for col in df_featured.columns if col not in ['Close', 'Date', 'Stock Splits', 'Dividends']]
                    #     df_final = df_featured.dropna()
                    #     X, y = df_final[feature_cols], df_final['Close']
                    #     train_idx = int(len(X) * split_ratio)
                    #     X_train, X_test = X.iloc[:train_idx], X.iloc[train_idx:]
                    #     y_train, y_test = y.iloc[:train_idx], y.iloc[train_idx:]

                    #     scaler = RobustScaler()
                    #     X_train_scaled = scaler.fit_transform(X_train)
                    #     X_test_scaled = scaler.transform(X_test)
                        
                    #     joblib.dump(scaler, os.path.join(self.scalers_dir, f"{stock_code}_{model_type}_feature_scaler.pkl"))
                    #     joblib.dump(feature_cols, os.path.join(self.scalers_dir, f"{stock_code}_{model_type}_feature_cols.pkl"))

                    #     model_path = os.path.join(self.models_dir, f"{stock_code}_{model_type}_model.pkl")
                    #     train_func = getattr(self, f"_train_{model_type.replace('_regressor', '')}")
                    #     model = train_func(X_train_scaled, y_train, model_path, retrain)
                        
                    #     predictions_unscaled = model.predict(X_test_scaled)
                    #     y_test_unscaled = y_test
                    elif model_type in ['linear_regression', 'random_forest_regressor', 'svm_regressor']:
                        # df_featured = self._add_technical_features(df)
                        # feature_cols = [col for col in df_featured.columns if col not in ['Close', 'Date', 'Stock Splits', 'Dividends']]
                        # df_final = df_featured.dropna()
                        df_featured = self._add_technical_features(df)
                        # === SỬA LỖI: LOẠI BỎ 'index' KHỎI DANH SÁCH FEATURES ===
                        feature_cols = [
                            col for col in df_featured.columns 
                            if col not in ['Close', 'Date', 'Stock Splits', 'Dividends', 'index'] # Thêm 'index' vào danh sách loại trừ
                        ]
                        # =========================================================
                        df_final = df_featured.dropna()
                        X, y_unscaled = df_final[feature_cols], df_final['Close'] # Đổi tên y thành y_unscaled

                        train_idx = int(len(X) * split_ratio)
                        X_train, X_test = X.iloc[:train_idx], X.iloc[train_idx:]
                        y_train_unscaled, y_test = y_unscaled.iloc[:train_idx], y_unscaled.iloc[train_idx:]

                        # Scaler cho features (dữ liệu đầu vào X)
                        feature_scaler = RobustScaler()
                        X_train_scaled = feature_scaler.fit_transform(X_train)
                        X_test_scaled = feature_scaler.transform(X_test)

                        # === THÊM SCALER CHO TARGET (DỮ LIỆU MỤC TIÊU y) ===
                        target_scaler = RobustScaler()
                        y_train_scaled = target_scaler.fit_transform(y_train_unscaled.values.reshape(-1, 1))
                        # ======================================================

                        # Lưu tất cả các scaler và feature columns
                        joblib.dump(feature_scaler, os.path.join(self.scalers_dir, f"{stock_code}_{model_type}_feature_scaler.pkl"))
                        joblib.dump(target_scaler, os.path.join(self.scalers_dir, f"{stock_code}_{model_type}_target_scaler.pkl")) # LƯU LẠI TARGET SCALER
                        joblib.dump(feature_cols, os.path.join(self.scalers_dir, f"{stock_code}_{model_type}_feature_cols.pkl"))

                        model_path = os.path.join(self.models_dir, f"{stock_code}_{model_type}_model.pkl")
                        train_func = getattr(self, f"_train_{model_type.replace('_regressor', '')}")

                        # Huấn luyện model trên dữ liệu đã được scale
                        model = train_func(X_train_scaled, y_train_scaled.flatten(), model_path, retrain)

                        predictions_scaled = model.predict(X_test_scaled)

                        # Đảo ngược scale của prediction để so sánh
                        predictions_unscaled = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
                        y_test_unscaled = y_test # y_test đã là unscaled rồi

                    # --- TÍNH TOÁN METRICS (DÙNG CHUNG) ---
                    if y_test_unscaled is not None and predictions_unscaled is not None:
                        mae = mean_absolute_error(y_test_unscaled, predictions_unscaled)
                        rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions_unscaled))
                        r2 = r2_score(y_test_unscaled, predictions_unscaled)
                        result.update({
                            "status": "success",
                            "evaluation": {
                                "metrics": {
                                    "R2": round(r2, 4),
                                    "RMSE": round(rmse, 4),
                                    "MAE": round(mae, 4)
                                }
                            }
                        })
                    else:
                        raise ValueError("Test values or predictions were not generated.")

                except Exception as e:
                    result.update({"status": "error", "error_message": str(e)})
                
                results.append(result)

            return {"status": "success", "results": results}
        except Exception as e:
            return {"status": "error", "message": f"Pipeline failed: {str(e)}"}


ml_service = MLService(
    models_dir=Config.ML_MODELS_DIR,
    scalers_dir=Config.SCALERS_DIR
)