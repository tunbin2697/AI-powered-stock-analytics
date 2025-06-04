import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import pickle
from typing import Dict, List, Any, Tuple, Optional
from .stock_data_service import StockDataService
from config.settings import Config

# New model imports
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import tensorflow as tf

class MLService:
    def __init__(self, models_dir: str = 'models', scalers_dir: str = 'scalers'):
        self.models_dir = models_dir
        self.scalers_dir = scalers_dir
        
        self.supported_models = {
            'linear_regression': LinearRegression,
            'random_forest_regressor': RandomForestRegressor,
            'svm_regressor': SVR,
            'prophet': Prophet,
            'arima': ARIMA,
            'lstm': Sequential
        }
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        df_copy = df.copy()
        
        # Price-based features
        df_copy['Price_Change'] = df_copy['Close'].pct_change()
        df_copy['High_Low_Ratio'] = df_copy['High'] / df_copy['Low']
        df_copy['Close_Open_Ratio'] = df_copy['Close'] / df_copy['Open']
        
        # Moving averages (multiple periods)
        for period in [5, 10, 20, 50, 200]:
            df_copy[f'SMA{period}'] = df_copy['Close'].rolling(window=period).mean()
            df_copy[f'Close_SMA{period}_Ratio'] = df_copy['Close'] / df_copy[f'SMA{period}']
        
        # Exponential moving averages
        for period in [12, 26]:
            df_copy[f'EMA{period}'] = df_copy['Close'].ewm(span=period).mean()
        
        # MACD
        df_copy['MACD'] = df_copy['EMA12'] - df_copy['EMA26']
        df_copy['MACD_Signal'] = df_copy['MACD'].ewm(span=9).mean()
        df_copy['MACD_Histogram'] = df_copy['MACD'] - df_copy['MACD_Signal']
        
        # RSI (multiple periods)
        for period in [14, 21]:
            df_copy[f'RSI{period}'] = self._calculate_rsi(df_copy['Close'], period)
        
        # Bollinger Bands
        df_copy['BB_Middle'] = df_copy['Close'].rolling(window=20).mean()
        bb_std = df_copy['Close'].rolling(window=20).std()
        df_copy['BB_Upper'] = df_copy['BB_Middle'] + (bb_std * 2)
        df_copy['BB_Lower'] = df_copy['BB_Middle'] - (bb_std * 2)
        df_copy['BB_Position'] = (df_copy['Close'] - df_copy['BB_Lower']) / (df_copy['BB_Upper'] - df_copy['BB_Lower'])
        
        # Volume indicators
        df_copy['Volume_SMA20'] = df_copy['Volume'].rolling(window=20).mean()
        df_copy['Volume_Ratio'] = df_copy['Volume'] / df_copy['Volume_SMA20']
        df_copy['Price_Volume'] = df_copy['Close'] * df_copy['Volume']
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df_copy[f'Close_Lag_{lag}'] = df_copy['Close'].shift(lag)
            df_copy[f'Volume_Lag_{lag}'] = df_copy['Volume'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df_copy[f'Close_Rolling_Mean_{window}'] = df_copy['Close'].rolling(window=window).mean()
            df_copy[f'Close_Rolling_Std_{window}'] = df_copy['Close'].rolling(window=window).std()
            df_copy[f'Close_Rolling_Min_{window}'] = df_copy['Close'].rolling(window=window).min()
            df_copy[f'Close_Rolling_Max_{window}'] = df_copy['Close'].rolling(window=window).max()
        
        # Time-based features
        df_copy['days_since_start'] = range(len(df_copy))
        df_copy['month'] = pd.to_datetime(df_copy.index).month if hasattr(df_copy.index, 'month') else (pd.to_datetime('2006-12-13') + pd.to_timedelta(df_copy.index, unit='D')).month
        df_copy['quarter'] = df_copy['month'].apply(lambda x: (x-1)//3 + 1)
        
        return df_copy
    
    def _calculate_rsi(self, data, window=14):
        """Calculate RSI indicator"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # ==== MODEL-SPECIFIC DATA PROCESSING FUNCTIONS ====
    
    def process_linear_regression_data(self, df: pd.DataFrame, stock_code: str, fit_scalers: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """Process data optimized for Linear Regression - focus on linear relationships"""
        try:
            df_processed = self._add_technical_features(df)
            
            # Select features that work well with linear regression
            feature_cols = [
                'days_since_start', 'Open', 'High', 'Low', 'Volume',
                'SMA5', 'SMA10', 'SMA20', 'SMA50', 
                'Close_SMA20_Ratio', 'Close_SMA50_Ratio',
                'Price_Change', 'High_Low_Ratio', 'Close_Open_Ratio',
                'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3',
                'Close_Rolling_Mean_5', 'Close_Rolling_Mean_10',
                'Volume_Ratio', 'month', 'quarter'
            ]
            
            # Fill NaN values
            for col in feature_cols:
                if col in df_processed.columns:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            
            # Use StandardScaler for linear regression
            scalers = {}
            if fit_scalers:
                scaler_close = StandardScaler()
                scaler_features = StandardScaler()
                
                df_processed['Close'] = scaler_close.fit_transform(df_processed[['Close']]).flatten()
                df_processed[feature_cols] = scaler_features.fit_transform(df_processed[feature_cols])
                
                joblib.dump(scaler_close, os.path.join(self.models_dir, f"{stock_code}_linear_regression_scaler_close.pkl"))
                joblib.dump(scaler_features, os.path.join(self.models_dir, f"{stock_code}_linear_regression_scaler_features.pkl"))
                
                scalers = {'Close': scaler_close, 'features': scaler_features}
            else:
                scaler_close = joblib.load(os.path.join(self.models_dir, f"{stock_code}_linear_regression_scaler_close.pkl"))
                scaler_features = joblib.load(os.path.join(self.models_dir, f"{stock_code}_linear_regression_scaler_features.pkl"))
                
                df_processed['Close'] = scaler_close.transform(df_processed[['Close']]).flatten()
                df_processed[feature_cols] = scaler_features.transform(df_processed[feature_cols])
                
                scalers = {'Close': scaler_close, 'features': scaler_features}
            
            final_cols = feature_cols + ['Close']
            return df_processed[final_cols].dropna(), scalers
            
        except Exception as e:
            print(f"Error in linear regression preprocessing: {e}")
            return None, None
    
    def process_random_forest_data(self, df: pd.DataFrame, stock_code: str, fit_scalers: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """Process data optimized for Random Forest - use raw features, no scaling needed"""
        try:
            df_processed = self._add_technical_features(df)
            
            # Random Forest can handle more features and doesn't need scaling
            feature_cols = [
                'days_since_start', 'Open', 'High', 'Low', 'Volume',
                'SMA5', 'SMA10', 'SMA20', 'SMA50', 'SMA200',
                'EMA12', 'EMA26', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                'RSI14', 'RSI21', 'BB_Position',
                'Close_SMA5_Ratio', 'Close_SMA20_Ratio', 'Close_SMA50_Ratio',
                'Price_Change', 'High_Low_Ratio', 'Close_Open_Ratio',
                'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5',
                'Volume_Lag_1', 'Volume_Lag_2',
                'Close_Rolling_Mean_5', 'Close_Rolling_Std_5',
                'Close_Rolling_Mean_10', 'Close_Rolling_Std_10',
                'Close_Rolling_Mean_20', 'Close_Rolling_Std_20',
                'Close_Rolling_Min_5', 'Close_Rolling_Max_5',
                'Volume_Ratio', 'Price_Volume',
                'month', 'quarter'
            ]
            
            # Fill NaN values
            for col in feature_cols:
                if col in df_processed.columns:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            
            # Random Forest doesn't need scaling, but we keep target scaling for consistency
            scalers = {}
            if fit_scalers:
                scaler_close = StandardScaler()
                df_processed['Close'] = scaler_close.fit_transform(df_processed[['Close']]).flatten()
                joblib.dump(scaler_close, os.path.join(self.models_dir, f"{stock_code}_random_forest_scaler_close.pkl"))
                scalers = {'Close': scaler_close}
            else:
                scaler_close = joblib.load(os.path.join(self.models_dir, f"{stock_code}_random_forest_scaler_close.pkl"))
                df_processed['Close'] = scaler_close.transform(df_processed[['Close']]).flatten()
                scalers = {'Close': scaler_close}
            
            final_cols = feature_cols + ['Close']
            return df_processed[final_cols].dropna(), scalers
            
        except Exception as e:
            print(f"Error in random forest preprocessing: {e}")
            return None, None
    
    def process_svm_data(self, df: pd.DataFrame, stock_code: str, fit_scalers: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """Process data optimized for SVM - robust scaling and feature selection"""
        try:
            df_processed = self._add_technical_features(df)
            
            # SVM works best with fewer, well-scaled features
            feature_cols = [
                'Open', 'High', 'Low', 'Volume',
                'SMA20', 'SMA50', 'RSI14', 'MACD',
                'Close_SMA20_Ratio', 'BB_Position',
                'Price_Change', 'High_Low_Ratio',
                'Close_Lag_1', 'Close_Lag_2',
                'Close_Rolling_Mean_10', 'Close_Rolling_Std_10',
                'Volume_Ratio'
            ]
            
            # Fill NaN values
            for col in feature_cols:
                if col in df_processed.columns:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            
            # Use RobustScaler for SVM (better with outliers)
            scalers = {}
            if fit_scalers:
                scaler_close = RobustScaler()
                scaler_features = RobustScaler()
                
                df_processed['Close'] = scaler_close.fit_transform(df_processed[['Close']]).flatten()
                df_processed[feature_cols] = scaler_features.fit_transform(df_processed[feature_cols])
                
                joblib.dump(scaler_close, os.path.join(self.models_dir, f"{stock_code}_svm_scaler_close.pkl"))
                joblib.dump(scaler_features, os.path.join(self.models_dir, f"{stock_code}_svm_scaler_features.pkl"))
                
                scalers = {'Close': scaler_close, 'features': scaler_features}
            else:
                scaler_close = joblib.load(os.path.join(self.models_dir, f"{stock_code}_svm_scaler_close.pkl"))
                scaler_features = joblib.load(os.path.join(self.models_dir, f"{stock_code}_svm_scaler_features.pkl"))
                
                df_processed['Close'] = scaler_close.transform(df_processed[['Close']]).flatten()
                df_processed[feature_cols] = scaler_features.transform(df_processed[feature_cols])
                
                scalers = {'Close': scaler_close, 'features': scaler_features}
            
            final_cols = feature_cols + ['Close']
            return df_processed[final_cols].dropna(), scalers
            
        except Exception as e:
            print(f"Error in SVM preprocessing: {e}")
            return None, None
    
    def process_prophet_data(self, df: pd.DataFrame, stock_code: str, fit_scalers: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """Process data for Prophet - simple date-value format with additional regressors"""
        try:
            # Prophet expects specific column names: 'ds' (date) and 'y' (value)
            start_date = pd.to_datetime('2006-12-13')
            df_prophet = df.copy()
            df_prophet['ds'] = start_date + pd.to_timedelta(df_prophet.index, unit='D')
            df_prophet['y'] = df_prophet['Close']
            
            # Add technical indicators as additional regressors for better R²
            df_enhanced = self._add_technical_features(df)
            
            # Select key indicators as regressors
            regressor_cols = ['SMA20', 'RSI14', 'Volume_Ratio', 'MACD', 'BB_Position']
            for col in regressor_cols:
                if col in df_enhanced.columns:
                    df_prophet[col] = df_enhanced[col].fillna(df_enhanced[col].median())
            
            # Prophet doesn't need target scaling but we store scalers for consistency
            scalers = {}
            if fit_scalers:
                # Save regressor scalers for prediction consistency
                for col in regressor_cols:
                    if col in df_prophet.columns:
                        scaler = StandardScaler()
                        df_prophet[col] = scaler.fit_transform(df_prophet[[col]]).flatten()
                        joblib.dump(scaler, os.path.join(self.models_dir, f"{stock_code}_prophet_scaler_{col}.pkl"))
                        scalers[col] = scaler
            else:
                for col in regressor_cols:
                    try:
                        scaler = joblib.load(os.path.join(self.models_dir, f"{stock_code}_prophet_scaler_{col}.pkl"))
                        df_prophet[col] = scaler.transform(df_prophet[[col]]).flatten()
                        scalers[col] = scaler
                    except:
                        pass
            
            return df_prophet[['ds', 'y'] + regressor_cols].dropna(), scalers
            
        except Exception as e:
            print(f"Error in Prophet preprocessing: {e}")
            return None, None
    
    def process_arima_data(self, df: pd.DataFrame, stock_code: str, fit_scalers: bool = True) -> Tuple[pd.Series, Dict]:
        """Process data for ARIMA - stationary time series"""
        try:
            close_series = df['Close'].copy()
            
            # Make series stationary by differencing if needed
            # Check stationarity and apply log transformation + differencing
            close_series = np.log(close_series + 1e-8)  # Log transform to stabilize variance
            close_series = close_series.diff().dropna()   # First difference to make stationary
            
            scalers = {'transformation': 'log_diff'}  # Store transformation info
            
            return close_series, scalers
            
        except Exception as e:
            print(f"Error in ARIMA preprocessing: {e}")
            return None, None
    
    def process_lstm_data(self, df: pd.DataFrame, stock_code: str, fit_scalers: bool = True, look_back: int = 60) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict]:
        """Process data for LSTM - sequences with multiple features"""
        try:
            # Use multiple features for better LSTM performance
            df_enhanced = self._add_technical_features(df)
            
            # Select key features for LSTM
            feature_cols = ['Close', 'Volume', 'SMA20', 'RSI14', 'MACD', 'BB_Position']
            lstm_df = df_enhanced[feature_cols].dropna()
            
            # Scale data using MinMaxScaler
            scalers = {}
            if fit_scalers:
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(lstm_df.values)
                joblib.dump(scaler, os.path.join(self.models_dir, f"{stock_code}_lstm_scaler.pkl"))
                scalers['features'] = scaler
            else:
                scaler = joblib.load(os.path.join(self.models_dir, f"{stock_code}_lstm_scaler.pkl"))
                scaled_data = scaler.transform(lstm_df.values)
                scalers['features'] = scaler
            
            # Create sequences
            X, y = [], []
            for i in range(look_back, len(scaled_data)):
                X.append(scaled_data[i-look_back:i])  # Use all features for sequence
                y.append(scaled_data[i, 0])  # Predict Close price (first column)
            
            X, y = np.array(X), np.array(y)
            
            return (X, y), scalers
            
        except Exception as e:
            print(f"Error in LSTM preprocessing: {e}")
            return None, None

    # ==== UPDATED TRAINING METHOD ====
    
    def _train_single_model(self, model_type: str, original_df: pd.DataFrame, train_size: int, 
                           stock_code: str, epochs: int, retrain: bool) -> Dict[str, Any]:
        """Train or load a single model with optimized data processing"""
        
        model_name = model_type.replace("_", " ").title()
        result = {
            "model_type": model_type,
            "model_name": model_name,
            "status": "pending",
            "retrained": False,
            "evaluation": {},
            "error_message": None,
            "logs": {"training": [], "evaluation": []}
        }
        
        try:
            # Use model-specific data processing
            if model_type == 'linear_regression':
                processed_data, scalers = self.process_linear_regression_data(original_df, stock_code, retrain)
                if processed_data is None:
                    raise ValueError("Linear regression data processing failed")
                
                X_all = processed_data.drop(columns=['Close'])
                y_all = processed_data['Close']
                train_size_adj = min(train_size, len(processed_data) - 1)
                X_train, X_test = X_all.iloc[:train_size_adj], X_all.iloc[train_size_adj:]
                y_train, y_test = y_all.iloc[:train_size_adj], y_all.iloc[train_size_adj:]
                
                model_path = os.path.join(self.models_dir, f"{stock_code}_linear_regression_model.pkl")
                model = self._train_linear_regression(X_train, y_train, model_path, retrain)
                predictions = model.predict(X_test)
                
            elif model_type == 'random_forest_regressor':
                processed_data, scalers = self.process_random_forest_data(original_df, stock_code, retrain)
                if processed_data is None:
                    raise ValueError("Random forest data processing failed")
                
                X_all = processed_data.drop(columns=['Close'])
                y_all = processed_data['Close']
                train_size_adj = min(train_size, len(processed_data) - 1)
                X_train, X_test = X_all.iloc[:train_size_adj], X_all.iloc[train_size_adj:]
                y_train, y_test = y_all.iloc[:train_size_adj], y_all.iloc[train_size_adj:]
                
                model_path = os.path.join(self.models_dir, f"{stock_code}_random_forest_regressor_model.pkl")
                model = self._train_random_forest(X_train, y_train, model_path, retrain, epochs)
                predictions = model.predict(X_test)
                
            elif model_type == 'svm_regressor':
                processed_data, scalers = self.process_svm_data(original_df, stock_code, retrain)
                if processed_data is None:
                    raise ValueError("SVM data processing failed")
                
                X_all = processed_data.drop(columns=['Close'])
                y_all = processed_data['Close']
                train_size_adj = min(train_size, len(processed_data) - 1)
                X_train, X_test = X_all.iloc[:train_size_adj], X_all.iloc[train_size_adj:]
                y_train, y_test = y_all.iloc[:train_size_adj], y_all.iloc[train_size_adj:]
                
                model_path = os.path.join(self.models_dir, f"{stock_code}_svm_regressor_model.pkl")
                model = self._train_svm(X_train, y_train, model_path, retrain)
                predictions = model.predict(X_test)
                
            elif model_type == 'prophet':
                processed_data, scalers = self.process_prophet_data(original_df, stock_code, retrain)
                if processed_data is None:
                    raise ValueError("Prophet data processing failed")
                
                train_size_adj = min(train_size, len(processed_data) - 1)
                train_df = processed_data.iloc[:train_size_adj]
                test_df = processed_data.iloc[train_size_adj:]
                
                model_path = os.path.join(self.models_dir, f"{stock_code}_prophet_model.pkl")
                model = self._train_prophet(train_df, model_path, retrain, list(scalers.keys()))
                
                future_test = test_df[['ds'] + list(scalers.keys())]
                forecast_test = model.predict(future_test)
                y_test = test_df['y'].values
                predictions = forecast_test['yhat'].values
                
            elif model_type == 'arima':
                processed_data, scalers = self.process_arima_data(original_df, stock_code, retrain)
                if processed_data is None:
                    raise ValueError("ARIMA data processing failed")
                
                train_size_adj = min(train_size, len(processed_data) - 1)
                train_series = processed_data.iloc[:train_size_adj]
                test_series = processed_data.iloc[train_size_adj:]
                
                model_path = os.path.join(self.models_dir, f"{stock_code}_arima_model.pkl")
                model = self._train_arima(train_series, model_path, retrain)
                
                forecast_test = model.get_forecast(steps=len(test_series))
                predictions = forecast_test.predicted_mean.values
                y_test = test_series.values
                
            elif model_type == 'lstm':
                processed_data, scalers = self.process_lstm_data(original_df, stock_code, retrain)
                if processed_data is None:
                    raise ValueError("LSTM data processing failed")
                
                X, y = processed_data
                train_size_adj = min(int(len(X) * (train_size / len(original_df))), len(X) - 1)
                X_train, X_test = X[:train_size_adj], X[train_size_adj:]
                y_train, y_test = y[:train_size_adj], y[train_size_adj:]
                
                model_path = os.path.join(self.models_dir, f"{stock_code}_lstm_model.h5")
                model = self._train_lstm(X_train, y_train, model_path, retrain, epochs, X.shape)
                predictions = model.predict(X_test, verbose=0).flatten()
                
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Calculate evaluation metrics
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            mape = np.mean(np.abs((y_test - predictions) / np.abs(y_test))) * 100
            
            result["evaluation"] = {
                "metrics": {
                    "MAE": round(mae, 6),
                    "RMSE": round(rmse, 6),
                    "R2": round(r2, 6),
                    "MAPE": round(mape, 2)
                },
                "test_samples": len(predictions)
            }
            
            result["status"] = "success"
            if retrain:
                result["retrained"] = True
                
        except Exception as e:
            result["status"] = "error"
            result["error_message"] = str(e)
        
        return result
    
    # ==== MODEL TRAINING HELPER METHODS ====
    
    def _train_linear_regression(self, X_train, y_train, model_path, retrain):
        """Train optimized linear regression"""
        if not retrain and os.path.exists(model_path):
            return joblib.load(model_path)
        
        # Remove normalize parameter (deprecated in newer sklearn versions)
        model = LinearRegression(fit_intercept=True)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        return model
    
    def _train_random_forest(self, X_train, y_train, model_path, retrain, epochs):
        """Train optimized random forest"""
        if not retrain and os.path.exists(model_path):
            return joblib.load(model_path)
        
        # Optimized parameters for better R²
        n_estimators = min(max(epochs, 200), 500)
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        return model
    
    def _train_svm(self, X_train, y_train, model_path, retrain):
        """Train optimized SVM"""
        if not retrain and os.path.exists(model_path):
            return joblib.load(model_path)
        
        # Optimized SVM parameters
        model = SVR(
            kernel='rbf',
            C=100,
            gamma='scale',
            epsilon=0.01,
            cache_size=1000
        )
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        return model
    
    def _train_prophet(self, train_df, model_path, retrain, regressor_cols):
        """Train optimized Prophet"""
        if not retrain and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        
        # Optimized Prophet with regressors
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10
        )
        
        # Add regressors for better performance
        for col in regressor_cols:
            if col in train_df.columns:
                model.add_regressor(col)
        
        model.fit(train_df)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        return model
    
    def _train_arima(self, train_series, model_path, retrain):
        """Train optimized ARIMA"""
        if not retrain and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        
        # Use auto ARIMA order selection for better performance
        # Simple fixed order for stability
        arima_model = ARIMA(train_series, order=(2, 1, 2))
        model = arima_model.fit()
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        return model
    
    def _train_lstm(self, X_train, y_train, model_path, retrain, epochs, input_shape):
        """Train optimized LSTM"""
        if not retrain and os.path.exists(model_path):
            return load_model(model_path)
        
        # Optimized LSTM architecture
        tf.random.set_seed(42)
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(input_shape[1], input_shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='huber',  # More robust to outliers
            metrics=['mae']
        )
        
        es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            epochs=min(epochs, 200),
            batch_size=32,
            validation_split=0.2,
            callbacks=[es],
            verbose=0
        )
        
        model.save(model_path)
        return model

    # Update the main training method to not use the old preprocess_data
    def train_and_evaluate_models(self, stock_code: str, period: str, models_to_train: List[str], 
                                split_ratio: float, epochs: int, retrain: bool = True) -> Dict[str, Any]:
        """Main training and evaluation pipeline with model-specific processing"""
        try:
            # Get data
            stock_service = StockDataService()
            df = stock_service.get_historical_data(stock_code, period)
            
            if df is None or df.empty:
                raise ValueError(f"No data available for {stock_code}")
            
            # Calculate base train size
            train_size = int(len(df) * split_ratio)
            train_size = max(1, min(train_size, len(df) - 1))
            
            # Process each model with its specific data processing
            results = []
            for model_type in models_to_train:
                result = self._train_single_model(
                    model_type, df, train_size, 
                    stock_code, epochs, retrain
                )
                results.append(result)
            
            # Determine overall status
            error_count = sum(1 for r in results if r['status'] == 'error')
            
            if error_count == len(results):
                status = "error"
                message = "All models failed"
            elif error_count > 0:
                status = "partial_success"
                message = f"{error_count}/{len(results)} models failed"
            else:
                status = "success"
                message = "All models completed successfully"
            
            return {
                "status": status,
                "message": message,
                "stock_code": stock_code,
                "period": period,
                "split_ratio": split_ratio,
                "retrain_mode": retrain,
                "results": results
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Pipeline failed: {str(e)}",
                "results": []
            }

    def get_model_status(self, stock_code: str) -> Dict[str, Any]:
        """Get available models for a stock"""
        available_models = []
        
        for model_type in self.supported_models.keys():
            model_paths = [
                os.path.join(self.models_dir, f"{stock_code}_{model_type}_model.pkl"),
                os.path.join(self.models_dir, f"{stock_code}_{model_type}_model.h5")
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    available_models.append({
                        "model_type": model_type,
                        "model_name": model_type.replace("_", " ").title(),
                        "path": model_path,
                        "size_kb": round(os.path.getsize(model_path) / 1024, 2)
                    })
                    break
        
        return {
            "stock_code": stock_code,
            "available_models": available_models,
            "supported_models": list(self.supported_models.keys()),
            "total_available": len(available_models)
        }

# Create singleton instance
ml_service = MLService(
    models_dir=Config.ML_MODELS_DIR,
    scalers_dir=Config.SCALERS_DIR
)