import os
import pickle
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging
from services.stock_data_service import StockDataService
from services.ml_service import MLService
import tensorflow as tf

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.models_dir = "src/models"
        self.ml_service = MLService(models_dir=self.models_dir)
    
    def predict(self, symbol: str, period: str = "1y", days: int = 7, model_type: str = "linear_regression") -> Dict[str, Any]:
        """
        Simple prediction using trained models - no validation
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Historical data period ('1mo', '3mo', '6mo', '1y', '2y', '5y')
            days: Number of days to predict (1-30)
            model_type: Model to use ('linear_regression', 'random_forest_regressor', 'svm_regressor', 'lstm')
        """
        try:
            # Validate inputs
            symbol = symbol.upper().strip()
            days = max(1, min(30, days))
            
            # Get historical data directly - no validation
            df = StockDataService.get_historical_data(symbol, period)
            if df is None or df.empty:
                return {
                    "error": f"No historical data available for {symbol}",
                    "symbol": symbol,
                    "period": period
                }
            
            # Check if model exists
            model_file = f"{symbol}_{model_type}_model"
            model_path = f"{self.models_dir}/{model_file}.pkl"
            if model_type == 'lstm':
                model_path = f"{self.models_dir}/{model_file}.h5"
            
            if not os.path.exists(model_path):
                return {"error": f"No {model_type} model found for {symbol}"}
            
            # Process data using ML service methods
            predictions = self._predict_with_model(symbol, df, model_type, days)
            if predictions is None:
                return {"error": f"Prediction failed for {model_type}"}
            
            # Generate future dates - fix the date handling
            future_dates = self._generate_future_dates(df, days)
            current_price = float(df['Close'].iloc[-1])
            
            # Format response
            result = {
                "symbol": symbol,
                "current_price": current_price,
                "period": period,
                "prediction_days": days,
                "model": model_type,
                "predictions": [
                    {
                        "date": date.strftime('%Y-%m-%d'),
                        "predicted_price": float(price),
                        "day": i + 1,
                        "change_percent": round(((float(price) - current_price) / current_price) * 100, 2)
                    }
                    for i, (date, price) in enumerate(zip(future_dates, predictions))
                ]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            return {
                "error": f"Prediction failed: {str(e)}",
                "symbol": symbol,
                "period": period
            }
    
    def _predict_with_model(self, symbol: str, df: pd.DataFrame, model_type: str, days: int) -> List[float]:
        """Generate predictions using specific model type"""
        try:
            if model_type == 'linear_regression':
                return self._predict_linear_regression(symbol, df, days)
            elif model_type == 'random_forest_regressor':
                return self._predict_random_forest(symbol, df, days)
            elif model_type == 'svm_regressor':
                return self._predict_svm(symbol, df, days)
            elif model_type == 'lstm':
                return self._predict_lstm(symbol, df, days)
            else:
                return None
        except Exception as e:
            logger.error(f"Error predicting with {model_type}: {e}")
            return None
    
    def _predict_linear_regression(self, symbol: str, df: pd.DataFrame, days: int) -> List[float]:
        """Predict using linear regression model"""
        # Process data using ML service
        processed_data, scalers = self.ml_service.process_linear_regression_data(df, symbol, fit_scalers=False)
        if processed_data is None:
            return None
        
        # Load model
        model_path = f"{self.models_dir}/{symbol}_linear_regression_model.pkl"
        model = joblib.load(model_path)
        
        # Get features and make predictions
        features = processed_data.drop(columns=['Close'])
        close_scaler = scalers['Close']
        feature_scaler = scalers['features']
        
        predictions = []
        current_features = features.iloc[-1:].copy()
        
        for _ in range(days):
            # Predict next price
            next_pred_scaled = model.predict(current_features)[0]
            
            # Inverse transform to get actual price
            next_pred = close_scaler.inverse_transform([[next_pred_scaled]])[0][0]
            predictions.append(next_pred)
            
            # Simple feature update (use trend from last few values)
            # This is a simplified approach - in practice you'd update features properly
        
        return predictions
    
    def _predict_random_forest(self, symbol: str, df: pd.DataFrame, days: int) -> List[float]:
        """Predict using random forest model"""
        # Process data using ML service
        processed_data, scalers = self.ml_service.process_random_forest_data(df, symbol, fit_scalers=False)
        if processed_data is None:
            return None
        
        # Load model
        model_path = f"{self.models_dir}/{symbol}_random_forest_regressor_model.pkl"
        model = joblib.load(model_path)
        
        # Get features and make predictions
        features = processed_data.drop(columns=['Close'])
        close_scaler = scalers['Close']
        
        predictions = []
        current_features = features.iloc[-1:].copy()
        
        for _ in range(days):
            next_pred_scaled = model.predict(current_features)[0]
            next_pred = close_scaler.inverse_transform([[next_pred_scaled]])[0][0]
            predictions.append(next_pred)
        
        return predictions
    
    def _predict_svm(self, symbol: str, df: pd.DataFrame, days: int) -> List[float]:
        """Predict using SVM model"""
        # Process data using ML service
        processed_data, scalers = self.ml_service.process_svm_data(df, symbol, fit_scalers=False)
        if processed_data is None:
            return None
        
        # Load model
        model_path = f"{self.models_dir}/{symbol}_svm_regressor_model.pkl"
        model = joblib.load(model_path)
        
        # Get features and make predictions
        features = processed_data.drop(columns=['Close'])
        close_scaler = scalers['Close']
        
        predictions = []
        current_features = features.iloc[-1:].copy()
        
        for _ in range(days):
            next_pred_scaled = model.predict(current_features)[0]
            next_pred = close_scaler.inverse_transform([[next_pred_scaled]])[0][0]
            predictions.append(next_pred)
        
        return predictions
    
    def _predict_lstm(self, symbol: str, df: pd.DataFrame, days: int) -> List[float]:
        """Predict using LSTM model"""
        # Process data using ML service
        processed_data, scalers = self.ml_service.process_lstm_data(df, symbol, fit_scalers=False, look_back=60)
        if processed_data is None:
            return None
        
        # Load model
        model_path = f"{self.models_dir}/{symbol}_lstm_model.h5"
        model = tf.keras.models.load_model(model_path)
        
        # Get the last sequence for prediction
        X, _ = processed_data
        feature_scaler = scalers['features']
        
        # Use last sequence to predict
        last_sequence = X[-1:].copy()  # Get the last sequence
        
        predictions = []
        for _ in range(days):
            # Predict next value
            next_pred_scaled = model.predict(last_sequence, verbose=0)[0][0]
            
            # Create next input sequence
            # Add the prediction to the sequence and remove the oldest value
            next_input = np.zeros((1, last_sequence.shape[1], last_sequence.shape[2]))
            next_input[0, :-1] = last_sequence[0, 1:]  # Shift sequence
            next_input[0, -1, 0] = next_pred_scaled  # Add prediction as Close price
            
            # For other features, use the last known values (simple approach)
            if last_sequence.shape[2] > 1:
                next_input[0, -1, 1:] = last_sequence[0, -1, 1:]
            
            last_sequence = next_input
            
            # Inverse transform to get actual price
            # Create dummy array for inverse transform
            dummy_features = np.zeros((1, feature_scaler.n_features_in_))
            dummy_features[0, 0] = next_pred_scaled
            actual_price = feature_scaler.inverse_transform(dummy_features)[0, 0]
            
            predictions.append(actual_price)
        
        return predictions
    
    def _generate_future_dates(self, df: pd.DataFrame, days: int) -> List[datetime]:
        """Generate future trading dates starting from tomorrow"""
        future_dates = []
        current_date = datetime.now().date()
        days_added = 0
        
        while days_added < days:
            current_date += timedelta(days=1)
            if current_date.weekday() < 5:  # Skip weekends
                future_dates.append(datetime.combine(current_date, datetime.min.time()))
                days_added += 1
        
        return future_dates
    
    def get_available_models(self, symbol: str) -> List[str]:
        """Get list of available models for a symbol"""
        return self.ml_service.get_model_status(symbol)['available_models']