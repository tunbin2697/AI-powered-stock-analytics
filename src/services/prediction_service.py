# import os
# import pickle
# import numpy as np
# import pandas as pd
# import joblib
# from datetime import datetime, timedelta
# from typing import Dict, Any, List
# import logging
# from services.stock_data_service import StockDataService
# from services.ml_service import MLService
# import tensorflow as tf

# logger = logging.getLogger(__name__)

# class PredictionService:
#     def __init__(self):
#             # Get the absolute path to the directory containing this script (services)
#             current_script_dir = os.path.dirname(os.path.abspath(__file__))
#             # Construct the absolute path to the 'src/models' directory
#             # This assumes 'models' is a sibling directory to 'services' parent 'src'
#             # i.e., src/models and src/services
#             self.models_dir = os.path.abspath(os.path.join(current_script_dir, "..", "models"))
#             self.ml_service = MLService(models_dir=self.models_dir)
    
#     def predict(self, symbol: str, period: str = "1y", days: int = 7, model_type: str = "linear_regression") -> Dict[str, Any]:
#         """
#         Simple prediction using trained models - no validation
        
#         Args:
#             symbol: Stock symbol (e.g., 'AAPL')
#             period: Historical data period ('1mo', '3mo', '6mo', '1y', '2y', '5y')
#             days: Number of days to predict (1-30)
#             model_type: Model to use ('linear_regression', 'random_forest_regressor', 'svm_regressor', 'lstm')
#         """
#         try:
#             # Validate inputs
#             symbol = symbol.upper().strip()
#             days = max(1, min(30, days))
            
#             # Get historical data directly - no validation
#             df = StockDataService.get_historical_data(symbol, period)
#             if df is None or df.empty:
#                 return {
#                     "error": f"No historical data available for {symbol}",
#                     "symbol": symbol,
#                     "period": period
#                 }
            
#             # Check if model exists
#             model_file = f"{symbol}_{model_type}_model"
#             model_path = f"{self.models_dir}/{model_file}.pkl"
#             if model_type == 'lstm':
#                 model_path = f"{self.models_dir}/{model_file}.h5"
            
#             if not os.path.exists(model_path):
#                 return {"error": f"No {model_type} model found for {symbol}"}
            
#             # Process data using ML service methods
#             predictions = self._predict_with_model(symbol, df, model_type, days)
#             if predictions is None:
#                 return {"error": f"Prediction failed for {model_type}"}
            
#             # Generate future dates - fix the date handling
#             future_dates = self._generate_future_dates(df, days)
#             current_price = float(df['Close'].iloc[-1])
            
#             # Format response
#             result = {
#                 "symbol": symbol,
#                 "current_price": current_price,
#                 "period": period,
#                 "prediction_days": days,
#                 "model": model_type,
#                 "predictions": [
#                     {
#                         "date": date.strftime('%Y-%m-%d'),
#                         "predicted_price": float(price),
#                         "day": i + 1,
#                         "change_percent": round(((float(price) - current_price) / current_price) * 100, 2)
#                     }
#                     for i, (date, price) in enumerate(zip(future_dates, predictions))
#                 ]
#             }
            
#             return result
            
#         except Exception as e:
#             logger.error(f"Prediction error for {symbol}: {e}")
#             return {
#                 "error": f"Prediction failed: {str(e)}",
#                 "symbol": symbol,
#                 "period": period
#             }
    
#     # Thêm/thay thế vào prediction_service.py

#     def _update_features_for_next_step(self, last_feature_row: pd.DataFrame, new_predicted_price: float, last_close_price: float) -> pd.DataFrame:
#         # ... (giữ nguyên như hướng dẫn trước)
#         next_feature_row = last_feature_row.copy()
#         if 'Close_Lag_1' in next_feature_row.columns:
#             next_feature_row['Close_Lag_1'] = last_close_price
#         if 'Close_Lag_2' in next_feature_row.columns:
#             next_feature_row['Close_Lag_2'] = last_feature_row['Close_Lag_1'].values[0]
#         if 'Open' in next_feature_row.columns:
#             next_feature_row['Open'] = last_close_price
#         if 'High' in next_feature_row.columns:
#             next_feature_row['High'] = new_predicted_price * 1.01
#         if 'Low' in next_feature_row.columns:
#             next_feature_row['Low'] = new_predicted_price * 0.99
#         return next_feature_row

#     def _predict_sklearn_generic(self, symbol: str, df: pd.DataFrame, days: int, model_type: str) -> List[float]:
#         """Hàm dự đoán chung cho các mô hình sklearn với logic cập nhật feature."""
#         process_func = getattr(self.ml_service, f"process_{model_type.replace('_regressor', '')}_data")
#         processed_data, scalers = process_func(df, symbol, fit_scalers=False)
#         if processed_data is None: return None
        
#         model_path = f"{self.models_dir}/{symbol}_{model_type}_model.pkl"
#         model = joblib.load(model_path)
        
#         features = processed_data.drop(columns=['Close'])
#         close_scaler = scalers['Close']
        
#         predictions = []
#         current_features = features.iloc[-1:].copy()
#         last_close_price_scaled = processed_data['Close'].iloc[-1]
        
#         for _ in range(days):
#             next_pred_scaled = model.predict(current_features)[0]
#             next_pred_unscaled = close_scaler.inverse_transform(np.array(next_pred_scaled).reshape(1, -1))[0][0]
#             predictions.append(next_pred_unscaled)
            
#             last_close_unscaled = close_scaler.inverse_transform(np.array(last_close_price_scaled).reshape(1, -1))[0][0]
#             current_features = self._update_features_for_next_step(current_features, next_pred_unscaled, last_close_unscaled)
#             last_close_price_scaled = next_pred_scaled
                
#         return predictions

#     def _predict_with_model(self, symbol: str, df: pd.DataFrame, model_type: str, days: int) -> List[float]:
#         """Generate predictions using specific model type"""
#         try:
#             if model_type == 'linear_regression':
#                 return self._predict_linear_regression(symbol, df, days)
#             elif model_type == 'random_forest_regressor':
#                 return self._predict_random_forest(symbol, df, days)
#             elif model_type == 'svm_regressor':
#                 return self._predict_svm(symbol, df, days)
#             elif model_type == 'lstm':
#                 return self._predict_lstm(symbol, df, days)
#             else:
#                 return None
#         except Exception as e:
#             logger.error(f"Error predicting with {model_type}: {e}")
#             return None
    
#     def _predict_linear_regression(self, symbol: str, df: pd.DataFrame, days: int) -> List[float]:
#         """Predict using linear regression model"""
#         # Process data using ML service
#         processed_data, scalers = self.ml_service.process_linear_regression_data(df, symbol, fit_scalers=False)
#         if processed_data is None:
#             return None
        
#         # Load model
#         model_path = f"{self.models_dir}/{symbol}_linear_regression_model.pkl"
#         model = joblib.load(model_path)
        
#         # Get features and make predictions
#         features = processed_data.drop(columns=['Close'])
#         close_scaler = scalers['Close']
#         feature_scaler = scalers['features']
        
#         predictions = []
#         current_features = features.iloc[-1:].copy()
        
#         for _ in range(days):
#             # Predict next price
#             next_pred_scaled = model.predict(current_features)[0]
            
#             # Inverse transform to get actual price
#             next_pred = close_scaler.inverse_transform([[next_pred_scaled]])[0][0]
#             predictions.append(next_pred)
            
#             # Simple feature update (use trend from last few values)
#             # This is a simplified approach - in practice you'd update features properly
        
#         return predictions
    
#     def _predict_random_forest(self, symbol: str, df: pd.DataFrame, days: int) -> List[float]:
#         """Predict using random forest model"""
#         # Process data using ML service
#         processed_data, scalers = self.ml_service.process_random_forest_data(df, symbol, fit_scalers=False)
#         if processed_data is None:
#             return None
        
#         # Load model
#         model_path = f"{self.models_dir}/{symbol}_random_forest_regressor_model.pkl"
#         model = joblib.load(model_path)
        
#         # Get features and make predictions
#         features = processed_data.drop(columns=['Close'])
#         close_scaler = scalers['Close']
        
#         predictions = []
#         current_features = features.iloc[-1:].copy()
        
#         for _ in range(days):
#             next_pred_scaled = model.predict(current_features)[0]
#             next_pred = close_scaler.inverse_transform([[next_pred_scaled]])[0][0]
#             predictions.append(next_pred)
        
#         return predictions
    
#     def _predict_svm(self, symbol: str, df: pd.DataFrame, days: int) -> List[float]:
#         """Predict using SVM model"""
#         # Process data using ML service
#         processed_data, scalers = self.ml_service.process_svm_data(df, symbol, fit_scalers=False)
#         if processed_data is None:
#             return None
        
#         # Load model
#         model_path = f"{self.models_dir}/{symbol}_svm_regressor_model.pkl"
#         model = joblib.load(model_path)
        
#         # Get features and make predictions
#         features = processed_data.drop(columns=['Close'])
#         close_scaler = scalers['Close']
        
#         predictions = []
#         current_features = features.iloc[-1:].copy()
        
#         for _ in range(days):
#             next_pred_scaled = model.predict(current_features)[0]
#             next_pred = close_scaler.inverse_transform([[next_pred_scaled]])[0][0]
#             predictions.append(next_pred)
        
#         return predictions
    
#     def _predict_lstm(self, symbol: str, df: pd.DataFrame, days: int) -> List[float]:
#         """Predict using LSTM model"""
#         # Process data using ML service
#         processed_data, scalers = self.ml_service.process_lstm_data(df, symbol, fit_scalers=False, look_back=60)
#         if processed_data is None:
#             return None
        
#         # Load model
#         model_path = f"{self.models_dir}/{symbol}_lstm_model.h5"
#         model = tf.keras.models.load_model(model_path)
        
#         # Get the last sequence for prediction
#         X, _ = processed_data
#         feature_scaler = scalers['features']
        
#         # Use last sequence to predict
#         last_sequence = X[-1:].copy()  # Get the last sequence
        
#         predictions = []
#         for _ in range(days):
#             # Predict next value
#             next_pred_scaled = model.predict(last_sequence, verbose=0)[0][0]
            
#             # Create next input sequence
#             # Add the prediction to the sequence and remove the oldest value
#             next_input = np.zeros((1, last_sequence.shape[1], last_sequence.shape[2]))
#             next_input[0, :-1] = last_sequence[0, 1:]  # Shift sequence
#             next_input[0, -1, 0] = next_pred_scaled  # Add prediction as Close price
            
#             # For other features, use the last known values (simple approach)
#             if last_sequence.shape[2] > 1:
#                 next_input[0, -1, 1:] = last_sequence[0, -1, 1:]
            
#             last_sequence = next_input
            
#             # Inverse transform to get actual price
#             # Create dummy array for inverse transform
#             dummy_features = np.zeros((1, feature_scaler.n_features_in_))
#             dummy_features[0, 0] = next_pred_scaled
#             actual_price = feature_scaler.inverse_transform(dummy_features)[0, 0]
            
#             predictions.append(actual_price)
        
#         return predictions
    
#     def _generate_future_dates(self, df: pd.DataFrame, days: int) -> List[datetime]:
#         """Generate future trading dates starting from tomorrow"""
#         future_dates = []
#         current_date = datetime.now().date()
#         days_added = 0
        
#         while days_added < days:
#             current_date += timedelta(days=1)
#             if current_date.weekday() < 5:  # Skip weekends
#                 future_dates.append(datetime.combine(current_date, datetime.min.time()))
#                 days_added += 1
        
#         return future_dates
    
#     def get_available_models(self, symbol: str) -> List[str]:
#         """Get list of available models for a symbol"""
#         return self.ml_service.get_model_status(symbol)['available_models']

# import os
# import pickle
# import numpy as np
# import pandas as pd
# import joblib
# from datetime import datetime, timedelta
# from typing import Dict, Any, List
# import logging
# from .stock_data_service import StockDataService
# from .ml_service import MLService
# from config.settings import Config

# logger = logging.getLogger(__name__)

# class PredictionService:
#     def __init__(self):
#         self.models_dir = Config.ML_MODELS_DIR
#         self.scalers_dir = Config.SCALERS_DIR
#         self.ml_service = MLService(models_dir=self.models_dir, scalers_dir=self.scalers_dir)
    
#     # Thay thế hàm predict trong prediction_service.py
#     def predict(self, symbol: str, period: str = "1y", days: int = 7, model_type: str = "linear_regression") -> Dict[str, Any]:
#         try:
#             symbol = symbol.upper().strip()
#             days = max(1, min(30, days))

#             model_path = os.path.join(self.models_dir, f"{symbol}_{model_type}_model.pkl")
#             if not os.path.exists(model_path):
#                 return {"error": f"Model for '{model_type}' not found. Please train it first."}
            
#             hist_df = StockDataService.get_historical_data(symbol, period="6mo") 
#             if hist_df is None or hist_df.empty:
#                 return {"error": f"Could not retrieve historical data for {symbol}."}

#             # Xác định hàm dự đoán sẽ gọi
#             if model_type == 'arima':
#                 predictions = self._predict_arima(symbol, hist_df, days)
#             else:
#                 predictions = self._predict_sklearn_iterative(symbol, hist_df, days, model_type)
            
#             if not predictions:
#                 return {"error": "Prediction failed."}

#             current_price = float(hist_df['Close'].iloc[-1])
#             future_dates = self._generate_future_dates(hist_df, days)
            
#             # === FIX LỖI MÀN HÌNH TRẮNG ===
#             # Thêm lại các trường `day` và `change_percent` mà frontend đang cần
#             formatted_predictions = []
#             for i, (date, price) in enumerate(zip(future_dates, predictions)):
#                 formatted_predictions.append({
#                     "date": date.strftime('%Y-%m-%d'),
#                     "predicted_price": float(price),
#                     "day": i + 1,
#                     "change_percent": round(((float(price) - current_price) / current_price) * 100, 2)
#                 })

#             return {
#                 "symbol": symbol,
#                 "current_price": current_price,
#                 "prediction_days": days,
#                 "model": model_type,
#                 "predictions": formatted_predictions  # Trả về dữ liệu đã được định dạng đầy đủ
#             }
            
#         except Exception as e:
#             logger.error(f"Prediction error for {symbol} with {model_type}: {e}", exc_info=True)
#             return {"error": f"An unexpected error occurred: {str(e)}"}

#     # Thay thế hàm _predict_arima trong prediction_service.py
#     def _predict_arima(self, symbol: str, df: pd.DataFrame, days: int) -> List[float]:
#         """Dự đoán sử dụng mô hình ARIMA đã lưu."""
#         # 1. Tải model
#         model_path = os.path.join(self.models_dir, f"{symbol}_arima_model.pkl")
#         model_fit = joblib.load(model_path)

#         # 2. Dự đoán các giá trị sai phân (differenced values)
#         forecast_diff = model_fit.forecast(steps=days)
        
#         # 3. Đảo ngược phép biến đổi (log -> diff)
#         # Lấy giá trị log cuối cùng của giá đóng cửa
#         last_log_close = np.log(df['Close'].iloc[-1])
        
#         # Cộng dồn các giá trị dự đoán vào giá trị log cuối cùng
#         last_val = last_log_close
#         log_predictions = []
#         for diff_val in forecast_diff:
#             last_val += diff_val
#             log_predictions.append(last_val)
            
#         # 4. Đảo ngược log để có giá trị thực
#         predictions = np.exp(log_predictions)
#         return predictions.tolist()

#     def _predict_sklearn_iterative(self, symbol: str, df: pd.DataFrame, days: int, model_type: str) -> List[float]:
#         # 1. Tải model và các thành phần cần thiết
#         model = joblib.load(os.path.join(self.models_dir, f"{symbol}_{model_type}_model.pkl"))
#         feature_scaler = joblib.load(os.path.join(self.scalers_dir, f"{symbol}_{model_type}_feature_scaler.pkl"))
#         target_scaler = joblib.load(os.path.join(self.scalers_dir, f"{symbol}_{model_type}_target_scaler.pkl"))
#         feature_cols = joblib.load(os.path.join(self.scalers_dir, f"{symbol}_{model_type}_feature_cols.pkl"))

#         # 2. Tạo một bản sao của dataframe để thực hiện dự đoán lặp
#         df_dynamic = df.copy()
#         predictions_unscaled = []

#         for _ in range(days):
#             # 3. Thêm features cho toàn bộ dữ liệu hiện có
#             df_featured = self.ml_service._add_technical_features(df_dynamic)
            
#             # 4. Lấy dòng dữ liệu cuối cùng để dự đoán
#             last_features_row = df_featured[feature_cols].iloc[-1:]
            
#             # 5. Scale features
#             last_features_scaled = feature_scaler.transform(last_features_row)
            
#             # 6. Dự đoán giá trị đã scale
#             next_pred_scaled = model.predict(last_features_scaled)
            
#             # 7. Đảo ngược scale để có giá trị thực
#             next_pred_unscaled = target_scaler.inverse_transform(next_pred_scaled.reshape(-1, 1)).flatten()[0]
#             predictions_unscaled.append(next_pred_unscaled)
            
#             # 8. Cập nhật dataframe với giá trị dự đoán mới để dùng cho vòng lặp sau
#             last_row = df_dynamic.iloc[-1:].copy()
#             last_date = pd.to_datetime(last_row.index[0])
#             next_day = last_date + timedelta(days=1)
            
#             # Tạo dòng mới với giá dự đoán
#             new_row = pd.DataFrame({
#                 'Open': [last_row['Close'].values[0]],
#                 'High': [next_pred_unscaled * 1.01], # Ước tính
#                 'Low': [next_pred_unscaled * 0.99],  # Ước tính
#                 'Close': [next_pred_unscaled],
#                 'Volume': [last_row['Volume'].values[0]] # Giữ nguyên volume
#             }, index=[next_day])

#             df_dynamic = pd.concat([df_dynamic, new_row])

#         return predictions_unscaled

#     def _generate_future_dates(self, df: pd.DataFrame, days: int) -> List[datetime]:
#         future_dates = []
#         last_date = pd.to_datetime(df.index[-1]).date()
#         current_date = last_date
#         while len(future_dates) < days:
#             current_date += timedelta(days=1)
#             # Bỏ qua cuối tuần
#             if current_date.weekday() < 5:
#                 future_dates.append(datetime.combine(current_date, datetime.min.time()))
#         return future_dates

#     def get_available_models(self, symbol: str) -> List[str]:
# #         """Get list of available models for a symbol"""
#         return self.ml_service.get_model_status(symbol)['available_models']

# # Singleton instance
# prediction_service = PredictionService()

import os
import pickle
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging
from .stock_data_service import StockDataService
from .ml_service import MLService
from config.settings import Config

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.models_dir = Config.ML_MODELS_DIR
        self.scalers_dir = Config.SCALERS_DIR
        self.ml_service = MLService(models_dir=self.models_dir, scalers_dir=self.scalers_dir)
    
    # def predict(self, symbol: str, period: str = "1y", days: int = 7, model_type: str = "linear_regression") -> Dict[str, Any]:
    #     try:
    #         symbol = symbol.upper().strip()
    #         days = max(1, min(30, days))

    #         model_path = os.path.join(self.models_dir, f"{symbol}_{model_type}_model.pkl")
    #         if not os.path.exists(model_path):
    #             return {"error": f"Model for '{model_type}' not found. Please train it first."}
            
    #         # === FIX LỖI DỰ ĐOÁN ===
    #         # Lấy 2 năm dữ liệu để đảm bảo tính được tất cả các features
    #         hist_df = StockDataService.get_historical_data(symbol, period="2y") 
    #         if hist_df is None or len(hist_df) < 60: # Cần ít nhất 60 ngày
    #             return {"error": f"Not enough historical data to make a prediction for {symbol}."}

    #         predictions = []
    #         if model_type == 'arima':
    #             predictions = self._predict_arima(symbol, hist_df, days)
    #         elif model_type in ['linear_regression', 'random_forest_regressor', 'svm_regressor']:
    #             predictions = self._predict_sklearn_iterative(symbol, hist_df, days, model_type)
    #         else:
    #             return {"error": f"Model type '{model_type}' is not supported for prediction."}

    #         if not predictions:
    #              return {"error": "Prediction failed. The prediction function returned no results."}

    #         current_price = float(hist_df['Close'].iloc[-1])
    #         future_dates = self._generate_future_dates(hist_df, days)
            
    #         formatted_predictions = [
    #             {
    #                 "date": date.strftime('%Y-%m-%d'),
    #                 "predicted_price": float(price),
    #                 "day": i + 1,
    #                 "change_percent": round(((float(price) - current_price) / current_price) * 100, 2)
    #             }
    #             for i, (date, price) in enumerate(zip(future_dates, predictions))
    #         ]

    #         return {
    #             "symbol": symbol, "current_price": current_price, "prediction_days": days, "model": model_type,
    #             "predictions": formatted_predictions
    #         }
            
    #     except Exception as e:
    #         logger.error(f"Prediction error for {symbol} with {model_type}: {e}", exc_info=True)
    #         return {"error": f"An unexpected error occurred during prediction: {str(e)}"}

    def predict(self, symbol: str, period: str = "1y", days: int = 7, model_type: str = "linear_regression") -> Dict[str, Any]:
        try:
            symbol = symbol.upper().strip()
            days = max(1, min(30, days))

            # Sửa lỗi đường dẫn cho model LSTM
            model_file = f"{symbol}_{model_type}_model"
            model_path = os.path.join(self.models_dir, f"{model_file}.h5" if model_type == 'lstm' else f"{model_file}.pkl")

            if not os.path.exists(model_path):
                return {"error": f"Model for '{model_type}' not found. Please train it first."}

            hist_df = StockDataService.get_historical_data(symbol, period="2y") 
            if hist_df is None or len(hist_df) < 60:
                return {"error": f"Not enough historical data to make a prediction for {symbol}."}

            predictions = []
            if model_type == 'arima':
                predictions = self._predict_arima(symbol, hist_df, days)
            # === THÊM LOGIC GỌI HÀM DỰ ĐOÁN LSTM ===
            elif model_type == 'lstm':
                predictions = self._predict_lstm_iterative(symbol, hist_df, days)
            # =========================================
            elif model_type in ['linear_regression', 'random_forest_regressor', 'svm_regressor']:
                predictions = self._predict_sklearn_iterative(symbol, hist_df, days, model_type)
            else:
                return {"error": f"Model type '{model_type}' is not supported for prediction."}

            if not predictions:
                return {"error": "Prediction failed. The prediction function returned no results."}

            current_price = float(hist_df['Close'].iloc[-1])
            future_dates = self._generate_future_dates(hist_df, days)

            # Cần import datetime ở đầu file prediction_service.py
            formatted_predictions = [
                {
                    "date": date.strftime('%Y-%m-%d'),
                    "predicted_price": float(price),
                    "day": i + 1,
                    "change_percent": round(((float(price) - current_price) / current_price) * 100, 2)
                }
                for i, (date, price) in enumerate(zip(future_dates, predictions))
            ]

            return {
                "symbol": symbol, "current_price": current_price, "prediction_days": days, "model": model_type,
                "predictions": formatted_predictions
            }

        except Exception as e:
            logger.error(f"Prediction error for {symbol} with {model_type}: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred during prediction: {str(e)}"}

    # def _predict_sklearn_iterative(self, symbol: str, df: pd.DataFrame, days: int, model_type: str) -> List[float]:
    #     try:
    #         model = joblib.load(os.path.join(self.models_dir, f"{symbol}_{model_type}_model.pkl"))
    #         feature_scaler = joblib.load(os.path.join(self.scalers_dir, f"{symbol}_{model_type}_feature_scaler.pkl"))
    #         target_scaler = joblib.load(os.path.join(self.scalers_dir, f"{symbol}_{model_type}_target_scaler.pkl"))
    #         feature_cols = joblib.load(os.path.join(self.scalers_dir, f"{symbol}_{model_type}_feature_cols.pkl"))

    #         df_dynamic = df.copy()
    #         predictions_unscaled = []

    #         for _ in range(days):
    #             df_featured = self.ml_service._add_technical_features(df_dynamic)
    #             last_features_row = df_featured[feature_cols].iloc[-1:]

    #             if last_features_row.isnull().values.any():
    #                 raise ValueError("NaN values found in features before prediction. Check historical data.")

    #             last_features_scaled = feature_scaler.transform(last_features_row)
    #             next_pred_scaled = model.predict(last_features_scaled)
    #             next_pred_unscaled = target_scaler.inverse_transform(next_pred_scaled.reshape(-1, 1)).flatten()[0]
    #             predictions_unscaled.append(next_pred_unscaled)
                
    #             last_row = df_dynamic.iloc[-1:].copy()
    #             # Dùng index của df_dynamic thay vì tạo mới để tránh lỗi timezone
    #             next_day = last_row.index[0] + timedelta(days=1)
                
    #             new_row = pd.DataFrame({
    #                 'Open': [last_row['Close'].values[0]], 'High': [next_pred_unscaled * 1.01],
    #                 'Low': [next_pred_unscaled * 0.99], 'Close': [next_pred_unscaled],
    #                 'Volume': [last_row['Volume'].values[0]]
    #             }, index=[next_day])

    #             df_dynamic = pd.concat([df_dynamic, new_row])

    #         return predictions_unscaled
    #     except Exception as e:
    #         logger.error(f"Error inside _predict_sklearn_iterative for {model_type}: {e}", exc_info=True)
    #         return [] # Trả về list rỗng nếu có lỗi

    def _predict_lstm_iterative(self, symbol: str, df: pd.DataFrame, days: int, look_back: int = 60) -> List[float]:
        try:
            from keras.models import load_model

            model_path = os.path.join(self.models_dir, f"{symbol}_lstm_model.h5")
            model = load_model(model_path)

            scaler_path = os.path.join(self.scalers_dir, f"{symbol}_lstm_scaler.pkl")
            scaler = joblib.load(scaler_path)

            # Lấy dữ liệu giá đóng cửa và scale nó
            close_prices = df['Close'].values.reshape(-1, 1)
            scaled_data = scaler.transform(close_prices)

            # Lấy chuỗi cuối cùng từ dữ liệu lịch sử để bắt đầu dự đoán
            last_sequence = scaled_data[-look_back:]

            predictions_scaled = []

            for _ in range(days):
                # Reshape lại cho đúng input của LSTM (1, look_back, 1)
                current_input = np.reshape(last_sequence, (1, look_back, 1))

                # Dự đoán giá trị scale tiếp theo
                next_pred_scaled = model.predict(current_input)[0][0]
                predictions_scaled.append(next_pred_scaled)

                # Cập nhật chuỗi: bỏ giá trị cũ nhất, thêm giá trị mới dự đoán vào cuối
                last_sequence = np.append(last_sequence[1:], [[next_pred_scaled]], axis=0)

            # Đảo ngược scale để có giá trị thực
            predictions_unscaled = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))

            return predictions_unscaled.flatten().tolist()

        except Exception as e:
            logger.error(f"Error inside _predict_lstm_iterative: {e}", exc_info=True)
            return []

    def _predict_sklearn_iterative(self, symbol: str, df: pd.DataFrame, days: int, model_type: str) -> List[float]:
        try:
            model = joblib.load(os.path.join(self.models_dir, f"{symbol}_{model_type}_model.pkl"))
            feature_scaler = joblib.load(os.path.join(self.scalers_dir, f"{symbol}_{model_type}_feature_scaler.pkl"))
            target_scaler = joblib.load(os.path.join(self.scalers_dir, f"{symbol}_{model_type}_target_scaler.pkl"))
            feature_cols = joblib.load(os.path.join(self.scalers_dir, f"{symbol}_{model_type}_feature_cols.pkl"))

            # Sử dụng df.copy() để tránh SettingWithCopyWarning
            df_dynamic = df.copy()
            predictions_unscaled = []

            for _ in range(days):
                df_featured = self.ml_service._add_technical_features(df_dynamic)
                # Đảm bảo các cột có trong features trước khi dự đoán
                last_features_row = df_featured[feature_cols].iloc[-1:]

                if last_features_row.isnull().values.any():
                    # Nếu có NaN, không thể dự đoán, dừng vòng lặp
                    break

                last_features_scaled = feature_scaler.transform(last_features_row)
                next_pred_scaled = model.predict(last_features_scaled)
                next_pred_unscaled = target_scaler.inverse_transform(next_pred_scaled.reshape(-1, 1)).flatten()[0]
                predictions_unscaled.append(next_pred_unscaled)

                # Lấy dòng cuối cùng của dataframe để tạo dòng mới
                last_row = df_dynamic.iloc[-1]

                # === SỬA LỖI LOGIC NGÀY THÁNG BÊN TRONG VÒNG LẶP ===
                # Lấy ngày cuối cùng từ cột 'Date' và cộng thêm một ngày
                next_day = pd.to_datetime(last_row['Date']) + timedelta(days=1)

                # Tạo dòng mới (new_row) dưới dạng DataFrame
                new_row_df = pd.DataFrame({
                    'Date': [next_day],  # Giá trị phải là một list
                    'Open': [last_row['Close']], 
                    'High': [next_pred_unscaled * 1.01],
                    'Low': [next_pred_unscaled * 0.99], 
                    'Close': [next_pred_unscaled],
                    'Volume': [last_row['Volume']]
                })

                # Nối DataFrame mới vào df_dynamic
                df_dynamic = pd.concat([df_dynamic, new_row_df], ignore_index=True)

            return predictions_unscaled
        except Exception as e:
            logger.error(f"Error inside _predict_sklearn_iterative for {model_type}: {e}", exc_info=True)
            return [] # Trả về list rỗng nếu có lỗi

    def _predict_arima(self, symbol: str, df: pd.DataFrame, days: int) -> List[float]:
        # ... (Hàm này giữ nguyên như lần trước)
        model_path = os.path.join(self.models_dir, f"{symbol}_arima_model.pkl")
        model_fit = joblib.load(model_path)
        forecast_diff = model_fit.forecast(steps=days)
        last_log_close = np.log(df['Close'].iloc[-1])
        last_val = last_log_close
        log_predictions = []
        for diff_val in forecast_diff:
            last_val += diff_val
            log_predictions.append(last_val)
        predictions = np.exp(log_predictions)
        return predictions.tolist()


    # def _generate_future_dates(self, df: pd.DataFrame, days: int) -> List[datetime]:
    #     # ... (Hàm này giữ nguyên như lần trước)
    #     future_dates = []
    #     last_date = df.index[-1].date()
    #     current_date = last_date
    #     while len(future_dates) < days:
    #         current_date += timedelta(days=1)
    #         if current_date.weekday() < 5:
    #             future_dates.append(datetime.combine(current_date, datetime.min.time()))
    #     return future_dates
    def _generate_future_dates(self, df: pd.DataFrame, days: int) -> List[datetime]:
        future_dates = []
        # === SỬA LỖI: LẤY NGÀY TỪ CỘT 'Date' THAY VÌ INDEX ===
        last_date = pd.to_datetime(df['Date'].iloc[-1]).date()
        # ========================================================
        current_date = last_date
        while len(future_dates) < days:
            current_date += timedelta(days=1)
            if current_date.weekday() < 5:
                future_dates.append(datetime.combine(current_date, datetime.min.time()))
        return future_dates

prediction_service = PredictionService()