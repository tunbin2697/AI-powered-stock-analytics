import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

class ARIMAService:
    def __init__(self):
        pass

    @staticmethod
    def process_data(df: pd.DataFrame, target_code: str, test_size: float = 0.2):
        """
        Prepare train/test split for ARIMA.
        Returns train_data, test_data, test_dates.
        """
        target_col = f"{target_code}_close_stock"
        df_arima = df[[target_col, 'Date']].copy()
        df_arima.dropna(subset=[target_col], inplace=True)
        df_arima['Date'] = pd.to_datetime(df_arima['Date'])
        df_arima.set_index('Date', inplace=True)

        train_size = int(len(df_arima) * (1 - test_size))
        train_data = df_arima[target_col][:train_size]
        test_data = df_arima[target_col][train_size:]
        test_dates = test_data.index
        return train_data, test_data, test_dates

    @staticmethod
    def train_and_save_model(train_data: pd.Series, order=(5,1,0), model_path='arima_model.joblib'):
        """
        Train ARIMA model and save it.
        """
        model = ARIMA(train_data.values, order=order)
        model_fit = model.fit()
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        joblib.dump(model_fit, model_path)
        return model_fit

    @staticmethod
    def load_and_predict(df: pd.DataFrame, target_code: str, model_path='arima_model.joblib', order=(5,1,0), test_size=0.2, is_test=True):
        """
        Load ARIMA model and predict.
        If is_test=True: predict on test set (walk-forward validation).
        If is_test=False: predict next day's close (inference).
        Returns:
            - If is_test: y_pred, y_true, test_dates
            - If inference: pred_price, pred_date
        """
        train_data, test_data, test_dates = ARIMAService.process_data(df, target_code, test_size)
        if not os.path.exists(model_path):
            ARIMAService.train_and_save_model(train_data, order, model_path)
        model_fit = joblib.load(model_path)

        if is_test:
            # Walk-forward validation on test set
            history = [x for x in train_data.values]
            predictions = []
            for t in range(len(test_data)):
                try:
                    model = ARIMA(history, order=order)
                    model_fit = model.fit()
                    yhat = model_fit.forecast()[0]
                    predictions.append(yhat)
                    history.append(test_data.values[t])
                except Exception as e:
                    if history:
                        predictions.append(history[-1])
                        history.append(test_data.values[t])
                    else:
                        predictions.append(np.nan)
                        history.append(test_data.values[t])
            rmse = np.sqrt(mean_squared_error(test_data.values, predictions))
            print(f'ARIMA RMSE ({target_code}): {rmse:.3f}')
            return np.array(predictions), test_data.values, test_dates
        else:
            # Inference: predict next day's close
            full_series = pd.concat([train_data, test_data])
            model = ARIMA(full_series.values, order=order)
            model_fit = model.fit()
            pred_price = model_fit.forecast(steps=1)[0]
            last_date = full_series.index[-1]
            pred_date = last_date + pd.Timedelta(days=1)
            return pred_price, pred_date

    @staticmethod
    def plot_prediction(dates, y_true, y_pred, title="ARIMA Actual vs Predicted"):
        """
        Plot actual vs predicted values.
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(dates, y_true, label='Actual Price', color='blue')
        ax.plot(dates, y_pred, label='ARIMA Forecast', color='red', linestyle='--')
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        fig.tight_layout()
        return fig