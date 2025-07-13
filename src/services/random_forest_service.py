import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os

class RandomForestService:
    def __init__(self):
        pass
      
    @staticmethod
    def process_rf_features(merged_df, target_code, is_training=True, test_size=0.2):
        """
        Prepares data and scales features. The model will predict the price change.

        Args:
            merged_df (pd.DataFrame): Full data set with stock info.
            target_code (str): e.g., 'AAPL'.
            is_training (bool): Split train/test if True.
            test_size (float): Proportion of test data.

        Returns:
            For training:
                X_train_scaled, X_test_scaled, y_train_diff, y_test_diff, scaler_X, date_test, y_test_actual
            For inference:
                X_scaled, last_price, scaler_X
        """
        df = merged_df.copy()
        target_col = f"{target_code}_close_stock"
        df["target_diff"] = df[target_col].diff().shift(-1)  # predict next-day change
        df.dropna(inplace=True)

        features = [col for col in df.columns if col not in ['Date', target_col, 'target_diff']]
        X = df[features]
        y_diff = df["target_diff"]
        dates = df["Date"]
        y_actual = df[target_col]

        scaler_X = StandardScaler()

        if is_training:
            X_train, X_test, y_train_diff, y_test_diff, _, date_test = train_test_split(
                X, y_diff, dates, test_size=test_size, shuffle=False
            )
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            y_test_actual = y_actual.loc[y_test_diff.index]  # original price for plotting
            return X_train_scaled, X_test_scaled, y_train_diff, y_test_diff, scaler_X, date_test, y_test_actual
        else:
            latest_row = df.tail(1)
            X_scaled = scaler_X.fit_transform(latest_row[features])
            last_price = latest_row[target_col].values[0]
            return X_scaled, last_price, scaler_X

    @staticmethod
    def train_and_save_rf_model(X_train, y_train, scaler_X, model_path='rf_model.joblib', scaler_path='rf_scaler.joblib'):
        """
        Trains a RandomForest model and saves it with the feature scaler.

        Args:
            X_train (np.ndarray): Scaled feature matrix.
            y_train (np.ndarray): Target diffs.
            scaler_X (StandardScaler): Fitted scaler.
        """
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        joblib.dump(model, model_path)
        joblib.dump(scaler_X, scaler_path)

        print(f"Model saved to {model_path} and scaler to {scaler_path}")
        print(f"Training R²: {model.score(X_train, y_train):.4f}")

    def load_rf_model_and_predict(self, data_df, target_code, model_path='rf_model.joblib', scaler_path='rf_scaler.joblib', is_test=False):
        """
        Loads the model and scaler, predicts either on test set or the latest row.

        Args:
            data_df (pd.DataFrame): Full dataset.
            target_code (str): Ticker.
            is_test (bool): Predict on test set if True.

        Returns:
            y_pred_prices, y_actual_prices, date_series
        """
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Model or scaler not found.")

        model = joblib.load(model_path)
        scaler_X = joblib.load(scaler_path)

        if is_test:
            _, X_test_scaled, _, y_test_diff, _, date_test, y_test_actual = self.process_rf_features(data_df, target_code, is_training=True)
            y_pred_diff = model.predict(X_test_scaled)
            y_pred_price = y_test_actual.values - y_test_diff.values + y_pred_diff  # reconstruct predicted price
            return y_pred_price, y_test_actual.values, date_test
        else:
            X_latest_scaled, last_price, _ = self.process_rf_features(data_df, target_code, is_training=False)
            y_pred_diff = model.predict(X_latest_scaled)[0]
            predicted_price = last_price + y_pred_diff
            return predicted_price

    @staticmethod
    def plot_rf_prediction(y_true, y_pred, date_series, title="Actual vs Predicted Close (Random Forest delta Close)"):
        """
        Plots actual and predicted prices.

        Args:
            y_true (array): Ground truth close prices.
            y_pred (array): Predicted close prices.
            date_series (array): Corresponding dates.
        """
        r2 = r2_score(y_true, y_pred)
        print(f"Test R² score: {r2:.4f}")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(date_series, y_true, label="Actual Close", color='black')
        ax.plot(date_series, y_pred, label="Predicted Close", color='green')
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        return fig

# target_code = 'AAPL'
# X_train, X_test, y_train_diff, y_test_diff, scaler_X, date_test, y_test_actual = process_rf_features(merged_df, target_code, is_training=True)
# train_and_save_rf_model(X_train, y_train_diff, scaler_X)

# y_pred_prices, y_actual_prices, date_series = load_rf_model_and_predict(merged_df, target_code, is_test=True)
# plot_rf_prediction(y_actual_prices, y_pred_prices, date_series)

# predicted_price = load_rf_model_and_predict(merged_df, target_code, is_test=False)
# print(f"Predicted next-day close price: {predicted_price:.2f}")
