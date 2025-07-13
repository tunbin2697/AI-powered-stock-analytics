  
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os

class LinearRegressionService:
  def __init__(self):
    pass
    
# 1) PROCESS DATA — train/test split & scaling, or extract latest raw row for inference
  @staticmethod
  def process_data( df: pd.DataFrame,
                target_code: str,
                is_training: bool = True,
                test_size: float = 0.2):
    df = df.copy()
    target_col = f"{target_code}_close_stock"
    features = [c for c in df.columns if c not in ("Date", target_col)]
    X, y, dates = df[features], df[target_col], df["Date"]

    if is_training:
        # time‐series split
        X_tr, X_te, y_tr, y_te, _, dates_te = train_test_split(
            X, y, dates, test_size=test_size, shuffle=False
        )
        scaler = StandardScaler().fit(X_tr)
        return (scaler.transform(X_tr),
                scaler.transform(X_te),
                y_tr,
                y_te,
                scaler,
                dates_te)

    # inference: grab last row
    latest = df.tail(1)
    X_raw = latest[features]
    last_date = latest["Date"].iloc[0]
    return X_raw, last_date

  # 2) TRAIN & SAVE
  @staticmethod
  def train_and_save_model(X_train: np.ndarray,
                          y_train: pd.Series,
                          scaler: StandardScaler,
                          model_path: str = "lr_model.joblib",
                          scaler_path: str = "lr_scaler.joblib"):
      model = LinearRegression().fit(X_train, y_train)
      os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
      joblib.dump(model, model_path)
      joblib.dump(scaler, scaler_path)
      print(f"Saved model → {model_path}")
      print(f"Saved scaler → {scaler_path}")
      print(f"Train R² = {model.score(X_train, y_train):.4f}")

  # 3) LOAD & PREDICT — always accepts merged_df
  def load_and_predict(self, df: pd.DataFrame,
                      target_code: str,
                      model_path: str ,
                      scaler_path: str,
                      is_test: bool = False):
      model  = joblib.load(model_path)
      scaler = joblib.load(scaler_path)

      if is_test:
          # regenerate test split
          _, X_te, _, y_te, _, dates_te = self.process_data(df, target_code, is_training=True)
          y_pred = model.predict(X_te)
          return y_pred, y_te.values, dates_te

      # inference: get last row raw, scale, predict one step
      X_raw, last_date = self.process_data(df, target_code, is_training=False)
      X_s   = scaler.transform(X_raw)
      y_pred = model.predict(X_s)[0]
      return y_pred, last_date

  # 4) PLOT ACTUAL vs PREDICTED (test‑set only)
  @staticmethod
  def create_actual_vs_predicted_figure(y_true, y_pred, dates, title="Actual vs Predicted"):
      """
      Create a line plot comparing actual vs predicted values over time.

      Args:
          y_true (array-like): True values.
          y_pred (array-like): Predicted values.
          dates (array-like): Corresponding datetime values.
          title (str): Title for the plot.

      Returns:
          matplotlib.figure.Figure: The resulting figure object.
      """
      r2 = r2_score(y_true, y_pred)
      print(f"Test R² = {r2:.4f}")

      fig, ax = plt.subplots(figsize=(12, 6))
      ax.plot(dates, y_true, label="Actual", linewidth=2)
      ax.plot(dates, y_pred, label="Predicted", linestyle="--")
      ax.set_xlabel("Date")
      ax.set_ylabel("Close Price")
      ax.set_title(title)
      ax.tick_params(axis='x', rotation=45)
      ax.legend()
      ax.grid(True)
      fig.tight_layout()
      
      return fig

# target = "AAPL"

# # — TRAIN & SAVE —
# X_tr, X_te, y_tr, y_te, scaler, dates_te = process_data(merged_df, target, is_training=True)
# train_and_save_model(X_tr, y_tr, scaler)

# # — EVALUATE & PLOT —
# y_pred, y_true, dates = load_and_predict(merged_df, target, is_test=True)
# plot_actual_vs_predicted(y_true, y_pred, dates)

# # — INFERENCE (next day) —
# pred_price, pred_date = load_and_predict(merged_df, target, is_test=False)
# print(f"[{pred_date}] Predicted next‑day close: {pred_price:.2f}")
