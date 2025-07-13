import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt


class LSTMService:
    def __init__(self):
        pass
      
    # ─── 1) PREPARE DATA 
    @staticmethod
    def process_lstm_data(df: pd.DataFrame,
                          target_col: str,
                          sequence_length: int = 30,
                          is_training: bool = True,
                          train_ratio: float = 0.8):
        df = df.copy()
        df.dropna(subset=[target_col], inplace=True)

        # feature matrix & target
        features = [c for c in df.columns if c not in ("Date", target_col) and not c.endswith('_close_stock')]
        X_all = df[features].values
        y_all = df[[target_col]].values
        dates = df["Date"].reset_index(drop=True)

        # scale separately
        scaler_X = MinMaxScaler((0,1))
        scaler_y = MinMaxScaler((0,1))
        X_scaled = scaler_X.fit_transform(X_all)
        y_scaled = scaler_y.fit_transform(y_all)

        # build sequences
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - sequence_length):
            X_seq.append(X_scaled[i:i+sequence_length])
            y_seq.append(y_scaled[i+sequence_length,0])
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        split_idx = int(len(X_seq) * train_ratio)
        if is_training:
            X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
            # map test indices to original dates & unscaled y
            test_start = split_idx + sequence_length
            test_dates = dates.iloc[test_start:].reset_index(drop=True)
            original_y_test = df[target_col].iloc[test_start:].reset_index(drop=True)
            return X_train, y_train, X_test, y_test, scaler_X, scaler_y, test_dates, original_y_test

        # inference: last sequence only
        X_last = X_seq[-1].reshape(1, sequence_length, -1)
        last_date = dates.iloc[-1]
        return X_last, last_date, scaler_X, scaler_y

    # ─── 2) TRAIN & SAVE ──────────
    @staticmethod
    def train_and_save_lstm(X_train, y_train,
                            scaler_X, scaler_y,
                            model_path: str,
                            scaler_path: str,
                            sequence_length: int = 30,
                            units: int = 50,
                            dropout: float = 0.2,
                            epochs: int = 50,
                            batch_size: int = 32):
        # build model
        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=(sequence_length, X_train.shape[2])),
            Dropout(dropout),
            LSTM(units),
            Dropout(dropout),
            Dense(1)
        ])
        model.compile("adam", "mean_squared_error")

        # train
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        # save
        model.save(model_path)
        joblib.dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, scaler_path)

        print(f"Saved LSTM model → {model_path}")
        print(f"Saved scalers     → {scaler_path}")

    # ─── 3) LOAD & PREDICT 
    @staticmethod
    def load_and_predict_lstm(df: pd.DataFrame,
                              target_col: str,
                              model_path: str,
                              scaler_path: str,
                              sequence_length: int = 30,
                              is_test: bool = False):
        # load artifacts
        model = load_model(model_path)
        scalers = joblib.load(scaler_path)
        scaler_y = scalers["scaler_y"]

        if is_test:
            _, _, X_te, _, _, _, dates_te, y_te_true = LSTMService.process_lstm_data(
                df, target_col, sequence_length, is_training=True
            )
            # predict & inverse-scale
            y_pred_scaled = model.predict(X_te).flatten()[:,None]
            y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
            return y_pred, y_te_true.values, dates_te

        # inference on last row
        X_last, last_date, _, _ = LSTMService.process_lstm_data(df, target_col, sequence_length, is_training=False)
        y_pred_scaled = model.predict(X_last)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()[0]
        return y_pred, last_date

    # ─── 4) PLOT RESULTS 
    @staticmethod
    def plot_lstm_results(y_true, y_pred, dates, title="LSTM Actual vs Predicted"):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates, y_true, label="Actual", linewidth=2)
        ax.plot(dates, y_pred, label="Predicted", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.tick_params(axis='x', rotation=45)
        ax.legend(); 
        ax.grid(True); 
        fig.tight_layout()
        return fig

# target = "AAPL_close_stock"
# seq_len = 60

# # — 1) Prepare & train —
# X_tr, y_tr, X_te, y_te_scaled, scX, scY, dates_te, y_te_true = process_lstm_data(
#     merged_df, target, sequence_length=seq_len, is_training=True
# )
# train_and_save_lstm(X_tr, y_tr, scX, scY, sequence_length=seq_len, epochs=20)

# # — 2) Test & plot —
# y_pred, y_true, dates = load_and_predict_lstm(
#     merged_df, target, sequence_length=seq_len, is_test=True
# )
# plot_lstm_results(y_true, y_pred, dates)

# # — 3) Inference (next day) —
# pred_price, pred_date = load_and_predict_lstm(
#     merged_df, target, sequence_length=seq_len, is_test=False
# )
# print(f"[{pred_date}] Predicted next-day close: {pred_price:.2f}")
