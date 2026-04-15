"""Shared LSTM helpers copied from the example_LSTM workflow.

These helpers are frozen here so teammates can run the packaged best model
without chasing the original LSTM example files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


LSTM_SEQUENCE_WINDOW = 20
LSTM_TRAIN_YEAR = 2018


def _resolve_price_frame(df_inp: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with a canonical PriceUSD column for shared LSTM helpers."""
    if "PriceUSD" in df_inp.columns:
        return df_inp[["PriceUSD"]].copy()
    if "PriceUSD_coinmetrics" in df_inp.columns:
        out = df_inp[["PriceUSD_coinmetrics"]].copy()
        return out.rename(columns={"PriceUSD_coinmetrics": "PriceUSD"})
    raise KeyError(
        "Expected a PriceUSD-like column for LSTM training, found: "
        f"{list(df_inp.columns)}"
    )


def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i + window])
        y.append(data[i + window, 0])
    return np.array(X), np.array(y)


def build_lstm_training_frame(df_inp: pd.DataFrame) -> pd.DataFrame:
    """Build the LSTM feature frame used by the original example runner."""
    df_small = _resolve_price_frame(df_inp)
    df_small["Momentum"] = df_small["PriceUSD"].diff()
    df_small["Acceleration"] = df_small["Momentum"].diff()
    df_small["MA5"] = df_small["PriceUSD"].rolling(5).mean()
    df_small["MA20"] = df_small["PriceUSD"].rolling(20).mean()
    df_small["MomentumMA"] = df_small["Momentum"].rolling(10).mean()
    df_small["AccelerationMA"] = df_small["Acceleration"].rolling(10).mean()
    df_small["Volatility"] = df_small["PriceUSD"].rolling(10).std()
    return df_small.dropna()


def compute_qty_from_buy_points(df: pd.DataFrame, buy_points: list[int]) -> np.ndarray:
    """Convert LSTM turning points into normalized DCA weights."""
    df = _resolve_price_frame(df)
    weights = []
    amount = 10000.0
    n = len(df)
    prev_pt = 0
    buy_map = set(buy_points)

    for idx, row in enumerate(df.itertuples(index=False)):
        qty_for_day = 1e-6
        if idx in buy_map:
            price = getattr(row, "PriceUSD", None)
            if price and np.isfinite(price):
                qty_for_day = (idx - prev_pt) * (amount / n) / price
            prev_pt = idx
        weights.append(qty_for_day)

    weights = np.array(weights, dtype=float)
    total = weights.sum()
    if total <= 0:
        return np.full(len(df), 1.0 / max(len(df), 1))
    return weights / total


def train_lstm_model(df_inp: pd.DataFrame):
    """Train the original example LSTM and return the trained network."""
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow import keras

    df_train = df_inp[df_inp.index.year == LSTM_TRAIN_YEAR].copy()
    df = build_lstm_training_frame(df_train)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = create_sequences(scaled_data, LSTM_SEQUENCE_WINDOW)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]

    model = keras.Sequential()
    model.add(keras.layers.LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1))
    model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)
    return model
