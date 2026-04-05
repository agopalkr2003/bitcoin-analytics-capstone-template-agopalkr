"""MA model with macro-only rolling Polymarket overlay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from template.model_development_template import _clean_array, allocate_sequential_stable

PRICE_COL = "PriceUSD_coinmetrics"
MIN_W = 1e-6
MA_WINDOW = 200
MA_SIGNAL_WEIGHT = 2.0
POLYMARKET_SIGNAL_WEIGHT = 2.0

FEATURES_FILE = (
    Path(__file__).resolve().parent.parent
    / "eda"
    / "outputs"
    / "rolling_macro_selector"
    / "rolling_macro_question_features.csv"
)

FEATS = ["price_vs_ma", "polymarket_overlay_signal", "polymarket_available"]


def load_macro_feature_matrix() -> pd.DataFrame:
    if not FEATURES_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(FEATURES_FILE)
    if "time" not in df.columns:
        return pd.DataFrame()
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    df.index = df.index.normalize().tz_localize(None)
    return df


def build_overlay_signal(poly_features: pd.DataFrame, price_index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    if poly_features.empty:
        return pd.Series(0.0, index=price_index), pd.Series(0.0, index=price_index)

    req = [
        "open_question_mean_return_z20_lag1",
        "open_question_mean_price_lag1",
        "open_question_mean_abs_corr_lag1",
        "selected_question_count_lag1",
    ]
    if any(col not in poly_features.columns for col in req):
        return pd.Series(0.0, index=price_index), pd.Series(0.0, index=price_index)

    z_signal = poly_features["open_question_mean_return_z20_lag1"].astype(float).clip(-3, 3) / 3.0
    price_signal = (poly_features["open_question_mean_price_lag1"].astype(float).clip(0, 1) - 0.5) * 2.0
    corr_weight = poly_features["open_question_mean_abs_corr_lag1"].astype(float).clip(0, 1)
    count_weight = np.tanh(poly_features["selected_question_count_lag1"].fillna(0.0).astype(float).clip(lower=0) / 3.0)

    overlay = ((0.7 * z_signal + 0.3 * price_signal) * corr_weight * count_weight).reindex(price_index).fillna(0.0).clip(-1, 1)
    available = poly_features["selected_question_count_lag1"].fillna(0.0).astype(float).gt(0).astype(float)
    available = available.reindex(price_index).fillna(0.0)
    return overlay, available


def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    if PRICE_COL not in df.columns:
        raise KeyError(f"'{PRICE_COL}' not found. Available: {list(df.columns)}")

    price = df[PRICE_COL].loc["2010-07-18":].copy()
    ma = price.rolling(MA_WINDOW, min_periods=MA_WINDOW // 2).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        price_vs_ma = ((price / ma) - 1).clip(-1, 1).fillna(0)

    poly_raw = load_macro_feature_matrix()
    overlay, available = build_overlay_signal(poly_raw, price.index)

    return pd.DataFrame(
        {
            PRICE_COL: price,
            "price_ma": ma,
            "price_vs_ma": price_vs_ma.shift(1).fillna(0),
            "polymarket_overlay_signal": overlay.shift(1).fillna(0),
            "polymarket_available": available.shift(1).fillna(0),
        },
        index=price.index,
    )


def compute_dynamic_multiplier(price_vs_ma: np.ndarray, overlay_signal: np.ndarray, available: np.ndarray) -> np.ndarray:
    ma_signal = -price_vs_ma
    poly_signal = overlay_signal * available
    adjustment = np.clip(MA_SIGNAL_WEIGHT * ma_signal + POLYMARKET_SIGNAL_WEIGHT * poly_signal, -3, 3)
    multiplier = np.exp(adjustment)
    return np.where(np.isfinite(multiplier), multiplier, 1.0)


def compute_weights_fast(features_df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp, n_past: int | None = None, locked_weights: np.ndarray | None = None) -> pd.Series:
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)
    n = len(df)
    base = np.ones(n) / n
    dyn = compute_dynamic_multiplier(
        _clean_array(df["price_vs_ma"].values),
        _clean_array(df["polymarket_overlay_signal"].values),
        _clean_array(df["polymarket_available"].values),
    )
    raw = base * dyn
    if n_past is None:
        n_past = n
    weights = allocate_sequential_stable(raw, n_past, locked_weights)
    return pd.Series(weights, index=df.index)


def compute_window_weights(features_df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp, current_date: pd.Timestamp, locked_weights: np.ndarray | None = None) -> pd.Series:
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")
    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        placeholder = pd.DataFrame({col: 0.0 for col in features_df.columns}, index=missing)
        features_df = pd.concat([features_df, placeholder]).sort_index()
    past_end = min(current_date, end_date)
    n_past = len(pd.date_range(start=start_date, end=past_end, freq="D")) if start_date <= past_end else 0
    weights = compute_weights_fast(features_df, start_date, end_date, n_past, locked_weights)
    return weights.reindex(full_range, fill_value=0.0)

