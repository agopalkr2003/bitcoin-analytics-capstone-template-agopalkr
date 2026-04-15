"""Compact shareable version of the current best honest BTC DCA model.

This file collapses the winning model stack into one main model module so
teammates do not need to chase the full historical dependency chain.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from template.model_development_template import _clean_array, allocate_sequential_stable
from lstm_helpers import (
    LSTM_SEQUENCE_WINDOW,
    build_lstm_training_frame,
    compute_qty_from_buy_points,
    create_sequences,
    train_lstm_model,
)


PRICE_COL = "PriceUSD_coinmetrics"

REPO_DIR = Path(__file__).resolve().parents[1]
SHAYAN_DIR = REPO_DIR / "Shayan's work"
POLYMARKET_FEATURES_FILE = (
    SHAYAN_DIR
    / "eda"
    / "outputs"
    / "rolling_open_selector"
    / "rolling_open_question_features.csv"
)

MA_WINDOW = 200
DYNAMIC_STRENGTH = 2.0
POLYMARKET_OVERLAY_STRENGTH = 2.0

MA_SIGNAL_WEIGHT = 2.0
FGI_SIGNAL_WEIGHT = 2.0
SNP_SIGNAL_WEIGHT = 2.0
POLY_FGI_BLEND = 0.8
SINGLE_MA_WINDOWS = [20, 50, 100, 150, 200]
DUAL_MA_PAIRS = [(20, 100), (50, 150), (50, 200), (100, 200)]

HALVING_SIGNAL_WEIGHT = 1.5
POLY_SIGNAL_THRESHOLD = 0.12
POLY_MIN_SELECTED_QUESTIONS = 2.0
POLY_MIN_ABS_CORR = 0.08
LEARNED_PRE_HALVING_DAYS = 30
LEARNED_POST_HALVING_DAYS = 180
HALVING_DATES = [
    pd.Timestamp("2012-11-28"),
    pd.Timestamp("2016-07-09"),
    pd.Timestamp("2020-05-11"),
    pd.Timestamp("2024-04-20"),
    pd.Timestamp("2028-04-20"),
]

RSI_WINDOW = 14
RSI_EWM_SPAN = 10
BUY_ZONE_WINDOW = 90
BUY_ZONE_QUANTILE = 0.20
VOLUME_WINDOW = 30
CONFIRMATION_SIGNAL_WEIGHT = 2.0

RECOVERY_LOOKBACK_DAYS = 365
RECOVERY_MOMENTUM_DAYS = 30
RECOVERY_TREND_SLOPE_DAYS = 21

DEFAULT_CONFIG = {
    "ma_mode": "dual_100_200",
    "base_ma_weight": 0.33,
    "base_sentiment_weight": 0.15,
    "base_lstm_weight": 0.34,
    "base_snp_weight": 0.08,
    "base_halving_weight": 0.05,
    "confirmation_weight": 0.03,
    "recovery_target_ma_weight": 0.40,
    "recovery_target_sentiment_weight": 0.05,
    "recovery_target_lstm_weight": 0.45,
    "recovery_target_snp_weight": 0.00,
    "recovery_target_halving_weight": 0.05,
    "anchor_weight": 0.10,
    "use_regime_confirmation_switch": True,
    "confirmation_weight_low": 0.00,
    "confirmation_weight_high": 0.05,
    "confirmation_recovery_threshold": 0.35,
    "confirmation_trend_threshold": 0.03,
}


def load_polymarket_feature_matrix() -> pd.DataFrame:
    if not POLYMARKET_FEATURES_FILE.exists():
        return pd.DataFrame()

    features = pd.read_csv(POLYMARKET_FEATURES_FILE)
    if "time" not in features.columns:
        return pd.DataFrame()

    features["time"] = pd.to_datetime(features["time"])
    features = features.set_index("time").sort_index()
    features.index = features.index.normalize().tz_localize(None)
    return features


def build_polymarket_overlay_signal(
    poly_features: pd.DataFrame, price_index: pd.DatetimeIndex
) -> pd.Series:
    if poly_features.empty:
        return pd.Series(0.0, index=price_index)

    required_cols = [
        "open_question_mean_return_z20_lag1",
        "open_question_mean_price_lag1",
        "open_question_mean_abs_corr_lag1",
        "selected_question_count_lag1",
    ]
    if any(col not in poly_features.columns for col in required_cols):
        return pd.Series(0.0, index=price_index)

    z_signal = (
        poly_features["open_question_mean_return_z20_lag1"].astype(float).clip(-3, 3) / 3.0
    )
    price_signal = (
        poly_features["open_question_mean_price_lag1"].astype(float).clip(0, 1) - 0.5
    ) * 2.0
    corr_weight = poly_features["open_question_mean_abs_corr_lag1"].astype(float).clip(0, 1)
    count_weight = poly_features["selected_question_count_lag1"].astype(float).clip(lower=0.0)
    count_weight = np.tanh(count_weight.fillna(0.0) / 3.0)

    overlay = (0.7 * z_signal + 0.3 * price_signal) * corr_weight * count_weight
    return overlay.reindex(price_index).fillna(0.0).clip(-1.0, 1.0)


def build_polymarket_availability_flag(
    poly_features: pd.DataFrame, price_index: pd.DatetimeIndex
) -> pd.Series:
    if poly_features.empty or "selected_question_count_lag1" not in poly_features.columns:
        return pd.Series(0.0, index=price_index)

    available = (
        poly_features["selected_question_count_lag1"]
        .fillna(0.0)
        .astype(float)
        .gt(0)
        .astype(float)
    )
    return available.reindex(price_index).fillna(0.0)


def load_fgi_data() -> pd.DataFrame:
    file_path = REPO_DIR / "data" / "crypto_fear_and_greed_index_2019_2025.csv"
    if not file_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    if "date" not in df.columns or "value" not in df.columns:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["fgi_normalized"] = (df["value"] / 100.0).clip(0.0, 1.0)
    return (
        df.set_index("date")
        .sort_index()[["value", "fgi_normalized"]]
        .dropna(subset=["fgi_normalized"])
    )


def load_snp_data() -> pd.DataFrame:
    file_path = REPO_DIR / "data" / "SP500.csv"
    if not file_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    if "Date" not in df.columns or "Close" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.set_index("Date").sort_index()
    df["snp_ma_20"] = df["Close"].rolling(20, min_periods=10).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["snp_vs_ma"] = ((df["Close"] / df["snp_ma_20"]) - 1.0).clip(-1.0, 1.0)
    return df[["Close", "snp_ma_20", "snp_vs_ma"]].dropna(subset=["snp_vs_ma"], how="all")


def _compute_rsi(price: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0).clip(0.0, 100.0)


def _build_halving_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    idx = pd.DatetimeIndex(index)
    days_since_last = pd.Series(np.nan, index=idx, dtype=float)
    days_to_next = pd.Series(np.nan, index=idx, dtype=float)

    for ts in idx:
        last_halving = max((d for d in HALVING_DATES if d <= ts), default=None)
        next_halving = min((d for d in HALVING_DATES if d > ts), default=None)
        if last_halving is not None:
            days_since_last.loc[ts] = float((ts - last_halving).days)
        if next_halving is not None:
            days_to_next.loc[ts] = float((next_halving - ts).days)

    post_halving = np.where(
        days_since_last.notna(),
        np.maximum(1.0 - (days_since_last / float(LEARNED_POST_HALVING_DAYS)), 0.0),
        0.0,
    )
    pre_halving = np.where(
        days_to_next.notna(),
        np.maximum(1.0 - (days_to_next / float(LEARNED_PRE_HALVING_DAYS)), 0.0),
        0.0,
    )
    base_halving_signal = np.clip(post_halving + 0.5 * pre_halving, 0.0, 1.0)
    return pd.DataFrame(
        {
            "days_since_last_halving": days_since_last,
            "days_to_next_halving": days_to_next,
            "base_halving_signal": pd.Series(base_halving_signal, index=idx),
        },
        index=idx,
    )


def available_ma_modes() -> list[str]:
    single = [f"single_{window}" for window in SINGLE_MA_WINDOWS]
    dual = [f"dual_{fast}_{slow}" for fast, slow in DUAL_MA_PAIRS]
    return single + dual


def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    if PRICE_COL not in df.columns:
        raise KeyError(f"'{PRICE_COL}' not found. Available: {list(df.columns)}")

    price = df[PRICE_COL].loc["2010-07-18":].copy()
    ma = price.rolling(MA_WINDOW, min_periods=MA_WINDOW // 2).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        price_vs_ma = ((price / ma) - 1.0).clip(-1.0, 1.0).fillna(0.0)

    poly_raw = load_polymarket_feature_matrix()
    polymarket_overlay = build_polymarket_overlay_signal(poly_raw, price.index)
    polymarket_available = build_polymarket_availability_flag(poly_raw, price.index)

    features = pd.DataFrame(
        {
            PRICE_COL: price,
            "price_ma": ma,
            "price_vs_ma": price_vs_ma.shift(1).fillna(0.0),
            "polymarket_overlay_signal": polymarket_overlay.shift(1).fillna(0.0),
            "polymarket_available": polymarket_available.shift(1).fillna(0.0),
        },
        index=price.index,
    )

    for window in SINGLE_MA_WINDOWS:
        ma_window = price.rolling(window, min_periods=max(window // 2, 5)).mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            price_vs_ma_window = ((price / ma_window) - 1.0).clip(-1.0, 1.0)
        features[f"price_ma_{window}"] = ma_window
        features[f"price_vs_ma_{window}"] = price_vs_ma_window.shift(1).fillna(0.0)

    for fast_window, slow_window in DUAL_MA_PAIRS:
        fast_ma = features[f"price_ma_{fast_window}"]
        slow_ma = features[f"price_ma_{slow_window}"]
        with np.errstate(divide="ignore", invalid="ignore"):
            spread = ((fast_ma / slow_ma) - 1.0).clip(-1.0, 1.0)
        features[f"ma_spread_{fast_window}_{slow_window}"] = spread.shift(1).fillna(0.0)

    fgi_df = load_fgi_data()
    if not fgi_df.empty:
        fgi = fgi_df["fgi_normalized"].reindex(price.index).ffill().fillna(0.5)
        fgi_available = fgi_df["fgi_normalized"].reindex(price.index).notna().astype(float)
    else:
        fgi = pd.Series(0.5, index=price.index)
        fgi_available = pd.Series(0.0, index=price.index)
    features["fgi_sentiment"] = fgi.shift(1).fillna(0.5)
    features["fgi_available"] = fgi_available.shift(1).fillna(0.0)

    snp_df = load_snp_data()
    if not snp_df.empty:
        snp_vs_ma = snp_df["snp_vs_ma"].reindex(price.index).ffill().fillna(0.0)
        snp_available = snp_df["snp_vs_ma"].reindex(price.index).notna().astype(float)
    else:
        snp_vs_ma = pd.Series(0.0, index=price.index)
        snp_available = pd.Series(0.0, index=price.index)
    features["snp_vs_ma"] = snp_vs_ma.shift(1).fillna(0.0)
    features["snp_available"] = snp_available.shift(1).fillna(0.0)

    idx = features.index
    if not poly_raw.empty:
        selected_count = (
            poly_raw.get("selected_question_count_lag1", pd.Series(index=poly_raw.index, dtype=float))
            .astype(float)
            .reindex(idx)
            .fillna(0.0)
        )
        abs_corr = (
            poly_raw.get("open_question_mean_abs_corr_lag1", pd.Series(index=poly_raw.index, dtype=float))
            .astype(float)
            .reindex(idx)
            .fillna(0.0)
        )
    else:
        selected_count = pd.Series(0.0, index=idx)
        abs_corr = pd.Series(0.0, index=idx)

    signal_strength = features["polymarket_overlay_signal"].abs().reindex(idx).fillna(0.0)
    polymarket_gate = (
        features["polymarket_available"].reindex(idx).fillna(0.0).gt(0)
        & signal_strength.ge(POLY_SIGNAL_THRESHOLD)
        & selected_count.ge(POLY_MIN_SELECTED_QUESTIONS)
        & abs_corr.ge(POLY_MIN_ABS_CORR)
    ).astype(float)

    halving = _build_halving_features(idx)
    price_vs_ma_200 = features["price_vs_ma_200"].reindex(idx).fillna(0.0)
    ma_spread_100_200 = features["ma_spread_100_200"].reindex(idx).fillna(0.0)
    valuation_room = np.clip(1.0 - np.maximum(price_vs_ma_200, 0.0) / 0.20, 0.0, 1.0)
    trend_balance = np.clip(1.0 - np.maximum(ma_spread_100_200, 0.0) / 0.10, 0.0, 1.0)
    event_confirmation = np.where(polymarket_gate > 0, 1.0, 0.6)
    regime_scale = np.clip(
        0.5 * valuation_room + 0.3 * trend_balance + 0.2 * event_confirmation,
        0.0,
        1.0,
    )
    conditional_halving_signal = halving["base_halving_signal"] * regime_scale

    features["polymarket_signal_strength"] = signal_strength
    features["polymarket_selected_count"] = selected_count
    features["polymarket_abs_corr"] = abs_corr
    features["polymarket_gate"] = polymarket_gate
    features["days_since_last_halving"] = halving["days_since_last_halving"].shift(1)
    features["days_to_next_halving"] = halving["days_to_next_halving"].shift(1)
    features["base_halving_signal"] = halving["base_halving_signal"].shift(1).fillna(0.0)
    features["halving_regime_scale"] = pd.Series(regime_scale, index=idx).shift(1).fillna(0.0)
    features["halving_signal"] = conditional_halving_signal.shift(1).fillna(0.0)

    rsi = _compute_rsi(price)
    ew_rsi = rsi.ewm(span=RSI_EWM_SPAN, adjust=False, min_periods=RSI_WINDOW).mean()
    buy_zone_threshold = (
        ew_rsi.rolling(BUY_ZONE_WINDOW, min_periods=max(20, BUY_ZONE_WINDOW // 3))
        .quantile(BUY_ZONE_QUANTILE)
        .shift(1)
        .fillna(ew_rsi.expanding().quantile(BUY_ZONE_QUANTILE))
        .fillna(40.0)
    )

    volume = df["volume_reported_spot_usd_1d"].reindex(features.index).astype(float)
    volume = volume.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    volume_baseline = volume.rolling(VOLUME_WINDOW, min_periods=max(10, VOLUME_WINDOW // 3)).median()
    volume_ratio = (volume / volume_baseline.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    volume_ratio = volume_ratio.fillna(1.0).clip(0.0, 5.0)
    abnormal_volume = volume_ratio.ge(1.20).astype(float)

    buy_zone_gap = (buy_zone_threshold - ew_rsi).clip(lower=0.0)
    confidence_score = (buy_zone_gap / 15.0).clip(0.0, 1.0) * abnormal_volume

    features["ew_rsi"] = ew_rsi.shift(1).fillna(50.0)
    features["ew_rsi_buy_zone_threshold"] = buy_zone_threshold.shift(1).fillna(40.0)
    features["volume_ratio"] = volume_ratio.shift(1).fillna(1.0)
    features["abnormal_volume"] = abnormal_volume.shift(1).fillna(0.0)
    features["confirmation_confidence"] = confidence_score.shift(1).fillna(0.0)
    features["confirmation_signal"] = confidence_score.shift(1).fillna(0.0)

    rolling_peak = price.rolling(
        RECOVERY_LOOKBACK_DAYS,
        min_periods=max(90, RECOVERY_LOOKBACK_DAYS // 3),
    ).max()
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown_365 = ((price / rolling_peak) - 1.0).clip(-1.0, 0.0)

    recent_return_30d = price.pct_change(RECOVERY_MOMENTUM_DAYS, fill_method=None)
    ma_spread = features["ma_spread_100_200"].astype(float)
    ma_spread_slope_21d = ma_spread.diff(RECOVERY_TREND_SLOPE_DAYS)
    deep_drawdown_score = np.clip(((-drawdown_365) - 0.20) / 0.40, 0.0, 1.0)
    momentum_score = np.clip((recent_return_30d + 0.02) / 0.18, 0.0, 1.0)
    trend_improvement_score = np.clip((ma_spread_slope_21d + 0.005) / 0.04, 0.0, 1.0)
    not_extended_score = np.clip(
        1.0 - np.maximum(price_vs_ma_200, 0.0) / 0.12,
        0.0,
        1.0,
    )
    recovery_signal = (
        deep_drawdown_score
        * not_extended_score
        * (0.55 * momentum_score + 0.45 * trend_improvement_score)
    ).clip(0.0, 1.0)

    features["drawdown_365"] = drawdown_365.shift(1).fillna(0.0)
    features["recent_return_30d"] = recent_return_30d.shift(1).fillna(0.0)
    features["ma_spread_slope_21d"] = ma_spread_slope_21d.shift(1).fillna(0.0)
    features["recovery_signal"] = recovery_signal.shift(1).fillna(0.0)
    return features


def compute_uniform_window_weights(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(full_range)
    if n == 0:
        return pd.Series(dtype=float)
    return pd.Series(np.ones(n) / n, index=full_range)


def _compute_single_lstm_window_weights(
    lstm_model,
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    from sklearn.preprocessing import MinMaxScaler

    df_inp = features_df.loc[start_date:end_date].copy()
    if df_inp.empty:
        return pd.Series(dtype=float)

    feature_df = build_lstm_training_frame(df_inp)
    if len(feature_df) <= LSTM_SEQUENCE_WINDOW:
        return pd.Series(np.ones(len(df_inp)) / len(df_inp), index=df_inp.index)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(feature_df)
    x_test, _ = create_sequences(scaled, LSTM_SEQUENCE_WINDOW)
    if len(x_test) == 0:
        return pd.Series(np.ones(len(df_inp)) / len(df_inp), index=df_inp.index)

    pred = lstm_model.predict(x_test, verbose=0)
    pred_inv = scaler.inverse_transform(
        np.c_[pred, np.zeros((len(pred), feature_df.shape[1] - 1))]
    )[:, 0]

    down_trend = False
    buy_points: list[int] = []
    for i in range(1, len(pred_inv)):
        if pred_inv[i] > pred_inv[i - 1]:
            if down_trend:
                buy_points.append(i + LSTM_SEQUENCE_WINDOW)
            down_trend = False
        elif pred_inv[i] < pred_inv[i - 1]:
            down_trend = True

    lstm_weights = compute_qty_from_buy_points(df_inp, buy_points)
    return pd.Series(lstm_weights, index=df_inp.index)


def compute_lstm_window_weights(
    lstm_model,
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    if isinstance(lstm_model, (list, tuple)):
        if len(lstm_model) == 0:
            return pd.Series(dtype=float)
        runs = [
            _compute_single_lstm_window_weights(model, features_df, start_date, end_date)
            for model in lstm_model
        ]
        full_index = runs[0].index
        averaged = pd.concat(
            [run.reindex(full_index, fill_value=0.0) for run in runs],
            axis=1,
        ).mean(axis=1)
        total = averaged.sum()
        if total <= 0:
            return pd.Series(np.ones(len(full_index)) / max(len(full_index), 1), index=full_index)
        return averaged / total
    return _compute_single_lstm_window_weights(lstm_model, features_df, start_date, end_date)


def compute_ma_variant_multiplier(features_df: pd.DataFrame, ma_mode: str) -> np.ndarray:
    if ma_mode.startswith("single_"):
        window = int(ma_mode.split("_", 1)[1])
        signal = -_clean_array(features_df[f"price_vs_ma_{window}"].values)
    elif ma_mode.startswith("dual_"):
        fast_window, slow_window = ma_mode.split("_", 1)[1].split("_")
        fast_window = int(fast_window)
        slow_window = int(slow_window)
        price_signal = -_clean_array(features_df[f"price_vs_ma_{slow_window}"].values)
        crossover_signal = -_clean_array(
            features_df[f"ma_spread_{fast_window}_{slow_window}"].values
        )
        signal = 0.5 * price_signal + 0.5 * crossover_signal
    else:
        raise ValueError(f"Unknown ma_mode: {ma_mode}")

    adjustment = np.clip(signal * MA_SIGNAL_WEIGHT, -3.0, 3.0)
    multiplier = np.exp(adjustment)
    return np.where(np.isfinite(multiplier), multiplier, 1.0)


def compute_ma_variant_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    ma_mode: str,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n
    raw = base * compute_ma_variant_multiplier(df, ma_mode)
    past_end = min(current_date, end_date)
    n_past = len(pd.date_range(start=start_date, end=past_end, freq="D")) if start_date <= past_end else 0
    weights = allocate_sequential_stable(raw, n_past, locked_weights)
    return pd.Series(weights, index=df.index)


def compute_sentiment_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n
    poly_signal = _clean_array(df["polymarket_overlay_signal"].values)
    fgi_sentiment = _clean_array(df["fgi_sentiment"].values)
    fgi_available = _clean_array(df["fgi_available"].values)
    fgi_signal = np.where(fgi_available > 0, (0.5 - fgi_sentiment) * 2.0, 0.0)
    sentiment_signal = POLY_FGI_BLEND * poly_signal + (1.0 - POLY_FGI_BLEND) * fgi_signal
    adjustment = np.clip(sentiment_signal * FGI_SIGNAL_WEIGHT, -3.0, 3.0)
    multiplier = np.exp(adjustment)
    raw = base * np.where(np.isfinite(multiplier), multiplier, 1.0)

    past_end = min(current_date, end_date)
    n_past = len(pd.date_range(start=start_date, end=past_end, freq="D")) if start_date <= past_end else 0
    weights = allocate_sequential_stable(raw, n_past, locked_weights)
    return pd.Series(weights, index=df.index)


def compute_snp_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n
    snp_signal = _clean_array(df["snp_vs_ma"].values)
    snp_available = _clean_array(df["snp_available"].values)
    adjustment = np.clip(snp_signal * SNP_SIGNAL_WEIGHT, -3.0, 3.0)
    multiplier = np.exp(adjustment)
    multiplier = np.where(snp_available > 0, multiplier, 1.0)
    raw = base * multiplier

    past_end = min(current_date, end_date)
    n_past = len(pd.date_range(start=start_date, end=past_end, freq="D")) if start_date <= past_end else 0
    weights = allocate_sequential_stable(raw, n_past, locked_weights)
    return pd.Series(weights, index=df.index)


def compute_halving_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n
    signal = _clean_array(df["halving_signal"].values)
    adjustment = np.clip(signal * HALVING_SIGNAL_WEIGHT, -3.0, 3.0)
    multiplier = np.exp(adjustment)
    raw = base * np.where(np.isfinite(multiplier), multiplier, 1.0)

    past_end = min(current_date, end_date)
    n_past = len(pd.date_range(start=start_date, end=past_end, freq="D")) if start_date <= past_end else 0
    weights = allocate_sequential_stable(raw, n_past, locked_weights)
    return pd.Series(weights, index=df.index)


def compute_confirmation_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
) -> pd.Series:
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n
    signal = _clean_array(df["confirmation_signal"].values)
    adjustment = np.clip(signal * CONFIRMATION_SIGNAL_WEIGHT, -3.0, 3.0)
    multiplier = np.exp(adjustment)
    raw = base * np.where(np.isfinite(multiplier), multiplier, 1.0)

    past_end = min(current_date, end_date)
    n_past = len(pd.date_range(start=start_date, end=past_end, freq="D")) if start_date <= past_end else 0
    weights = allocate_sequential_stable(raw, n_past, locked_weights)
    return pd.Series(weights, index=df.index)


def normalize_four_weights(
    ma_weight: float,
    sentiment_weight: float,
    lstm_weight: float,
    snp_weight: float,
) -> tuple[float, float, float, float]:
    weights = np.array([ma_weight, sentiment_weight, lstm_weight, snp_weight], dtype=float)
    weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    if total <= 0:
        return (1.0, 0.0, 0.0, 0.0)
    weights = weights / total
    return tuple(weights.tolist())


def normalize_five_weights(
    ma_weight: float,
    sentiment_weight: float,
    lstm_weight: float,
    snp_weight: float,
    halving_weight: float,
) -> tuple[float, float, float, float, float]:
    weights = np.array(
        [ma_weight, sentiment_weight, lstm_weight, snp_weight, halving_weight], dtype=float
    )
    weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    if total <= 0:
        return (1.0, 0.0, 0.0, 0.0, 0.0)
    weights = weights / total
    return tuple(weights.tolist())


def blend_four_component_weights(
    ma_weights: pd.Series,
    sentiment_weights: pd.Series,
    lstm_weights: pd.Series,
    snp_weights: pd.Series,
    ma_weight: float,
    sentiment_weight: float,
    lstm_weight: float,
    snp_weight: float,
) -> pd.Series:
    ma_weight, sentiment_weight, lstm_weight, snp_weight = normalize_four_weights(
        ma_weight, sentiment_weight, lstm_weight, snp_weight
    )

    full_index = (
        ma_weights.index.union(sentiment_weights.index)
        .union(lstm_weights.index)
        .union(snp_weights.index)
    )
    ma_weights = ma_weights.reindex(full_index, fill_value=0.0)
    sentiment_weights = sentiment_weights.reindex(full_index, fill_value=0.0)
    lstm_weights = lstm_weights.reindex(full_index, fill_value=0.0)
    snp_weights = snp_weights.reindex(full_index, fill_value=0.0)

    blended = (
        ma_weight * ma_weights
        + sentiment_weight * sentiment_weights
        + lstm_weight * lstm_weights
        + snp_weight * snp_weights
    )
    total = blended.sum()
    if total <= 0:
        return pd.Series(np.ones(len(full_index)) / max(len(full_index), 1), index=full_index)
    return blended / total


def compute_sentiment_macro_window_weights(
    lstm_model,
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    ma_mode: str,
    locked_weights: np.ndarray | None,
    ma_weight: float,
    sentiment_weight: float,
    lstm_weight: float,
    snp_weight: float,
) -> pd.Series:
    ma_weights = compute_ma_variant_window_weights(
        features_df, start_date, end_date, current_date, ma_mode, locked_weights
    )
    sentiment_weights = compute_sentiment_window_weights(
        features_df, start_date, end_date, current_date, locked_weights
    )
    lstm_weights = compute_lstm_window_weights(lstm_model, features_df, start_date, end_date)
    snp_weights = compute_snp_window_weights(
        features_df, start_date, end_date, current_date, locked_weights
    )
    blended = blend_four_component_weights(
        ma_weights,
        sentiment_weights,
        lstm_weights,
        snp_weights,
        ma_weight,
        sentiment_weight,
        lstm_weight,
        snp_weight,
    )
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")
    return blended.reindex(full_range, fill_value=0.0)


def compute_conditional_halving_window_weights(
    lstm_model,
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    ma_mode: str,
    locked_weights: np.ndarray | None,
    ma_weight: float,
    sentiment_weight: float,
    lstm_weight: float,
    snp_weight: float,
    halving_weight: float,
) -> pd.Series:
    (
        ma_weight,
        sentiment_weight,
        lstm_weight,
        snp_weight,
        halving_weight,
    ) = normalize_five_weights(
        ma_weight,
        sentiment_weight,
        lstm_weight,
        snp_weight,
        halving_weight,
    )

    base_weights = compute_sentiment_macro_window_weights(
        lstm_model,
        features_df,
        start_date,
        end_date,
        current_date,
        ma_mode,
        locked_weights,
        ma_weight,
        sentiment_weight,
        lstm_weight,
        snp_weight,
    )
    halving_weights = compute_halving_window_weights(
        features_df,
        start_date,
        end_date,
        current_date,
        locked_weights,
    )

    full_index = base_weights.index.union(halving_weights.index)
    base_weights = base_weights.reindex(full_index, fill_value=0.0)
    halving_weights = halving_weights.reindex(full_index, fill_value=0.0)

    halving_active = float(features_df.loc[start_date:end_date, "halving_signal"].max()) > 0.0
    effective_halving_weight = halving_weight if halving_active else 0.0
    base_stack_weight = 1.0 - effective_halving_weight

    blended = base_stack_weight * base_weights + effective_halving_weight * halving_weights
    total = blended.sum()
    if total <= 0:
        return pd.Series(np.ones(len(full_index)) / max(len(full_index), 1), index=full_index)

    full_range = pd.date_range(start=start_date, end=end_date, freq="D")
    return (blended / total).reindex(full_range, fill_value=0.0)


def _compute_recovery_strength(window_features: pd.DataFrame) -> float:
    recovery_signal = window_features["recovery_signal"].astype(float)
    if recovery_signal.empty:
        return 0.0
    recent = recovery_signal.tail(min(60, len(recovery_signal)))
    strength = 0.5 * recent.mean() + 0.5 * recent.max()
    return float(np.clip(strength, 0.0, 1.0))


def _compute_trend_strength(window_features: pd.DataFrame) -> float:
    if window_features.empty:
        return 0.0
    ma_spread = window_features["ma_spread_100_200"].astype(float).tail(min(60, len(window_features)))
    price_vs_ma = window_features["price_vs_ma_200"].astype(float).tail(min(60, len(window_features)))
    if ma_spread.empty or price_vs_ma.empty:
        return 0.0
    spread_strength = np.clip(ma_spread.mean() / 0.05, 0.0, 1.0)
    price_strength = np.clip(price_vs_ma.mean() / 0.10, 0.0, 1.0)
    return float(np.clip(0.6 * spread_strength + 0.4 * price_strength, 0.0, 1.0))


def _resolve_confirmation_weight(cfg: dict, recovery_strength: float, trend_strength: float) -> float:
    if not cfg.get("use_regime_confirmation_switch", False):
        return float(cfg["confirmation_weight"])

    high_weight = float(cfg.get("confirmation_weight_high", cfg["confirmation_weight"]))
    low_weight = float(cfg.get("confirmation_weight_low", 0.0))
    recovery_threshold = float(cfg.get("confirmation_recovery_threshold", 0.35))
    trend_threshold = float(cfg.get("confirmation_trend_threshold", 0.03))

    if recovery_strength >= recovery_threshold or trend_strength >= trend_threshold:
        return low_weight
    return high_weight


def _blend_weight(default_value: float, target_value: float, strength: float) -> float:
    return (1.0 - strength) * default_value + strength * target_value


def _merged_config(overrides: dict | None = None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if overrides:
        cfg.update(overrides)
    return cfg


def compute_window_weights(
    lstm_model,
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
    config: dict | None = None,
) -> pd.Series:
    cfg = _merged_config(config)
    window_features = features_df.loc[start_date:end_date]
    recovery_strength = _compute_recovery_strength(window_features)
    trend_strength = _compute_trend_strength(window_features)

    ma_weight = _blend_weight(
        cfg["base_ma_weight"], cfg["recovery_target_ma_weight"], recovery_strength
    )
    sentiment_weight = _blend_weight(
        cfg["base_sentiment_weight"],
        cfg["recovery_target_sentiment_weight"],
        recovery_strength,
    )
    lstm_weight = _blend_weight(
        cfg["base_lstm_weight"], cfg["recovery_target_lstm_weight"], recovery_strength
    )
    snp_weight = _blend_weight(
        cfg["base_snp_weight"], cfg["recovery_target_snp_weight"], recovery_strength
    )
    halving_weight = _blend_weight(
        cfg["base_halving_weight"], cfg["recovery_target_halving_weight"], recovery_strength
    )

    base_weights = compute_conditional_halving_window_weights(
        lstm_model,
        features_df,
        start_date,
        end_date,
        current_date,
        ma_mode=cfg["ma_mode"],
        locked_weights=locked_weights,
        ma_weight=ma_weight,
        sentiment_weight=sentiment_weight,
        lstm_weight=lstm_weight,
        snp_weight=snp_weight,
        halving_weight=halving_weight,
    )
    confirmation_weights = compute_confirmation_window_weights(
        features_df,
        start_date,
        end_date,
        current_date,
        locked_weights=locked_weights,
    )

    full_index = base_weights.index.union(confirmation_weights.index)
    base_weights = base_weights.reindex(full_index, fill_value=0.0)
    confirmation_weights = confirmation_weights.reindex(full_index, fill_value=0.0)
    confirmation_weight = _resolve_confirmation_weight(cfg, recovery_strength, trend_strength)

    blended = (
        (1.0 - confirmation_weight) * base_weights
        + confirmation_weight * confirmation_weights
    )
    blended_total = blended.sum()
    if blended_total <= 0:
        blended = pd.Series(np.ones(len(full_index)) / max(len(full_index), 1), index=full_index)
    else:
        blended = blended / blended_total

    uniform_weights = compute_uniform_window_weights(start_date, end_date)
    full_index = blended.index.union(uniform_weights.index)
    blended = blended.reindex(full_index, fill_value=0.0)
    uniform_weights = uniform_weights.reindex(full_index, fill_value=0.0)

    final_weights = (
        (1.0 - cfg["anchor_weight"]) * blended
        + cfg["anchor_weight"] * uniform_weights
    )
    total = final_weights.sum()
    if total <= 0:
        return uniform_weights
    return final_weights / total


__all__ = [
    "DEFAULT_CONFIG",
    "PRICE_COL",
    "available_ma_modes",
    "compute_uniform_window_weights",
    "compute_window_weights",
    "load_fgi_data",
    "load_polymarket_feature_matrix",
    "load_snp_data",
    "precompute_features",
    "train_lstm_model",
]
