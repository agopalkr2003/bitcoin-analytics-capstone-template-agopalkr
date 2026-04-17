"""Dynamic DCA using 200-day MA plus a rolling open-question Polymarket overlay.

This keeps the template model as the primary driver and adds a secondary,
lower-strength signal derived from rolling open-question Polymarket features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from template.model_development_template import (
    _clean_array,
    allocate_sequential_stable,
)

# =============================================================================
# Constants
# =============================================================================

PRICE_COL = "PriceUSD_coinmetrics"
MIN_W = 1e-6
MA_WINDOW = 200
DYNAMIC_STRENGTH = 2.0
# Chosen from the overlay-strength sweep on the active Polymarket period
# (2025-03-12 to 2025-12-31) with 180-day windows.
# We decided to use 2.0 for the Polymarket question overlay because it
# delivered the highest tested win rate versus the MA baseline.
# Higher values (for example 5.0) degraded performance.
POLYMARKET_OVERLAY_STRENGTH = 2.0

FEATURES_FILE = (
    Path(__file__).resolve().parent.parent
    / "eda"
    / "outputs"
    / "rolling_open_selector"
    / "rolling_open_question_features.csv"
)

FEATS = [
    "price_vs_ma",
    "polymarket_overlay_signal",
    "polymarket_available",
]


def load_polymarket_feature_matrix() -> pd.DataFrame:
    """Load pre-generated rolling Polymarket features if available."""
    if not FEATURES_FILE.exists():
        return pd.DataFrame()

    features = pd.read_csv(FEATURES_FILE)
    if "time" not in features.columns:
        return pd.DataFrame()

    features["time"] = pd.to_datetime(features["time"])
    features = features.set_index("time").sort_index()
    features.index = features.index.normalize().tz_localize(None)
    return features


def build_polymarket_overlay_signal(
    poly_features: pd.DataFrame, price_index: pd.DatetimeIndex
) -> pd.Series:
    """Build one overlay signal from rolling open-question aggregates.

    The rolling selector already compresses the currently open questions into
    daily aggregate series. We combine:
    - lagged z-scored mean return
    - lagged mean price level
    - lagged average historical absolute correlation

    This avoids the cancellation problem from averaging many fixed historical
    candidate questions together.
    """
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
    count_weight = (
        poly_features["selected_question_count_lag1"].astype(float).clip(lower=0).fillna(0.0)
    )
    count_weight = np.tanh(count_weight / 3.0)

    overlay = (0.7 * z_signal + 0.3 * price_signal) * corr_weight * count_weight
    overlay = overlay.reindex(price_index).fillna(0.0).clip(-1, 1)
    return overlay


def build_polymarket_availability_flag(
    poly_features: pd.DataFrame, price_index: pd.DatetimeIndex
) -> pd.Series:
    """Flag dates where rolling Polymarket inputs are actually available.

    This makes the MA-only fallback explicit in backtests and production.
    """
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


def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute primary MA feature plus a secondary Polymarket overlay."""
    if PRICE_COL not in df.columns:
        raise KeyError(f"'{PRICE_COL}' not found. Available: {list(df.columns)}")

    price = df[PRICE_COL].loc["2010-07-18":].copy()
    ma = price.rolling(MA_WINDOW, min_periods=MA_WINDOW // 2).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        price_vs_ma = ((price / ma) - 1).clip(-1, 1).fillna(0)

    poly_raw = load_polymarket_feature_matrix()
    polymarket_overlay = build_polymarket_overlay_signal(poly_raw, price.index)
    polymarket_available = build_polymarket_availability_flag(poly_raw, price.index)

    features = pd.DataFrame(
        {
            PRICE_COL: price,
            "price_ma": ma,
            "price_vs_ma": price_vs_ma.shift(1).fillna(0),
            "polymarket_overlay_signal": polymarket_overlay.shift(1).fillna(0),
            "polymarket_available": polymarket_available.shift(1).fillna(0),
        },
        index=price.index,
    )
    return features


def compute_dynamic_multiplier(
    price_vs_ma: np.ndarray,
    polymarket_overlay_signal: np.ndarray,
    polymarket_available: np.ndarray,
    overlay_strength: float = POLYMARKET_OVERLAY_STRENGTH,
) -> np.ndarray:
    """Compute MA-driven multiplier with explicit MA-only fallback.

    When Polymarket inputs are unavailable, the model uses only the MA signal.
    """
    ma_signal = -price_vs_ma
    overlay_signal = polymarket_overlay_signal * polymarket_available

    combined_signal = ma_signal + overlay_strength * overlay_signal
    adjustment = np.clip(combined_signal * DYNAMIC_STRENGTH, -3, 3)

    multiplier = np.exp(adjustment)
    return np.where(np.isfinite(multiplier), multiplier, 1.0)


def compute_weights_fast(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n_past: int | None = None,
    locked_weights: np.ndarray | None = None,
    overlay_strength: float = POLYMARKET_OVERLAY_STRENGTH,
) -> pd.Series:
    """Compute weights for a date window using the overlay model."""
    df = features_df.loc[start_date:end_date]
    if df.empty:
        return pd.Series(dtype=float)

    n = len(df)
    base = np.ones(n) / n

    price_vs_ma = _clean_array(df["price_vs_ma"].values)
    polymarket_overlay_signal = _clean_array(df["polymarket_overlay_signal"].values)
    polymarket_available = _clean_array(df["polymarket_available"].values)

    dyn = compute_dynamic_multiplier(
        price_vs_ma,
        polymarket_overlay_signal,
        polymarket_available,
        overlay_strength=overlay_strength,
    )
    raw = base * dyn

    if n_past is None:
        n_past = n
    weights = allocate_sequential_stable(raw, n_past, locked_weights)
    return pd.Series(weights, index=df.index)


def compute_window_weights(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    current_date: pd.Timestamp,
    locked_weights: np.ndarray | None = None,
    overlay_strength: float = POLYMARKET_OVERLAY_STRENGTH,
) -> pd.Series:
    """Compute weights for a date range with lock-on-compute stability."""
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")

    missing = full_range.difference(features_df.index)
    if len(missing) > 0:
        placeholder = pd.DataFrame(
            {col: 0.0 for col in features_df.columns},
            index=missing,
        )
        features_df = pd.concat([features_df, placeholder]).sort_index()

    past_end = min(current_date, end_date)
    if start_date <= past_end:
        n_past = len(pd.date_range(start=start_date, end=past_end, freq="D"))
    else:
        n_past = 0

    weights = compute_weights_fast(
        features_df,
        start_date,
        end_date,
        n_past,
        locked_weights,
        overlay_strength=overlay_strength,
    )
    return weights.reindex(full_range, fill_value=0.0)
