import logging
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from template.prelude_template import backtest_dynamic_dca, load_data
from example_polymarket_overlay.model_development_polymarket_overlay import (
    compute_window_weights,
    precompute_features,
)

_FEATURES_DF = None
WINDOW_DAYS = 180


def compute_weights_wrapper(df_window: pd.DataFrame) -> pd.Series:
    """Adapt the broader non-crypto overlay model to the shared backtest engine."""
    global _FEATURES_DF

    if _FEATURES_DF is None:
        raise ValueError("Features not precomputed. Call precompute_features() first.")
    if df_window.empty:
        return pd.Series(dtype=float)

    start_date = df_window.index.min()
    end_date = df_window.index.max()
    current_date = end_date
    return compute_window_weights(_FEATURES_DF, start_date, end_date, current_date)


def main():
    global _FEATURES_DF

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info(
        "Starting Bitcoin DCA Strategy Analysis - MA + broader non-crypto Polymarket overlay (active window only)"
    )

    btc_df = load_data()
    _FEATURES_DF = precompute_features(btc_df)

    active_overlay = _FEATURES_DF["polymarket_overlay_signal"].ne(0)
    if active_overlay.any():
        analysis_start = _FEATURES_DF.index[active_overlay].min()
    else:
        raise ValueError("No active Polymarket overlay dates were found in the feature set.")

    logging.info(
        "Using active overlay window only, starting %s with %d-day windows",
        analysis_start.date(),
        WINDOW_DAYS,
    )

    btc_df = btc_df.loc[analysis_start:].copy()
    _FEATURES_DF = _FEATURES_DF.loc[analysis_start:].copy()

    output_dir = Path(__file__).parent / "output_active_window"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_spd, exp_decay_percentile = backtest_dynamic_dca(
        btc_df,
        compute_weights_wrapper,
        features_df=_FEATURES_DF,
        strategy_label="MA + Broader Non-Crypto Polymarket Overlay (Active Window)",
        start_date=analysis_start.strftime("%Y-%m-%d"),
        end_date="2025-12-31",
        window_days=WINDOW_DAYS,
    )

    win_rate = (
        df_spd["dynamic_percentile"] > df_spd["uniform_percentile"]
    ).mean() * 100
    score = 0.5 * win_rate + 0.5 * exp_decay_percentile
    excess_percentile = df_spd["dynamic_percentile"] - df_spd["uniform_percentile"]

    metrics = pd.DataFrame(
        [
            {
                "start_date": analysis_start.strftime("%Y-%m-%d"),
                "end_date": "2025-12-31",
                "window_days": WINDOW_DAYS,
                "total_windows": len(df_spd),
                "win_rate": win_rate,
                "exp_decay_percentile": exp_decay_percentile,
                "score": score,
                "mean_excess": excess_percentile.mean(),
                "median_excess": excess_percentile.median(),
                "mean_ratio": (
                    df_spd["dynamic_percentile"] / df_spd["uniform_percentile"]
                ).mean(),
                "median_ratio": (
                    df_spd["dynamic_percentile"] / df_spd["uniform_percentile"]
                ).median(),
            }
        ]
    )

    metrics.to_csv(output_dir / "active_window_summary.csv", index=False)
    df_spd.to_csv(output_dir / "active_window_spd.csv")

    logging.info("Saved active-window summary to %s", output_dir / "active_window_summary.csv")
    logging.info("Saved active-window SPD detail to %s", output_dir / "active_window_spd.csv")
    logging.info("Active-window win rate: %.2f%%", win_rate)
    logging.info("Active-window model score: %.2f%%", score)


if __name__ == "__main__":
    main()
