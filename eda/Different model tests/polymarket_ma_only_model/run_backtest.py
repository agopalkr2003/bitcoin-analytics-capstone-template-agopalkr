import logging
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from template.backtest_template import run_full_analysis
from template.prelude_template import load_data

from example_polymarket_overlay.model_development_polymarket_overlay import (
    compute_window_weights,
    precompute_features,
)

_FEATURES_DF = None
MIN_BACKTEST_START = pd.Timestamp("2018-01-01")
# Selected after testing multiple overlay strengths on the active
# Polymarket window. We chose 180-day windows for the experiment and kept
# overlay strength at 2.0 because that setting achieved the highest win rate.
EXPERIMENT_WINDOW_DAYS = 180


def compute_weights_wrapper(df_window: pd.DataFrame) -> pd.Series:
    """Adapt overlay model to the template backtest engine."""
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

    logging.info("Starting Bitcoin DCA Strategy Analysis - MA + Polymarket Overlay")
    btc_df = load_data()

    logging.info("Precomputing features (MA + Polymarket overlay)...")
    _FEATURES_DF = precompute_features(btc_df)

    active_overlay = _FEATURES_DF["polymarket_overlay_signal"].ne(0)
    if active_overlay.any():
        overlay_start = _FEATURES_DF.index[active_overlay].min()
        analysis_start = max(MIN_BACKTEST_START, overlay_start)
        logging.info(
            "Restricting overlay backtest to active-signal period starting %s",
            analysis_start.date(),
        )
        btc_df = btc_df.loc[analysis_start:].copy()
        _FEATURES_DF = _FEATURES_DF.loc[analysis_start:].copy()
    else:
        logging.warning(
            "Polymarket overlay signal is zero for all dates; using full backtest range."
        )
        analysis_start = MIN_BACKTEST_START

    output_dir = Path(__file__).parent / "output"
    run_full_analysis(
        btc_df=btc_df,
        features_df=_FEATURES_DF,
        compute_weights_fn=compute_weights_wrapper,
        output_dir=output_dir,
        strategy_label="MA + Polymarket Overlay",
        start_date=analysis_start.strftime("%Y-%m-%d"),
        window_days=EXPERIMENT_WINDOW_DAYS,
    )


if __name__ == "__main__":
    main()
