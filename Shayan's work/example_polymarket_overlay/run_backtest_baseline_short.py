import logging
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from template.backtest_template import run_full_analysis
from template.model_development_template import compute_window_weights, precompute_features
from template.prelude_template import load_data

_FEATURES_DF = None
START_DATE = "2025-03-12"
WINDOW_DAYS = 180


def compute_weights_wrapper(df_window: pd.DataFrame) -> pd.Series:
    """Adapt baseline template model to the shared backtest engine."""
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

    logging.info("Starting Bitcoin DCA Strategy Analysis - MA Baseline (short window)")
    btc_df = load_data().loc[START_DATE:].copy()

    logging.info("Precomputing features (MA baseline)...")
    _FEATURES_DF = precompute_features(load_data()).loc[START_DATE:].copy()

    output_dir = Path(__file__).parent / "output_baseline_short"
    run_full_analysis(
        btc_df=btc_df,
        features_df=_FEATURES_DF,
        compute_weights_fn=compute_weights_wrapper,
        output_dir=output_dir,
        strategy_label="MA Baseline (180d)",
        start_date=START_DATE,
        window_days=WINDOW_DAYS,
    )


if __name__ == "__main__":
    main()
