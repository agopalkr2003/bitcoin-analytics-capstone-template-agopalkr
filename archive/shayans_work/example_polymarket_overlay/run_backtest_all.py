import logging
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from template.backtest_template import run_full_analysis
from template.prelude_template import load_data
from example_polymarket_overlay.model_development_all_polymarket_overlay import (
    compute_window_weights,
    precompute_features,
)

_FEATURES_DF = None
MIN_BACKTEST_START = pd.Timestamp("2018-01-01")
EXPERIMENT_WINDOW_DAYS = 180


def compute_weights_wrapper(df_window: pd.DataFrame) -> pd.Series:
    global _FEATURES_DF
    if _FEATURES_DF is None:
        raise ValueError("Features not precomputed. Call precompute_features() first.")
    if df_window.empty:
        return pd.Series(dtype=float)
    return compute_window_weights(
        _FEATURES_DF,
        df_window.index.min(),
        df_window.index.max(),
        df_window.index.max(),
    )


def main():
    global _FEATURES_DF
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    btc_df = load_data()
    _FEATURES_DF = precompute_features(btc_df)

    active = _FEATURES_DF["polymarket_overlay_signal"].ne(0)
    analysis_start = max(MIN_BACKTEST_START, _FEATURES_DF.index[active].min()) if active.any() else MIN_BACKTEST_START
    btc_df = btc_df.loc[analysis_start:].copy()
    _FEATURES_DF = _FEATURES_DF.loc[analysis_start:].copy()

    output_dir = Path(__file__).parent / "output_all"
    run_full_analysis(
        btc_df=btc_df,
        features_df=_FEATURES_DF,
        compute_weights_fn=compute_weights_wrapper,
        output_dir=output_dir,
        strategy_label="MA + All Polymarket Overlay",
        start_date=analysis_start.strftime("%Y-%m-%d"),
        window_days=EXPERIMENT_WINDOW_DAYS,
    )


if __name__ == "__main__":
    main()
