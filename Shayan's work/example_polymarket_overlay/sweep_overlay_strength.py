import json
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


MIN_BACKTEST_START = pd.Timestamp("2018-01-01")
EXPERIMENT_WINDOW_DAYS = 180
STRENGTHS = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
_FEATURES_DF = None
_CURRENT_STRENGTH = 0.0


def compute_weights_wrapper(df_window: pd.DataFrame) -> pd.Series:
    """Adapt overlay model to template backtest engine with tunable strength."""
    global _FEATURES_DF, _CURRENT_STRENGTH

    if _FEATURES_DF is None:
        raise ValueError("Features not precomputed. Call precompute_features() first.")
    if df_window.empty:
        return pd.Series(dtype=float)

    start_date = df_window.index.min()
    end_date = df_window.index.max()
    current_date = end_date
    return compute_window_weights(
        _FEATURES_DF,
        start_date,
        end_date,
        current_date,
        overlay_strength=_CURRENT_STRENGTH,
    )


def main():
    global _FEATURES_DF, _CURRENT_STRENGTH

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    btc_df = load_data()
    _FEATURES_DF = precompute_features(btc_df)

    active_overlay = _FEATURES_DF["polymarket_overlay_signal"].ne(0)
    if active_overlay.any():
        analysis_start = max(MIN_BACKTEST_START, _FEATURES_DF.index[active_overlay].min())
        btc_df = btc_df.loc[analysis_start:].copy()
        _FEATURES_DF = _FEATURES_DF.loc[analysis_start:].copy()
    else:
        analysis_start = MIN_BACKTEST_START

    output_root = Path(__file__).parent / "sweep_output"
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for strength in STRENGTHS:
        _CURRENT_STRENGTH = strength
        run_dir = output_root / f"strength_{str(strength).replace('.', '_')}"
        logging.info("Running sweep for overlay strength=%s", strength)
        run_full_analysis(
            btc_df=btc_df,
            features_df=_FEATURES_DF,
            compute_weights_fn=compute_weights_wrapper,
            output_dir=run_dir,
            strategy_label=f"MA + Overlay ({strength})",
            start_date=analysis_start.strftime("%Y-%m-%d"),
            window_days=EXPERIMENT_WINDOW_DAYS,
        )
        metrics = json.load(open(run_dir / "metrics.json"))["summary_metrics"]
        rows.append({"overlay_strength": strength, **metrics})

    results = pd.DataFrame(rows).sort_values(
        ["win_rate", "score", "exp_decay_percentile"],
        ascending=[False, False, False],
    )
    results.to_csv(output_root / "overlay_strength_sweep.csv", index=False)
    print(results.to_string(index=False))
    print(f"\nSaved outputs to: {output_root}")


if __name__ == "__main__":
    main()
