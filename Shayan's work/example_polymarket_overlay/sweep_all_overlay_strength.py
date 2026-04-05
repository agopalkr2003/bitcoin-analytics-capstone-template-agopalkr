import logging
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from template.model_development_template import (
    compute_window_weights as compute_base_weights,
    precompute_features as precompute_base_features,
)
from template.prelude_template import backtest_dynamic_dca, load_data
from example_polymarket_overlay.model_development_all_polymarket_overlay import (
    compute_window_weights,
    precompute_features,
)


MIN_BACKTEST_START = pd.Timestamp("2018-01-01")
EXPERIMENT_WINDOW_DAYS = 180
STRENGTHS = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]


def summarize_run(
    btc_df: pd.DataFrame,
    features_df: pd.DataFrame,
    compute_fn,
    start_date: str,
    label: str,
):
    """Run SPD backtest and return a compact metric summary."""

    def wrapper(df_window: pd.DataFrame, current_date=None) -> pd.Series:
        if df_window.empty:
            return pd.Series(dtype=float)
        start = df_window.index.min()
        end = df_window.index.max()
        current = end if current_date is None else current_date
        return compute_fn(features_df, start, end, current)

    df_spd, exp_decay = backtest_dynamic_dca(
        btc_df,
        wrapper,
        features_df=features_df,
        strategy_label=label,
        start_date=start_date,
        end_date="2025-12-31",
        window_days=EXPERIMENT_WINDOW_DAYS,
    )

    win_rate = (df_spd["dynamic_percentile"] > df_spd["uniform_percentile"]).mean() * 100
    score = 0.5 * win_rate + 0.5 * exp_decay
    excess = df_spd["dynamic_percentile"] - df_spd["uniform_percentile"]

    return {
        "windows": len(df_spd),
        "win_rate": float(win_rate),
        "exp_decay_percentile": float(exp_decay),
        "score": float(score),
        "mean_excess": float(excess.mean()),
        "median_excess": float(excess.median()),
        "mean_ratio": float(
            (df_spd["dynamic_percentile"] / df_spd["uniform_percentile"]).mean()
        ),
        "median_ratio": float(
            (df_spd["dynamic_percentile"] / df_spd["uniform_percentile"]).median()
        ),
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    btc_df = load_data()
    all_features_df = precompute_features(btc_df)

    active_overlay = all_features_df["polymarket_overlay_signal"].ne(0)
    if active_overlay.any():
        analysis_start = max(MIN_BACKTEST_START, all_features_df.index[active_overlay].min())
    else:
        analysis_start = MIN_BACKTEST_START
    analysis_start_str = analysis_start.strftime("%Y-%m-%d")

    output_root = Path(__file__).parent / "sweep_output_all"
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for strength in STRENGTHS:
        logging.info("Running all-question sweep for overlay strength=%s", strength)

        def compute_strength(features_df, start_date, end_date, current_date):
            return compute_window_weights(
                features_df,
                start_date,
                end_date,
                current_date,
                overlay_strength=strength,
            )

        metrics = summarize_run(
            btc_df=btc_df,
            features_df=all_features_df,
            compute_fn=compute_strength,
            start_date=analysis_start_str,
            label=f"MA + All Polymarket ({strength})",
        )
        rows.append({"overlay_strength": strength, **metrics})

    baseline_metrics = summarize_run(
        btc_df=btc_df,
        features_df=precompute_base_features(btc_df),
        compute_fn=compute_base_weights,
        start_date=analysis_start_str,
        label="MA baseline",
    )

    results = pd.DataFrame(rows).sort_values(
        ["win_rate", "score", "exp_decay_percentile"],
        ascending=[False, False, False],
    )
    baseline = pd.DataFrame([{"overlay_strength": "baseline", **baseline_metrics}])

    results.to_csv(output_root / "all_overlay_strength_sweep.csv", index=False)
    baseline.to_csv(output_root / "baseline_short_window_metrics.csv", index=False)

    print("\n=== All-Question Overlay Strength Sweep ===")
    print(results.to_string(index=False))
    print("\n=== Baseline ===")
    print(baseline.to_string(index=False))
    print(f"\nSaved outputs to: {output_root}")


if __name__ == "__main__":
    main()
