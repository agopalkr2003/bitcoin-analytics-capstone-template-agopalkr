#!/usr/bin/env python3
"""Compare baseline MA weights vs MA + Polymarket overlay weights.

This script computes one consistent full-horizon allocation schedule for both
models across the shared backtest date range and exports a daily comparison.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from template.prelude_template import load_data
from template.model_development_template import (
    compute_window_weights as compute_base_weights,
    precompute_features as precompute_base_features,
)
from example_polymarket_overlay.model_development_polymarket_overlay import (
    compute_window_weights as compute_overlay_weights,
    precompute_features as precompute_overlay_features,
)


BACKTEST_START = pd.Timestamp("2018-01-01")
BACKTEST_END = pd.Timestamp("2025-12-31")


def build_weight_comparison() -> pd.DataFrame:
    """Compute baseline and overlay weights over the full backtest horizon."""
    btc = load_data()
    base_features = precompute_base_features(btc)
    overlay_features = precompute_overlay_features(btc)

    base_weights = compute_base_weights(
        base_features, BACKTEST_START, BACKTEST_END, BACKTEST_END
    )
    overlay_weights = compute_overlay_weights(
        overlay_features, BACKTEST_START, BACKTEST_END, BACKTEST_END
    )

    comparison = pd.DataFrame(
        {
            "base_weight": base_weights,
            "overlay_weight": overlay_weights,
        }
    )
    comparison["weight_diff"] = comparison["overlay_weight"] - comparison["base_weight"]
    comparison["abs_weight_diff"] = comparison["weight_diff"].abs()
    comparison["base_cum_weight"] = comparison["base_weight"].cumsum()
    comparison["overlay_cum_weight"] = comparison["overlay_weight"].cumsum()
    return comparison


def plot_weight_comparison(df: pd.DataFrame, output_file: Path) -> None:
    """Plot daily weight comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(df.index, df["base_weight"], label="MA Baseline", linewidth=1.5)
    axes[0].plot(df.index, df["overlay_weight"], label="MA + Overlay", linewidth=1.5)
    axes[0].set_ylabel("Daily Weight")
    axes[0].set_title("Daily Weight Comparison")
    axes[0].legend()

    axes[1].plot(df.index, df["weight_diff"], color="darkred", linewidth=1.2)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Overlay - Base")
    axes[1].set_title("Daily Weight Difference")

    plt.tight_layout()
    plt.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    output_dir = ROOT_DIR / "eda" / "outputs" / "weight_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Computing baseline vs overlay weight comparison...")
    comparison = build_weight_comparison()
    comparison.to_csv(output_dir / "ma_vs_overlay_weight_comparison.csv", index=True)
    plot_weight_comparison(comparison, output_dir / "ma_vs_overlay_weight_comparison.png")

    summary = {
        "mean_abs_weight_diff": float(comparison["abs_weight_diff"].mean()),
        "max_abs_weight_diff": float(comparison["abs_weight_diff"].max()),
        "days_with_diff_gt_1bp": int((comparison["abs_weight_diff"] > 0.0001).sum()),
        "days_with_diff_gt_10bp": int((comparison["abs_weight_diff"] > 0.001).sum()),
    }
    print(summary)
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
