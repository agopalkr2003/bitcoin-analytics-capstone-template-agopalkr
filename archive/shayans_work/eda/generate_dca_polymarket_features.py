#!/usr/bin/env python3
"""
generate_dca_polymarket_features.py

Create candidate Polymarket feature series for the DCA model from the
non-BTC correlation scan outputs.

This script focuses on non-crypto questions that:
1) are relatively liquid and have enough observations
2) show positive lead lags versus BTC returns
3) can therefore be treated as candidate predictive features rather than
   questions that simply react after BTC moves
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from polymarket_non_btc_correlation_scan import (  # type: ignore
    build_daily_token_panel,
    filter_non_btc_markets,
    load_btc,
    load_polymarket_tables,
)


def select_candidate_questions(
    rankings: pd.DataFrame,
    min_obs: int,
    min_volume: float,
    top_n: int,
) -> pd.DataFrame:
    """Keep the most plausible non-crypto questions with predictive lags.

    Positive best_lag means past question moves align with later BTC moves.
    Negative best_lag means the question tends to react after BTC and is less
    useful as a forward-looking feature.
    """
    if rankings.empty:
        return rankings.copy()

    selected = rankings.copy()
    selected = selected[
        (selected["n_obs"] >= min_obs)
        & (selected["volume"].fillna(0) >= min_volume)
        & (selected["best_lag"] > 0)
    ].copy()
    selected = selected.sort_values(
        ["abs_best_lag_corr", "best_lag_corr", "n_obs", "volume"],
        ascending=[False, False, False, False],
    ).head(top_n)
    return selected.reset_index(drop=True)


def slugify_question(question: str, max_len: int = 60) -> str:
    """Create a compact feature-safe identifier from question text."""
    slug = (
        question.lower()
        .replace("%", "pct")
        .replace("$", "usd")
        .replace("-", " ")
    )
    slug = "".join(ch if ch.isalnum() else "_" for ch in slug)
    slug = "_".join(part for part in slug.split("_") if part)
    return slug[:max_len].rstrip("_")


def make_candidate_feature_panel(
    daily: pd.DataFrame,
    candidates: pd.DataFrame,
    btc: pd.DataFrame,
) -> pd.DataFrame:
    """Build aligned daily features for selected candidate questions."""
    features = pd.DataFrame(index=btc.index.copy())
    features["PriceUSD"] = btc["PriceUSD"]
    features["btc_return"] = btc["btc_return"]

    for row in candidates.itertuples(index=False):
        market_id = str(row.market_id)
        token_id = str(row.token_id)
        lead_days = int(row.best_lag)
        feature_base = slugify_question(str(row.question))
        if not feature_base:
            feature_base = f"market_{market_id}"

        subset = daily[
            (daily["market_id"] == market_id) & (daily["token_id"] == token_id)
        ].sort_values("trade_date")
        if subset.empty:
            continue

        series = subset.set_index("trade_date")
        raw_return = series["token_return"].reindex(features.index)
        raw_price = series["price"].reindex(features.index)

        # Shift by one day to respect no-lookahead when used in a daily model.
        # Positive lead_days means the question historically moved before BTC,
        # so yesterday's question move is a plausible candidate feature today.
        features[f"{feature_base}_return_lag1"] = raw_return.shift(1)
        features[f"{feature_base}_z20_lag1"] = (
            ((raw_return - raw_return.rolling(20, min_periods=10).mean())
            / raw_return.rolling(20, min_periods=10).std())
            .shift(1)
        )
        features[f"{feature_base}_price_lag1"] = raw_price.shift(1)
        features[f"{feature_base}_lead_days"] = lead_days
        features[f"{feature_base}_corr"] = float(row.best_lag_corr)

    return features


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--btc-csv",
        type=Path,
        default=repo_root / "data" / "Coin Metrics" / "coinmetrics_btc.csv",
    )
    parser.add_argument(
        "--polymarket-dir",
        type=Path,
        default=repo_root / "data" / "Polymarket",
    )
    parser.add_argument(
        "--rankings-csv",
        type=Path,
        default=repo_root
        / "eda"
        / "outputs"
        / "non_btc_polymarket_corr"
        / "plausible_non_crypto_market_correlations.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "eda" / "outputs" / "dca_polymarket_features",
    )
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--min-obs", type=int, default=60)
    parser.add_argument("--min-volume", type=float, default=1000.0)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading BTC data...")
    btc = load_btc(args.btc_csv, args.start_date, args.end_date)

    logging.info("Loading saved ranking results...")
    rankings = pd.read_csv(args.rankings_csv)

    logging.info("Selecting candidate questions with positive lead lags...")
    candidates = select_candidate_questions(
        rankings, args.min_obs, args.min_volume, args.top_n
    )
    if candidates.empty:
        raise ValueError(
            "No candidate questions met the filters. Try lowering min_obs/min_volume."
        )

    logging.info("Rebuilding daily Polymarket panel for selected questions...")
    markets, tokens, odds, summary = load_polymarket_tables(args.polymarket_dir)
    markets = filter_non_btc_markets(markets)
    daily = build_daily_token_panel(markets, tokens, odds, summary, btc.index)
    daily["market_id"] = daily["market_id"].astype(str)
    daily["token_id"] = daily["token_id"].astype(str)
    candidates["market_id"] = candidates["market_id"].astype(str)
    candidates["token_id"] = candidates["token_id"].astype(str)

    selected_daily = daily.merge(
        candidates[["market_id", "token_id", "question", "best_lag", "best_lag_corr"]],
        on=["market_id", "token_id"],
        how="inner",
    )
    selected_daily.to_csv(args.output_dir / "selected_question_daily_panel.csv", index=False)

    features = make_candidate_feature_panel(selected_daily, candidates, btc)
    features.to_csv(args.output_dir / "candidate_polymarket_features.csv", index=True)
    candidates.to_csv(args.output_dir / "candidate_question_summary.csv", index=False)

    print("\n=== Candidate Polymarket Features For DCA ===")
    print(
        candidates[
            [
                "question",
                "outcome",
                "n_obs",
                "best_lag",
                "best_lag_corr",
                "volume",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
