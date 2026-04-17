#!/usr/bin/env python3
"""Rolling selector for macro-focused non-BTC Polymarket questions."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rolling_open_polymarket_selector import (  # type: ignore
    build_forward_feature_rows,
    currently_open_markets,
    filter_non_crypto_questions,
    get_rebalance_dates,
    normalize_ids,
    score_open_questions,
)
from polymarket_non_btc_correlation_scan import (  # type: ignore
    build_daily_token_panel,
    filter_non_btc_markets,
    load_btc,
    load_polymarket_tables,
)


MACRO_PATTERN = re.compile(
    r"\b(fed|rate cuts?|interest rates?|recession|gdp|inflation|cpi|pce|"
    r"unemployment|jobs report|payrolls|treasury|yield|bank fail|banking|"
    r"market cap|liquidity|deficit|debt ceiling|soft landing|hard landing|"
    r"economic growth|contraction)\b",
    flags=re.IGNORECASE,
)


def filter_macro_questions(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only macro / broad-risk / financial-stress questions."""
    question = df["question"].fillna("").astype(str)
    category = df["category"].fillna("").astype(str)
    mask = question.str.contains(MACRO_PATTERN, na=False) | category.str.contains(
        MACRO_PATTERN, na=False
    )
    return df.loc[mask].copy()


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent.parent
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
        "--output-dir",
        type=Path,
        default=repo_root / "Shayan's work" / "eda" / "outputs" / "rolling_macro_selector",
    )
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--train-window-days", type=int, default=365)
    parser.add_argument("--rebalance-freq", default="MS")
    parser.add_argument("--min-obs", type=int, default=45)
    parser.add_argument("--min-volume", type=float, default=1000.0)
    parser.add_argument("--max-lag", type=int, default=7)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    btc = load_btc(args.btc_csv, args.start_date, args.end_date)
    markets, tokens, odds, summary = load_polymarket_tables(args.polymarket_dir)
    markets = normalize_ids(filter_non_btc_markets(markets))
    markets = filter_non_crypto_questions(markets)
    markets = filter_macro_questions(markets)
    tokens = normalize_ids(tokens)
    odds = normalize_ids(odds)
    summary = normalize_ids(summary)

    daily = build_daily_token_panel(markets, tokens, odds, summary, btc.index)
    daily = normalize_ids(daily)

    merged = daily.merge(
        btc[["PriceUSD", "btc_return"]],
        left_on="trade_date",
        right_index=True,
        how="inner",
    ).dropna(subset=["token_return", "btc_return"])

    rebalance_dates = get_rebalance_dates(
        btc.index,
        args.train_window_days,
        args.start_date,
        args.end_date,
        args.rebalance_freq,
    )

    selections = []
    for rebalance_date in rebalance_dates:
        open_markets = currently_open_markets(markets, rebalance_date)
        selected = score_open_questions(
            merged,
            open_markets,
            rebalance_date,
            args.train_window_days,
            args.min_obs,
            args.min_volume,
            args.max_lag,
            args.top_n,
        )
        if not selected.empty:
            selections.append(selected)

    if not selections:
        raise ValueError("No macro questions met the rolling selection criteria.")

    selections_df = pd.concat(selections, axis=0, ignore_index=True)
    selections_df.to_csv(args.output_dir / "rolling_macro_selected_questions.csv", index=False)

    feature_panel = build_forward_feature_rows(merged, selections_df, rebalance_dates, btc.index)
    feature_panel.insert(0, "PriceUSD", btc["PriceUSD"])
    feature_panel.insert(1, "btc_return", btc["btc_return"])
    feature_panel.to_csv(args.output_dir / "rolling_macro_question_features.csv", index=True)

    print("\n=== Rolling Macro Polymarket Selector ===")
    print(
        selections_df[
            ["rebalance_date", "question", "best_lag", "best_lag_corr", "volume"]
        ].head(15).to_string(index=False)
    )
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
