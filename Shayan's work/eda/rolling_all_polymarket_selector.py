#!/usr/bin/env python3
"""
rolling_all_polymarket_selector.py

Rolling historical selector for all open Polymarket questions, including
crypto-adjacent markets. This lets us test whether using the full Polymarket
universe improves the DCA overlay versus filtering to non-crypto or macro-only
questions.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rolling_open_polymarket_selector import (  # type: ignore
    build_forward_feature_rows,
    currently_open_markets,
    get_rebalance_dates,
    normalize_ids,
    score_open_questions,
)
from polymarket_non_btc_correlation_scan import (  # type: ignore
    build_daily_token_panel,
    load_btc,
    load_polymarket_tables,
)


def parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parent.parent.parent
    shayan_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--btc-csv",
        type=Path,
        default=workspace_root / "data" / "Coin Metrics" / "coinmetrics_btc.csv",
    )
    parser.add_argument(
        "--polymarket-dir",
        type=Path,
        default=workspace_root / "data" / "Polymarket",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=shayan_root / "eda" / "outputs" / "rolling_all_selector",
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
        format="%(levelname)s: %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading BTC data...")
    btc = load_btc(args.btc_csv, args.start_date, args.end_date)

    logging.info("Loading Polymarket tables...")
    markets, tokens, odds, summary = load_polymarket_tables(args.polymarket_dir)
    markets = normalize_ids(markets)
    tokens = normalize_ids(tokens)
    odds = normalize_ids(odds)
    summary = normalize_ids(summary)

    logging.info("Building daily all-question Polymarket panel...")
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
    if len(rebalance_dates) == 0:
        raise ValueError("No rebalance dates available with the current settings.")

    logging.info("Scoring all open questions across %d rebalance dates...", len(rebalance_dates))
    selections_list: list[pd.DataFrame] = []
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
            selections_list.append(selected)

    if not selections_list:
        raise ValueError("No open questions met the rolling selection criteria.")

    selections = pd.concat(selections_list, axis=0, ignore_index=True)
    selections.to_csv(args.output_dir / "rolling_selected_all_questions.csv", index=False)

    feature_panel = build_forward_feature_rows(merged, selections, rebalance_dates, btc.index)
    feature_panel.insert(0, "PriceUSD", btc["PriceUSD"])
    feature_panel.insert(1, "btc_return", btc["btc_return"])
    feature_panel.to_csv(args.output_dir / "rolling_all_question_features.csv", index=True)

    latest = selections.sort_values("rebalance_date").groupby("rebalance_date").head(args.top_n)
    print("\n=== Rolling All-Question Polymarket Selector ===")
    print(f"Rebalance dates processed: {len(rebalance_dates)}")
    print("\nLatest selections:")
    print(
        latest.tail(args.top_n)[
            [
                "rebalance_date",
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
