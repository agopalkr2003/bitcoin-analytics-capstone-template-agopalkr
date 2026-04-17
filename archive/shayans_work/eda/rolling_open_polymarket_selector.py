#!/usr/bin/env python3
"""
rolling_open_polymarket_selector.py

Rolling historical selector for open non-BTC Polymarket questions that may be
useful as candidate DCA features.

The key idea is to avoid picking fixed historical question IDs. Instead, on
each rebalance date we:
1) look only at questions that were open on that date
2) score them using only historical data available up to that date
3) keep the strongest positive-lead questions
4) build DCA-ready daily aggregate features from the selected open questions
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
    NON_CRYPTO_PATTERN,
    build_daily_token_panel,
    filter_non_btc_markets,
    load_btc,
    load_polymarket_tables,
    safe_corr,
)


def normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize market/token identifiers to strings."""
    out = df.copy()
    if "market_id" in out.columns:
        out["market_id"] = out["market_id"].astype(str)
    if "token_id" in out.columns:
        out["token_id"] = out["token_id"].astype(str)
    return out


def filter_non_crypto_questions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove obvious crypto-adjacent questions."""
    out = df.copy()
    question = out["question"].fillna("").astype(str)
    category = out["category"].fillna("").astype(str)
    mask = ~(
        question.str.contains(NON_CRYPTO_PATTERN, na=False)
        | category.str.contains(NON_CRYPTO_PATTERN, na=False)
    )
    return out.loc[mask].copy()


def get_rebalance_dates(
    btc_index: pd.DatetimeIndex,
    train_window_days: int,
    start_date: str | None,
    end_date: str | None,
    freq: str,
) -> pd.DatetimeIndex:
    """Build rebalance dates after enough training history is available."""
    start = btc_index.min() + pd.Timedelta(days=train_window_days)
    end = btc_index.max()
    if start_date:
        start = max(start, pd.to_datetime(start_date))
    if end_date:
        end = min(end, pd.to_datetime(end_date))

    if start > end:
        return pd.DatetimeIndex([])

    raw = pd.date_range(start=start, end=end, freq=freq)
    # Align to actual BTC dates.
    aligned = []
    for date in raw:
        valid = btc_index[btc_index >= date]
        if len(valid) > 0:
            aligned.append(valid[0])
    return pd.DatetimeIndex(sorted(set(aligned)))


def currently_open_markets(markets: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Approximate which markets were open on a given date."""
    created_ok = markets["created_at"].isna() | (markets["created_at"] <= as_of)
    end_ok = markets["end_date"].isna() | (markets["end_date"] >= as_of)

    # Historical active/closed may not be point-in-time perfect, so use them
    # only as soft filters combined with date bounds.
    active_ok = (~markets["active"].eq(False)) if "active" in markets.columns else True
    closed_ok = (~markets["closed"].eq(True)) if "closed" in markets.columns else True

    return markets.loc[created_ok & end_ok & active_ok & closed_ok].copy()


def score_open_questions(
    merged: pd.DataFrame,
    open_markets: pd.DataFrame,
    train_end: pd.Timestamp,
    train_window_days: int,
    min_obs: int,
    min_volume: float,
    max_lag: int,
    top_n: int,
) -> pd.DataFrame:
    """Score currently open questions using only trailing historical data."""
    train_start = train_end - pd.Timedelta(days=train_window_days)
    train = merged[
        (merged["trade_date"] >= train_start) & (merged["trade_date"] <= train_end)
    ].copy()
    if train.empty:
        return pd.DataFrame()

    open_ids = set(open_markets["market_id"].astype(str))
    train = train[train["market_id"].astype(str).isin(open_ids)].copy()
    if train.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    meta_cols = ["question", "outcome", "category", "volume", "created_at", "end_date"]

    for (market_id, token_id), grp in train.groupby(["market_id", "token_id"], sort=False):
        grp = grp.sort_values("trade_date").copy()
        n_obs = len(grp.dropna(subset=["token_return", "btc_return"]))
        if n_obs < min_obs:
            continue

        volume = float(grp["volume"].iloc[-1]) if pd.notna(grp["volume"].iloc[-1]) else 0.0
        if volume < min_volume:
            continue

        same_day = safe_corr(grp["token_return"], grp["btc_return"])

        best_lag = None
        best_corr = None
        best_n = None
        for lag in range(1, max_lag + 1):
            shifted = grp["token_return"].shift(lag)
            valid = pd.DataFrame(
                {"shifted_token_return": shifted, "btc_return": grp["btc_return"]}
            ).dropna()
            if len(valid) < min_obs:
                continue
            corr = safe_corr(valid["shifted_token_return"], valid["btc_return"])
            if np.isnan(corr):
                continue
            if best_corr is None or abs(corr) > abs(best_corr):
                best_lag = lag
                best_corr = corr
                best_n = len(valid)

        if best_corr is None or best_lag is None:
            continue

        meta = grp.iloc[-1][meta_cols].to_dict()
        rows.append(
            {
                "rebalance_date": train_end,
                "market_id": str(market_id),
                "token_id": str(token_id),
                "n_obs": n_obs,
                "same_day_corr": same_day,
                "best_lag": best_lag,
                "best_lag_corr": best_corr,
                "abs_best_lag_corr": abs(best_corr),
                "best_lag_n": best_n,
                **meta,
            }
        )

    scores = pd.DataFrame(rows)
    if scores.empty:
        return scores

    scores = scores.sort_values(
        ["abs_best_lag_corr", "n_obs", "volume"], ascending=[False, False, False]
    )
    return scores.head(top_n).reset_index(drop=True)


def build_forward_feature_rows(
    merged: pd.DataFrame,
    selections: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    btc_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Aggregate selected open-question features over each forward holding period."""
    all_rows: list[pd.DataFrame] = []

    for i, rebalance_date in enumerate(rebalance_dates):
        next_date = (
            rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else btc_index.max() + pd.Timedelta(days=1)
        )
        current = selections[selections["rebalance_date"] == rebalance_date].copy()
        if current.empty:
            continue

        selected_keys = set(zip(current["market_id"], current["token_id"]))
        period = merged[
            (merged["trade_date"] >= rebalance_date)
            & (merged["trade_date"] < next_date)
        ].copy()
        period = period[
            period.apply(lambda r: (str(r["market_id"]), str(r["token_id"])) in selected_keys, axis=1)
        ].copy()
        if period.empty:
            continue

        period["market_id"] = period["market_id"].astype(str)
        period["token_id"] = period["token_id"].astype(str)
        period = period.merge(
            current[
                [
                    "market_id",
                    "token_id",
                    "question",
                    "best_lag",
                    "best_lag_corr",
                    "abs_best_lag_corr",
                ]
            ],
            on=["market_id", "token_id"],
            how="left",
        )

        grouped = (
            period.groupby("trade_date", as_index=True)
            .agg(
                selected_question_count=("market_id", "nunique"),
                open_question_mean_return=("token_return", "mean"),
                open_question_mean_price=("price", "mean"),
                open_question_mean_abs_corr=("abs_best_lag_corr", "mean"),
                open_question_max_abs_corr=("abs_best_lag_corr", "max"),
            )
            .sort_index()
        )
        grouped["rebalance_date"] = rebalance_date
        all_rows.append(grouped)

    if not all_rows:
        return pd.DataFrame(index=btc_index)

    features = pd.concat(all_rows, axis=0).sort_index()
    features = features[~features.index.duplicated(keep="last")]
    features = features.reindex(btc_index)

    features["open_question_mean_return_lag1"] = features["open_question_mean_return"].shift(1)
    rolling_mean = features["open_question_mean_return"].rolling(20, min_periods=10).mean()
    rolling_std = features["open_question_mean_return"].rolling(20, min_periods=10).std()
    features["open_question_mean_return_z20_lag1"] = (
        ((features["open_question_mean_return"] - rolling_mean) / rolling_std).shift(1)
    )
    features["open_question_mean_price_lag1"] = features["open_question_mean_price"].shift(1)
    features["open_question_mean_abs_corr_lag1"] = features["open_question_mean_abs_corr"].shift(1)
    features["open_question_max_abs_corr_lag1"] = features["open_question_max_abs_corr"].shift(1)
    features["selected_question_count_lag1"] = features["selected_question_count"].shift(1)

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
        "--output-dir",
        type=Path,
        default=repo_root / "eda" / "outputs" / "rolling_open_selector",
    )
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--train-window-days", type=int, default=365)
    parser.add_argument("--rebalance-freq", default="MS", help="Pandas date freq, e.g. MS or W")
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
    markets = normalize_ids(filter_non_btc_markets(markets))
    markets = filter_non_crypto_questions(markets)
    tokens = normalize_ids(tokens)
    odds = normalize_ids(odds)
    summary = normalize_ids(summary)

    logging.info("Building daily non-crypto Polymarket panel...")
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

    logging.info("Scoring open questions across %d rebalance dates...", len(rebalance_dates))
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
    selections.to_csv(args.output_dir / "rolling_selected_open_questions.csv", index=False)

    feature_panel = build_forward_feature_rows(merged, selections, rebalance_dates, btc.index)
    feature_panel.insert(0, "PriceUSD", btc["PriceUSD"])
    feature_panel.insert(1, "btc_return", btc["btc_return"])
    feature_panel.to_csv(args.output_dir / "rolling_open_question_features.csv", index=True)

    latest = selections.sort_values("rebalance_date").groupby("rebalance_date").head(args.top_n)
    print("\n=== Rolling Open Question Selector ===")
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
