#!/usr/bin/env python3
"""
polymarket_non_btc_correlation_scan.py

Scan non-BTC Polymarket questions and rank the markets whose price changes are
most correlated with daily Bitcoin returns.

What it does
------------
1) Loads daily BTC price data from CoinMetrics
2) Loads Polymarket markets / tokens / odds history / summary tables
3) Excludes BTC-related markets
4) Merges Polymarket tables so each token price series is tied back to a question
5) Builds daily closing prices and daily returns per token
6) Computes same-day and lead-lag correlations versus BTC daily returns
7) Saves ranked CSV outputs for further EDA

Example
-------
python eda/polymarket_non_btc_correlation_scan.py \
    --btc-csv "data/Coin Metrics/coinmetrics_btc.csv" \
    --polymarket-dir "data/Polymarket" \
    --output-dir "eda/outputs/non_btc_polymarket_corr"
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BTC_PATTERN = re.compile(r"\b(?:bitcoin|btc)\b", flags=re.IGNORECASE)
CRYPTO_PATTERN = re.compile(r"crypto", flags=re.IGNORECASE)
NON_CRYPTO_PATTERN = re.compile(
    r"bitcoin|btc|crypto|ethereum|eth\b|solana|\bsol\b|doge|dogecoin|xrp|ripple|"
    r"cardano|ada\b|avax|avalanche|sui\b|litecoin|ltc\b|binance|bnb\b",
    flags=re.IGNORECASE,
)


def fix_timestamp(df: pd.DataFrame) -> None:
    """Fix likely corrupted timestamps in-place."""
    candidate_terms = ["timestamp", "trade", "created_at", "end_date", "time", "date"]

    for col in df.columns:
        if any(term in col.lower() for term in candidate_terms):
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                if not df[col].empty and df[col].max() < pd.Timestamp("2020-01-01"):
                    logging.info("Fixing corrupted timestamps in column: %s", col)
                    ns_values = df[col].values.astype("datetime64[ns]").astype("int64")
                    df[col] = pd.to_datetime(ns_values * 1000)

                mask = df[col] < pd.Timestamp("2020-01-01")
                if mask.any():
                    df.loc[mask, col] = pd.NaT


def load_btc(btc_csv: Path, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    """Load BTC daily data and compute daily returns."""
    df_btc = pd.read_csv(btc_csv, usecols=["time", "PriceUSD"])
    df_btc["time"] = pd.to_datetime(df_btc["time"])
    df_btc = df_btc.set_index("time")
    df_btc.index = df_btc.index.normalize().tz_localize(None)
    df_btc = df_btc.loc[~df_btc.index.duplicated(keep="last")].sort_index()

    if start_date:
        df_btc = df_btc[df_btc.index >= pd.to_datetime(start_date)]
    if end_date:
        df_btc = df_btc[df_btc.index <= pd.to_datetime(end_date)]

    if df_btc.empty:
        raise ValueError("BTC data is empty after date filtering.")

    df_btc["btc_return"] = df_btc["PriceUSD"].pct_change(fill_method=None)
    df_btc["btc_log_return"] = np.log(df_btc["PriceUSD"]).diff()
    return df_btc


def load_polymarket_tables(
    polymarket_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load required Polymarket tables."""
    files = {
        "markets": polymarket_dir / "finance_politics_markets.parquet",
        "tokens": polymarket_dir / "finance_politics_tokens.parquet",
        "odds": polymarket_dir / "finance_politics_odds_history.parquet",
        "summary": polymarket_dir / "finance_politics_summary.parquet",
    }

    for name, file_path in files.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required {name} file: {file_path}")

    df_markets = pd.read_parquet(
        files["markets"],
        columns=[
            "market_id",
            "question",
            "slug",
            "event_slug",
            "category",
            "volume",
            "active",
            "closed",
            "created_at",
            "end_date",
        ],
    )
    df_tokens = pd.read_parquet(files["tokens"], columns=["market_id", "token_id", "outcome"])
    df_odds = pd.read_parquet(
        files["odds"], columns=["market_id", "token_id", "timestamp", "price"]
    )
    df_summary = pd.read_parquet(
        files["summary"],
        columns=[
            "market_id",
            "trade_count",
            "token_count",
            "first_trade",
            "last_trade",
            "active",
            "volume",
        ],
    )

    for df in [df_markets, df_tokens, df_odds, df_summary]:
        for col in df.columns:
            if any(k in col.lower() for k in ["timestamp", "created_at", "end_date", "time", "date"]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass
        fix_timestamp(df)

    df_odds["timestamp"] = pd.to_datetime(df_odds["timestamp"], errors="coerce")
    return df_markets, df_tokens, df_odds, df_summary


def filter_non_btc_markets(df_markets: pd.DataFrame) -> pd.DataFrame:
    """Exclude BTC-related markets based on question and metadata fields."""
    question_match = df_markets["question"].astype("string").str.contains(BTC_PATTERN, na=False)
    slug_match = df_markets["slug"].astype("string").str.contains(BTC_PATTERN, na=False)
    event_match = df_markets["event_slug"].astype("string").str.contains(BTC_PATTERN, na=False)
    category_match = df_markets["category"].astype("string").str.contains(CRYPTO_PATTERN, na=False)

    keep_mask = ~(question_match | slug_match | event_match | category_match)
    out = df_markets.loc[keep_mask].copy()
    if out.empty:
        raise ValueError("No non-BTC markets found after filtering.")
    return out


def build_daily_token_panel(
    df_markets: pd.DataFrame,
    df_tokens: pd.DataFrame,
    df_odds: pd.DataFrame,
    df_summary: pd.DataFrame,
    btc_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build daily closing prices and returns for non-BTC Polymarket tokens."""
    panel = df_odds.merge(df_tokens, on=["market_id", "token_id"], how="inner")
    panel = panel.merge(
        df_markets[
            [
                "market_id",
                "question",
                "slug",
                "event_slug",
                "category",
                "volume",
                "active",
                "closed",
                "created_at",
                "end_date",
            ]
        ],
        on="market_id",
        how="inner",
    )
    panel = panel.merge(
        df_summary[
            ["market_id", "trade_count", "token_count", "first_trade", "last_trade"]
        ],
        on="market_id",
        how="left",
    )

    panel = panel.dropna(subset=["timestamp", "price"]).copy()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"]).dt.tz_localize(None)
    panel["trade_date"] = panel["timestamp"].dt.normalize()
    panel = panel[
        (panel["trade_date"] >= btc_index.min()) & (panel["trade_date"] <= btc_index.max())
    ].copy()

    if panel.empty:
        raise ValueError("No odds-history rows remain after merging and date filtering.")

    panel = panel.sort_values(["market_id", "token_id", "trade_date", "timestamp"])

    # Use the final observed price each day as the daily close for that token.
    daily = panel.groupby(["market_id", "token_id", "trade_date"], as_index=False).last()
    daily = daily.sort_values(["market_id", "token_id", "trade_date"])

    daily["token_return"] = daily.groupby(["market_id", "token_id"])["price"].pct_change(
        fill_method=None
    )
    daily["token_price_change"] = daily.groupby(["market_id", "token_id"])["price"].diff()
    daily["obs_count"] = daily.groupby(["market_id", "token_id"])["price"].transform("size")

    return daily


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    """Compute correlation safely."""
    if len(x) < 2 or x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return np.nan
    return float(x.corr(y))


def correlation_by_group(
    daily: pd.DataFrame,
    df_btc: pd.DataFrame,
    min_obs: int,
    max_lag: int,
) -> pd.DataFrame:
    """Rank token-level and market-level relationships to BTC returns."""
    merged = daily.merge(
        df_btc[["PriceUSD", "btc_return", "btc_log_return"]],
        left_on="trade_date",
        right_index=True,
        how="inner",
    )
    merged = merged.dropna(subset=["token_return", "btc_return"]).copy()

    if merged.empty:
        raise ValueError("No overlapping BTC and Polymarket return observations found.")

    rows: list[dict[str, object]] = []
    group_cols = ["market_id", "token_id"]
    metadata_cols = [
        "question",
        "outcome",
        "category",
        "slug",
        "event_slug",
        "volume",
        "trade_count",
        "token_count",
        "created_at",
        "end_date",
        "first_trade",
        "last_trade",
    ]

    for (market_id, token_id), grp in merged.groupby(group_cols, sort=False):
        grp = grp.sort_values("trade_date").copy()
        n_obs = len(grp)
        if n_obs < min_obs:
            continue

        same_day_corr = safe_corr(grp["token_return"], grp["btc_return"])
        price_level_corr = safe_corr(grp["price"], grp["PriceUSD"])

        best_lag = 0
        best_lag_corr = same_day_corr
        best_lag_n = n_obs

        for lag in range(-max_lag, max_lag + 1):
            shifted = grp["token_return"].shift(lag)
            valid = pd.DataFrame(
                {"shifted_token_return": shifted, "btc_return": grp["btc_return"]}
            ).dropna()
            if len(valid) < min_obs:
                continue
            lag_corr = safe_corr(valid["shifted_token_return"], valid["btc_return"])
            if np.isnan(lag_corr):
                continue
            if np.isnan(best_lag_corr) or abs(lag_corr) > abs(best_lag_corr):
                best_lag = lag
                best_lag_corr = lag_corr
                best_lag_n = len(valid)

        metadata = grp.iloc[-1][metadata_cols].to_dict()
        rows.append(
            {
                "market_id": market_id,
                "token_id": token_id,
                "n_obs": n_obs,
                "same_day_corr": same_day_corr,
                "abs_same_day_corr": abs(same_day_corr) if pd.notna(same_day_corr) else np.nan,
                "price_level_corr": price_level_corr,
                "best_lag": best_lag,
                "best_lag_corr": best_lag_corr,
                "abs_best_lag_corr": abs(best_lag_corr) if pd.notna(best_lag_corr) else np.nan,
                **metadata,
            }
        )

    corr_df = pd.DataFrame(rows)
    if corr_df.empty:
        raise ValueError(
            f"No token series met the minimum observation threshold of {min_obs}."
        )

    corr_df = corr_df.sort_values(
        ["abs_best_lag_corr", "abs_same_day_corr", "n_obs"], ascending=[False, False, False]
    )

    market_best = (
        corr_df.sort_values(["market_id", "abs_best_lag_corr"], ascending=[True, False])
        .drop_duplicates(subset=["market_id"])
        .sort_values(["abs_best_lag_corr", "abs_same_day_corr", "n_obs"], ascending=[False, False, False])
        .reset_index(drop=True)
    )

    return corr_df.reset_index(drop=True), market_best, merged


def correlation_by_category(
    merged: pd.DataFrame,
    min_obs: int,
    max_lag: int,
) -> pd.DataFrame:
    """Rank category-level relationships to BTC returns using average daily token returns."""
    normalized_category = (
        merged["category"].fillna("").astype(str).str.strip().replace("", "Uncategorized")
    )

    category_daily = (
        merged.assign(category=normalized_category)
        .groupby(["trade_date", "category"], as_index=False)
        .agg(
            category_return=("token_return", "mean"),
            category_price_change=("token_price_change", "mean"),
            btc_return=("btc_return", "first"),
            btc_price=("PriceUSD", "first"),
            market_count=("market_id", "nunique"),
            token_count=("token_id", "nunique"),
        )
        .sort_values(["category", "trade_date"])
    )

    rows: list[dict[str, object]] = []
    for category, grp in category_daily.groupby("category", sort=False):
        grp = grp.dropna(subset=["category_return", "btc_return"]).copy()
        n_obs = len(grp)
        if n_obs < min_obs:
            continue

        same_day_corr = safe_corr(grp["category_return"], grp["btc_return"])
        price_change_corr = safe_corr(grp["category_price_change"], grp["btc_return"])

        best_lag = 0
        best_lag_corr = same_day_corr
        best_lag_n = n_obs

        for lag in range(-max_lag, max_lag + 1):
            shifted = grp["category_return"].shift(lag)
            valid = pd.DataFrame(
                {"shifted_category_return": shifted, "btc_return": grp["btc_return"]}
            ).dropna()
            if len(valid) < min_obs:
                continue
            lag_corr = safe_corr(valid["shifted_category_return"], valid["btc_return"])
            if np.isnan(lag_corr):
                continue
            if np.isnan(best_lag_corr) or abs(lag_corr) > abs(best_lag_corr):
                best_lag = lag
                best_lag_corr = lag_corr
                best_lag_n = len(valid)

        rows.append(
            {
                "category": category,
                "n_obs": n_obs,
                "same_day_corr": same_day_corr,
                "abs_same_day_corr": abs(same_day_corr) if pd.notna(same_day_corr) else np.nan,
                "price_change_corr": price_change_corr,
                "best_lag": best_lag,
                "best_lag_corr": best_lag_corr,
                "abs_best_lag_corr": abs(best_lag_corr) if pd.notna(best_lag_corr) else np.nan,
                "best_lag_n": best_lag_n,
                "avg_market_count": grp["market_count"].mean(),
                "avg_token_count": grp["token_count"].mean(),
            }
        )

    category_rankings = pd.DataFrame(rows)
    if category_rankings.empty:
        raise ValueError(
            f"No category series met the minimum observation threshold of {min_obs}."
        )

    return category_rankings.sort_values(
        ["abs_best_lag_corr", "abs_same_day_corr", "n_obs"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_top_results_table(market_rankings: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Create a clean, sorted top-results table for export and display."""
    table = market_rankings[
        [
            "market_id",
            "token_id",
            "question",
            "outcome",
            "category",
            "n_obs",
            "same_day_corr",
            "best_lag",
            "best_lag_corr",
            "abs_best_lag_corr",
            "volume",
            "trade_count",
        ]
    ].head(top_n).copy()
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table["same_day_corr"] = table["same_day_corr"].round(4)
    table["best_lag_corr"] = table["best_lag_corr"].round(4)
    table["abs_best_lag_corr"] = table["abs_best_lag_corr"].round(4)
    table["volume"] = table["volume"].round(2)
    return table


def build_top_category_table(category_rankings: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Create a clean category-level table for export and display."""
    filtered = category_rankings[category_rankings["category"] != "Uncategorized"].copy()
    if filtered.empty:
        filtered = category_rankings.copy()

    table = filtered[
        [
            "category",
            "n_obs",
            "same_day_corr",
            "best_lag",
            "best_lag_corr",
            "abs_best_lag_corr",
            "avg_market_count",
            "avg_token_count",
        ]
    ].head(top_n).copy()
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table["same_day_corr"] = table["same_day_corr"].round(4)
    table["best_lag_corr"] = table["best_lag_corr"].round(4)
    table["abs_best_lag_corr"] = table["abs_best_lag_corr"].round(4)
    table["avg_market_count"] = table["avg_market_count"].round(1)
    table["avg_token_count"] = table["avg_token_count"].round(1)
    return table


def plot_top_correlations(top_table: pd.DataFrame, output_file: Path) -> None:
    """Plot highest-correlation non-BTC markets as a horizontal bar chart."""
    plot_df = top_table.copy().iloc[::-1]
    plot_df["label"] = plot_df.apply(
        lambda row: f"#{int(row['rank'])} {str(row['question'])[:80]}",
        axis=1,
    )
    colors = np.where(plot_df["best_lag_corr"] >= 0, "#1f77b4", "#d62728")

    plt.figure(figsize=(14, max(6, len(plot_df) * 0.45)))
    plt.barh(plot_df["label"], plot_df["best_lag_corr"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Best Lag Correlation With BTC Daily Return")
    plt.ylabel("Non-BTC Polymarket Question")
    plt.title("Top Non-BTC Polymarket Questions by Correlation With BTC")

    for i, value in enumerate(plot_df["best_lag_corr"]):
        x_pos = value + 0.01 if value >= 0 else value - 0.01
        ha = "left" if value >= 0 else "right"
        plt.text(x_pos, i, f"{value:.3f}", va="center", ha=ha, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close()


def plot_top_category_correlations(top_table: pd.DataFrame, output_file: Path) -> None:
    """Plot highest-correlation categories as a horizontal bar chart."""
    plot_df = top_table.copy().iloc[::-1]
    colors = np.where(plot_df["best_lag_corr"] >= 0, "#2a9d8f", "#e76f51")

    plt.figure(figsize=(12, max(5, len(plot_df) * 0.5)))
    plt.barh(plot_df["category"], plot_df["best_lag_corr"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Best Lag Correlation With BTC Daily Return")
    plt.ylabel("Polymarket Category")
    plt.title("Top Polymarket Categories by Correlation With BTC")

    for i, value in enumerate(plot_df["best_lag_corr"]):
        x_pos = value + 0.01 if value >= 0 else value - 0.01
        ha = "left" if value >= 0 else "right"
        plt.text(x_pos, i, f"{value:.3f}", va="center", ha=ha, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close()


def build_plausible_non_crypto_table(
    market_rankings: pd.DataFrame,
    min_non_crypto_obs: int,
    min_non_crypto_volume: float,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter out crypto-adjacent questions and keep stronger, more liquid series."""
    question = market_rankings["question"].fillna("")
    category = market_rankings["category"].fillna("")
    non_crypto_mask = ~(
        question.astype(str).str.contains(NON_CRYPTO_PATTERN, na=False)
        | category.astype(str).str.contains(NON_CRYPTO_PATTERN, na=False)
    )

    plausible = market_rankings.loc[non_crypto_mask].copy()
    plausible = plausible[
        (plausible["n_obs"] >= min_non_crypto_obs)
        & (plausible["volume"].fillna(0) >= min_non_crypto_volume)
    ].copy()
    plausible = plausible.sort_values(
        ["abs_best_lag_corr", "n_obs", "volume"], ascending=[False, False, False]
    ).reset_index(drop=True)

    top_table = plausible[
        [
            "market_id",
            "token_id",
            "question",
            "outcome",
            "category",
            "n_obs",
            "same_day_corr",
            "best_lag",
            "best_lag_corr",
            "abs_best_lag_corr",
            "volume",
            "trade_count",
        ]
    ].head(top_n).copy()
    if not top_table.empty:
        top_table.insert(0, "rank", np.arange(1, len(top_table) + 1))
        top_table["same_day_corr"] = top_table["same_day_corr"].round(4)
        top_table["best_lag_corr"] = top_table["best_lag_corr"].round(4)
        top_table["abs_best_lag_corr"] = top_table["abs_best_lag_corr"].round(4)
        top_table["volume"] = top_table["volume"].round(2)

    return plausible, top_table


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--btc-csv",
        type=Path,
        default=repo_root / "data" / "Coin Metrics" / "coinmetrics_btc.csv",
        help="Path to BTC CoinMetrics CSV",
    )
    parser.add_argument(
        "--polymarket-dir",
        type=Path,
        default=repo_root / "data" / "Polymarket",
        help="Directory with Polymarket parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "eda" / "outputs" / "non_btc_polymarket_corr",
        help="Directory where CSV outputs will be written",
    )
    parser.add_argument("--start-date", default="2022-01-01", help="Inclusive analysis start date")
    parser.add_argument("--end-date", default=None, help="Inclusive analysis end date")
    parser.add_argument(
        "--min-obs",
        type=int,
        default=30,
        help="Minimum overlapping daily observations required per token series",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=7,
        help="Maximum absolute lag used when searching for strongest lead-lag correlation",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Number of top rows to print")
    parser.add_argument(
        "--min-non-crypto-obs",
        type=int,
        default=60,
        help="Minimum observations for the plausible non-crypto table",
    )
    parser.add_argument(
        "--min-non-crypto-volume",
        type=float,
        default=1000.0,
        help="Minimum volume for the plausible non-crypto table",
    )
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
    df_btc = load_btc(args.btc_csv, args.start_date, args.end_date)

    logging.info("Loading Polymarket tables...")
    df_markets, df_tokens, df_odds, df_summary = load_polymarket_tables(args.polymarket_dir)

    logging.info("Filtering to non-BTC markets...")
    df_markets_non_btc = filter_non_btc_markets(df_markets)

    logging.info("Building merged daily token panel from odds history...")
    daily = build_daily_token_panel(
        df_markets_non_btc, df_tokens, df_odds, df_summary, df_btc.index
    )

    logging.info("Computing correlations to BTC returns...")
    token_rankings, market_rankings, merged = correlation_by_group(
        daily, df_btc, args.min_obs, args.max_lag
    )
    category_rankings = correlation_by_category(merged, args.min_obs, args.max_lag)

    token_rankings.to_csv(args.output_dir / "non_btc_token_correlations.csv", index=False)
    market_rankings.to_csv(args.output_dir / "non_btc_market_correlations.csv", index=False)
    category_rankings.to_csv(args.output_dir / "non_btc_category_correlations.csv", index=False)
    top_table = build_top_results_table(market_rankings, args.top_n)
    top_category_table = build_top_category_table(category_rankings, args.top_n)
    plausible_non_crypto, plausible_non_crypto_top = build_plausible_non_crypto_table(
        market_rankings,
        args.min_non_crypto_obs,
        args.min_non_crypto_volume,
        args.top_n,
    )
    top_table.to_csv(args.output_dir / "top_non_btc_market_correlations_table.csv", index=False)
    top_category_table.to_csv(
        args.output_dir / "top_non_btc_category_correlations_table.csv", index=False
    )
    plausible_non_crypto.to_csv(
        args.output_dir / "plausible_non_crypto_market_correlations.csv", index=False
    )
    plausible_non_crypto_top.to_csv(
        args.output_dir / "top_plausible_non_crypto_market_correlations_table.csv",
        index=False,
    )
    plot_top_correlations(top_table, args.output_dir / "top_non_btc_market_correlations_chart.png")
    plot_top_category_correlations(
        top_category_table,
        args.output_dir / "top_non_btc_category_correlations_chart.png",
    )

    top_market_ids = market_rankings["market_id"].head(args.top_n).tolist()
    merged[merged["market_id"].isin(top_market_ids)].to_csv(
        args.output_dir / "top_non_btc_market_daily_panel.csv", index=False
    )

    print("\n=== Non-BTC Polymarket Questions Most Correlated With BTC ===")
    print(f"BTC rows in analysis window: {len(df_btc)}")
    print(f"Non-BTC markets considered: {df_markets_non_btc['market_id'].nunique()}")
    print(f"Token series meeting min_obs={args.min_obs}: {len(token_rankings)}")

    print("\nTop market-level results:")
    print(top_table.to_string(index=False))

    print("\nTop category-level results:")
    print(top_category_table.to_string(index=False))

    if not plausible_non_crypto_top.empty:
        print("\nTop plausible non-crypto results:")
        print(plausible_non_crypto_top.to_string(index=False))
    else:
        print(
            "\nTop plausible non-crypto results:\n"
            "No rows met the configured non-crypto observation and volume thresholds."
        )

    print(f"\nSaved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
