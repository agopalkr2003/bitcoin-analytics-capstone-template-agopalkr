"""Live data fetchers for all signal sources.

Free APIs used (no keys required):
  - BTC Price + MVRV + Miner Revenue: Coin Metrics Community API
  - Fear & Greed Index: alternative.me
  - Polymarket BTC sentiment: Gamma API (public, no key)
  - Halving dates: hardcoded (known schedule)
  - RSI / MA: computed from price

Optional (key required):
  - Exchange price: CCXT (Coinbase/Binance/Kraken)
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HALVING_DATES = [
    pd.Timestamp("2012-11-28"),
    pd.Timestamp("2016-07-09"),
    pd.Timestamp("2020-05-11"),
    pd.Timestamp("2024-04-19"),
    pd.Timestamp("2028-03-20"),  # estimated
]

# ══════════════════════════════════════════════════════════════
# Coin Metrics Community API (free, no key needed)
# ══════════════════════════════════════════════════════════════

COINMETRICS_BASE = "https://community-api.coinmetrics.io/v4"
COINMETRICS_METRICS = [
    "PriceUSD",
    "CapMVRVCur",
    "IssTotNtv",
    "FeeTotNtv",
]


def fetch_coinmetrics(
    start_date: str = "2017-01-01",
    end_date: str | None = None,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch BTC metrics from Coin Metrics community API."""
    if end_date is None:
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    metrics_str = ",".join(COINMETRICS_METRICS)
    url = (
        f"{COINMETRICS_BASE}/timeseries/asset-metrics"
        f"?assets=btc&metrics={metrics_str}"
        f"&start_time={start_date}&end_time={end_date}"
        f"&frequency=1d&page_size=10000"
    )

    all_rows = []
    next_url = url

    while next_url:
        for attempt in range(max_retries):
            try:
                req = Request(next_url, headers={"User-Agent": "btc-dca-bot/1.0"})
                with urlopen(req, timeout=30) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                break
            except (HTTPError, URLError, TimeoutError) as e:
                logger.warning(f"Coin Metrics attempt {attempt+1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

        all_rows.extend(payload.get("data", []))
        next_url = payload.get("next_page_url")

    if not all_rows:
        raise RuntimeError("No data returned from Coin Metrics")

    df = pd.DataFrame(all_rows)
    df["time"] = pd.to_datetime(df["time"]).dt.normalize().dt.tz_localize(None)
    df = df.set_index("time").sort_index()

    for col in COINMETRICS_METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.rename(columns={"PriceUSD": "PriceUSD_coinmetrics"})
    logger.info(f"Coin Metrics: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    return df


# ══════════════════════════════════════════════════════════════
# Fear & Greed Index (alternative.me — free, no key)
# ══════════════════════════════════════════════════════════════

FGI_URL = "https://api.alternative.me/fng/?limit=0&format=json"


def fetch_fgi(max_retries: int = 3) -> pd.DataFrame:
    """Fetch full history of Crypto Fear & Greed Index."""
    for attempt in range(max_retries):
        try:
            req = Request(FGI_URL, headers={"User-Agent": "btc-dca-bot/1.0"})
            with urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except (HTTPError, URLError, TimeoutError) as e:
            logger.warning(f"FGI attempt {attempt+1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise

    rows = payload.get("data", [])
    records = []
    for row in rows:
        ts = row.get("timestamp")
        val = row.get("value")
        if ts is None or val is None:
            continue
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).date()
        records.append({"date": pd.Timestamp(dt), "fgi": int(val) / 100.0})

    df = pd.DataFrame(records).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    logger.info(f"FGI: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    return df


# ══════════════════════════════════════════════════════════════
# BTC spot price (real-time via CoinGecko — free, no key)
# ══════════════════════════════════════════════════════════════

COINGECKO_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"


def fetch_current_btc_price(max_retries: int = 3) -> float:
    """Get current BTC/USD spot price from CoinGecko."""
    for attempt in range(max_retries):
        try:
            req = Request(COINGECKO_PRICE_URL, headers={"User-Agent": "btc-dca-bot/1.0"})
            with urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            price = float(data["bitcoin"]["usd"])
            logger.info(f"Current BTC price: ${price:,.2f}")
            return price
        except (HTTPError, URLError, TimeoutError, KeyError) as e:
            logger.warning(f"Price fetch attempt {attempt+1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    return 0.0  # unreachable


# ══════════════════════════════════════════════════════════════
# Polymarket Gamma API (free, no key needed)
# ══════════════════════════════════════════════════════════════

POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com/markets"
BTC_KEYWORDS = ["Bitcoin", "BTC", "btc", "bitcoin"]


def fetch_polymarket_btc_sentiment(max_retries: int = 3) -> pd.DataFrame:
    """Fetch active BTC-related Polymarket markets and compute daily sentiment.

    Sentiment score (0-1):
      - Aggregates all active BTC markets
      - Uses volume-weighted average of Yes-token prices
      - Higher = more bullish BTC sentiment on Polymarket
    """
    # Fetch crypto-tagged markets
    all_markets = []
    for offset in range(0, 500, 100):
        url = (
            f"{POLYMARKET_GAMMA_URL}?active=true&closed=false"
            f"&limit=100&offset={offset}"
        )
        for attempt in range(max_retries):
            try:
                req = Request(url, headers={"User-Agent": "btc-dca-bot/1.0"})
                with urlopen(req, timeout=30) as resp:
                    markets = json.loads(resp.read().decode("utf-8"))
                break
            except (HTTPError, URLError, TimeoutError) as e:
                logger.warning(f"Polymarket attempt {attempt+1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error("Failed to fetch Polymarket data")
                    return pd.DataFrame()

        if not markets:
            break
        all_markets.extend(markets)

    # Filter to BTC-related questions
    btc_markets = []
    for m in all_markets:
        q = m.get("question", "")
        if any(kw in q for kw in BTC_KEYWORDS):
            btc_markets.append(m)

    if not btc_markets:
        logger.warning("No BTC-related Polymarket markets found")
        return pd.DataFrame()

    # Extract current sentiment snapshot
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    total_volume = 0.0
    weighted_price_sum = 0.0
    market_details = []

    for m in btc_markets:
        try:
            outcomes = m.get("outcomes", "")
            outcome_prices = m.get("outcomePrices", "")
            volume = float(m.get("volume", 0) or 0)
            volume_24h = float(m.get("volume24hr", 0) or 0)

            if isinstance(outcome_prices, str):
                # Parse JSON string like '["0.95", "0.05"]'
                prices = json.loads(outcome_prices) if outcome_prices else []
            else:
                prices = outcome_prices or []

            if isinstance(outcomes, str):
                outcomes_list = json.loads(outcomes) if outcomes else []
            else:
                outcomes_list = outcomes or []

            # Find Yes price (first outcome is typically Yes)
            yes_price = float(prices[0]) if prices else 0.5

            # Determine if "Yes" is bullish for BTC
            question = m.get("question", "").lower()
            is_bullish_question = any(w in question for w in [
                "hit", "reach", "above", "break", "surge", "rise",
                "ath", "new high", "100k", "150k", "200k",
            ])
            is_bearish_question = any(w in question for w in [
                "crash", "below", "dip", "fall", "drop", "bear",
            ])

            # Bullish question: Yes price = bullish
            # Bearish question: Yes price = bearish (invert)
            if is_bearish_question:
                sentiment_contribution = 1.0 - yes_price
            else:
                sentiment_contribution = yes_price

            weight = max(volume, 1.0)
            total_volume += weight
            weighted_price_sum += sentiment_contribution * weight

            market_details.append({
                "question": m.get("question", "")[:80],
                "yes_price": round(yes_price, 3),
                "volume": round(volume, 0),
                "volume_24h": round(volume_24h, 0),
                "sentiment": round(sentiment_contribution, 3),
                "bullish": is_bullish_question,
                "bearish": is_bearish_question,
            })
        except (ValueError, IndexError, TypeError) as e:
            logger.debug(f"Skipping market: {e}")
            continue

    if total_volume > 0:
        aggregate_sentiment = weighted_price_sum / total_volume
    else:
        aggregate_sentiment = 0.5

    logger.info(
        f"Polymarket: {len(btc_markets)} BTC markets, "
        f"sentiment={aggregate_sentiment:.3f}, "
        f"total_volume=${total_volume:,.0f}"
    )

    # Return as single-row DataFrame with today's date
    result = pd.DataFrame([{
        "date": pd.Timestamp(today),
        "polymarket_sentiment": aggregate_sentiment,
        "polymarket_n_markets": len(btc_markets),
        "polymarket_total_volume": total_volume,
    }]).set_index("date")

    # Also store market details for inspection
    result.attrs["market_details"] = market_details
    return result


# ══════════════════════════════════════════════════════════════
# Derived features (computed from raw data)
# ══════════════════════════════════════════════════════════════

def compute_features(
    cm_df: pd.DataFrame,
    fgi_df: pd.DataFrame,
    poly_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build all signal features from raw API data.

    Returns DataFrame with columns:
        price, price_vs_ma, mvrv_zscore, puell_multiple, fgi,
        polymarket_sentiment, days_since_halving, drawdown_from_ath, rsi
    """
    price = cm_df["PriceUSD_coinmetrics"].astype(float)

    # 200-day MA
    ma200 = price.rolling(200, min_periods=100).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        price_vs_ma = ((price / ma200) - 1).clip(-1, 1).fillna(0)

    # MVRV z-score + gradient + acceleration
    if "CapMVRVCur" in cm_df.columns:
        mvrv = cm_df["CapMVRVCur"].astype(float)
        mvrv_mean = mvrv.rolling(365, min_periods=180).mean()
        mvrv_std = mvrv.rolling(365, min_periods=180).std()
        mvrv_zscore = ((mvrv - mvrv_mean) / mvrv_std.replace(0, np.nan)).fillna(0).clip(-4, 4)

        # MVRV gradient (30d smoothed rate of change)
        gradient_raw = mvrv_zscore.diff(30)
        mvrv_gradient = np.tanh(
            gradient_raw.ewm(span=30, adjust=False).mean() * 2
        ).fillna(0)

        # MVRV acceleration (14d second derivative)
        accel_raw = mvrv_gradient.diff(14)
        mvrv_acceleration = np.tanh(
            accel_raw.ewm(span=14, adjust=False).mean() * 3
        ).fillna(0)
    else:
        mvrv_zscore = pd.Series(0.0, index=price.index)
        mvrv_gradient = pd.Series(0.0, index=price.index)
        mvrv_acceleration = pd.Series(0.0, index=price.index)

    # Puell multiple
    if {"IssTotNtv", "FeeTotNtv"}.issubset(cm_df.columns):
        iss = cm_df["IssTotNtv"].astype(float)
        fees = cm_df["FeeTotNtv"].astype(float).fillna(0)
        daily_iss = iss.diff().clip(lower=0).fillna(0)
        miner_rev = (daily_iss + fees) * price
        miner_rev = miner_rev.replace([np.inf, -np.inf], np.nan).fillna(0)
        baseline = miner_rev.rolling(365, min_periods=90).mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            puell = (miner_rev / baseline).replace([np.inf, -np.inf], np.nan)
        puell = puell.fillna(1.0).clip(0, 5)
    else:
        puell = pd.Series(1.0, index=price.index)

    # FGI — join on date
    fgi_aligned = fgi_df["fgi"].reindex(price.index).ffill().fillna(0.5)

    # Days since halving
    days_since_halving = pd.Series(0.0, index=price.index)
    for date in price.index:
        past = [h for h in HALVING_DATES if h <= date]
        if past:
            days_since_halving[date] = (date - past[-1]).days

    # ATH + drawdown
    ath = price.expanding().max()
    drawdown_from_ath = (price / ath) - 1.0

    # RSI-14 (exponential)
    delta = price.diff()
    gain = delta.where(delta > 0, 0.0).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50)

    # Polymarket sentiment — join live snapshot
    if poly_df is not None and not poly_df.empty:
        poly_aligned = (
            poly_df["polymarket_sentiment"]
            .reindex(price.index)
            .ffill()
            .fillna(0.5)
        )
    else:
        poly_aligned = pd.Series(0.5, index=price.index)

    features = pd.DataFrame({
        "price": price,
        "price_vs_ma": price_vs_ma,
        "mvrv_zscore": mvrv_zscore,
        "mvrv_gradient": mvrv_gradient,
        "mvrv_acceleration": mvrv_acceleration,
        "puell_multiple": puell,
        "fgi": fgi_aligned,
        "polymarket_sentiment": poly_aligned,
        "days_since_halving": days_since_halving,
        "ath": ath,
        "drawdown_from_ath": drawdown_from_ath,
        "rsi": rsi,
    }, index=price.index)

    # Lag signals by 1 day (no lookahead)
    signal_cols = ["price_vs_ma", "mvrv_zscore", "mvrv_gradient", "mvrv_acceleration",
                   "puell_multiple", "fgi",
                   "polymarket_sentiment", "days_since_halving",
                   "drawdown_from_ath", "rsi"]
    features[signal_cols] = features[signal_cols].shift(1)
    features["ath"] = features["ath"].shift(1)
    features = features.ffill().fillna(0)
    return features


# ══════════════════════════════════════════════════════════════
# Unified data loader
# ══════════════════════════════════════════════════════════════

def fetch_all_live_data(
    start_date: str = "2017-01-01",
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Fetch all data from APIs and return computed features.

    Optionally caches raw API responses to disk to avoid re-fetching.
    """
    cache_path = Path(cache_dir) if cache_dir else None

    # 1. Coin Metrics
    cm_cache = cache_path / "coinmetrics_raw.parquet" if cache_path else None
    if cm_cache and cm_cache.exists():
        cm_df = pd.read_parquet(cm_cache)
        # Fetch only new data since last cached date
        last_date = cm_df.index[-1].strftime("%Y-%m-%d")
        try:
            new_data = fetch_coinmetrics(start_date=last_date)
            if not new_data.empty:
                cm_df = pd.concat([cm_df, new_data[~new_data.index.isin(cm_df.index)]])
                cm_df = cm_df.sort_index()
        except Exception as e:
            logger.warning(f"Failed to fetch new Coin Metrics data: {e}. Using cache.")
    else:
        cm_df = fetch_coinmetrics(start_date=start_date)

    if cm_cache:
        cm_cache.parent.mkdir(parents=True, exist_ok=True)
        cm_df.to_parquet(cm_cache)

    # 2. Fear & Greed
    fgi_cache = cache_path / "fgi_raw.parquet" if cache_path else None
    if fgi_cache and fgi_cache.exists():
        fgi_df = pd.read_parquet(fgi_cache)
        try:
            new_fgi = fetch_fgi()
            if not new_fgi.empty:
                fgi_df = pd.concat([fgi_df, new_fgi[~new_fgi.index.isin(fgi_df.index)]])
                fgi_df = fgi_df.sort_index()
        except Exception as e:
            logger.warning(f"Failed to fetch new FGI data: {e}. Using cache.")
    else:
        fgi_df = fetch_fgi()

    if fgi_cache:
        fgi_cache.parent.mkdir(parents=True, exist_ok=True)
        fgi_df.to_parquet(fgi_cache)

    # 3. Polymarket BTC sentiment
    poly_cache = cache_path / "polymarket_raw.parquet" if cache_path else None
    try:
        new_poly = fetch_polymarket_btc_sentiment()
    except Exception as e:
        logger.warning(f"Failed to fetch Polymarket data: {e}")
        new_poly = pd.DataFrame()

    if poly_cache and poly_cache.exists():
        poly_df = pd.read_parquet(poly_cache)
        if not new_poly.empty:
            poly_df = pd.concat([poly_df, new_poly[~new_poly.index.isin(poly_df.index)]])
            poly_df = poly_df.sort_index()
    else:
        poly_df = new_poly

    if poly_cache and not poly_df.empty:
        poly_cache.parent.mkdir(parents=True, exist_ok=True)
        poly_df.to_parquet(poly_cache)

    # 4. Compute features
    features = compute_features(cm_df, fgi_df, poly_df if not poly_df.empty else None)
    logger.info(f"Features: {len(features)} rows, {features.index[0].date()} to {features.index[-1].date()}")
    return features
