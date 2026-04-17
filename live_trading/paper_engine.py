"""Paper trading engine with persistent state.

Runs the optimized cycle-top sell strategy against live API data.
Tracks portfolio state, lot history, and trade log in a JSON file.
Designed to be run once per day (e.g., via cron).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data_fetchers import fetch_all_live_data, fetch_current_btc_price, HALVING_DATES

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Signal-weighted DCA — ported from erick_best_puell_model.py
# ══════════════════════════════════════════════════════════════

DEFAULT_BUY_WEIGHTS = {
    "mvrv": 1.0,
    "fgi": 7.0,
    "poly": 6.0,
    "puell": 2.0,
}

# Enhanced v2 weights — adds RSI, drawdown, MVRV gradient, price vs MA
DEFAULT_BUY_WEIGHTS_V2 = {
    "mvrv": 1.0,
    "fgi": 7.0,
    "poly": 6.0,
    "puell": 2.0,
    "rsi": 2.0,
    "drawdown": 3.0,
    "mvrv_gradient": 1.5,
    "price_vs_ma": 1.0,
}

DEFAULT_PUELL_PARAMS = {
    "low_input": 0.3,
    "neutral_input": 1.0,
    "high_input": 2.0,
    "low_output": 1.55,
    "high_output": 0.65,
}


def compute_buy_conviction(signals: dict, buy_weights: dict | None = None,
                           puell_params: dict | None = None) -> float:
    """Compute conviction multiplier for today's DCA buy.

    Returns multiplier > 1 when conditions favor buying (fear, low MVRV, low Puell),
    < 1 when overheated. Matches the best backtest model's weighting logic.

    Enhanced v2 signals (used when weights include rsi/drawdown/mvrv_gradient/price_vs_ma):
      - RSI < 30 → buy more (oversold), RSI > 70 → buy less (overbought)
      - Drawdown from ATH: deeper drawdown → buy more aggressively
      - MVRV gradient: falling MVRV → buy more (trend improving)
      - Price vs 200d MA: below MA → buy more (undervalued)
    """
    w = buy_weights or DEFAULT_BUY_WEIGHTS
    pp = puell_params or DEFAULT_PUELL_PARAMS

    mvrv_z = signals.get("mvrv_zscore", 0.0)
    fgi = signals.get("fgi", 0.5)
    poly = signals.get("polymarket_sentiment", 0.5)
    puell = signals.get("puell_multiple", 1.0)

    # Core signals (from erick_best_puell_model)
    m_raw_mvrv = float(np.interp(mvrv_z, [-2.0, 2.5], [1.5, 0.5]))
    m_raw_fgi = float(np.interp(fgi, [0.0, 1.0], [1.5, 0.5]))
    m_raw_poly = float(np.interp(poly, [0.0, 1.0], [0.8, 1.2]))

    puell_clamped = float(np.clip(puell, pp["low_input"], pp["high_input"]))
    m_raw_puell = float(np.interp(
        puell_clamped,
        [pp["low_input"], pp["neutral_input"], pp["high_input"]],
        [pp["low_output"], 1.0, pp["high_output"]],
    ))

    # Apply weights: 1 + weight * (raw - 1)
    m_mvrv = 1.0 + w.get("mvrv", 0.0) * (m_raw_mvrv - 1.0)
    m_fgi = 1.0 + w.get("fgi", 0.0) * (m_raw_fgi - 1.0)
    m_poly = 1.0 + w.get("poly", 0.0) * (m_raw_poly - 1.0)
    m_puell = 1.0 + w.get("puell", 0.0) * (m_raw_puell - 1.0)

    conviction = m_mvrv * m_fgi * m_poly * m_puell

    # ── Enhanced v2 signals (only active if weights > 0) ──

    # RSI: oversold (< 30) → buy more, overbought (> 70) → buy less
    if w.get("rsi", 0.0) > 0:
        rsi = signals.get("rsi", 50.0)
        m_raw_rsi = float(np.interp(rsi, [20, 50, 80], [1.5, 1.0, 0.5]))
        conviction *= 1.0 + w["rsi"] * (m_raw_rsi - 1.0)

    # Drawdown from ATH: -50%+ → buy much more, near ATH → reduce
    if w.get("drawdown", 0.0) > 0:
        dd = signals.get("drawdown_from_ath", 0.0)
        m_raw_dd = float(np.interp(dd, [-0.6, -0.2, 0.0], [1.5, 1.1, 0.8]))
        conviction *= 1.0 + w["drawdown"] * (m_raw_dd - 1.0)

    # MVRV gradient: falling (< 0) → improving → buy more
    if w.get("mvrv_gradient", 0.0) > 0:
        grad = signals.get("mvrv_gradient", 0.0)
        m_raw_grad = float(np.interp(grad, [-0.5, 0.0, 0.5], [1.3, 1.0, 0.7]))
        conviction *= 1.0 + w["mvrv_gradient"] * (m_raw_grad - 1.0)

    # Price vs 200d MA: below MA → buy more
    if w.get("price_vs_ma", 0.0) > 0:
        pvm = signals.get("price_vs_ma", 0.0)
        m_raw_pvm = float(np.interp(pvm, [-0.3, 0.0, 0.3], [1.4, 1.0, 0.6]))
        conviction *= 1.0 + w["price_vs_ma"] * (m_raw_pvm - 1.0)

    return max(0.01, conviction)


# ══════════════════════════════════════════════════════════════
# Portfolio state management
# ══════════════════════════════════════════════════════════════

def load_state(path: Path) -> dict:
    """Load or initialize paper trading state."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {
        "btc_held": 0.0,
        "cash": 0.0,
        "reinvest_pool": 0.0,
        "total_contributed": 0.0,
        "total_btc_sold": 0.0,
        "peak_btc_held": 0.0,
        "total_sell_proceeds": 0.0,
        "total_realized_gain": 0.0,
        "total_tax_paid": 0.0,
        "total_reinvested": 0.0,
        "sell_cooldown_remaining": 0,
        "reinvest_cooldown_remaining": 0,
        "trailing_active": False,
        "trailing_peak_price": 0.0,
        "trailing_days": 0,
        "pending_sell_fraction": 0.0,
        "pending_n_signals": 0,
        "lots": [],
        "trade_log": [],
        "daily_log": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_run": None,
    }


def save_state(state: dict, path: Path) -> None:
    """Persist state to disk."""
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to temp file first then rename for atomicity
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.rename(path)


# ══════════════════════════════════════════════════════════════
# Signal detection
# ══════════════════════════════════════════════════════════════

def count_sell_signals(row: dict, params: dict) -> int:
    """Count how many of 5 conjunction conditions are met."""
    count = 0
    if row.get("mvrv_zscore", 0) >= params["mvrv_zscore_min"]:
        count += 1
    if row.get("puell_multiple", 1) >= params["puell_min"]:
        count += 1
    if row.get("fgi", 0.5) >= params["fgi_min"] / 100.0:
        count += 1
    if row.get("price_vs_ma", 0) >= params["price_vs_ma_min"]:
        count += 1
    if row.get("days_since_halving", 0) >= params["min_days_post_halving"]:
        count += 1
    return count


def get_sell_fraction(n_signals: int) -> float:
    """Sell fraction based on signal count."""
    tiers = [(5, 0.15), (4, 0.10), (3, 0.05)]
    for min_sig, frac in tiers:
        if n_signals >= min_sig:
            return frac
    return 0.0


# ══════════════════════════════════════════════════════════════
# Tax-optimized lot consumption
# ══════════════════════════════════════════════════════════════

def consume_lots_tax_optimized(
    lots: list[dict],
    sell_qty: float,
    sell_price: float,
    sell_date: str,
    fee_rate: float,
    st_tax: float,
    lt_tax: float,
    state_tax: float,
) -> tuple[float, float, float]:
    """Sell lots: long-term + highest-basis first."""
    sell_dt = pd.Timestamp(sell_date)
    scored = []
    for i, lot in enumerate(lots):
        if lot["qty"] <= 1e-12:
            continue
        holding_days = (sell_dt - pd.Timestamp(lot["date"])).days
        is_lt = holding_days > 365
        cost_per_unit = lot["cost"] / lot["qty"] if lot["qty"] > 1e-12 else 0
        scored.append((0 if is_lt else 1, -cost_per_unit, i))
    scored.sort()

    remaining = sell_qty
    total_proceeds = 0.0
    realized_gain = 0.0
    tax_due = 0.0

    for _, _, idx in scored:
        if remaining <= 1e-12:
            break
        lot = lots[idx]
        qty_from_lot = min(remaining, lot["qty"])
        lot_frac = qty_from_lot / lot["qty"]
        cost_basis = lot["cost"] * lot_frac
        proceeds = qty_from_lot * sell_price * (1.0 - fee_rate)
        gain = proceeds - cost_basis
        holding_days = (sell_dt - pd.Timestamp(lot["date"])).days
        federal_rate = lt_tax if holding_days > 365 else st_tax

        total_proceeds += proceeds
        realized_gain += gain
        if gain > 0:
            tax_due += gain * (federal_rate + state_tax)

        lot["qty"] -= qty_from_lot
        lot["cost"] -= cost_basis
        remaining -= qty_from_lot

    lots[:] = [l for l in lots if l["qty"] > 1e-12]
    return total_proceeds, realized_gain, tax_due


# ══════════════════════════════════════════════════════════════
# Daily tick — the main trading logic
# ══════════════════════════════════════════════════════════════

def run_daily_tick(state: dict, config: dict, features: pd.DataFrame, current_price: float) -> dict:
    """Execute one day of the paper trading strategy.

    Returns dict of actions taken this tick.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sell_params = config["sell_parameters"]
    reinvest_cfg = config["reinvestment"]
    fee_rate = config["fees"]["taker_fee_rate"]
    st_tax = config["tax"]["short_term_rate"]
    lt_tax = config["tax"]["long_term_rate"]
    state_tax = config["tax"]["state_rate"]
    daily_usd = config["daily_contribution_usd"]

    actions = {
        "date": today,
        "price": current_price,
        "action": "none",
        "details": {},
    }

    # Get latest feature row
    if features.empty:
        logger.warning("No features available")
        return actions

    latest = features.iloc[-1]
    signals = {
        "mvrv_zscore": float(latest.get("mvrv_zscore", 0)),
        "mvrv_gradient": float(latest.get("mvrv_gradient", 0)),
        "mvrv_acceleration": float(latest.get("mvrv_acceleration", 0)),
        "puell_multiple": float(latest.get("puell_multiple", 1)),
        "fgi": float(latest.get("fgi", 0.5)),
        "price_vs_ma": float(latest.get("price_vs_ma", 0)),
        "days_since_halving": float(latest.get("days_since_halving", 0)),
        "polymarket_sentiment": float(latest.get("polymarket_sentiment", 0.5)),
        "drawdown_from_ath": float(latest.get("drawdown_from_ath", 0)),
        "rsi": float(latest.get("rsi", 50)),
    }

    # ── Phase 1: Daily DCA buy (signal-weighted) ──
    buy_weights = config.get("buy_weights", DEFAULT_BUY_WEIGHTS)
    puell_params = config.get("puell_params", DEFAULT_PUELL_PARAMS)
    conviction = compute_buy_conviction(signals, buy_weights, puell_params)

    state["total_contributed"] += daily_usd
    state["cash"] += daily_usd

    # ── Rolling normalization: approximate backtest's weight normalization ──
    # The backtest normalizes all weights over a 365-day window so total spend is constant.
    # We approximate this by comparing today's conviction to the rolling mean.
    # This preserves the signal (Spearman ≈ 0.96) and concentrates ~50% of spend on
    # the top-5 conviction days per month — much better than raw conviction (only 20%).
    # Unspent cash rolls over, building a war chest for the next major fear event.

    # Track conviction history for rolling normalization
    if "conviction_history" not in state:
        state["conviction_history"] = []
    state["conviction_history"].append({"date": today, "conviction": conviction})
    # Keep 60 days of history
    state["conviction_history"] = state["conviction_history"][-60:]

    # Rolling mean (fallback to 1.0 until we have 3+ days of data)
    hist_vals = [h["conviction"] for h in state["conviction_history"]]
    mean_conv = float(np.mean(hist_vals)) if len(hist_vals) >= 3 else 1.0
    mean_conv = max(mean_conv, 0.1)  # avoid division by zero

    # Tilt: today's conviction relative to recent average, capped at [0.2x, 5.0x]
    MAX_TILT = 5.0
    MIN_TILT = 0.2
    tilt = float(np.clip(conviction / mean_conv, MIN_TILT, MAX_TILT))

    buy_amount = daily_usd * tilt

    # Track DCA benchmark (pure hold, no sells — always uses flat $daily_usd)
    if "dca_benchmark_btc" not in state:
        state["dca_benchmark_btc"] = 0.0
    if daily_usd > 0 and current_price > 0:
        state["dca_benchmark_btc"] += (daily_usd * (1.0 - fee_rate)) / current_price

    # Safety floor: never spend more than 90% of available cash
    buy_amount = min(buy_amount, state["cash"] * 0.90)

    if buy_amount > 0 and state["cash"] >= buy_amount:
        btc_bought = (buy_amount * (1.0 - fee_rate)) / current_price
        state["btc_held"] += btc_bought
        state["lots"].append({
            "date": today,
            "qty": btc_bought,
            "cost": buy_amount,
        })
        state["cash"] -= buy_amount
        actions["details"]["dca_buy"] = {
            "usd": round(buy_amount, 2),
            "btc": round(btc_bought, 8),
            "price": round(current_price, 2),
            "conviction": round(conviction, 3),
            "tilt": round(tilt, 3),
            "mean_conviction": round(mean_conv, 3),
        }

    state["peak_btc_held"] = max(state["peak_btc_held"], state["btc_held"])

    # ── Phase 2: Check trailing stop ──
    if state["trailing_active"]:
        state["trailing_days"] += 1
        if current_price > state["trailing_peak_price"]:
            state["trailing_peak_price"] = current_price

        drop = (current_price / state["trailing_peak_price"]) - 1.0
        force = state["trailing_days"] >= sell_params["trailing_stop_max_days"]

        if drop <= -sell_params["trailing_stop_pct"] or force:
            btc_to_sell = state["btc_held"] * state["pending_sell_fraction"]
            cum_room = max(
                state["peak_btc_held"] * sell_params["max_cumulative_sell_pct"]
                - state["total_btc_sold"], 0
            )
            btc_to_sell = min(btc_to_sell, cum_room)

            if btc_to_sell > 1e-12:
                proceeds, gain, tax = consume_lots_tax_optimized(
                    state["lots"], btc_to_sell, current_price, today,
                    fee_rate, st_tax, lt_tax, state_tax,
                )
                state["btc_held"] -= btc_to_sell
                state["total_btc_sold"] += btc_to_sell
                state["reinvest_pool"] += proceeds - tax
                state["total_sell_proceeds"] += proceeds
                state["total_realized_gain"] += gain
                state["total_tax_paid"] += tax
                state["sell_cooldown_remaining"] = sell_params["sell_cooldown_days"]

                actions["action"] = "SELL"
                actions["details"]["sell"] = {
                    "btc_sold": round(btc_to_sell, 8),
                    "proceeds": round(proceeds, 2),
                    "gain": round(gain, 2),
                    "tax": round(tax, 2),
                    "net": round(proceeds - tax, 2),
                    "n_signals": state["pending_n_signals"],
                    "trigger": "trailing_stop" if not force else "max_days_forced",
                }

                state["trade_log"].append({
                    "date": today,
                    "type": "sell",
                    "price": round(current_price, 2),
                    "btc": round(btc_to_sell, 8),
                    "usd": round(proceeds, 2),
                    "signals": state["pending_n_signals"],
                })

            state["trailing_active"] = False
            state["trailing_peak_price"] = 0
            state["trailing_days"] = 0
            state["pending_sell_fraction"] = 0
            state["pending_n_signals"] = 0

    # ── Phase 3: Check conjunction sell signals ──
    elif state["btc_held"] > 0 and state["sell_cooldown_remaining"] <= 0:
        n_signals = count_sell_signals(signals, sell_params)
        sell_frac = get_sell_fraction(n_signals)

        if n_signals >= sell_params["min_signals_required"] and sell_frac > 0:
            state["trailing_active"] = True
            state["trailing_peak_price"] = current_price
            state["trailing_days"] = 0
            state["pending_sell_fraction"] = sell_frac
            state["pending_n_signals"] = n_signals

            actions["action"] = "TRAILING_STOP_ACTIVATED"
            actions["details"]["trailing"] = {
                "n_signals": n_signals,
                "sell_fraction": sell_frac,
                "signals": signals,
            }

    # ── Phase 4: Reinvestment engine ──
    if (reinvest_cfg["enabled"]
            and state["reinvest_pool"] > 1.0
            and state["reinvest_cooldown_remaining"] <= 0):
        drawdown = signals.get("drawdown_from_ath", 0)
        if drawdown <= reinvest_cfg["drawdown_threshold"]:
            severity = min(abs(drawdown) / 0.60, 1.0)
            deploy_frac = reinvest_cfg["fraction_per_day"] * (0.5 + 0.5 * severity)
            deploy_amount = state["reinvest_pool"] * deploy_frac

            if deploy_amount > 1.0:
                btc_bought = (deploy_amount * (1.0 - fee_rate)) / current_price
                state["btc_held"] += btc_bought
                state["lots"].append({
                    "date": today,
                    "qty": btc_bought,
                    "cost": deploy_amount,
                })
                state["reinvest_pool"] -= deploy_amount
                state["total_reinvested"] += deploy_amount
                state["reinvest_cooldown_remaining"] = reinvest_cfg["cooldown_days"]

                if actions["action"] == "none":
                    actions["action"] = "REINVEST"
                actions["details"]["reinvest"] = {
                    "usd": round(deploy_amount, 2),
                    "btc": round(btc_bought, 8),
                    "drawdown": round(drawdown * 100, 1),
                }

                state["trade_log"].append({
                    "date": today,
                    "type": "reinvest",
                    "price": round(current_price, 2),
                    "btc": round(btc_bought, 8),
                    "usd": round(deploy_amount, 2),
                })

    # Decrement cooldowns
    state["sell_cooldown_remaining"] = max(state["sell_cooldown_remaining"] - 1, 0)
    state["reinvest_cooldown_remaining"] = max(state["reinvest_cooldown_remaining"] - 1, 0)

    # ── Daily snapshot ──
    portfolio_value = (
        state["cash"]
        + state["reinvest_pool"]
        + state["btc_held"] * current_price
    )
    actions["portfolio"] = {
        "btc_held": round(state["btc_held"], 8),
        "cash": round(state["cash"], 2),
        "reinvest_pool": round(state["reinvest_pool"], 2),
        "portfolio_value": round(portfolio_value, 2),
        "total_contributed": round(state["total_contributed"], 2),
        "profit": round(portfolio_value - state["total_contributed"], 2),
        "return_pct": round(
            ((portfolio_value / state["total_contributed"]) - 1) * 100, 2
        ) if state["total_contributed"] > 0 else 0,
    }
    actions["signals"] = signals

    # Pure DCA benchmark: what if we just held all BTC bought, never sold
    dca_btc = state.get("dca_benchmark_btc", 0.0)
    dca_cost = state["total_contributed"]
    dca_value = dca_btc * current_price
    actions["portfolio"]["dca_benchmark_value"] = round(dca_value, 2)

    # Keep daily log trimmed to last 730 entries (2 years)
    state["daily_log"].append({
        "date": today,
        "price": round(current_price, 2),
        "portfolio_value": round(portfolio_value, 2),
        "total_contributed": round(state["total_contributed"], 2),
        "btc_held": round(state["btc_held"], 8),
        "cash": round(state["cash"], 2),
        "reinvest_pool": round(state["reinvest_pool"], 2),
        "dca_benchmark_value": round(dca_value, 2),
        "return_pct": actions["portfolio"]["return_pct"],
        "action": actions["action"],
    })
    if len(state["daily_log"]) > 730:
        state["daily_log"] = state["daily_log"][-730:]

    return actions


# ══════════════════════════════════════════════════════════════
# Signal dashboard — current state of all indicators
# ══════════════════════════════════════════════════════════════

def format_dashboard(actions: dict, state: dict) -> str:
    """Format a human-readable dashboard string."""
    p = actions.get("portfolio", {})
    s = actions.get("signals", {})
    sell_params_ref = {
        "mvrv_zscore_min": 2.2,
        "puell_min": 1.4,
        "fgi_min": 0.75,
        "price_vs_ma_min": 0.30,
        "min_days_post_halving": 300,
    }

    lines = [
        "══════════════════════════════════════════════════",
        f"  BTC Paper Trader — {actions.get('date', 'N/A')}",
        "══════════════════════════════════════════════════",
        f"  BTC Price:      ${actions.get('price', 0):>12,.2f}",
        f"  BTC Held:       {p.get('btc_held', 0):.8f}",
        f"  Portfolio:      ${p.get('portfolio_value', 0):>12,.2f}",
        f"  Contributed:    ${p.get('total_contributed', 0):>12,.2f}",
        f"  Profit:         ${p.get('profit', 0):>12,.2f} ({p.get('return_pct', 0):.1f}%)",
        f"  Cash:           ${p.get('cash', 0):>12,.2f}",
        f"  Reinvest Pool:  ${p.get('reinvest_pool', 0):>12,.2f}",
        "",
        "  ── Sell Signals (need 4/5) ──",
        f"  MVRV Z-Score:   {s.get('mvrv_zscore', 0):>8.2f}  (trigger ≥ 2.20)  "
        f"{'🔴' if s.get('mvrv_zscore', 0) >= 2.2 else '⚪'}",
        f"  Puell Multiple: {s.get('puell_multiple', 1):>8.2f}  (trigger ≥ 1.40)  "
        f"{'🔴' if s.get('puell_multiple', 1) >= 1.4 else '⚪'}",
        f"  Fear & Greed:   {s.get('fgi', 0.5)*100:>7.0f}%  (trigger ≥ 75%)   "
        f"{'🔴' if s.get('fgi', 0.5) >= 0.75 else '⚪'}",
        f"  Price vs MA200: {s.get('price_vs_ma', 0)*100:>7.1f}%  (trigger ≥ 30%)   "
        f"{'🔴' if s.get('price_vs_ma', 0) >= 0.30 else '⚪'}",
        f"  Days Post Halv: {s.get('days_since_halving', 0):>7.0f}d  (trigger ≥ 300d)  "
        f"{'🔴' if s.get('days_since_halving', 0) >= 300 else '⚪'}",
        f"  Polymarket:     {s.get('polymarket_sentiment', 0.5)*100:>7.1f}%  (BTC sentiment)",
        f"  RSI:            {s.get('rsi', 50):>8.1f}",
        f"  ATH Drawdown:   {s.get('drawdown_from_ath', 0)*100:>7.1f}%",
        "",
        f"  Action: {actions.get('action', 'none')}",
    ]

    if actions.get("action") == "SELL":
        d = actions["details"]["sell"]
        lines.append(f"    Sold {d['btc_sold']:.8f} BTC for ${d['proceeds']:,.2f}")
        lines.append(f"    Tax: ${d['tax']:,.2f}  Net: ${d['net']:,.2f}")
    elif actions.get("action") == "TRAILING_STOP_ACTIVATED":
        d = actions["details"]["trailing"]
        lines.append(f"    {d['n_signals']}/5 signals active — trailing stop armed")
    elif actions.get("action") == "REINVEST":
        d = actions["details"]["reinvest"]
        lines.append(f"    Deployed ${d['usd']:,.2f} at {d['drawdown']:.1f}% drawdown")

    if "dca_buy" in actions.get("details", {}):
        d = actions["details"]["dca_buy"]
        conv = d.get("conviction", 1.0)
        conv_label = "🟢 BUY MORE" if conv > 1.3 else ("🔻 BUY LESS" if conv < 0.7 else "⚪ NORMAL")
        lines.append(f"    DCA Buy: ${d['usd']:.2f} → {d['btc']:.8f} BTC @ ${d['price']:,.2f}")
        lines.append(f"    Conviction: {conv:.2f}x  ({conv_label})")

    # DCA benchmark comparison
    dca_bench = p.get("dca_benchmark_value", 0)
    if dca_bench > 0:
        edge = p.get("portfolio_value", 0) - dca_bench
        lines.append(f"  vs DCA Bench:   ${edge:>+12,.2f}  {'✅' if edge > 0 else '❌'}")

    lines.append("")
    lines.append(f"  Lifetime: {len(state.get('trade_log', []))} trades, "
                 f"${state.get('total_tax_paid', 0):,.2f} tax paid")
    lines.append("══════════════════════════════════════════════════")
    return "\n".join(lines)
