#!/usr/bin/env python3
"""BTC Paper Trader — run once daily via cron or manually.

Usage:
    python run_paper_trader.py                  # Normal daily tick
    python run_paper_trader.py --status         # Show current signals + portfolio
    python run_paper_trader.py --history        # Show trade history
    python run_paper_trader.py --reset          # Reset portfolio (fresh start)
    python run_paper_trader.py --backfill 30    # Backfill last N days from API data

Schedule with cron (runs daily at 9am UTC):
    0 9 * * * cd /path/to/live_trading && python run_paper_trader.py >> cron.log 2>&1

Or run continuously:
    python run_paper_trader.py --daemon
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from data_fetchers import fetch_all_live_data, fetch_current_btc_price  # noqa: E402
from paper_engine import (  # noqa: E402
    load_state,
    save_state,
    run_daily_tick,
    format_dashboard,
)

import statistics  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def cmd_tick(config: dict, state_path: Path, force: bool = False) -> None:
    """Run one daily tick."""
    state = load_state(state_path)

    # Check if already ran today
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if not force and state.get("last_run") and state["last_run"][:10] == today:
        logger.info(f"Already ran today ({today}). Use --force to override.")
        # Still show dashboard
        price = fetch_current_btc_price()
        portfolio_value = state["cash"] + state["reinvest_pool"] + state["btc_held"] * price
        print(f"\nCurrent portfolio: ${portfolio_value:,.2f} "
              f"({state['btc_held']:.8f} BTC @ ${price:,.2f})")
        return

    logger.info("Fetching live data from APIs...")
    cache_dir = THIS_DIR / config.get("data_dir", "live_data")
    features = fetch_all_live_data(cache_dir=str(cache_dir))

    logger.info("Fetching current BTC price...")
    price = fetch_current_btc_price()

    logger.info("Running daily tick...")
    actions = run_daily_tick(state, config, features, price)

    save_state(state, state_path)

    dashboard = format_dashboard(actions, state)
    print(dashboard)

    # Send webhook alert if configured and action taken
    webhook = config.get("alerts", {}).get("webhook_url", "")
    if webhook and actions["action"] != "none":
        _send_webhook(webhook, actions, dashboard)


def cmd_status(config: dict, state_path: Path) -> None:
    """Show current signals and portfolio without executing a trade."""
    state = load_state(state_path)

    logger.info("Fetching live data...")
    cache_dir = THIS_DIR / config.get("data_dir", "live_data")
    features = fetch_all_live_data(cache_dir=str(cache_dir))
    price = fetch_current_btc_price()

    if features.empty:
        print("No feature data available.")
        return

    latest = features.iloc[-1]
    portfolio_value = state["cash"] + state["reinvest_pool"] + state["btc_held"] * price

    signals = {
        "mvrv_zscore": float(latest.get("mvrv_zscore", 0)),
        "puell_multiple": float(latest.get("puell_multiple", 1)),
        "fgi": float(latest.get("fgi", 0.5)),
        "price_vs_ma": float(latest.get("price_vs_ma", 0)),
        "days_since_halving": float(latest.get("days_since_halving", 0)),
        "polymarket_sentiment": float(latest.get("polymarket_sentiment", 0.5)),
        "drawdown_from_ath": float(latest.get("drawdown_from_ath", 0)),
        "rsi": float(latest.get("rsi", 50)),
    }

    # Build a fake actions dict for the dashboard formatter
    actions = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "price": price,
        "action": "STATUS CHECK (no trade)",
        "details": {},
        "portfolio": {
            "btc_held": round(state["btc_held"], 8),
            "cash": round(state["cash"], 2),
            "reinvest_pool": round(state["reinvest_pool"], 2),
            "portfolio_value": round(portfolio_value, 2),
            "total_contributed": round(state["total_contributed"], 2),
            "profit": round(portfolio_value - state["total_contributed"], 2),
            "return_pct": round(
                ((portfolio_value / state["total_contributed"]) - 1) * 100, 2
            ) if state["total_contributed"] > 0 else 0,
        },
        "signals": signals,
    }
    print(format_dashboard(actions, state))

    # Show signal proximity warnings
    sell_params = config["sell_parameters"]
    n = 0
    if signals["mvrv_zscore"] >= sell_params["mvrv_zscore_min"] * 0.8:
        n += 1
    if signals["puell_multiple"] >= sell_params["puell_min"] * 0.8:
        n += 1
    if signals["fgi"] >= (sell_params["fgi_min"] / 100.0) * 0.8:
        n += 1
    if signals["price_vs_ma"] >= sell_params["price_vs_ma_min"] * 0.8:
        n += 1
    if signals["days_since_halving"] >= sell_params["min_days_post_halving"] * 0.8:
        n += 1
    if n >= 3:
        print(f"\n  ⚠️  WARNING: {n}/5 signals within 80% of sell threshold")


def cmd_history(state_path: Path) -> None:
    """Show trade history."""
    state = load_state(state_path)
    trades = state.get("trade_log", [])
    if not trades:
        print("No trades yet.")
        return

    print(f"\n{'Date':<12} {'Type':<10} {'Price':>12} {'BTC':>14} {'USD':>12}")
    print("-" * 62)
    for t in trades:
        print(f"{t['date']:<12} {t['type']:<10} ${t['price']:>10,.2f} "
              f"{t['btc']:>13.8f} ${t['usd']:>10,.2f}")

    print(f"\nTotal trades: {len(trades)}")
    print(f"Total tax paid: ${state.get('total_tax_paid', 0):,.2f}")
    print(f"Total reinvested: ${state.get('total_reinvested', 0):,.2f}")


def cmd_report(config: dict, state_path: Path) -> None:
    """Full performance report with return, win rate, drawdown, vs DCA benchmark."""
    state = load_state(state_path)
    daily_log = state.get("daily_log", [])

    if len(daily_log) < 2:
        print("Need at least 2 days of data for a report. Run --backfill first.")
        return

    price = fetch_current_btc_price()
    portfolio_value = state["cash"] + state["reinvest_pool"] + state["btc_held"] * price
    contributed = state["total_contributed"]
    dca_btc = state.get("dca_benchmark_btc", 0.0)
    dca_value = dca_btc * price

    # ── Daily returns ──
    values = [d["portfolio_value"] for d in daily_log]
    daily_returns = []
    for i in range(1, len(values)):
        if values[i - 1] > 0:
            daily_returns.append((values[i] / values[i - 1]) - 1.0)

    winning_days = sum(1 for r in daily_returns if r > 0)
    losing_days = sum(1 for r in daily_returns if r < 0)
    flat_days = sum(1 for r in daily_returns if r == 0)
    total_days = len(daily_returns)
    win_rate = (winning_days / total_days * 100) if total_days > 0 else 0

    avg_return = sum(daily_returns) / len(daily_returns) * 100 if daily_returns else 0
    best_day = max(daily_returns) * 100 if daily_returns else 0
    worst_day = min(daily_returns) * 100 if daily_returns else 0

    # ── Max drawdown ──
    peak = 0.0
    max_dd = 0.0
    max_dd_date = ""
    for d in daily_log:
        v = d["portfolio_value"]
        if v > peak:
            peak = v
        dd = (v / peak - 1.0) if peak > 0 else 0
        if dd < max_dd:
            max_dd = dd
            max_dd_date = d["date"]

    # ── Sharpe ratio (annualized, risk-free = 0) ──
    if len(daily_returns) > 1:
        mean_r = statistics.mean(daily_returns)
        std_r = statistics.stdev(daily_returns)
        sharpe = (mean_r / std_r) * (365 ** 0.5) if std_r > 0 else 0
    else:
        sharpe = 0

    # ── Trades breakdown ──
    trades = state.get("trade_log", [])
    sells = [t for t in trades if t["type"] == "sell"]
    reinvests = [t for t in trades if t["type"] == "reinvest"]
    buys = [t for t in trades if t["type"] == "buy"]

    # ── Rolling periods ──
    def period_return(n_days):
        if len(daily_log) < n_days + 1:
            return None
        old_v = daily_log[-(n_days + 1)]["portfolio_value"]
        new_v = daily_log[-1]["portfolio_value"]
        return ((new_v / old_v) - 1.0) * 100 if old_v > 0 else 0

    r_7d = period_return(7)
    r_30d = period_return(30)
    r_90d = period_return(90)

    # ── Print report ──
    first_date = daily_log[0]["date"]
    last_date = daily_log[-1]["date"]

    print()
    print("══════════════════════════════════════════════════════════")
    print("  BTC PAPER TRADER — PERFORMANCE REPORT")
    print(f"  Period: {first_date} to {last_date} ({total_days} days)")
    print("══════════════════════════════════════════════════════════")
    print()
    print("  ── Portfolio ──")
    print(f"  Current Value:     ${portfolio_value:>12,.2f}")
    print(f"  Total Contributed: ${contributed:>12,.2f}")
    print(f"  Total Profit:      ${portfolio_value - contributed:>12,.2f}")
    print(f"  Total Return:      {((portfolio_value / contributed) - 1) * 100 if contributed > 0 else 0:>11.2f}%")
    print(f"  BTC Held:          {state['btc_held']:>12.8f}")
    print(f"  BTC Price:         ${price:>12,.2f}")
    print()
    print("  ── vs DCA Benchmark (pure hold, no sells) ──")
    print(f"  DCA Bench Value:   ${dca_value:>12,.2f}")
    print(f"  DCA Bench Return:  {((dca_value / contributed) - 1) * 100 if contributed > 0 else 0:>11.2f}%")
    edge = portfolio_value - dca_value
    print(f"  Strategy Edge:     ${edge:>12,.2f}  {'✅' if edge > 0 else '❌'}")
    print()
    print("  ── Daily Stats ──")
    print(f"  Win Rate:          {win_rate:>11.1f}%  ({winning_days}W / {losing_days}L / {flat_days}F)")
    print(f"  Avg Daily Return:  {avg_return:>11.3f}%")
    print(f"  Best Day:          {best_day:>11.3f}%")
    print(f"  Worst Day:         {worst_day:>11.3f}%")
    print(f"  Max Drawdown:      {max_dd * 100:>11.2f}%  ({max_dd_date})")
    print(f"  Sharpe Ratio:      {sharpe:>11.2f}  (annualized)")
    print()
    print("  ── Rolling Returns ──")
    print(f"  7-Day:             {f'{r_7d:.2f}%':>12}" if r_7d is not None else "  7-Day:              N/A")
    print(f"  30-Day:            {f'{r_30d:.2f}%':>12}" if r_30d is not None else "  30-Day:             N/A")
    print(f"  90-Day:            {f'{r_90d:.2f}%':>12}" if r_90d is not None else "  90-Day:             N/A")
    print()
    print("  ── Trade Summary ──")
    print(f"  Total Trades:      {len(trades):>6}")
    print(f"  DCA Buys:          {len(buys):>6}")
    print(f"  Sells:             {len(sells):>6}")
    print(f"  Reinvestments:     {len(reinvests):>6}")
    print(f"  Tax Paid:          ${state.get('total_tax_paid', 0):>12,.2f}")
    print(f"  Sell Proceeds:     ${state.get('total_sell_proceeds', 0):>12,.2f}")
    print(f"  Reinvested:        ${state.get('total_reinvested', 0):>12,.2f}")

    # ── Sell trade detail ──
    if sells:
        print()
        print("  ── Sell History ──")
        print(f"  {'Date':<12} {'Price':>10} {'BTC Sold':>12} {'Proceeds':>10} {'Gain':>10}")
        print(f"  {'-'*56}")
        for s in sells:
            print(f"  {s['date']:<12} ${s['price']:>8,.0f} {s['btc']:>11.8f} "
                  f"${s['usd']:>8,.2f} ${s.get('gain', 0):>8,.2f}")

    print()
    print("══════════════════════════════════════════════════════════")
    print()


def cmd_reset(state_path: Path) -> None:
    """Reset portfolio state."""
    if state_path.exists():
        backup = state_path.with_suffix(".backup.json")
        state_path.rename(backup)
        print(f"Old state backed up to {backup}")
    state = load_state(state_path)  # Creates fresh state
    save_state(state, state_path)
    print("Portfolio reset to zero.")


def cmd_backfill(config: dict, state_path: Path, days: int) -> None:
    """Backfill the last N days of trading from historical data."""
    state = load_state(state_path)
    if state["total_contributed"] > 0:
        print("WARNING: State is not empty. Reset first with --reset if you want a clean backfill.")
        return

    logger.info(f"Fetching data for {days}-day backfill...")
    cache_dir = THIS_DIR / config.get("data_dir", "live_data")
    features = fetch_all_live_data(cache_dir=str(cache_dir))

    if features.empty:
        print("No data available for backfill.")
        return

    # Use the last N days of features
    end_idx = len(features)
    start_idx = max(0, end_idx - days)
    backfill_dates = features.index[start_idx:end_idx]

    print(f"Backfilling {len(backfill_dates)} days: "
          f"{backfill_dates[0].date()} to {backfill_dates[-1].date()}")

    for date in backfill_dates:
        row = features.loc[date]
        price = float(row["price"])
        if price <= 0:
            continue

        actions = run_daily_tick(state, config, features.loc[:date], price)

    save_state(state, state_path)

    portfolio_value = (state["cash"] + state["reinvest_pool"]
                       + state["btc_held"] * float(features.iloc[-1]["price"]))
    print(f"\nBackfill complete:")
    print(f"  Contributed: ${state['total_contributed']:,.2f}")
    print(f"  Portfolio:   ${portfolio_value:,.2f}")
    print(f"  BTC held:    {state['btc_held']:.8f}")
    print(f"  Trades:      {len(state['trade_log'])}")


def cmd_daemon(config: dict, state_path: Path) -> None:
    """Run continuously, executing once per day."""
    print("Starting daemon mode. Press Ctrl+C to stop.")
    while True:
        try:
            cmd_tick(config, state_path)
        except KeyboardInterrupt:
            print("\nStopping daemon.")
            break
        except Exception as e:
            logger.error(f"Tick failed: {e}")

        # Sleep until next day 9:00 UTC
        now = datetime.now(timezone.utc)
        next_run = now.replace(hour=9, minute=0, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)
        sleep_seconds = (next_run - now).total_seconds()
        logger.info(f"Next run at {next_run.isoformat()}. Sleeping {sleep_seconds/3600:.1f}h...")
        try:
            time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            print("\nStopping daemon.")
            break


def _send_webhook(url: str, actions: dict, dashboard: str) -> None:
    """Send alert via webhook (Discord/Slack compatible)."""
    import urllib.request

    payload = {
        "content": f"```\n{dashboard}\n```",
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            logger.info(f"Webhook sent: {resp.status}")
    except Exception as e:
        logger.warning(f"Webhook failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC Paper Trader")
    parser.add_argument("--config", type=Path, default=THIS_DIR / "config.json")
    parser.add_argument("--state", type=Path, default=None,
                        help="Override state file path")
    parser.add_argument("--status", action="store_true",
                        help="Show current signals without trading")
    parser.add_argument("--history", action="store_true",
                        help="Show trade history")
    parser.add_argument("--report", action="store_true",
                        help="Full performance report (return, win rate, drawdown, Sharpe)")
    parser.add_argument("--reset", action="store_true",
                        help="Reset portfolio to zero")
    parser.add_argument("--backfill", type=int, metavar="DAYS",
                        help="Backfill last N days from API data")
    parser.add_argument("--daemon", action="store_true",
                        help="Run continuously (once per day)")
    parser.add_argument("--force", action="store_true",
                        help="Force run even if already ran today")
    args = parser.parse_args()

    config = load_config(args.config)
    state_path = args.state or (THIS_DIR / config.get("state_file", "paper_state.json"))

    if args.reset:
        cmd_reset(state_path)
    elif args.report:
        cmd_report(config, state_path)
    elif args.history:
        cmd_history(state_path)
    elif args.status:
        cmd_status(config, state_path)
    elif args.backfill:
        cmd_backfill(config, state_path, args.backfill)
    elif args.daemon:
        cmd_daemon(config, state_path)
    else:
        cmd_tick(config, state_path, force=args.force)


if __name__ == "__main__":
    main()
