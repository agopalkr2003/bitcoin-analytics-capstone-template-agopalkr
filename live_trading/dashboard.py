"""BTC Paper Trader — Streamlit Dashboard"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

THIS_DIR = Path(__file__).resolve().parent
STATE_FILE = THIS_DIR / "paper_state.json"
CONFIG_FILE = THIS_DIR / "config.json"


@st.cache_data(ttl=60)
def load_state() -> dict:
    with open(STATE_FILE) as f:
        return json.load(f)


@st.cache_data(ttl=60)
def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return json.load(f)


def main():
    st.set_page_config(page_title="BTC Paper Trader", page_icon="₿", layout="wide")

    state = load_state()
    config = load_config()
    daily_log = state.get("daily_log", [])
    trade_log = state.get("trade_log", [])
    lots = state.get("lots", [])

    if not daily_log:
        st.warning("No trading data yet. Run the paper trader first.")
        return

    df = pd.DataFrame(daily_log)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    latest = df.iloc[-1]

    # ── Current price from most recent data ──
    price = latest["price"]
    portfolio_value = latest["portfolio_value"]
    contributed = latest["total_contributed"]
    profit = portfolio_value - contributed
    return_pct = latest.get("return_pct", 0)
    dca_bench = latest.get("dca_benchmark_value", 0)

    # ══════════════════════════════════════════
    # Header
    # ══════════════════════════════════════════
    st.title("₿ BTC Paper Trader")
    st.caption(f"Last updated: {latest['date'].strftime('%B %d, %Y')}  ·  "
               f"BTC @ ${price:,.0f}")

    # ══════════════════════════════════════════
    # KPI Row
    # ══════════════════════════════════════════
    k1, k2, k3, k4, k5 = st.columns(5)

    # Compute deltas from previous day
    if len(df) >= 2:
        prev = df.iloc[-2]
        pv_delta = portfolio_value - prev["portfolio_value"]
        ret_delta = return_pct - prev.get("return_pct", 0)
        btc_delta = latest["btc_held"] - prev["btc_held"]
    else:
        pv_delta = ret_delta = btc_delta = 0

    k1.metric("Portfolio Value", f"${portfolio_value:,.2f}", f"${pv_delta:+,.2f}")
    k2.metric("Total Return", f"{return_pct:+.1f}%", f"{ret_delta:+.2f}%")
    k3.metric("Total Profit", f"${profit:+,.2f}")
    k4.metric("BTC Held", f"₿{latest['btc_held']:.6f}", f"{btc_delta:+.8f}")
    k5.metric("Cash", f"${latest['cash']:,.2f}")

    st.divider()

    # ══════════════════════════════════════════
    # Charts
    # ══════════════════════════════════════════
    tab_perf, tab_btc, tab_lots, tab_log = st.tabs(
        ["📈 Performance", "₿ BTC Holdings", "📦 Cost Basis", "📋 Daily Log"]
    )

    with tab_perf:
        st.subheader("Portfolio vs DCA Benchmark")
        chart_df = df[["date", "portfolio_value", "total_contributed"]].copy()
        chart_df = chart_df.rename(columns={
            "portfolio_value": "Portfolio",
            "total_contributed": "Contributed",
        })
        if "dca_benchmark_value" in df.columns:
            chart_df["DCA Benchmark"] = df["dca_benchmark_value"].values
        chart_df = chart_df.set_index("date")
        st.line_chart(chart_df, use_container_width=True)

        # Strategy vs DCA comparison
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Strategy Value", f"${portfolio_value:,.2f}",
                       f"{return_pct:+.1f}%")
        with col2:
            if dca_bench > 0:
                dca_ret = ((dca_bench / contributed) - 1) * 100 if contributed > 0 else 0
                st.metric("DCA Benchmark", f"${dca_bench:,.2f}", f"{dca_ret:+.1f}%")
        with col3:
            edge = portfolio_value - dca_bench if dca_bench > 0 else 0
            st.metric("Strategy Edge", f"${edge:+,.2f}",
                       "Outperforming" if edge > 0 else "Underperforming")

        # Return over time
        st.subheader("Daily Return %")
        ret_df = df[["date", "return_pct"]].set_index("date")
        st.area_chart(ret_df, use_container_width=True)

    with tab_btc:
        st.subheader("BTC Holdings Over Time")
        btc_df = df[["date", "btc_held"]].set_index("date")
        st.area_chart(btc_df, use_container_width=True, color="#f7931a")

        st.subheader("BTC Price")
        price_df = df[["date", "price"]].set_index("date")
        st.line_chart(price_df, use_container_width=True, color="#f7931a")

    with tab_lots:
        st.subheader("Tax Lots (Cost Basis)")
        if lots:
            lots_df = pd.DataFrame(lots)
            lots_df["date"] = pd.to_datetime(lots_df["date"])
            lots_df["cost_per_btc"] = lots_df["cost"] / lots_df["qty"]
            lots_df["current_value"] = lots_df["qty"] * price
            lots_df["unrealized_pnl"] = lots_df["current_value"] - lots_df["cost"]
            lots_df["pnl_pct"] = (lots_df["unrealized_pnl"] / lots_df["cost"] * 100)

            # Summary metrics
            total_cost = lots_df["cost"].sum()
            total_value = lots_df["current_value"].sum()
            total_pnl = total_value - total_cost
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Cost Basis", f"${total_cost:,.2f}")
            c2.metric("Current Value", f"${total_value:,.2f}")
            c3.metric("Unrealized P&L", f"${total_pnl:+,.2f}")
            c4.metric("Avg Cost/BTC", f"${total_cost / lots_df['qty'].sum():,.0f}")

            st.dataframe(
                lots_df[["date", "qty", "cost", "cost_per_btc",
                          "current_value", "unrealized_pnl", "pnl_pct"]]
                .sort_values("date", ascending=False)
                .style.format({
                    "qty": "{:.8f}",
                    "cost": "${:,.2f}",
                    "cost_per_btc": "${:,.0f}",
                    "current_value": "${:,.2f}",
                    "unrealized_pnl": "${:+,.2f}",
                    "pnl_pct": "{:+.1f}%",
                })
                .map(lambda v: "color: green" if v > 0 else "color: red",
                     subset=["unrealized_pnl", "pnl_pct"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No lots yet.")

    with tab_log:
        st.subheader("Daily Trading Log")
        log_df = df[["date", "price", "btc_held", "cash", "portfolio_value",
                      "total_contributed", "return_pct", "action"]].copy()
        log_df = log_df.sort_values("date", ascending=False)
        st.dataframe(
            log_df.style.format({
                "price": "${:,.0f}",
                "btc_held": "{:.8f}",
                "cash": "${:,.2f}",
                "portfolio_value": "${:,.2f}",
                "total_contributed": "${:,.2f}",
                "return_pct": "{:+.1f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

        if trade_log:
            st.subheader("Trade History")
            trades_df = pd.DataFrame(trade_log)
            st.dataframe(trades_df, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════
    # Sidebar — Config & Stats
    # ══════════════════════════════════════════
    with st.sidebar:
        st.header("Strategy Config")
        weights = config.get("buy_weights", {})
        for k, v in weights.items():
            st.text(f"{k}: {v}")

        st.divider()
        st.header("Quick Stats")
        st.text(f"Days Trading: {len(df)}")
        st.text(f"Total Lots: {len(lots)}")
        st.text(f"DCA/day: ${config.get('daily_contribution_usd', 10):.0f}")
        st.text(f"Taker Fee: {config.get('fees', {}).get('taker_fee_rate', 0.006)*100:.1f}%")

        st.divider()
        st.header("Sell Triggers")
        sell = config.get("sell_parameters", {})
        st.text(f"MVRV ≥ {sell.get('mvrv_zscore_min', '?')}")
        st.text(f"Puell ≥ {sell.get('puell_min', '?')}")
        st.text(f"FGI ≥ {sell.get('fgi_min', '?')}%")
        st.text(f"Price/MA ≥ {sell.get('price_vs_ma_min', '?')}")
        st.text(f"Days post halving ≥ {sell.get('min_days_post_halving', '?')}")


if __name__ == "__main__":
    main()
