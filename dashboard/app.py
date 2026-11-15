"""Sentinel monitor: read-only Streamlit view of the bot's SQLite audit trail.

    streamlit run dashboard/app.py                       # live/paper database
    streamlit run dashboard/app.py -- --db data/replay.db  # a replay run
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="data/sentinel.db")
args, _ = ap.parse_known_args(sys.argv[1:])

st.set_page_config(page_title="Sentinel monitor", layout="wide")
db = st.sidebar.text_input("Database", args.db)
auto = st.sidebar.toggle("Auto-refresh (30 s)", value=False)
if not Path(db).exists():
    st.error(f"{db} not found. Run `sentinel live` or `sentinel replay` first.")
    st.stop()


@st.cache_data(ttl=10)
def q(sql: str, path: str) -> pd.DataFrame:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as c:
        df = pd.read_sql_query(sql, c)
    if "ts" in df:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601").dt.tz_convert("America/New_York")
    return df


eq = q("SELECT * FROM equity ORDER BY ts", db)
signals = q("SELECT * FROM signals ORDER BY ts DESC LIMIT 2000", db)
fills = q("SELECT * FROM fills ORDER BY ts DESC LIMIT 500", db)
orders = q("SELECT * FROM orders ORDER BY ts DESC LIMIT 200", db)
risk = q("SELECT * FROM risk_events ORDER BY ts DESC LIMIT 200", db)
news = q("""SELECT n.ts, n.symbols, n.headline, s.score, s.engine, s.rationale
            FROM news n LEFT JOIN sentiment s ON s.news_id = n.id ORDER BY n.ts DESC LIMIT 200""", db)

st.title("Sentinel monitor")
c1, c2, c3, c4, c5 = st.columns(5)
if not eq.empty:
    last = eq.iloc[-1]
    today = eq[eq["ts"].dt.date == last["ts"].date()]
    day_pnl = last["equity"] - today["equity"].iloc[0]
    c1.metric("Equity", f"${last['equity']:,.0f}", f"{day_pnl:+,.0f} today")
    c2.metric("Open positions", int(last["open_positions"]))
    c3.metric("Gross exposure", f"${last['gross_exposure']:,.0f}")
    dd = (eq["equity"] / eq["equity"].cummax() - 1).min()
    c4.metric("Max drawdown", f"{dd:.2%}")
closed = fills[fills["pnl"].notna()] if not fills.empty else fills
c5.metric("Closed trades", len(closed), f"win rate {(closed['pnl'] > 0).mean():.0%}" if len(closed) else None)

halts = risk[risk["level"] == "critical"] if not risk.empty else risk
if len(halts):
    st.error(f"Latest critical risk event: {halts.iloc[0]['ts']:%Y-%m-%d %H:%M} — {halts.iloc[0]['message']}")

if not eq.empty:
    fig = go.Figure(go.Scatter(x=eq["ts"], y=eq["equity"], mode="lines", name="equity"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), title="Equity")
    st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Decisions (non-hold)")
    act = signals[signals["action"] != "hold"] if not signals.empty else signals
    st.dataframe(act[["ts", "symbol", "action", "prob", "regime", "sentiment", "reason"]] if len(act) else act,
                 width="stretch", height=300, hide_index=True)
    st.subheader("Fills")
    st.dataframe(fills, width="stretch", height=260, hide_index=True)
with right:
    st.subheader("Risk events")
    st.dataframe(risk, width="stretch", height=300, hide_index=True)
    st.subheader("News & sentiment")
    st.dataframe(news, width="stretch", height=260, hide_index=True)

with st.expander("Latest model view per symbol"):
    if not signals.empty:
        latest = signals.sort_values("ts").groupby("symbol").tail(1)
        st.dataframe(latest, width="stretch", hide_index=True)
with st.expander("Orders"):
    st.dataframe(orders, width="stretch", hide_index=True)

if auto:
    time.sleep(30)
    st.rerun()
