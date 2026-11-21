"""Gold bot lab: the Sentinel intraday bot on GLD (the gold ETF Alpaca can trade), plus the paper account."""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).parent / "src"))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from sentinel.analyzer import data as D  # noqa: E402

st.title("Gold bot lab (GLD, 15-minute bars)")
st.caption("The paper-trading bot trades **GLD**, the largest gold ETF (it tracks spot gold and trades on Alpaca). "
           "Strategy: LightGBM on 15-minute candles and patterns, regime filter, bracket orders (take profit 2 ATR / "
           "stop 1 ATR), a risk officer with a 2% daily loss halt, and flat by the close.")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def run(symbols: tuple[str, ...], sessions: int):
    from sentinel.analyzer.intraday import run_lab
    bars = D.fetch_intraday(list(symbols))
    if bars.empty:
        raise ValueError("No intraday data returned.")
    return run_lab(bars, sessions), bars["ts"].min(), bars["ts"].max()


sessions = st.slider("Forward-simulation sessions: the bot trades these days bar by bar with a model that never saw them",
                     3, 15, 5)
if st.button("Run backtest + forward simulation on the last 60 days", type="primary"):
    st.session_state["gold_bot"] = sessions
if "gold_bot" in st.session_state:
    with st.spinner("Downloading GLD 15-minute bars, walk-forward training, simulating every bar..."):
        try:
            lab, t0, t1 = run(("GLD",), st.session_state["gold_bot"])
        except Exception as e:
            st.error(f"Bot lab failed: {e}")
            st.stop()
    st.caption(f"Data: {t0:%b %d} – {t1:%b %d, %Y} (Yahoo Finance, regular session). Model AUC {lab.auc:.3f}.")
    for title, res in (("Walk-forward backtest (out-of-sample)", lab.backtest),
                       (f"Forward simulation: last {lab.n_sessions} sessions", lab.forward)):
        st.subheader(title)
        m, b = res.metrics, res.benchmark
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Return", f"{m.get('total_return', 0):+.2%}", f"GLD buy & hold {b.get('total_return', 0):+.2%}",
                  delta_color="off")
        c2.metric("Trades", m.get("trades", 0))
        c3.metric("Win rate", f"{m.get('win_rate', 0):.0%}")
        pf = m.get("profit_factor", 0)
        c4.metric("Profit factor", "∞" if pf == float("inf") else f"{pf:.2f}")
        c5.metric("Max drawdown", f"{m.get('max_drawdown', 0):.2%}")
        if not res.equity.empty:
            eq = res.equity.set_index("ts")["equity"]
            fig = go.Figure(go.Scatter(x=eq.index, y=eq.values, line=dict(color="#f5c542")))
            fig.update_layout(height=240, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")
        if not res.fills.empty:
            with st.expander("Trades"):
                st.dataframe(res.fills, hide_index=True, width="stretch")
    st.info("Sixty days of one symbol gives only a handful of trades, far too few to prove a profit. "
            "Paper trade for 4–8 weeks first.")

st.divider()
st.subheader("Paper trading account")
try:
    key, secret = st.secrets["ALPACA_API_KEY"], st.secrets["ALPACA_SECRET_KEY"]
except Exception:
    key = secret = None
if not key:
    st.info("To see your Alpaca **paper** account here, add `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in the app's "
            "Settings → Secrets. The bot itself runs on your computer (`start_bot.bat`).")
else:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    client = TradingClient(key, secret, paper=True)
    a = client.get_account()
    m1, m2, m3 = st.columns(3)
    m1.metric("Equity", f"${float(a.equity):,.2f}", f"{float(a.equity) - float(a.last_equity):+,.2f} today")
    m2.metric("Cash", f"${float(a.cash):,.2f}")
    m3.metric("Buying power", f"${float(a.buying_power):,.2f}")
    st.dataframe(pd.DataFrame([{"Symbol": p.symbol, "Qty": p.qty, "Entry": p.avg_entry_price, "Price": p.current_price,
                                "P&L": p.unrealized_pl} for p in client.get_all_positions()]), hide_index=True,
                 width="stretch")
    orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=50))
    st.dataframe(pd.DataFrame([{"Time": o.submitted_at, "Symbol": o.symbol, "Side": str(o.side.value), "Qty": o.qty,
                                "Status": str(o.status.value), "Fill": o.filled_avg_price} for o in orders]),
                 hide_index=True, width="stretch")
