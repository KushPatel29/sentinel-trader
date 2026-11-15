"""Sentinel: market analyzer + trading-bot lab (Streamlit).

Run locally:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")   # before numpy/lightgbm load: avoids thread oversubscription

sys.path.insert(0, str(Path(__file__).parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from sentinel.analyzer import data as D  # noqa: E402
from sentinel.analyzer.engine import analyze  # noqa: E402
from sentinel.analyzer.factors import GROUPS  # noqa: E402

st.set_page_config(page_title="Sentinel · Market Analyzer", page_icon="📈", layout="wide")

GREEN, RED, GREY, BLUE, AMBER = "#22c55e", "#ef4444", "#94a3b8", "#3b82f6", "#f59e0b"
PRESETS = {
    "Stocks": ["AAPL", "NVDA", "MSFT", "TSLA", "AMZN", "JPM"],
    "ETFs": ["SPY", "QQQ", "TLT"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
    "Commodities": ["GC=F", "CL=F", "SI=F", "NG=F"],
}
COMMODITY_NAMES = {"GC=F": "Gold", "CL=F": "Crude oil (WTI)", "SI=F": "Silver", "NG=F": "Natural gas",
                   "HG=F": "Copper", "ZC=F": "Corn", "ZW=F": "Wheat"}
DISCLAIMER = ("Model output for research and education, not financial advice. Markets are mostly unpredictable "
              "in the short run; every number below comes from out-of-sample tests, and past results do not "
              "guarantee future returns.")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.sig-card {border-radius: 14px; padding: 18px 22px; border: 1px solid rgba(148,163,184,.25);}
.sig-label {font-size: 2.1rem; font-weight: 700; line-height: 1.1;}
.small {color: #94a3b8; font-size: .85rem;}
.pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.8rem; font-weight:600;}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------- cached work
@st.cache_data(ttl=3600, show_spinner=False)
def run_analysis(ticker: str):
    t0 = time.time()
    a = analyze(ticker)
    print(f"[sentinel] analyzed {ticker} in {time.time() - t0:.1f}s", flush=True)
    return a


@st.cache_data(ttl=600, show_spinner=False)
def run_search(q: str):
    return D.search(q)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def run_intraday(symbols: tuple[str, ...], sessions: int):
    from sentinel.analyzer.intraday import run_lab
    bars = D.fetch_intraday(list(symbols))
    if bars.empty:
        raise ValueError("No intraday data returned.")
    return run_lab(bars, sessions), bars["ts"].min(), bars["ts"].max()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def run_universe(tickers: tuple[str, ...]):
    rows = []
    for t in tickers:
        try:
            a = analyze(t)
        except Exception as e:  # keep going if one asset fails
            rows.append({"Asset": t, "error": str(e)})
            continue
        m, T = a.ml.metrics if a.ml else {}, a.strategy_table
        best = T.drop(index="buy_hold").iloc[0]
        rows.append({
            "Asset": t, "Class": a.asset, "OOS years": round(m.get("oos_days", 0) / 252, 1),
            "ML accuracy": m.get("accuracy"), "Always-guess-majority": m.get("majority_accuracy"),
            "ML AUC": m.get("auc"), "p-value": m.get("p_value"),
            "Buy & hold Sharpe": T.loc["buy_hold", "sharpe"], "ML strategy Sharpe": T.loc["ml", "sharpe"] if "ml" in T.index else np.nan,
            "Best strategy": best["strategy"], "Best Sharpe": best["sharpe"], "Best vs B&H (t)": best["excess_t"],
            "Signal now": a.signal["label"], "Evidence": a.signal["quality"],
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------- helpers
def pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}%}"


def color_for(label: str) -> str:
    return {"Bullish": GREEN, "Bearish": RED}.get(label, AMBER)


def price_chart(a) -> go.Figure:
    px = a.prices.iloc[-756:]
    c = px["close"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=px.index, open=px["open"], high=px["high"], low=px["low"], close=c,
                                 name="Price", increasing_line_color=GREEN, decreasing_line_color=RED), 1, 1)
    full = a.prices["close"]
    for n, col in ((50, BLUE), (200, AMBER)):
        fig.add_trace(go.Scatter(x=px.index, y=full.rolling(n).mean().loc[px.index], name=f"{n}-day avg",
                                 line=dict(width=1.4, color=col)), 1, 1)
    if px["volume"].sum() > 0:
        fig.add_trace(go.Bar(x=px.index, y=px["volume"], name="Volume", marker_color="rgba(148,163,184,.45)"), 2, 1)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.02, x=0), template="plotly_dark")
    return fig


def equity_chart(eq: pd.DataFrame, keys: list[str], names: dict[str, str]) -> go.Figure:
    fig = go.Figure()
    for k in keys:
        if k in eq:
            fig.add_trace(go.Scatter(x=eq.index, y=eq[k], name=names.get(k, k),
                                     line=dict(width=2.6 if k == "buy_hold" else 1.6,
                                               dash="dot" if k == "buy_hold" else None)))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Growth of $1 (log)",
                      yaxis_type="log", template="plotly_dark", legend=dict(orientation="h", y=1.05))
    return fig


# ----------------------------------------------------------------------------------------- pages
def page_analyze():
    st.title("Analyze any asset")
    st.caption("Stocks, ETFs, crypto and commodities. Pick or search an asset: Sentinel downloads up to 10 years of "
               "daily data, tests 10 strategies and a machine-learning ensemble on it out-of-sample, reads trend, "
               "demand, psychology and macro factors, and combines the evidence into one signal.")
    c1, c2 = st.columns([2, 3])
    with c1:
        q = st.text_input("Search by name or ticker", placeholder="e.g. Apple, bitcoin, gold, RELIANCE.NS, ^NSEI")
        choice = None
        if q:
            res = run_search(q)
            if res:
                opts = {f"{r['symbol']} — {r['name']} ({r['type'] or '?'}, {r['exchange']})": r["symbol"] for r in res}
                choice = opts[st.selectbox("Matches", list(opts))]
            else:
                st.caption("No matches; using what you typed as the ticker.")
                choice = q.strip().upper()
    with c2:
        st.write("Quick picks")
        cols = st.columns(len(PRESETS))
        for col, (grp, ticks) in zip(cols, PRESETS.items()):
            pick = col.selectbox(grp, ["—"] + ticks, key=f"preset_{grp}",
                                 format_func=lambda t: f"{t} · {COMMODITY_NAMES[t]}" if t in COMMODITY_NAMES else t)
            if pick != "—":
                choice = pick
    ticker = choice or st.session_state.get("last_ticker", "SPY")
    st.session_state["last_ticker"] = ticker

    with st.spinner(f"Downloading {ticker}, computing factors, running walk-forward tests..."):
        try:
            a = run_analysis(ticker)
        except Exception as e:
            st.error(f"Could not analyze {ticker}: {e}")
            return

    last, prev = a.prices["close"].iloc[-1], a.prices["close"].iloc[-2]
    sig = a.signal
    st.subheader(f"{a.name} ({a.ticker}) · {a.asset}")
    left, right = st.columns([1.1, 2])
    with left:
        col = color_for(sig["label"])
        st.markdown(f"""<div class="sig-card" style="border-color:{col}">
            <div class="small">Sentinel signal · {sig['horizon']}</div>
            <div class="sig-label" style="color:{col}">{sig['label']}</div>
            <div>Score <b>{sig['score']:+.2f}</b> (−1 … +1) &nbsp;·&nbsp; Evidence
            <span class="pill" style="background:{ {'Strong': GREEN, 'Moderate': AMBER}.get(sig['quality'], GREY)}33">
            {sig['quality']}</span></div>
            <div class="small" style="margin-top:8px">Typical 5-day move: ±{sig['expected_move_1sd']:.1%} (1 sd) ·
            annual volatility {sig['vol_annual']:.0%}</div></div>""", unsafe_allow_html=True)
        st.metric("Last close", f"{last:,.2f}", f"{last / prev - 1:+.2%}")
        if sig["quality"] == "Weak":
            st.info("**Weak evidence:** none of the models or strategies showed a statistically reliable edge on "
                    "this asset out-of-sample. Treat the signal as a coin flip.")
        comp = pd.DataFrame(sig["components"])
        if not comp.empty:
            comp["weight"] = comp["weight"] / comp["weight"].sum()
            st.dataframe(comp.rename(columns=str.capitalize), hide_index=True, width="stretch",
                         column_config={"Score": st.column_config.ProgressColumn(min_value=-1, max_value=1, format="%.2f"),
                                        "Weight": st.column_config.NumberColumn(format="percent")})
    with right:
        st.plotly_chart(price_chart(a), width="stretch")

    t1, t2, t3, t4, t5 = st.tabs(["Factors: trend · demand · psychology · macro", "Strategy tournament",
                                  "ML accuracy", "News & fundamentals", "How it works"])
    with t1:
        rd = pd.DataFrame([r.__dict__ for r in a.readings])
        if not rd.empty:
            for grp in [g for g in GROUPS if g in set(rd["group"])]:
                sub = rd[rd["group"] == grp]
                st.markdown(f"**{grp}**")
                st.dataframe(sub[["name", "value", "reading", "why"]].rename(columns=str.capitalize), hide_index=True,
                             width="stretch")
        if a.ml:
            gi = pd.Series(a.ml.group_importance).sort_values()
            fig = go.Figure(go.Bar(x=gi.values, y=gi.index, orientation="h", marker_color=BLUE))
            fig.update_layout(height=300, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10),
                              title="What the model leans on for this asset (share of importance)")
            st.plotly_chart(fig, width="stretch")
    with t2:
        T = a.strategy_table
        st.caption(f"All strategies traded the same out-of-sample period ({a.equity.index.min():%b %Y} – "
                   f"{a.equity.index.max():%b %Y}), next-day execution, {('20' if a.asset == 'crypto' else '5')} bps "
                   "cost per trade. Rule strategies use fixed textbook parameters; nothing was tuned on this data.")
        show = T[["strategy", "cagr", "sharpe", "max_drawdown", "volatility", "exposure", "excess_t", "description"]].copy()
        show.columns = ["Strategy", "CAGR", "Sharpe", "Max drawdown", "Volatility", "Time invested", "vs B&H (t-stat)",
                        "Rule"]
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="percent") for c in ["CAGR", "Max drawdown", "Volatility", "Time invested"]}
            | {"Sharpe": st.column_config.NumberColumn(format="%.2f"),
               "vs B&H (t-stat)": st.column_config.NumberColumn(format="%.2f")})
        names = dict(zip(T.index, T["strategy"]))
        default = ["buy_hold"] + [k for k in T.index if k != "buy_hold"][:3]
        keys = st.multiselect("Compare equity curves", list(T.index), default=default, format_func=lambda k: names[k])
        st.plotly_chart(equity_chart(a.equity, keys, names), width="stretch")
        st.caption("A t-stat above ~2 versus buy & hold is the usual bar for 'probably not luck'. "
                   "Positions held now: " + ", ".join(f"{names.get(k, k)}: {'LONG' if v > 0 else 'flat'}"
                                                     for k, v in a.current_positions.items() if k in names))
    with t3:
        if not a.ml:
            st.warning(" ".join(a.notes))
        else:
            m = a.ml.metrics
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Direction accuracy", pct(m["accuracy"]), f"{m['accuracy'] - m['majority_accuracy']:+.1%} vs always-guess")
            k2.metric("AUC", f"{m['auc']:.3f}", "0.5 = coin flip", delta_color="off")
            k3.metric("p-value", f"{m['p_value']:.3f}", "< 0.05 = significant", delta_color="off")
            k4.metric("Out-of-sample days", f"{m['oos_days']:,}", f"{m['oos_start']:%Y} – {m['oos_end']:%Y}", delta_color="off")
            st.caption(f"Target: will the price be higher in {m['horizon_days']} trading days? Ensemble of LightGBM and "
                       f"logistic regression, retrained every quarter on past data only (labels purged). "
                       f"AUC by model: LightGBM {m['auc_gbm']:.3f}, logistic {m['auc_lr']:.3f}. The p-value tests whether "
                       f"accuracy beats always guessing the more common outcome, on {m['independent_samples']} "
                       "non-overlapping samples.")
            cal = m["calibration"]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=cal["bucket"], y=cal["actual_up"], name="Actually went up", marker_color=BLUE))
            fig.add_trace(go.Scatter(x=cal["bucket"], y=cal["predicted"], name="Model said", mode="lines+markers",
                                     line=dict(color=AMBER)))
            fig.update_layout(height=320, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10),
                              title="Calibration: when the model says X%, how often did it go up?",
                              yaxis_tickformat=".0%")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(pd.DataFrame(a.ml.top_features, columns=["Feature", "Importance"]), hide_index=True)
    with t4:
        n1, n2 = st.columns(2)
        with n1:
            st.markdown("**Recent headlines** (lexicon sentiment)")
            if not a.news:
                st.caption("No headlines available.")
            for h in a.news[:12]:
                s = h["sentiment"]
                dot = GREEN if s > 0.15 else RED if s < -0.15 else GREY
                link = f"[{h['title']}]({h['link']})" if h.get("link") else h["title"]
                st.markdown(f"<span style='color:{dot}'>●</span> {link} <span class='small'>· {h.get('publisher', '')}"
                            f" · {s:+.2f}</span>", unsafe_allow_html=True)
        with n2:
            st.markdown("**Fundamentals & positioning**")
            if a.fundamentals:
                st.dataframe(pd.DataFrame({"Metric": list(a.fundamentals), "Value": [str(v) for v in a.fundamentals.values()]}),
                             hide_index=True, width="stretch")
                st.caption("Short interest and analyst ratings are crowd-positioning (psychology) inputs. "
                           "Point-in-time data like this cannot be backtested, so it carries a small weight.")
            else:
                st.caption("No fundamental data for this asset type.")
    with t5:
        st.markdown("""
**Pipeline**
1. **Data:** up to 10 years of daily prices, plus context series: S&P 500 (SPY), VIX (fear gauge), 10-year Treasury yield and the US dollar index (and Bitcoin for crypto).
2. **Factors (49):** trend, momentum, mean reversion, volatility, **demand & volume** (on-balance volume, accumulation/distribution, money flow, up/down volume), **psychology** (fear & greed composite, VIX percentile, 52-week anchoring, drawdown, streaks, gaps), **macro** (market trend, relative strength, beta, rates, dollar) and seasonality.
3. **Strategy tournament:** 9 rule strategies + walk-forward ML + an adaptive "follow the leader" meta-strategy, all on the same out-of-sample window, with trading costs.
4. **Signal:** each piece of evidence gets a vote weighted by how well it has proven itself out-of-sample. A model with no demonstrated edge gets no say.

**What the tests show across 17 assets (stocks, ETFs, crypto, commodities):** predicting the 5-day direction from daily data is close to a coin flip (mean AUC ≈ 0.505). Trend and volatility filters mainly *cut drawdowns*; they rarely beat buy & hold on risk-adjusted return. The app shows these numbers for every asset instead of hiding them.
""")
    st.caption(DISCLAIMER)


def page_lab():
    st.title("Accuracy Lab")
    st.caption("Runs the full analyzer on a basket of assets and reports out-of-sample accuracy honestly. "
               "Takes about a minute the first time, then it's cached.")
    sel = st.multiselect("Assets", sum(PRESETS.values(), []), default=sum(PRESETS.values(), []))
    if st.button("Run accuracy test", type="primary"):
        st.session_state["lab"] = tuple(sel)
    if "lab" in st.session_state:
        with st.spinner("Testing every asset out-of-sample..."):
            df = run_universe(st.session_state["lab"])
        if "error" in df:
            errs = df[df["error"].notna()]
            if len(errs):
                st.warning("Failed: " + ", ".join(errs["Asset"]))
            df = df[df["error"].isna()].drop(columns="error")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Mean ML accuracy", pct(df["ML accuracy"].mean()), f"{(df['ML accuracy'] - df['Always-guess-majority']).mean():+.1%} vs always-guess")
        a2.metric("Mean AUC", f"{df['ML AUC'].mean():.3f}")
        a3.metric("Significant (p<0.05)", f"{(df['p-value'] < 0.05).sum()} / {len(df)}")
        a4.metric("ML beats buy & hold Sharpe", f"{(df['ML strategy Sharpe'] > df['Buy & hold Sharpe']).sum()} / {len(df)}")
        st.dataframe(df, hide_index=True, width="stretch", column_config={
            "ML accuracy": st.column_config.NumberColumn(format="percent"),
            "Always-guess-majority": st.column_config.NumberColumn(format="percent"),
            "ML AUC": st.column_config.NumberColumn(format="%.3f"), "p-value": st.column_config.NumberColumn(format="%.3f"),
            "Buy & hold Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "ML strategy Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "Best Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "Best vs B&H (t)": st.column_config.NumberColumn(format="%.2f")})
        st.caption("How to read this: an AUC of 0.50 is a coin flip. With many assets tested, expect about 1 in 20 "
                   "to look 'significant' at p<0.05 by pure luck.")
    st.caption(DISCLAIMER)


def page_bot():
    st.title("Trading bot lab (15-minute intraday)")
    st.caption("The Sentinel paper-trading bot's strategy on the last ~60 days of real 15-minute bars: LightGBM + "
               "candlestick patterns + regime filter, bracket orders (take profit 2 ATR / stop 1 ATR), risk officer "
               "with a 2% daily loss halt, flat by the close.")
    syms = st.multiselect("Symbols", ["SPY", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "JPM"],
                          default=["SPY", "AAPL", "MSFT", "NVDA", "AMZN", "META"])
    sessions = st.slider("Forward-simulation sessions (bot trades these days bar by bar with a model that never saw them)",
                         3, 15, 5)
    if st.button("Run backtest + forward simulation", type="primary") and syms:
        st.session_state["bot"] = (tuple(sorted(syms)), sessions)
    if "bot" not in st.session_state:
        return
    with st.spinner("Downloading 15-minute bars, walk-forward training, simulating every bar..."):
        try:
            lab, t0, t1 = run_intraday(*st.session_state["bot"])
        except Exception as e:
            st.error(f"Intraday lab failed: {e}")
            return
    st.caption(f"Data: {t0:%b %d} – {t1:%b %d, %Y} (Yahoo Finance, regular session).")
    for title, res, note in [
        ("Walk-forward backtest (out-of-sample)", lab.backtest, f"Model AUC {lab.auc:.3f} (base rate {lab.base_rate:.1%} of setups hit the target)"),
        (f"Forward simulation: last {lab.n_sessions} sessions", lab.forward, f"From {lab.forward_start:%b %d}: rolling features, per-bar inference, simulated bracket fills"),
    ]:
        st.subheader(title)
        st.caption(note)
        m, b = res.metrics, res.benchmark
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Return", pct(m.get("total_return", 0), 2), f"buy & hold {pct(b.get('total_return', 0), 2)}", delta_color="off")
        c2.metric("Trades", m.get("trades", 0))
        c3.metric("Win rate", pct(m.get("win_rate", 0)))
        pf = m.get("profit_factor", 0)
        c4.metric("Profit factor", "∞" if pf == float("inf") else f"{pf:.2f}")
        c5.metric("Max drawdown", pct(m.get("max_drawdown", 0), 2))
        if not res.equity.empty:
            eq = res.equity.set_index("ts")["equity"]
            fig = go.Figure(go.Scatter(x=eq.index, y=eq.values, line=dict(color=BLUE)))
            fig.update_layout(height=260, template="plotly_dark", margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Equity ($)")
            st.plotly_chart(fig, width="stretch")
        if not res.fills.empty:
            with st.expander("Trades"):
                st.dataframe(res.fills, hide_index=True, width="stretch")
    st.info("A few dozen trades over a couple of months is far too small a sample to call the strategy profitable. "
            "Paper trade it for 4–8 weeks and compare.")
    st.caption(DISCLAIMER)


def page_paper():
    st.title("Paper trading account")
    try:
        key = st.secrets["ALPACA_API_KEY"]
        secret = st.secrets["ALPACA_SECRET_KEY"]
    except Exception:
        key = secret = None
    if not key:
        st.info("Connect a read-only view of your **Alpaca paper** account: in Streamlit Cloud open *Settings → Secrets* "
                "and add `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` (paper keys). The bot itself runs separately "
                "(`sentinel live` on your computer); this page only reads the account.")
        return
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest
    c = TradingClient(key, secret, paper=True)
    a = c.get_account()
    m1, m2, m3 = st.columns(3)
    m1.metric("Equity", f"${float(a.equity):,.2f}", f"{float(a.equity) - float(a.last_equity):+,.2f} today")
    m2.metric("Cash", f"${float(a.cash):,.2f}")
    m3.metric("Buying power", f"${float(a.buying_power):,.2f}")
    pos = c.get_all_positions()
    st.subheader("Positions")
    st.dataframe(pd.DataFrame([{"Symbol": p.symbol, "Qty": p.qty, "Entry": p.avg_entry_price, "Price": p.current_price,
                                "P&L": p.unrealized_pl} for p in pos]), hide_index=True, width="stretch")
    st.subheader("Recent orders")
    orders = c.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=50))
    st.dataframe(pd.DataFrame([{"Time": o.submitted_at, "Symbol": o.symbol, "Side": str(o.side.value), "Qty": o.qty,
                                "Type": str(o.order_type.value), "Status": str(o.status.value),
                                "Fill": o.filled_avg_price} for o in orders]), hide_index=True, width="stretch")


PAGES = {"🔎 Analyze any asset": page_analyze, "🧪 Accuracy Lab": page_lab, "🤖 Trading bot lab": page_bot,
         "💼 Paper account": page_paper}
with st.sidebar:
    st.markdown("### 📈 Sentinel")
    page = st.radio("Go to", list(PAGES), label_visibility="collapsed")
    st.markdown("---")
    st.caption("Built by Kush Patel · [GitHub](https://github.com/kushpatel29/sentinel-trader)")
    st.caption(DISCLAIMER)
PAGES[page]()
