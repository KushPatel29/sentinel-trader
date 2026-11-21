"""Sentinel Gold: one page, everything about gold, with the measured accuracy next to every forecast."""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from sentinel.gold import model as M  # noqa: E402
from sentinel.gold.data import load_universe  # noqa: E402
from sentinel.gold.plan import todays_plan  # noqa: E402

GOLD_C, GREEN, RED, GREY, BLUE = "#f5c542", "#22c55e", "#ef4444", "#94a3b8", "#3b82f6"
DISCLAIMER = ("Research tool, not financial advice. Gold's short-term direction has not been predictable in our tests; "
              "the price ranges are statistical estimates and will be wrong about 1 time in 5 at 80% confidence. "
              "Trade only money you can afford to lose.")

st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1400px}
.card{border:1px solid #27334a;border-radius:14px;padding:16px 20px;background:#111a2e;height:100%}
.big{font-size:clamp(1.15rem,2.1vw,1.7rem);font-weight:700;line-height:1.2}
[data-testid="stMetricValue"]{font-size:clamp(1.05rem,1.9vw,1.75rem)}
[data-testid="stMetricLabel"] p{white-space:normal}
.muted{color:#94a3b8;font-size:.86rem}
.tag{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.78rem;font-weight:600;margin-right:6px}
</style>""", unsafe_allow_html=True)


@st.cache_data(ttl=900, show_spinner=False)
def compute():
    t0 = time.time()
    gold, ctx = load_universe()
    out = {
        "gold": gold, "ctx": ctx, "ranges": M.ranges(gold, ctx), "direction": M.direction(gold, ctx),
        "drivers": M.drivers(gold, ctx), "sentiment": M.sentiment(gold, ctx),
        "strategies": M.strategies(gold), "strategies_cfd": M.strategies(gold, financing_annual=0.06),
        "season": M.seasonality(gold), "loaded": pd.Timestamp.utcnow(),
        "plan": todays_plan(gold, ctx),
    }
    print(f"[gold] computed in {time.time() - t0:.1f}s", flush=True)
    return out


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def extra_drivers():
    from sentinel.gold.extra import extended_drivers
    return extended_drivers()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def research_log():
    import json
    p = Path(__file__).parent / "research" / "gold" / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def money(x):
    return f"{x:,.2f}"


with st.spinner("Downloading gold and its drivers, recalibrating forecasts..."):
    try:
        D = compute()
    except Exception as e:
        st.error(f"Could not load gold data right now: {e}")
        st.stop()

gold, ctx, R = D["gold"], D["ctx"], D["ranges"]
last, prev = float(gold["close"].iloc[-1]), float(gold["close"].iloc[-2])
c = gold["close"]

# ------------------------------------------------------------------ sidebar: user's own price and account
with st.sidebar:
    st.markdown("### 🪙 Your setup")
    broker = st.number_input("Your broker's gold price now (optional)", min_value=0.0, value=0.0, step=1.0,
                             help="Every level is computed on COMEX gold futures (GC=F). Enter your XAU/USD quote "
                                  "and all levels are rescaled to your broker's price.")
    scale = broker / last if broker > 0 else 1.0
    account = st.number_input("Account size", min_value=100.0, value=10_000.0, step=500.0)
    risk_pct = st.slider("Risk per trade (% of account)", 0.25, 3.0, 1.0, 0.25) / 100
    lot_oz = st.selectbox("Contract size", [100.0, 10.0, 1.0], format_func=lambda x: {100.0: "Standard lot = 100 oz",
                          10.0: "Mini lot = 10 oz", 1.0: "Micro / 1 oz"}[x])
    st.caption(DISCLAIMER)

px = lambda v: v * scale  # noqa: E731

# ------------------------------------------------------------------ header
st.markdown('<div class="muted">SENTINEL / GOLD</div>', unsafe_allow_html=True)
st.title("Gold (XAU/USD) signals")
sent = D["sentiment"]
vol_ann = R.sigma * np.sqrt(252)
size_mult = float(np.clip(0.15 / vol_ann, 0, 1.5))
sma50, sma200 = c.rolling(50).mean().iloc[-1], c.rolling(200).mean().iloc[-1]
trend = "Uptrend" if last > sma50 and last > sma200 else "Downtrend" if last < sma50 and last < sma200 else "Mixed"
trend_col = GREEN if trend == "Uptrend" else RED if trend == "Downtrend" else GOLD_C
dirs = D["direction"]
any_edge = bool(dirs["reliable"].any())
r1 = R.bands[(R.bands.days == 1) & (R.bands.level == 0.8)].iloc[0]
r5 = R.bands[(R.bands.days == 5) & (R.bands.level == 0.8)].iloc[0]

k1, k2, k3 = st.columns(3)
k4, k5, k6 = st.columns(3)
k1.metric("Gold" + (" (your broker)" if scale != 1 else " (COMEX futures)"), money(px(last)), f"{last / prev - 1:+.2%}")
k2.metric("Trend", trend, f"{last / sma200 - 1:+.1%} vs 200-day", delta_color="off")
k3.metric("Daily volatility", f"±{R.sigma:.2%}", f"±${px(last) * R.sigma:,.0f} per oz, 1 sd", delta_color="off")
k4.metric("Fear & greed (gold)", f"{sent['greed_index']:.0f}/100",
          "extreme fear" if sent["greed_index"] < 20 else "fear" if sent["greed_index"] < 40 else
          "neutral" if sent["greed_index"] < 60 else "greed" if sent["greed_index"] < 80 else "extreme greed",
          delta_color="off")
k5.metric("From all-time high", f"{sent['drawdown_from_high']:.1%}")
k6.metric("Gold VIX (GVZ)", f"{sent['gvz']:.1f}" if sent["gvz"] else "—",
          f"{sent['gvz_percentile']:.0%} percentile" if sent["gvz_percentile"] is not None else None, delta_color="off")
st.caption(f"Last daily bar: {gold.index[-1]:%a %d %b %Y} (Yahoo Finance, delayed). Refreshed {D['loaded']:%H:%M} UTC; "
           "updates every 15 minutes.")

# ------------------------------------------------------------------ the signal, in plain words
st.markdown("### Today's gold signal")
a, b, cc = st.columns([1.15, 1, 1])
with a:
    st.markdown(f"""<div class="card"><div class="muted">Expected range: next day (80% confidence)</div>
    <div class="big" style="color:{GOLD_C}">{money(px(r1.low))} – {money(px(r1.high))}</div>
    <div class="muted">Next week: {money(px(r5.low))} – {money(px(r5.high))}<br>
    Accuracy on unseen 2019–2026 data: <b>{r1.measured_coverage:.0%}</b> of days ended inside the 80% range
    ({r1.tests:,} tests)</div></div>""", unsafe_allow_html=True)
with b:
    verdict = "Edge found" if any_edge else "No reliable edge"
    st.markdown(f"""<div class="card"><div class="muted">Direction (up or down)</div>
    <div class="big" style="color:{GREEN if any_edge else GREY}">{verdict}</div>
    <div class="muted">{"A model beat the 'always up' baseline with significance." if any_edge else
    "No model beat simply guessing 'up' on 2019–2026. Treat any up/down call, ours included, as a coin flip."}
    </div></div>""", unsafe_allow_html=True)
with cc:
    st.markdown(f"""<div class="card"><div class="muted">Risk signal</div>
    <div class="big" style="color:{trend_col}">{trend}<br>Size ×{size_mult:.1f}</div>
    <div class="muted">Volatility is at its {sent['vol_percentile']:.0%} percentile. Volatility-sizing
    (15% annual target) suggests <b>{size_mult:.1f}×</b> a normal position. That rule had the best risk-adjusted
    result of all tested strategies.</div></div>""", unsafe_allow_html=True)

P = D["plan"]
plan_col = {"ENTER LONG": GREEN, "HOLD LONG": GREEN, "EXIT": RED, "STAND ASIDE": GREY}[P.action]
if P.action == "STAND ASIDE":
    detail = (f"Gold is not in an uptrend. The plan buys only after a daily close above <b>{money(px(P.trigger_above))}</b> "
              f"(above both the 50-day average {money(px(P.sma50))} and the 200-day {money(px(P.sma200))}).")
elif P.action == "EXIT":
    detail = (f"Close the long position. Re-entry needs a close above {money(px(P.trigger_above))}.")
elif P.action == "ENTER LONG":
    detail = (f"Buy at the next open (~{money(px(P.entry))}). Initial stop-loss <b>{money(px(P.stop))}</b>. The stop "
              "then trails 2 daily volatilities below the best close. Exit at the stop, or at the open after gold "
              "closes below its 50- or 200-day average.")
else:
    detail = (f"Holding from {money(px(P.entry))}. Trailing stop-loss now <b>{money(px(P.stop))}</b>: move your stop "
              "here (never lower). Exit at the stop, or at the open after a close below the 50- or 200-day average.")
st.markdown(f"""<div class="card" style="margin:14px 0;border-color:{plan_col}">
<div class="muted">TODAY'S TRADE PLAN · rule: trend-following long with trailing stop (see 🗓️ Daily plan)</div>
<div class="big" style="color:{plan_col}">{P.action}</div><div>{detail}</div></div>""", unsafe_allow_html=True)

tabs = st.tabs(["🗓️ Daily plan", "📏 Price ranges", "🎯 Trade planner", "🧭 Direction tests", "🌍 What's moving gold",
                "🧠 Psychology & demand", "📈 Strategies", "📅 Seasonality", "🔬 What we tested", "ℹ️ Method"])

# ------------------------------------------------------------------ daily plan
with tabs[0]:
    st.markdown("""**The rule** (fixed in advance; chosen on 2006–2018 out of 12 plan designs, then tested on 2019–2026):
1. **Entry:** when gold closes above both its 50-day and 200-day averages, buy at the next open.
2. **Stop-loss:** 2 daily volatilities below the best close since entry. It only moves up (a trailing stop).
3. **Exit:** at the stop-loss, or at the next open after gold closes below either average.
4. **Size:** use the Trade planner so a stop-out costs at most your chosen % of the account.

The plan needs no up/down forecast. It follows trends and cuts losing trades early. It is checked every day after the
close, so act at the next open.""")
    stt = P.stats.copy()
    if not stt.empty:
        st.markdown("**Real track record of this exact rule** (spread, slippage and 6%/yr CFD financing included):")
        st.dataframe(stt, hide_index=True, width="stretch", column_config={
            k: st.column_config.NumberColumn(format="percent") for k in
            ["win_rate", "avg_win", "avg_loss", "annual_return", "worst_trade", "max_drawdown"]} |
            {"avg_days_held": st.column_config.NumberColumn(format="%.1f")})
        recent = stt.iloc[-1]
        st.warning(f"**What to expect:** only about {recent.win_rate:.0%} of trades win. Losing trades average "
                   f"{recent.avg_loss:.1%} and winners {recent.avg_win:+.1%}. Runs of {int(recent.worst_losing_streak)} losses "
                   f"in a row have happened, and the account has fallen {recent.max_drawdown:.0%} from a peak. In 2008–2018 "
                   "the rule roughly broke even. **It will lose money on some trades, some months and possibly some "
                   "years.** No rule can avoid losses.")
    tr = P.trades.tail(15).iloc[::-1].copy()
    if not tr.empty:
        st.markdown("**Last 15 trades of the rule**")
        for col in ("entry", "exit"):
            tr[col] = tr[col].map(lambda v: money(px(v)))
        tr["return"] = tr["return"].map(lambda v: f"{v:+.2%}")
        tr["entry_date"] = pd.to_datetime(tr["entry_date"]).dt.date
        tr["exit_date"] = pd.to_datetime(tr["exit_date"]).dt.date
        st.dataframe(tr[["entry_date", "entry", "exit_date", "exit", "reason", "days", "return"]], hide_index=True,
                     width="stretch")
    st.caption("Levels are on COMEX gold futures. Enter your broker's price in the sidebar to convert them. The plan updates "
               "with each new daily close.")

# ------------------------------------------------------------------ ranges
with tabs[1]:
    hist = c.iloc[-130:]
    future = pd.bdate_range(c.index[-1], periods=22)[1:]
    cone = R.cone
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=px(hist), name="Gold", line=dict(color=GOLD_C, width=2)))
    for lo, hi, op, nm in (("q5", "q95", .12, "90% range"), ("q10", "q90", .2, "80% range"), ("q25", "q75", .3, "50% range")):
        fig.add_trace(go.Scatter(x=list(future) + list(future[::-1]), y=list(px(cone[hi])) + list(px(cone[lo])[::-1]),
                                 fill="toself", fillcolor=f"rgba(245,197,66,{op})", line=dict(width=0), name=nm))
    fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.04), yaxis_title="Price")
    st.plotly_chart(fig, width="stretch")
    t = R.bands.copy()
    t["Range"] = t.apply(lambda r: f"{money(px(r.low))} – {money(px(r.high))}", axis=1)
    t["Confidence"] = (t["level"] * 100).astype(int).astype(str) + "%"
    t["Measured on 2019–2026"] = t.apply(lambda r: f"{r.measured_coverage:.1%} inside ({r.tests:,} tests)", axis=1)
    st.dataframe(t[["horizon", "Confidence", "Range", "Measured on 2019–2026"]].rename(columns={"horizon": "Horizon"}),
                 hide_index=True, width="stretch")
    hl = R.highlow.copy()
    st.markdown("**Tomorrow's high and low: how far can it stretch?**")
    hl["Level"] = hl["level"].map(lambda v: money(px(v)))
    hl["Meaning"] = hl.apply(lambda r: f"{int(r.confidence * 100)}% chance tomorrow's {r.side} stays "
                                       f"{'below' if r.side == 'high' else 'above'} this", axis=1)
    hl["Measured"] = hl["measured"].map(lambda v: f"{v:.1%}")
    st.dataframe(hl[["side", "Level", "Meaning", "Measured"]].rename(columns={"side": "Side"}), hide_index=True,
                 width="stretch")
    st.caption("Use these as stop-loss and take-profit distances: a stop just beyond the 90% level is hit "
               "on roughly 1 day in 10 by normal noise alone.")

# ------------------------------------------------------------------ planner
with tabs[2]:
    st.markdown("Plan a trade's risk before you take it. This sizes the position from **your** stop; it doesn't tell "
                "you which way to trade.")
    p1, p2, p3 = st.columns(3)
    side = p1.radio("Side", ["Long (buy)", "Short (sell)"], horizontal=True)
    entry = p2.number_input("Entry price", value=float(round(px(last), 2)), step=1.0)
    stop_mode = p3.selectbox("Stop placement", ["Beyond tomorrow's 90% level", "Beyond tomorrow's 80% level",
                                                "1.5 × daily volatility", "Custom"])
    long_ = side.startswith("Long")
    hlv = R.highlow.set_index(["side", "confidence"])["level"]
    if stop_mode.startswith("Beyond"):
        L = 0.9 if "90" in stop_mode else 0.8
        lvl = px(hlv[("low" if long_ else "high", L)])
        stop = lvl - 0.1 * px(last) * R.sigma if long_ else lvl + 0.1 * px(last) * R.sigma
    elif stop_mode.startswith("1.5"):
        stop = entry * (1 - 1.5 * R.sigma) if long_ else entry * (1 + 1.5 * R.sigma)
    else:
        stop = st.number_input("Stop price", value=float(round(entry * (0.98 if long_ else 1.02), 2)))
    rr = st.slider("Target as a multiple of risk (reward : risk)", 1.0, 4.0, 2.0, 0.5)
    target = entry + rr * (entry - stop) if long_ else entry - rr * (stop - entry)
    size = M.position_size(account, risk_pct * min(size_mult, 1.0) if st.toggle(
        "Apply volatility sizing (shrink size when gold is unusually volatile)", value=True) else risk_pct,
        entry, stop, lot_oz)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Stop", money(stop), f"{(stop / entry - 1):+.2%}")
    q2.metric("Target", money(target), f"{(target / entry - 1):+.2%}")
    q3.metric("Position", f"{size['ounces']:,.2f} oz", f"= {size['lots']:.3f} lots · ${size['notional']:,.0f} exposure",
              delta_color="off")
    q4.metric("Loss if stopped", f"${size['risk_cash']:,.2f}", f"{size['risk_cash'] / account:.2%} of account",
              delta_color="off")
    if 0 < size["lots"] < 0.01:
        st.warning(f"That is {size['lots']:.3f} lots, below the usual broker minimum of 0.01 lots. "
                   "At the minimum lot your loss at the stop would be "
                   f"${0.01 * lot_oz * abs(entry - stop):,.2f}, more than your chosen risk. Use a smaller contract "
                   "size if your broker offers one, or skip the trade.")
    stop_dist = abs(np.log(stop / entry)) / R.sigma
    st.caption(f"The stop is {stop_dist:.1f} daily standard deviations away. To be profitable at {rr:.1f}:1 "
               f"reward to risk, you need to win more than {1 / (1 + rr):.0%} of trades, before costs and spread.")

# ------------------------------------------------------------------ direction
with tabs[3]:
    st.markdown("Each January, a regularised logistic model is retrained on all earlier data, using 23 gold drivers "
                "(momentum, trend, the dollar, yields, real yields, VIX, silver, copper, oil, miners, ETF volume and "
                "seasonality). Its 2019–2026 predictions are compared with the simplest baseline: always saying 'up'.")
    show = dirs.copy()
    show["Horizon"] = show["days"].map(lambda d: f"{d} trading days")
    show["Model accuracy"] = show["accuracy"].map(lambda v: f"{v:.1%}")
    show["'Always up' accuracy"] = show["always_up_accuracy"].map(lambda v: f"{v:.1%}")
    show["Model says now"] = show["prob_up"].map(lambda v: f"{v:.0%} up")
    show["Verdict"] = show["reliable"].map(lambda r: "✅ use" if r else "❌ ignore: no proven edge")
    st.dataframe(show[["Horizon", "Model accuracy", "'Always up' accuracy", "auc", "p_value", "tests", "Model says now",
                       "Verdict"]].rename(columns={"auc": "AUC", "p_value": "p-value", "tests": "Tests"}),
                 hide_index=True, width="stretch")
    st.info("We also tested gradient boosting, simple trend and momentum rules, at 1, 5, 21 and 63 days. The earlier "
            "hourly XAU/USD study in this project tested LSTM and Temporal Fusion Transformer networks. None beat the "
            "baseline on data they hadn't seen. Most '90% accurate gold signal' claims are tested on the same data "
            "they were built on. This page shows the forecasts that held up on unseen data.")

# ------------------------------------------------------------------ drivers
with tabs[4]:
    dr = D["drivers"].copy()
    dr["1-month change"] = dr.apply(lambda r: f"{r['1-month change']:+.2f} pts" if r["kind"] == "diff"
                                    else f"{r['1-month change']:+.1%}", axis=1)
    dr["pushing gold"] = dr["pushing gold"].map({"up": "▲ up", "down": "▼ down", "neutral": "• neutral"})
    st.dataframe(dr.drop(columns="kind"), hide_index=True, width="stretch",
                 column_config={"correlation with gold (1y)": st.column_config.ProgressColumn(min_value=-1, max_value=1,
                                                                                              format="%.2f")})
    st.caption("'Pushing gold' combines each driver's 1-month move with how it has moved alongside gold over the past "
               "year. These relationships explain what has happened. They do not forecast what happens next.")
    st.markdown("**More drivers: positioning, real yields, policy, credit, Asian demand**")
    with st.spinner("Loading CFTC, FRED and market data..."):
        try:
            ex = extra_drivers()
        except Exception:
            ex = pd.DataFrame()
    if ex.empty:
        st.caption("These sources are unavailable right now.")
    else:
        ex = ex.copy()
        ex["latest"] = ex.apply(lambda r: f"{r['latest']:.1%}" if r["unit"] == "% of open interest"
                                else f"{r['latest']:,.2f}", axis=1)
        ex["1-month change"] = ex.apply(
            lambda r: f"{r['1-month change'] * 100:+.1f} pts" if r["unit"] == "% of open interest"
            else f"{r['1-month change']:+.1%}" if r["unit"] in ("ratio", "INR", "CNY", "USD")
            else f"{r['1-month change']:+.2f}", axis=1)
        st.dataframe(ex.drop(columns="unit"), hide_index=True, width="stretch", column_config={
            "3-year percentile": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")})
        st.caption("Each of these drivers was added to the forecasting models and tested on 2019–2026. None improved "
                   "the forecasts (see 'What we tested'), so they appear here as context only.")

# ------------------------------------------------------------------ psychology
with tabs[5]:
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Fear & greed", f"{sent['greed_index']:.0f}/100")
    s2.metric("RSI(14)", f"{sent['rsi14']:.0f}", "oversold < 30, overbought > 70", delta_color="off")
    s3.metric("Gold VIX (GVZ)", f"{sent['gvz']:.1f}" if sent["gvz"] else "—",
              f"{sent['gvz_percentile']:.0%} percentile" if sent["gvz_percentile"] is not None else None, delta_color="off")
    s4.metric("GLD ETF volume", f"{sent['gld_volume_z']:+.1f} sd" if sent["gld_volume_z"] is not None else "—",
              "vs 3-month normal (investor demand)", delta_color="off")
    t1, t2, t3 = st.columns(3)
    t1.metric("Position in 52-week range", f"{sent['pos52']:.0%}")
    t2.metric("Up/down streak", f"{sent['streak_days']:+d} days")
    t3.metric("Volatility percentile", f"{sent['vol_percentile']:.0%}")
    st.caption("Fear & greed combines gold's momentum against its 6-month average, its place in the 52-week range, "
               "calmness and RSI. Extreme readings show crowd emotion. They have not been reliable timing signals "
               "on their own.")

# ------------------------------------------------------------------ strategies
with tabs[6]:
    how = st.radio("How do you hold gold?", ["CFD / spot XAU/USD (pays ~6%/yr overnight financing on longs)",
                                             "ETF or physical (no financing)"], horizontal=True)
    S, curves = D["strategies_cfd"] if how.startswith("CFD") else D["strategies"]
    st.markdown("Simple, fixed rules on gold, traded at the next day's close with costs, **2019–2026**:")
    st.dataframe(S, hide_index=True, width="stretch", column_config={
        c_: st.column_config.NumberColumn(format="percent") for c_ in
        ["annual_return", "max_drawdown", "worst_year", "time_invested", "position_now"]} |
        {"sharpe": st.column_config.NumberColumn(format="%.2f")})
    fig = go.Figure()
    for col in curves:
        fig.add_trace(go.Scatter(x=curves.index, y=curves[col], name=col,
                                 line=dict(width=2.5 if col == "Buy & hold" else 1.5)))
    fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="Growth of $1", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")
    st.caption("No rule earned more than buying and holding through gold's 2019–2026 bull run. Volatility sizing "
               "(holding less when gold swings hard) gave the best return per unit of risk and the smaller drawdown. "
               "Trend filters cut drawdowns at the cost of return. Holding through a CFD costs roughly 4–5 points of "
               "return a year in financing, so for holds longer than a few days an ETF is usually cheaper.")

# ------------------------------------------------------------------ seasonality
with tabs[7]:
    se = D["season"]
    fig = go.Figure(go.Bar(x=se.index, y=se["avg_return"], marker_color=[GREEN if v > 0 else RED for v in se["avg_return"]],
                           text=[f"{u:.0%} up" for u in se["up_share"]], textposition="outside"))
    fig.update_layout(template="plotly_dark", height=340, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_tickformat=".1%", yaxis_title="Average monthly return since 2000")
    st.plotly_chart(fig, width="stretch")
    st.caption("About 26 observations per month: too few to trade on alone. January has historically been strong, "
               "March and the autumn weaker.")

with tabs[8]:
    log = research_log()
    if not log:
        st.caption("Research log not found.")
    else:
        st.markdown(log["summary"])
        for sec in log["sections"]:
            st.markdown(f"**{sec['title']}**")
            if sec.get("text"):
                st.markdown(sec["text"])
            if sec.get("table"):
                st.dataframe(pd.DataFrame(sec["table"]), hide_index=True, width="stretch")
        st.caption(f"Reproduce with the scripts in research/gold (data as of {log['data_as_of']}).")

with tabs[9]:
    st.markdown(f"""
**Price ranges:**
- Volatility forecast = average of realised volatility (exponentially weighted, λ = 0.94) and gold's implied volatility (CBOE GVZ). This blend scored best on 2009–2018 data. It was chosen before any later data was examined.
- Gold's moves have fat tails, so the ranges use the actual distribution of past moves divided by volatility over a rolling 3 years, not a bell curve. Only outcomes already known at each date are used.
- Accuracy is the share of 2019–2026 outcomes that landed inside each range, counted on non-overlapping periods.

**Direction:** the yearly walk-forward test described in that tab. Labels are purged so training never sees the test period.

**Reference price:** COMEX gold futures (GC=F), daily, from Yahoo Finance (delayed). Spot XAU/USD differs by the futures basis. Enter your broker's price in the sidebar to rescale every level.

**Current one-day volatility:** {R.sigma:.2%} (annualised {vol_ann:.0%}).
""")
st.caption(DISCLAIMER)
