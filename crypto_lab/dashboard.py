"""Read-only local monitor and portable historical research dashboard."""
from __future__ import annotations
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sentinel_crypto.config import ROOT, load_protocol

st.set_page_config(page_title="Sentinel | Gold, Oil & Crypto", page_icon="◈", layout="wide")
st.markdown("""<style>
.block-container{padding-top:3.8rem;max-width:1450px}h1{letter-spacing:-.04em}
[data-testid="stMetric"]{background:#111a2e;border:1px solid #24314b;border-radius:10px;padding:14px}
.eyebrow{letter-spacing:.15em;font-size:.75rem;color:#65c9d0;font-weight:600}
.subtle{color:#9ba9c2;font-size:.95rem;max-width:900px}
</style>""", unsafe_allow_html=True)
cfg = load_protocol()
report_dir = ROOT / "reports" / cfg["version"]
if not (report_dir / "summary.json").exists():
    report_dir = ROOT / "evidence"

st.markdown('<div class="eyebrow">SENTINEL / RESEARCH TERMINAL</div>', unsafe_allow_html=True)
st.title("Evidence before exposure.")
st.markdown('<p class="subtle">Gold, crude oil and crypto research with instrument-specific data checks, chronological model comparisons and separate future observations. No real-money execution.</p>', unsafe_allow_html=True)

if not (report_dir / "summary.json").exists():
    st.info("No research report yet. Run `python -m sentinel_crypto.cli fetch` then `python -m sentinel_crypto.cli research` from crypto_lab.")
    st.stop()
report = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
challengers, focus, hold, live, accuracy, audit, sources = st.tabs(["LSTM · TFT · RF", "Gold · Oil · Crypto", "Frozen crypto v1", "Forward paper", "Prediction quality v1", "Audit trail v1", "Research notes"])

with challengers:
    from sentinel_crypto.challenger_ui import render_challengers
    render_challengers()

with focus:
    from sentinel_crypto.focus_ui import render_focus
    render_focus()


def pct(x):
    return f"{x:+.2%}" if x is not None else "—"


def curve_chart(names):
    fig = go.Figure()
    colors = ["#67d5cf", "#617ca2", "#f0bb68", "#bb9aff"]
    for (key, label), color in zip(names, colors):
        file = report_dir / f"{key}-equity.csv"
        if file.exists():
            p = pd.read_csv(file, index_col=0, parse_dates=True)
            fig.add_trace(go.Scatter(x=p.index, y=p.equity * 10000, name=label,
                         line={"color": color, "width": 2}, hovertemplate="%{x}<br>$%{y:,.2f}<extra>%{fullData.name}</extra>"))
    fig.update_layout(height=390, margin={"l": 0, "r": 10, "t": 10, "b": 0},
        yaxis_title="Simulated value · $10,000 start", legend={"orientation": "h", "y": 1.12},
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


with hold:
    st.warning("RESEARCH ONLY · The selected portfolio stayed in cash. No profitable trading edge has been established.")
    st.caption(f"Frozen test: {report['holdout_start']} → {report['holdout_end']} · 180 days · public Binance spot data")
    a, b, c, d = st.columns(4)
    a.metric("Selected strategy net", pct(report["holdout"]["total_return"]))
    b.metric("Equal-weight buy & hold", pct(report["buy_hold"]["total_return"]))
    c.metric("Selected drawdown", f"{report['holdout']['max_drawdown']:.2%}")
    d.metric("Selected round trips", report["holdout"]["round_trips"])
    st.plotly_chart(curve_chart([("holdout", "Validation-selected policy"), ("buy_hold", "Buy & hold, full exposure"),
                                ("vol_matched", "Volatility-matched baseline"), ("stress", "Policy, doubled costs")]), width="stretch")
    left, right = st.columns([1, 1])
    with left:
        st.subheader("What was tested")
        st.write(f"{report['candidate_count_per_selection']} fixed candidates, {report['selection_events']} selection events and {report['candidate_evaluations']} validation evaluations. Each decision used earlier training and validation data.")
        st.write("10 trading rules plus cash; logistic regression, gradient boosting and extra trees at three probability thresholds each. Fees: 10 bps per side; slippage: 5 bps per side. These are assumptions, not verified account fees.")
    with right:
        st.subheader("Evidence requirements")
        checks = pd.DataFrame([{"Requirement": k.replace("_", " "), "Met": v} for k, v in report["checks"].items()])
        st.dataframe(checks, hide_index=True, width="stretch")
        st.caption("These diagnostics do not authorize real-money trading. Forward-history requirements remain unmet at inception.")

with live:
    st.subheader("New observations, separate ledger")
    st.write("The local worker polls public quotes every 60 seconds. Fresh four-hour candles trigger simulated decisions. Historical catch-up trades are skipped.")

    @st.fragment(run_every="30s")
    def render_live():
        status_path = ROOT / "runtime" / "status.json"
        db = ROOT / "runtime" / "paper.sqlite"
        if not status_path.exists():
            st.info("The local worker has not reported here. A hosted dashboard cannot see the SQLite database on your computer.")
            return
        status = json.loads(status_path.read_text())
        age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(status["heartbeat_utc"])).total_seconds()
        state = status["state"] if age < 180 else "heartbeat stale"
        st.caption(f"Worker: {state.upper()} · last heartbeat {status['heartbeat_utc']} · age {age:.0f}s")
        if status.get("errors"):
            st.error(str(status["errors"]))
        if not db.exists():
            return
        with sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True) as con:
            latest = pd.read_sql_query("""WITH latest AS (SELECT name,symbol,max(time) time FROM snapshots GROUP BY name,symbol)
                SELECT s.name,sum(s.equity) equity,sum(s.cash) cash FROM snapshots s
                JOIN latest l ON s.name=l.name AND s.symbol=l.symbol AND s.time=l.time GROUP BY s.name""", con)
            fills = pd.read_sql_query("SELECT time,name,symbol,reason,payload FROM fills ORDER BY time DESC LIMIT 200", con)
            forecasts = pd.read_sql_query("SELECT symbol,family,probability,actual_up,created,matured FROM predictions ORDER BY created DESC LIMIT 1000", con)
            snapshots = pd.read_sql_query("SELECT time,sum(equity) equity FROM snapshots WHERE name='selected_policy' GROUP BY time HAVING count(*)=? ORDER BY time", con, params=(len(cfg["symbols"]),))
        if not latest.empty:
            selected = latest[latest.name == "selected_policy"]
            current = float(selected.equity.iloc[0]) if not selected.empty else cfg["initial_cash"]
            a, b, c = st.columns(3)
            a.metric("Forward selected equity", f"${current:,.2f}", pct(current/cfg["initial_cash"]-1))
            b.metric("Logged simulated fills", len(fills), help="Latest 200 fills shown; shadow portfolios are independent experiments.")
            c.metric("Matured forecasts", int(forecasts.actual_up.notna().sum()) if not forecasts.empty else 0)
            latest["net_return"] = latest.equity/cfg["initial_cash"]-1
            st.dataframe(latest.sort_values("name"), hide_index=True, width="stretch")
        if len(snapshots) > 1:
            st.line_chart(snapshots.set_index("time")["equity"])
        st.caption("Shadow portfolios receive identical independent starting capital; their equity must not be added together. Paper fills include observed spread, assumed fees and slippage. Actual exchange fills may differ.")
        if not forecasts.empty:
            st.dataframe(forecasts, hide_index=True, width="stretch")
        if not fills.empty:
            st.dataframe(fills.drop(columns="payload"), hide_index=True, width="stretch")
        st.code("python -m sentinel_crypto.cli stop", language="bash")
        st.caption("Run the stop command from crypto_lab to request simulated liquidation. The worker requires an awake, online computer.")
    render_live()

with accuracy:
    st.subheader("Accuracy is a diagnostic, profit is a separate test")
    st.write("Direction forecasts cover the next 24 hours, entering at the next candle open. Samples are spaced 24 hours apart. Compare each model with the majority class and base rate learned before the test.")
    metrics = pd.DataFrame(report["prediction_metrics"])
    st.dataframe(metrics, hide_index=True, width="stretch")
    st.caption("AUC near 0.50 indicates weak ranking. Lower Brier score is better; a model should improve on its pre-test base-rate forecast. These diagnostic results never select the frozen policy.")
    chart = go.Figure()
    for family, group in metrics.groupby("family"):
        chart.add_trace(go.Bar(name=family, x=group.symbol, y=group.auc))
    chart.add_hline(y=.5, line_dash="dash", annotation_text="Chance ranking")
    chart.update_layout(barmode="group", yaxis={"range": [.4, .65], "title": "Holdout AUC"}, height=350)
    st.plotly_chart(chart, width="stretch")

with audit:
    st.subheader("Every attempt is visible")
    st.caption(f"Protocol SHA-256: {report['protocol_sha256']}")
    table = st.selectbox("Evidence table", ["fold-results", "all-validation-trials", "prediction-metrics"])
    file = report_dir / f"{table}.csv"
    if file.exists():
        frame = pd.read_csv(file)
        symbol = st.selectbox("Asset", ["All", *cfg["symbols"]])
        if symbol != "All" and "symbol" in frame:
            frame = frame[frame.symbol == symbol]
        st.dataframe(frame, hide_index=True, width="stretch")
        st.download_button("Download this evidence table", frame.to_csv(index=False), file_name=f"{table}.csv", mime="text/csv")
    st.download_button("Download sealed report", json.dumps(report, indent=2), file_name="sentinel-sealed-report.json", mime="application/json")
    st.json(report["data_quality"], expanded=False)
    st.write("Limitations")
    for item in report["limitations"]:
        st.write("• " + item)

with sources:
    st.subheader("Research and implementation decisions")
    doc = ROOT / "docs" / "RESEARCH.md"
    if doc.exists():
        st.markdown(doc.read_text(encoding="utf-8"))

st.caption("Simulated results do not predict future returns. This version sends no real orders and accepts no exchange credentials.")
