"""Focused instrument dashboard. Results and data readiness are deliberately separate."""
from __future__ import annotations
import json
import sqlite3

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .config import ROOT
from .focus_data import load_focus


def render_focus():
    cfg = load_focus()
    folder = ROOT / "reports" / cfg["version"]
    if not (folder / "summary.json").exists():
        folder = ROOT / "focus_evidence"
    st.subheader("Gold · Crude oil · Crypto")
    st.caption("XAUUSD and USOIL-style CFDs are the commodity targets. Broker not selected. Historical study and broker execution readiness are separate.")
    if not (folder / "summary.json").exists():
        st.info("The focused research run is being prepared. The original frozen crypto evidence and paper ledger remain available in the other tabs.")
        return
    if st.button("Refresh focus results", key="focus_refresh"):
        st.rerun()
    summary = json.loads((folder / "summary.json").read_text())
    symbols = list(summary["assets"])
    symbol = st.selectbox("Focus instrument", symbols, format_func=lambda s: f"{s} · {summary['assets'][s]['name']}", key="focus_symbol")
    asset = summary["assets"][symbol]
    if asset["status"] == "DATA_BLOCKED":
        st.error("USOIL/WTI CFD study blocked: the reference history has a 940-day gap. No current CFD model or executable recommendation is presented.")
        st.write("The WTI_REFERENCE panel is a separate daily futures direction study. It cannot establish USOIL CFD returns.")
    elif asset["status"] == "DIRECTION_ONLY":
        st.warning("Daily CL=F futures-reference direction study only. Contract-roll changes and CFD broker cash adjustments are unverified, so trading P&L is not reported.")
    else:
        st.info("Research only. A broker-specific replay and future observations are required before execution claims.")
    st.caption(asset["test_interpretation"])
    quality = asset["final_predictions"]
    if quality.get("samples", 0):
        a, b, c, d = st.columns(4)
        a.metric("Non-overlapping forecasts", quality["samples"])
        b.metric("Direction accuracy", f"{quality['accuracy']:.1%}", f"{100*(quality['accuracy']-quality['prior_accuracy']):+.1f} pp vs prior")
        c.metric("Probability skill vs prior", f"{quality['brier_skill_vs_prior']:+.1%}", help="Brier skill: positive improves on the pre-test base-rate forecast; negative is worse.")
        d.metric("Ranking AUC", f"{quality['auc']:.3f}" if quality.get("auc") is not None else "—")
        compare = [{"Predictor": "Validation-selected model", "Accuracy": quality["accuracy"], "Brier": quality["brier"]},
                   {"Predictor": "Pre-test prior", "Accuracy": quality["prior_accuracy"], "Brier": quality["prior_brier"]},
                   {"Predictor": "Price-only validation champion", "Accuracy": quality["price_only_accuracy"], "Brier": quality["price_only_brier"]}]
        if "v1_ensemble_accuracy" in quality:
            compare.append({"Predictor": "V1-family ensemble, same test dates", "Accuracy": quality["v1_ensemble_accuracy"], "Brier": quality["v1_ensemble_brier"]})
        st.dataframe(pd.DataFrame(compare), hide_index=True, width="stretch")
        st.caption("Accuracy thresholds are selected on pre-test calibration. Model selection minimizes validation Brier score. Overlapping forecasts are not counted as independent observations.")
        st.subheader("Probability reliability")
        calibration = pd.DataFrame(quality["calibration"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Perfect calibration", mode="lines", line={"dash":"dash", "color":"#617ca2"}))
        fig.add_trace(go.Scatter(x=calibration.predicted, y=calibration.actual_up, name="Observed", mode="markers+lines",
                                marker={"size":calibration.samples.clip(upper=200).pow(.5)*2,"color":"#67d5cf"},
                                text=calibration.samples, hovertemplate="Forecast %{x:.1%}<br>Observed %{y:.1%}<br>n=%{text}<extra></extra>"))
        fig.update_layout(height=310, xaxis_title="Predicted up probability", yaxis_title="Observed up frequency", margin={"t":10,"b":0})
        st.plotly_chart(fig, width="stretch")
        with st.expander("Uncertainty and coverage"):
            st.json({k:v for k,v in quality.items() if "ci" in k or "confident" in k})
            st.caption("Descriptive 14-day block bootstraps; no multiple-comparison-adjusted superiority claim. High-confidence accuracy must be considered with its coverage.")
    if asset["final_trading"]:
        st.subheader("Reference trading simulation")
        m = asset["final_trading"]
        a, b, c = st.columns(3)
        a.metric("Policy net return", f"{m['metrics']['total_return']:+.2%}")
        b.metric("Net at 4× assumed costs", f"{m['stress_4x']['total_return']:+.2%}")
        c.metric("Maximum drawdown", f"{m['metrics']['max_drawdown']:.2%}")
        fig = go.Figure()
        for suffix, label, color in [("equity","Validation-selected policy","#67d5cf"),
                                     ("baseline-equity","Session-matched baseline" if asset['spec']['intraday_only'] else "Vol-matched spot baseline","#f0bb68"),
                                     ("stress-equity","Policy, 4× costs","#bb9aff")]:
            path = folder / f"{symbol}-final-{suffix}.csv"
            if path.exists():
                e = pd.read_csv(path, index_col=0, parse_dates=True)
                fig.add_trace(go.Scatter(x=e.index,y=e.equity*10000,name=label,line={"color":color}))
        fig.update_layout(height=330,yaxis_title="Simulated value · $10,000 start",margin={"t":10,"b":0})
        st.plotly_chart(fig,width="stretch")
        st.caption(f"Selected policy: {m['policy']}. Commodity positions are session-limited; CFD spreads are assumptions, not a broker quote. A cash policy is not proof of an edge.")
    st.subheader("Source and execution readiness")
    st.dataframe(pd.DataFrame([{"Asset":s,"State":a["status"],"Forecast samples":a["final_predictions"].get("samples",0),
                               "Broker execution validated":False} for s,a in summary["assets"].items()]), hide_index=True,width="stretch")
    with st.expander("Source coverage and blocked windows"):
        st.json({"data":asset["data"],"stale_days_at_cutoff":asset["stale_days_at_cutoff"],"blocked_windows":asset["blocked_windows"]})
    qualification = ROOT / "focus_evidence" / "qualification.json"
    if qualification.exists():
        q = json.loads(qualification.read_text())["assets"][symbol]
        with st.expander("Model qualification checks"):
            st.dataframe(pd.DataFrame([{"Requirement":k.replace("_"," "),"Met":v} for k,v in q.items()]),hide_index=True,width="stretch")
    st.subheader("Prospective challenger observation")
    status = ROOT / "runtime" / "focus" / "status.json"
    if status.exists():
        state = json.loads(status.read_text())
        st.caption(f"Last observation: {state['heartbeat_utc']} · forecasts only; original paper portfolios remain separate")
        st.json(state.get("assets", {}), expanded=False)
        db = ROOT / "runtime" / "focus" / "forecasts.sqlite"
        if db.exists():
            with sqlite3.connect(f"file:{db.as_posix()}?mode=ro",uri=True) as con:
                forecasts = pd.read_sql_query("SELECT symbol,observed,due,model,probability,actual_up,label_state FROM forecasts ORDER BY observed DESC LIMIT 50",con)
            st.dataframe(forecasts,hide_index=True,width="stretch")
        st.caption("Prospective labels use newly observed mid quotes. Training uses bar-open targets; that timing difference is disclosed. Commodity observation requires a fresh broker feed.")
    else:
        st.info("The prospective challenger observer has not started in this environment.")
    st.subheader("Inspect the full experiment")
    evidence = st.selectbox("Focus evidence table", ["fold-results","all-model-validation","all-policy-validation"], key="focus_evidence")
    file = folder / f"{evidence}.csv"
    if file.exists():
        rows = pd.read_csv(file)
        st.dataframe(rows[rows.symbol == symbol],hide_index=True,width="stretch")
        st.download_button("Download focus audit table",rows.to_csv(index=False),file_name=f"{evidence}.csv",mime="text/csv",key="focus_download")
    st.download_button("Download focused research report",json.dumps(summary,indent=2),file_name="focus-research-report.json",mime="application/json")
    st.caption("Best-in-class accuracy and profitability are not established. Larger models and more trials do not by themselves establish an edge.")
