from pathlib import Path
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import pandas as pd
import streamlit as st

from sentinel_crypto.config import ROOT
from sentinel_crypto.public_live import quote_snapshot,recent_bars,load_release,inference

st.set_page_config(page_title="Sentinel · Gold, Oil & Crypto",page_icon="◈",layout="wide")
st.markdown("""<style>
.block-container{max-width:1400px;padding-top:3rem}h1{letter-spacing:-.045em}
.eyebrow{font-size:.75rem;letter-spacing:.15em;color:#68d0c9}
[data-testid="stMetric"]{border:1px solid #27334a;border-radius:10px;padding:12px;background:#111a2e}
</style>""",unsafe_allow_html=True)
st.markdown('<div class="eyebrow">SENTINEL / MARKET RESEARCH</div>',unsafe_allow_html=True)
st.title("Gold, oil & crypto.")
st.write("Live crypto market data, auditable model comparisons and clear commodity data readiness. All model outputs are research; this website sends no trading orders.")
markets,challengers,accuracy,commodities,operations=st.tabs(["Live crypto","LSTM · TFT · RF","Accuracy evidence","Gold & oil readiness","Paper testing & methods"])

with challengers:
    from sentinel_crypto.challenger_ui import render_challengers
    render_challengers()


@st.cache_data(ttl=30,show_spinner=False)
def quotes():
    return quote_snapshot()


@st.cache_data(ttl=600,show_spinner=False)
def history(symbol):
    return recent_bars(symbol)


@st.cache_resource(show_spinner=False)
def bundle(symbol):
    return load_release(symbol)


with markets:
    st.caption("Public Binance spot data · prices in USDT · quote refresh every 60 seconds while this page is open")

    @st.fragment(run_every="60s")
    def market_panel():
        try:
            data=quotes()
        except Exception as exc:
            st.warning(f"Live source unavailable: {exc}")
            st.caption("No cached or synthetic number is presented as a current quote. Historical evidence remains available.")
            return
        cols=st.columns(3)
        for col,row in zip(cols,data):
            col.metric(row["symbol"],f"{row['mid']:,.4f}" if row["mid"]<1000 else f"{row['mid']:,.2f}",help=f"Bid {row['bid']} · ask {row['ask']} · spread {row['spread_bps']:.2f} bps")
        st.caption("Observed UTC: "+data[0]["observed_utc"])
        symbol=st.selectbox("Research forecast",["BTCUSDT","ETHUSDT","SOLUSDT"],key="public_symbol")
        try:
            bars,_=history(symbol)
            now=pd.Timestamp(data[0]["observed_utc"])
            result=inference(symbol,bundle(symbol),bars,now)
            left,right=st.columns([1,2])
            left.metric("24h up probability from candle close",f"{result['up_probability']:.1%}")
            right.info(result["trade_status"])
            st.caption(f"Inputs end at {result['last_completed_candle_close']}; model fitted before {result['model_fit_before']}. The current incomplete candle is excluded.")
            st.line_chart(bars.close.tail(180),height=260)
            st.caption("A forecast from a completed candle is not an executable price or a guarantee. Probability quality, after-cost performance and future validation must be assessed separately.")
        except Exception as exc:
            st.info(f"Research forecast unavailable: {exc}")
    market_panel()

with accuracy:
    file=ROOT/"accuracy_evidence"/"summary.json"
    if file.exists():
        report=json.loads(file.read_text())
        st.caption("Matched-date comparisons against v2 · full forecast coverage · classification threshold 0.50")
        rows=[]
        for symbol,a in report["assets"].items():
            m=a["selected_retrospective"]
            rows.append({"Asset":symbol,"Development-selected profile":a["selected_profile"],"Samples":m["samples"],
                         "V3 accuracy":f"{m['accuracy']:.2%}","V2 accuracy":f"{m['baseline_accuracy']:.2%}","Change (pp)":round(m["accuracy_change_pp"],2),
                         "V3 Brier":round(m["brier"],5),"V2 Brier":round(m["baseline_brier"],5)})
        st.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch")
        st.warning("These dates were inspected in earlier research. Improvements here are retrospective and do not prove future accuracy, profitability or best-in-class performance.")
        asset=st.selectbox("Inspect accuracy uncertainty",list(report["assets"]),key="accuracy_asset")
        a=report["assets"][asset]
        with st.expander("Paired uncertainty, coverage and all candidate scores"):
            st.json(a)
        audit=ROOT/"accuracy_evidence"/"all-candidate-scores.csv"
        if audit.exists():
            table=pd.read_csv(audit)
            st.dataframe(table[table.symbol==asset],hide_index=True,width="stretch")
            st.download_button("Download all accuracy attempts",audit.read_bytes(),file_name="accuracy-candidate-scores.csv",mime="text/csv")
        st.download_button("Download accuracy report",file.read_bytes(),file_name="accuracy-report.json",mime="application/json")
    else:
        st.info("The versioned accuracy experiment is being finalized. No improved score is claimed until its report is available.")

with commodities:
    st.subheader("XAUUSD and USOIL-style CFDs")
    st.error("CFD live signals are disabled: a broker has not been selected and reference data is not execution-qualified.")
    st.dataframe(pd.DataFrame([
        {"Market":"Gold · XAUUSD","Reference":"Hourly bars from real bid-price minute quotes","Readiness":"Reference ends 6 November 2026; stale for current signals"},
        {"Market":"Crude · USOIL/WTI CFD","Reference":"WTIUSD reference history","Readiness":"940-day data gap; affected models blocked"},
        {"Market":"WTI daily reference","Reference":"CL=F continuous futures","Readiness":"Direction study only; no USOIL CFD P&L claim"}
    ]),hide_index=True,width="stretch")
    st.write("Broker-specific bid/ask history, sessions, contract sizes, fees, financing and oil-roll cash adjustments are needed for execution validation. No passwords or broker keys are needed to review this research.")
    st.page_link("crypto_lab/focus_dashboard.py",label="Open the complete commodity research and data audit")

with operations:
    st.write("The hosted app reads public prices and packaged research models. The continuous paper worker and challenger observer run on your computer; the hosted app does not claim access to their local SQLite ledgers.")
    st.write("Your computer must remain awake and online for those local processes. They preserve the original paper account and do not execute real-money trades.")
    st.code("python -m sentinel_crypto.cli status\npython -m sentinel_crypto.cli focus-status",language="bash")
    st.page_link("crypto_lab/dashboard.py",label="Open archived tests and the local paper-monitor view")
    st.caption("On the hosted site, local-worker panels show unavailable rather than inventing a cloud paper account.")

st.caption("Release research v3 · Public market data only · No real-order execution · Historical tests remain immutable")
