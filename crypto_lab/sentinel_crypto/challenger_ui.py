"""Read-only challenger evidence; the hosted UI never imports PyTorch."""

import json

import pandas as pd
import streamlit as st

from .config import ROOT


def render_challengers():
    st.subheader("LSTM, TFT, Random Forest & hybrid lab")
    st.write(
        "Standard LSTM and the PyTorch Forecasting TFT implementation are tested alongside Random Forest, LightGBM, Extra Trees, logistic regression and a base-rate forecast. The hybrid combines calibrated probabilities with validation-fitted weights and prior shrinkage."
    )
    cfg = json.loads((ROOT / "challenger_protocol.json").read_text())
    folder = ROOT / "reports" / cfg["version"]
    if not (folder / "summary.json").exists():
        folder = ROOT / "challenger_evidence"
    if (folder / "summary.json").exists():
        report = json.loads((folder / "summary.json").read_text())
        st.warning(
            "Previously inspected historical dates: retrospective research, not proof of future accuracy, profitability or best-in-class performance. All model results are retained. No real-money orders."
        )
        symbol = st.selectbox("Challenger instrument", list(report["assets"]), key="challenger_asset")
        asset = report["assets"][symbol]
        if asset["status"] == "DATA_BLOCKED":
            st.error("WTI CFD history fails the coverage check. No model result or live CFD forecast is fabricated.")
            st.json(asset["gate"])
        else:
            st.caption(
                f"Selected on earlier development Brier loss: {asset['selected']} · later periods are reported at full forecast coverage"
            )
            table = [
                {
                    "Model": name,
                    "Samples": m["samples"],
                    "Accuracy": f"{m['accuracy']:.2%}",
                    "Balanced accuracy": f"{m['balanced_accuracy']:.2%}",
                    "Brier": round(m["brier"], 5),
                    "Selected earlier": name == asset["selected"],
                }
                for name, m in asset["retrospective"].items()
            ]
            st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
            if symbol == "WTI_REFERENCE":
                st.info("CL=F daily direction study only. This is not USOIL CFD execution data or trading P&L.")
            with st.expander("Frozen hybrid weights, matched v3 comparison and uncertainty"):
                st.json(
                    {
                        "weights": asset["weights"],
                        "matched_v3": asset["matched_v3"],
                        "selected_metrics": asset["retrospective"][asset["selected"]],
                    }
                )
        st.download_button(
            "Download all challenger results",
            (folder / "summary.json").read_bytes(),
            file_name="challenger-results.json",
            mime="application/json",
        )
    else:
        st.info(
            "The versioned challenger experiment is running locally. No score is claimed until its evidence file is complete."
        )
    st.subheader("New real-data paper testing")
    st.caption(
        "Independent virtual books · no keys or order endpoint · real observed bid/ask · assumed 10 bps fee and 5 bps slippage per side · 10% maximum book exposure · long or cash"
    )

    @st.fragment(run_every="60s")
    def local_status():
        path = ROOT / "runtime" / "challengers" / "status.json"
        if not path.exists():
            st.info(
                "The paper worker runs on the owner's computer. This hosted page cannot read its local ledger; no live cloud paper result is invented."
            )
            return
        status = json.loads(path.read_text())
        age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(status["heartbeat_utc"])).total_seconds()
        st.caption(f"Local worker: {status['state']} · heartbeat {status['heartbeat_utc']} · age {age:.0f}s")
        if age > 180 or status["state"] == "degraded":
            st.warning("Worker heartbeat is stale or degraded. New labels are not assumed to exist.")
        st.json(status.get("assets", {}))
        st.caption(f"Recorded real quote observations: {status.get('quote_observations', 0)}")
        books = status.get("books", [])
        if books:
            st.dataframe(pd.DataFrame(books), hide_index=True, width="stretch")
            if sum(b["observed_outcomes"] for b in books) == 0:
                st.info("No new 24-hour outcome has matured yet. Paper accuracy is not available.")
        else:
            st.info(
                "Collecting current quotes; first scored forecasts require a fresh four-hour candle close. Outcomes take 24 hours to mature."
            )
        if status.get("startup_diagnostic_books"):
            with st.expander("Immediate quote-origin diagnostic books — separate timing"):
                st.warning(status["startup_diagnostic_note"])
                st.dataframe(pd.DataFrame(status["startup_diagnostic_books"]), hide_index=True, width="stretch")
        st.caption(
            "Missing the due-time observation by over five minutes expires its direction label. An open simulated trade closes at the actual later quote and records the delay. Restarts never backfill forecasts."
        )

    local_status()
