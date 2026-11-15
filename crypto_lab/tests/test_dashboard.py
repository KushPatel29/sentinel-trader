from pathlib import Path
import pytest


def test_dashboard_renders_evidence_without_execution():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1]
    if not (root / "evidence" / "summary.json").exists() and not (root / "reports").exists():
        pytest.skip("Evidence not available")
    app = AppTest.from_file(str(root / "dashboard.py"), default_timeout=30).run()
    assert not app.exception
    assert {"LSTM · TFT · RF","Gold · Oil · Crypto","Frozen crypto v1","Forward paper","Prediction quality v1","Audit trail v1","Research notes"} <= {tab.label for tab in app.tabs}
    assert any(m.label == "Selected strategy net" and m.value == "+0.00%" for m in app.metric)
    next(s for s in app.selectbox if s.label == "Evidence table").set_value("all-validation-trials").run()
    assert not app.exception


def test_focused_dashboard_labels_oil_data_failure_and_futures_separately():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    root = Path(__file__).resolve().parents[1]
    if not (root / "focus_evidence" / "summary.json").exists():
        pytest.skip("Focus evidence not available")
    app = AppTest.from_file(str(root / "focus_dashboard.py"), default_timeout=30).run()
    assert not app.exception
    next(s for s in app.selectbox if s.label == "Focus instrument").set_value("WTIUSD").run()
    assert not app.exception
    assert any("940-day gap" in e.value for e in app.error)
    next(s for s in app.selectbox if s.label == "Focus instrument").set_value("WTI_REFERENCE").run()
    assert not app.exception
    assert any("direction study only" in e.value for e in app.warning)
