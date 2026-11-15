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
    assert len(app.tabs) == 5
    assert app.metric[0].value == "+0.00%"
    app.selectbox[0].set_value("all-validation-trials").run()
    assert not app.exception
