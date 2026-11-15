from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel_crypto.accuracy import adaptive_prior,guarded_ensemble,load_accuracy
from sentinel_crypto.config import atomic_json
from sentinel_crypto.public_live import get_json,inference,load_release


def test_adaptive_base_rate_uses_actual_outcome_availability():
    cfg=load_accuracy()
    idx=pd.date_range("2026-01-01",periods=200,freq="4h",tz="UTC")
    y=pd.Series(np.where(np.arange(200)%2,.01,-.01),index=idx)
    end=pd.Series(idx+pd.Timedelta(hours=28),index=idx)
    before=adaptive_prior(y,end,idx,cfg)
    changed=y.copy();changed.loc[end>=idx[100]]=1
    after=adaptive_prior(changed,end,idx,cfg)
    pd.testing.assert_series_equal(before.iloc[:100],after.iloc[:100])
    assert before.iloc[0]==.5


def test_guarded_model_cannot_use_future_prediction_losses():
    cfg=load_accuracy()
    idx=pd.date_range("2024-01-01",periods=600,freq="1D",tz="UTC")
    y=pd.Series(np.where(np.arange(600)%3,.01,-.01),index=idx)
    ends=pd.Series(idx+pd.Timedelta(days=1),index=idx)
    p=pd.DataFrame({"adaptive_prior":.5,"ensemble":.65,"blend_25":.5375,"blend_50":.575,"static_prior":.51},index=idx)
    before,_=guarded_ensemble(p,y,ends,cfg)
    changed=y.copy();changed.iloc[400:]*=-1
    p2=p.copy();p2.loc[idx[400]:,"ensemble"] = .01
    after,_=guarded_ensemble(p2,changed,ends,cfg)
    np.testing.assert_array_equal(before.iloc[:400],after.iloc[:400])


def test_public_client_refuses_order_or_account_endpoints():
    with pytest.raises(ValueError,match="public market-data"):
        get_json("/api/v3/order")


def test_quote_query_uses_compact_json_and_returns_supported_assets(monkeypatch):
    import sentinel_crypto.public_live as p
    def reply(path,params=None):
        if path.endswith("time"):
            return {"serverTime":1794787200000}
        assert " " not in params["symbols"]
        return [{"symbol":s,"bidPrice":"100","askPrice":"100.01"} for s in p.SYMBOLS]
    monkeypatch.setattr(p,"get_json",reply)
    assert [q["symbol"] for q in p.quote_snapshot()]==list(p.SYMBOLS)


def test_model_integrity_is_verified_before_deserialization(tmp_path):
    folder=tmp_path/"release_models";folder.mkdir()
    (folder/"BTCUSDT.joblib").write_bytes(b"this is not a model")
    atomic_json(folder/"manifest.json",{"assets":{"BTCUSDT":{"sha256":"invalid"}}})
    with pytest.raises(ValueError,match="integrity"):
        load_release("BTCUSDT",tmp_path)


def test_live_cfd_model_loading_is_disabled():
    with pytest.raises(ValueError,match="CFD live inference"):
        load_release("XAUUSD")


def test_stale_model_or_candles_do_not_generate_forecast():
    now=pd.Timestamp("2026-11-16T10:00:00Z")
    bars=pd.DataFrame({"close":[100.]},index=[pd.Timestamp("2026-11-16T04:00:00Z")])
    bundle={"fit_before":"2026-01-01T00:00:00Z"}
    with pytest.raises(ValueError,match="45 days old"):
        inference("BTCUSDT",bundle,bars,now)
    old=bars.copy();old.index=[pd.Timestamp("2026-11-01T00:00:00Z")]
    with pytest.raises(ValueError,match="candles are stale"):
        inference("BTCUSDT",{"fit_before":"2026-11-14T00:00:00Z"},old,now)


def test_public_artifacts_are_bundled_and_all_remain_research_only():
    folder=Path(__file__).resolve().parents[1]/"release_models"
    if not (folder/"manifest.json").exists():
        pytest.skip("Release models not generated")
    for symbol in ("BTCUSDT","ETHUSDT","SOLUSDT"):
        model=load_release(symbol)
        assert model["orders_enabled"] is False
        assert model["spec"]["source"]=="binance"


def test_public_entrypoint_renders_without_live_network(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("SENTINEL_DISABLE_LIVE","1")
    root=Path(__file__).resolve().parents[2]
    app=AppTest.from_file(str(root/"streamlit_app.py"),default_timeout=30).run()
    assert not app.exception
    assert any(t.label=="Live crypto" for t in app.tabs)
    assert any("Live source unavailable" in w.value for w in app.warning)
