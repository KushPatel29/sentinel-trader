
import numpy as np
import pandas as pd

from sentinel_crypto.config import atomic_json
from sentinel_crypto.focus_data import load_focus
from sentinel_crypto.focus_watch import database, observe


class FixedModel:
    threshold = .5

    def predict(self, frame):
        return np.full(len(frame), .6)


def test_challenger_forecasts_are_prospective_and_restart_safe(tmp_path, monkeypatch):
    import sentinel_crypto.focus_watch as w
    cfg = load_focus()
    cfg["assets"] = {"BTCUSDT": cfg["assets"]["BTCUSDT"]}
    atomic_json(tmp_path/"runtime/focus/manifest.json", {"assets":{"BTCUSDT":{"status":"OBSERVATION_ONLY"}}})
    now = pd.Timestamp.now(tz="UTC")
    idx = pd.date_range(end=now.floor("4h")-pd.Timedelta(hours=4), periods=2000, freq="4h", tz="UTC")
    px = pd.DataFrame({"open":100.,"high":101.,"low":99.,"close":100.,"volume":100.,"taker_base":50.},index=idx)
    quote = {"mid":100.,"latency_s":.1,"spread_bps":1.}
    state = {"now":now}
    monkeypatch.setattr(w,"server_now",lambda:state["now"])
    monkeypatch.setattr(w,"read_context",lambda *args:{})
    monkeypatch.setattr(w,"rest_bars",lambda *args:pd.DataFrame())
    monkeypatch.setattr(w,"live_snapshot",lambda *args:quote)
    policies={"BTCUSDT":{"model":FixedModel(),"name":"test"}}
    con=database(tmp_path)
    first=observe(con,{"BTCUSDT":px},policies,cfg,tmp_path)
    assert not first["orders_enabled"]
    row=con.execute("SELECT observed,due,entry_price,actual_up FROM forecasts").fetchone()
    assert pd.Timestamp(row[1])>pd.Timestamp(row[0])
    assert row[2]==100 and row[3] is None
    observe(con,{"BTCUSDT":px},policies,cfg,tmp_path)
    assert con.execute("SELECT count(*) FROM forecasts").fetchone()[0]==1
    state["now"]=pd.Timestamp(row[1])+pd.Timedelta(seconds=60)
    fresh=px.copy();fresh.index=fresh.index+pd.Timedelta(days=1)
    quote["mid"]=110.
    observe(con,{"BTCUSDT":fresh},policies,cfg,tmp_path)
    matured=con.execute("SELECT actual_up,label_state FROM forecasts").fetchone()
    assert matured==(1,"observed")
    con.close()
    con=database(tmp_path)
    observe(con,{"BTCUSDT":px},policies,cfg,tmp_path)
    assert con.execute("SELECT count(*) FROM forecasts").fetchone()[0]==1
    con.close()


def test_stale_commodity_never_gets_current_signal(tmp_path,monkeypatch):
    import sentinel_crypto.focus_watch as w
    cfg=load_focus();cfg["assets"]={"XAUUSD":cfg["assets"]["XAUUSD"]}
    atomic_json(tmp_path/"runtime/focus/manifest.json",{"assets":{"XAUUSD":{"status":"OBSERVATION_ONLY"}}})
    now=pd.Timestamp.now(tz="UTC")
    px=pd.DataFrame({"open":[100.],"high":[101.],"low":[99.],"close":[100.]},index=[now-pd.Timedelta(days=10)])
    monkeypatch.setattr(w,"server_now",lambda:now)
    monkeypatch.setattr(w,"read_context",lambda *args:{})
    con=database(tmp_path)
    status=observe(con,{"XAUUSD":px},{"XAUUSD":{"model":FixedModel(),"name":"test"}},cfg,tmp_path)
    assert status["assets"]["XAUUSD"]["status"]=="STALE_REFERENCE_NO_SIGNAL"
    assert con.execute("SELECT count(*) FROM forecasts").fetchone()[0]==0
    con.close()


def test_outage_does_not_score_a_late_quote_as_a_24_hour_outcome(tmp_path,monkeypatch):
    import sentinel_crypto.focus_watch as w
    cfg=load_focus();cfg["assets"]={"BTCUSDT":cfg["assets"]["BTCUSDT"]}
    atomic_json(tmp_path/"runtime/focus/manifest.json",{"assets":{"BTCUSDT":{"status":"OBSERVATION_ONLY"}}})
    now=pd.Timestamp.now(tz="UTC")
    idx=pd.date_range(end=now.floor("4h")-pd.Timedelta(hours=4),periods=1000,freq="4h",tz="UTC")
    px=pd.DataFrame({"open":100.,"high":101.,"low":99.,"close":100.,"volume":100.,"taker_base":50.},index=idx)
    monkeypatch.setattr(w,"server_now",lambda:now)
    monkeypatch.setattr(w,"read_context",lambda *args:{})
    monkeypatch.setattr(w,"rest_bars",lambda *args:pd.DataFrame())
    monkeypatch.setattr(w,"live_snapshot",lambda *args:{"mid":110.,"latency_s":.1,"spread_bps":1.})
    con=database(tmp_path)
    con.execute("INSERT INTO forecasts(symbol,bar,observed,due,model,probability,threshold,entry_price) VALUES(?,?,?,?,?,?,?,?)",
                ("BTCUSDT","old",str(now-pd.Timedelta(days=2)),str(now-pd.Timedelta(days=1)),"test",.6,.5,100.))
    con.commit()
    observe(con,{"BTCUSDT":px},{"BTCUSDT":{"model":FixedModel(),"name":"test"}},cfg,tmp_path)
    assert con.execute("SELECT actual_up,label_state FROM forecasts WHERE bar='old'").fetchone()==(None,"expired_outage")
    con.close()
