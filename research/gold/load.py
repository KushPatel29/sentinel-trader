import json, os
import pandas as pd

# Folder holding the JSON files downloaded from Yahoo Finance / CFTC (see README.md)
DATA_DIR = os.environ.get("GOLD_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
RAW = json.load(open(os.path.join(DATA_DIR, "sentinel_gold_data_v2.json")))
def daily(sym):
    rows=[l.split(",") for l in RAW["daily"][sym].split("\n")]
    df=pd.DataFrame(rows,columns=["t","open","high","low","close","adj","volume"]).astype(float)
    df.index=pd.DatetimeIndex(pd.to_datetime(df.pop("t"),unit="s").dt.normalize()); df=df[~df.index.duplicated(keep="last")]
    f=(df.adj/df.close).fillna(1)
    for c in ("open","high","low"): df[c]*=f
    df["close"]=df.pop("adj"); return df
def hourly(sym):
    rows=[l.split(",") for l in RAW["hourly"][sym].split("\n")]
    df=pd.DataFrame(rows,columns=["t","open","high","low","close","volume"]).astype(float)
    df.index=pd.to_datetime(df.pop("t"),unit="s",utc=True); return df

EXTRA = json.load(open(os.path.join(DATA_DIR, "sentinel_gold_extra.json")))["daily"]
def daily_any(sym):
    if sym in RAW["daily"]: return daily(sym)
    rows=[l.split(",") for l in EXTRA[sym].split("\n")]
    df=pd.DataFrame(rows,columns=["t","open","high","low","close","adj","volume"]).astype(float)
    df.index=pd.DatetimeIndex(pd.to_datetime(df.pop("t"),unit="s").dt.normalize()); df=df[~df.index.duplicated(keep="last")]
    df["close"]=df.pop("adj"); return df
def cot():
    d=pd.DataFrame(json.load(open(os.path.join(DATA_DIR, "sentinel_cot_gold.json"))))
    d["date"]=pd.to_datetime(d.pop("report_date_as_yyyy_mm_dd"))
    for c_ in d.columns.drop("date"): d[c_]=d[c_].astype(float)
    d=d.set_index("date").sort_index()
    # Tuesday positions are published Friday 15:30 ET -> usable from the following Monday's close at the earliest safe point
    d.index = d.index + pd.Timedelta(days=6)
    return d
