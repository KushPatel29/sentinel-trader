"""analyze(ticker): download -> factors -> strategy tournament + walk-forward ML -> evidence-weighted signal.

The signal is a model output with its evidence attached, not investment advice."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sentinel.analyzer import data as D
from sentinel.analyzer.factors import compute_factors
from sentinel.analyzer.ml import MLResult, walk_forward
from sentinel.analyzer.strategies import COST_BPS, SPECS, adaptive, perf, positions, strategy_returns
from sentinel.models.sentiment import LexiconSentiment

HORIZON = 5


@dataclass
class FactorReading:
    group: str
    name: str
    value: str
    reading: str        # "bullish" | "bearish" | "neutral"
    why: str


@dataclass
class Analysis:
    ticker: str
    name: str
    asset: str
    prices: pd.DataFrame
    factors: pd.DataFrame
    ml: MLResult | None
    strategy_table: pd.DataFrame
    equity: pd.DataFrame
    current_positions: dict[str, float]
    readings: list[FactorReading]
    news: list[dict]
    fundamentals: dict
    signal: dict
    notes: list[str] = field(default_factory=list)
    adaptive_choice: pd.Series | None = None


# ------------------------------------------------------------------------------------------ readings
def factor_readings(f: pd.DataFrame, asset: str) -> list[FactorReading]:
    x = f.iloc[-1]
    R: list[FactorReading] = []

    def add(group, name, value, reading, why):
        R.append(FactorReading(group, name, value, reading, why))

    def ok(k):
        return k in x and pd.notna(x[k])

    if ok("dist_sma200"):
        add("Trend", "Price vs 200-day average", f"{x.dist_sma200:+.1%}",
            "bullish" if x.dist_sma200 > 0 else "bearish", "Long-term trend direction")
    if ok("sma50_vs_200"):
        add("Trend", "50-day vs 200-day average", f"{x.sma50_vs_200:+.1%}",
            "bullish" if x.sma50_vs_200 > 0 else "bearish", "Golden cross (above) or death cross (below)")
    if ok("adx14"):
        s = abs(x.adx14)
        add("Trend", "Trend strength (ADX)", f"{s:.0f}", "neutral" if s < 20 else ("bullish" if x.adx14 > 0 else "bearish"),
            "Below 20 = no clear trend; above 25 = strong trend")
    if ok("mom_12_1"):
        add("Momentum", "12-1 month momentum", f"{x.mom_12_1:+.1%}", "bullish" if x.mom_12_1 > 0 else "bearish",
            "Assets that rose over the past year tend to keep rising for a while")
    if ok("ret_21"):
        add("Momentum", "1-month return", f"{x.ret_21:+.1%}", "bullish" if x.ret_21 > 0 else "bearish", "Short-term momentum")
    if ok("rsi14"):
        r = x.rsi14
        add("Mean reversion", "RSI(14)", f"{r:.0f}", "bearish" if r > 70 else ("bullish" if r < 30 else "neutral"),
            "Above 70 overbought, below 30 oversold")
    if ok("bb_pos"):
        add("Mean reversion", "Bollinger position", f"{x.bb_pos:+.2f}",
            "bearish" if x.bb_pos > 1 else ("bullish" if x.bb_pos < -1 else "neutral"), "Beyond ±1 = outside the bands")
    if ok("vol_pctile"):
        add("Volatility", "Volatility vs last year", f"{x.vol_pctile:.0%} pctile",
            "bearish" if x.vol_pctile > 0.85 else "neutral", "Very high volatility often comes with falling prices")
    if ok("obv_slope"):
        add("Demand & volume", "On-balance volume trend", f"{x.obv_slope:+.2f}",
            "bullish" if x.obv_slope > 0.05 else ("bearish" if x.obv_slope < -0.05 else "neutral"),
            "Is volume flowing in on up days (buying) or down days (selling)?")
    if ok("ad_slope"):
        add("Demand & volume", "Accumulation / distribution", f"{x.ad_slope:+.2f}",
            "bullish" if x.ad_slope > 0.05 else ("bearish" if x.ad_slope < -0.05 else "neutral"),
            "Closes near the day's high on volume = accumulation")
    if ok("mfi14"):
        add("Demand & volume", "Money flow index", f"{x.mfi14:.0f}",
            "bearish" if x.mfi14 > 80 else ("bullish" if x.mfi14 < 20 else "neutral"), "Volume-weighted RSI")
    if ok("volume_z"):
        add("Demand & volume", "Volume vs 20-day normal", f"{x.volume_z:+.1f} sd", "neutral",
            "Unusual volume marks news or institutional activity")
    if ok("greed_index"):
        g = x.greed_index
        label = "Extreme greed" if g > 80 else "Greed" if g > 60 else "Neutral" if g > 40 else "Fear" if g > 20 else "Extreme fear"
        add("Psychology & sentiment", "Fear & greed (asset)", f"{g:.0f} — {label}",
            "bearish" if g > 85 else ("bullish" if g < 15 else "neutral"),
            "Contrarian at extremes: crowds overreact at euphoric tops and panicked bottoms")
    if ok("pos_52w"):
        add("Psychology & sentiment", "Position in 52-week range", f"{x.pos_52w:.0%}", "neutral",
            "Anchoring: traders fixate on 52-week highs and lows")
    if ok("drawdown"):
        add("Psychology & sentiment", "Drawdown from all-time high", f"{x.drawdown:.1%}", "neutral",
            "Deep drawdowns mean many holders are at a loss, creating selling pressure on rallies")
    if ok("streak"):
        add("Psychology & sentiment", "Up/down day streak", f"{int(x.streak):+d} days", "neutral",
            "Long streaks signal herding")
    if ok("vix"):
        add("Psychology & sentiment", "VIX (market fear gauge)", f"{x.vix:.1f} ({x.vix_pctile:.0%} pctile)",
            "bearish" if x.vix_pctile > 0.85 and x.vix_chg5 > 0.15 else "neutral",
            "Rising fear compresses prices of risk assets")
    if ok("mkt_trend"):
        add("Macro & cross-asset", "US market (SPY) vs 200-day", f"{x.mkt_trend:+.1%}",
            "bullish" if x.mkt_trend > 0 else "bearish", "Most assets struggle when the broad market is in a downtrend")
    if ok("rel_strength_63") and asset in ("stock", "etf"):
        add("Macro & cross-asset", "3-month return vs market", f"{x.rel_strength_63:+.1%}",
            "bullish" if x.rel_strength_63 > 0 else "bearish", "Relative strength leaders tend to keep leading")
    if ok("rates_chg21"):
        add("Macro & cross-asset", "10-year yield, 1-month change", f"{x.rates_chg21:+.2f} pts",
            "bearish" if x.rates_chg21 > 0.25 else ("bullish" if x.rates_chg21 < -0.25 else "neutral"),
            "Rising yields pressure valuations, especially growth stocks and gold")
    if ok("dollar_ret21"):
        bad = asset in ("commodity", "crypto")
        add("Macro & cross-asset", "US dollar, 1-month change", f"{x.dollar_ret21:+.1%}",
            ("bearish" if x.dollar_ret21 > 0.02 else "bullish" if x.dollar_ret21 < -0.02 else "neutral") if bad else "neutral",
            "A stronger dollar usually weighs on commodities and crypto priced in dollars")
    if ok("beta63") and asset in ("stock", "etf", "crypto"):
        add("Macro & cross-asset", "Beta to US market", f"{x.beta63:.2f}", "neutral", "1.5 = moves 1.5x the market")
    return R


def _reading_score(rs: list[FactorReading]) -> float:
    vals = [1 if r.reading == "bullish" else -1 if r.reading == "bearish" else 0 for r in rs]
    n = sum(1 for v in vals if v != 0)
    return float(sum(vals) / n) if n else 0.0


def _news_score(headlines: list[dict]) -> tuple[float, list[dict]]:
    lex = LexiconSentiment()
    scored = [{**h, "sentiment": lex.score(h["title"] + ". " + h.get("summary", ""))} for h in headlines]
    s = [h["sentiment"] for h in scored]
    return (float(np.mean(s)) if s else 0.0), scored


def _analyst_score(info: dict) -> tuple[float | None, dict]:
    keys = ["recommendationKey", "recommendationMean", "numberOfAnalystOpinions", "targetMeanPrice",
            "currentPrice", "trailingPE", "forwardPE", "priceToBook", "pegRatio", "profitMargins", "revenueGrowth",
            "earningsGrowth", "debtToEquity", "returnOnEquity", "shortPercentOfFloat", "heldPercentInstitutions",
            "marketCap", "sector", "industry", "dividendYield", "fiftyTwoWeekChange"]
    fund = {k: info.get(k) for k in keys if info.get(k) is not None}
    rm, n = info.get("recommendationMean"), info.get("numberOfAnalystOpinions") or 0
    if rm is None or n < 3:
        return None, fund
    return float(np.clip((3 - rm) / 2, -1, 1)), fund      # 1 strong buy .. 5 sell -> +1 .. -1


# ------------------------------------------------------------------------------------------ main
def analyze(ticker: str, fetch=D.fetch_history, snapshot=D.fetch_snapshot, period: str = "10y") -> Analysis:
    ticker = ticker.strip().upper()
    snap = snapshot(ticker)
    asset = D.asset_class(ticker, snap.quote_type)
    px = fetch(ticker) if fetch is not D.fetch_history else fetch(ticker, period)
    if px is None or len(px) < 300:
        raise ValueError(f"Not enough daily history for {ticker} ({0 if px is None else len(px)} days; need 300+).")
    ctx = D.fetch_context(asset, fetch)
    f = compute_factors(px, ctx)
    notes: list[str] = []

    # ---- strategy tournament
    pos = positions(px, f)
    ml = None
    try:
        ml = walk_forward(f, HORIZON)
        pos["ml"] = 0.0
        pos.loc[ml.oos.index, "ml"] = (ml.oos["prob"] > 0.5).astype(float)
    except ValueError as e:
        notes.append(str(e) + " Machine-learning test skipped.")
    cost = COST_BPS.get(asset, 5)
    rets = strategy_returns(px["close"], pos, cost)
    start = ml.oos.index.min() if ml else px.index[252]
    window = rets.loc[start:]
    rule_cols = [c for c in window.columns]
    adapt = adaptive(rets[rule_cols]).loc[start:]
    window = window.assign(adaptive=adapt)
    specs = {s.key: s for s in SPECS}
    rows = []
    for k in window.columns:
        p = perf(window[k], 365 if asset == "crypto" else 252, pos[k].loc[start:] if k in pos else None)
        if not p:
            continue
        name = specs[k].name if k in specs else ("Machine learning (walk-forward)" if k == "ml" else "Adaptive: follow the leader")
        fam = specs[k].family if k in specs else ("ml" if k == "ml" else "meta")
        desc = specs[k].description if k in specs else (
            f"Long when the ML ensemble predicts a rise over the next {HORIZON} days." if k == "ml" else
            "Each month, switch to whichever strategy had the best Sharpe over the past year.")
        rows.append({"key": k, "strategy": name, "family": fam, "description": desc, **p})
    table = pd.DataFrame(rows).set_index("key")
    bh = table.loc["buy_hold"]
    table["excess_cagr"] = table["cagr"] - bh["cagr"]
    excess = window.sub(window["buy_hold"], axis=0)
    table["excess_t"] = [float(excess[k].mean() / excess[k].std() * np.sqrt(len(excess[k])))
                         if excess[k].std() > 0 else 0.0 for k in table.index]
    table = table.sort_values("sharpe", ascending=False)
    equity = (1 + window).cumprod()

    current = {k: float(pos[k].iloc[-1]) for k in pos.columns}
    readings = factor_readings(f, asset)
    news_s, news = _news_score(snap.headlines)
    analyst_s, fund = _analyst_score(snap.info)

    signal = combine(table, current, ml, readings, news_s, len(news), analyst_s)
    # Volatility clusters, so recent volatility is a reliable guide to the size of the next move
    # (unlike its direction). One standard deviation covers ~68% of outcomes.
    days = 365 if asset == "crypto" else 252
    sigma = float(np.log(px["close"]).diff().ewm(span=21).std().iloc[-1] * np.sqrt(HORIZON))
    signal["expected_move_1sd"] = sigma
    signal["vol_annual"] = sigma / np.sqrt(HORIZON) * np.sqrt(days)
    return Analysis(ticker=ticker, name=snap.name or ticker, asset=asset, prices=px, factors=f, ml=ml,
                    strategy_table=table, equity=equity, current_positions=current, readings=readings, news=news,
                    fundamentals=fund, signal=signal, notes=notes, adaptive_choice=adapt.attrs.get("choice"))


def combine(table: pd.DataFrame, current: dict, ml: MLResult | None, readings, news_s: float, n_news: int,
            analyst_s: float | None) -> dict:
    """Blend the evidence. Each component is in [-1, 1]; weights scale with how much that component has
    proven itself out-of-sample, so a model with no demonstrated edge gets no say."""
    comps = {}
    # 1) ML, weighted by its out-of-sample AUC and significance
    if ml is not None:
        m = ml.metrics
        w = float(np.clip((m["auc"] - 0.5) / 0.05, 0, 1)) * (1.0 if m["p_value"] < 0.05 else 0.5)
        comps["Machine learning"] = (float(np.clip((ml.current_prob - 0.5) * 4, -1, 1)), 0.35 * w,
                                     f"P(up in {m['horizon_days']}d) = {ml.current_prob:.0%}; OOS AUC {m['auc']:.3f}, "
                                     f"p = {m['p_value']:.3f}")
    # 2) Strategy votes, weighted by out-of-sample Sharpe (only strategies that made money)
    votes, wsum, voters = 0.0, 0.0, []
    for k, row in table.iterrows():
        if k in ("buy_hold", "adaptive", "ml") or row["sharpe"] <= 0:
            continue
        p = current.get(k, 0.0)
        fam = row["family"]
        vote = (1.0 if p > 0 else -1.0) if fam == "trend" else (1.0 if p > 0 else 0.0)
        votes += row["sharpe"] * vote
        wsum += row["sharpe"]
        voters.append(f"{row['strategy']}: {'LONG' if p > 0 else 'flat'}")
    if wsum > 0:
        comps["Strategy votes"] = (votes / wsum, 0.35, "; ".join(voters[:5]))
    # 3) Factor readings (trend, demand, psychology, macro)
    comps["Factor readings"] = (_reading_score(readings), 0.20,
                                f"{sum(r.reading == 'bullish' for r in readings)} bullish / "
                                f"{sum(r.reading == 'bearish' for r in readings)} bearish")
    # 4) News + analysts (current snapshot only — cannot be backtested, so kept small)
    if n_news:
        comps["News sentiment"] = (float(np.clip(news_s * 2, -1, 1)), 0.05, f"{n_news} recent headlines, avg {news_s:+.2f}")
    if analyst_s is not None:
        comps["Analyst consensus"] = (analyst_s, 0.05, "Wall Street rating average")

    total_w = sum(w for _, w, _ in comps.values())
    score = float(sum(s * w for s, w, _ in comps.values()) / total_w) if total_w else 0.0
    label = "Bullish" if score > 0.25 else "Bearish" if score < -0.25 else "Neutral"

    # Evidence quality: did anything beat buy & hold, or predict direction, reliably?
    best = table.drop(index=[k for k in ("buy_hold",) if k in table.index])
    strong_strat = bool(((best["excess_t"] > 2) & (best["sharpe"] > table.loc["buy_hold", "sharpe"])).any())
    ml_sig = ml is not None and ml.metrics["p_value"] < 0.01 and ml.metrics["auc"] >= 0.55
    ml_some = ml is not None and ml.metrics["p_value"] < 0.10 and ml.metrics["auc"] > 0.52
    some_strat = bool((best["excess_t"] > 1).any())
    if strong_strat or ml_sig:
        quality = "Strong"
    elif ml_some or some_strat:
        quality = "Moderate"
    else:
        quality = "Weak"
    return {"score": score, "label": label, "quality": quality, "horizon": f"next ~{HORIZON} trading days",
            "components": [{"component": k, "score": s, "weight": w, "detail": d} for k, (s, w, d) in comps.items()]}
