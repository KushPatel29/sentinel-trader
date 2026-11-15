"""News sentiment: a fast local scorer, optional LLM escalation, and a time-decayed per-symbol state.

Stage 1 (every headline, milliseconds): FinBERT if `transformers` is installed, else a finance lexicon.
Stage 2 (rare, optional): ambiguous headlines that mention high-impact topics go to Claude for a
second opinion. Failures or rate limits fall back silently to the stage-1 score."""
from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import defaultdict, deque
from datetime import datetime

from sentinel.config import SentimentCfg
from sentinel.events import NewsItem, SentimentScore

log = logging.getLogger(__name__)

POSITIVE = {
    "beat": 1.0, "beats": 1.0, "tops": 0.8, "surge": 0.9, "surges": 0.9, "soar": 0.9, "soars": 0.9,
    "jump": 0.6, "jumps": 0.6, "rally": 0.6, "rallies": 0.6, "upgrade": 0.9, "upgraded": 0.9,
    "outperform": 0.7, "record": 0.6, "raises": 0.7, "raised": 0.6, "buyback": 0.6, "repurchase": 0.5,
    "wins": 0.6, "win": 0.5, "approval": 0.8, "approved": 0.8, "strong": 0.5, "growth": 0.4,
    "profit": 0.4, "partnership": 0.4, "expands": 0.4, "dividend": 0.3, "bullish": 0.7,
}
NEGATIVE = {
    "miss": 1.0, "misses": 1.0, "missed": 1.0, "plunge": 1.0, "plunges": 1.0, "tumble": 0.9,
    "tumbles": 0.9, "sink": 0.8, "sinks": 0.8, "slump": 0.8, "drop": 0.5, "drops": 0.5, "falls": 0.5,
    "downgrade": 0.9, "downgraded": 0.9, "underperform": 0.7, "cut": 0.7, "cuts": 0.7, "lowers": 0.7,
    "lawsuit": 0.8, "sued": 0.8, "probe": 0.8, "investigation": 0.8, "subpoena": 0.9, "fraud": 1.0,
    "recall": 0.8, "bankruptcy": 1.0, "default": 0.9, "layoffs": 0.6, "weak": 0.6, "warning": 0.7,
    "halt": 0.7, "halted": 0.7, "delay": 0.5, "delays": 0.5, "resigns": 0.6, "bearish": 0.7, "loss": 0.5,
}
NEGATORS = {"not", "no", "never", "without", "fails", "failed"}
HIGH_IMPACT = re.compile(
    r"\b(earnings|guidance|outlook|fda|merger|acquisition|acquire|buyout|sec|doj|ftc|antitrust|"
    r"bankruptcy|recall|lawsuit|downgrade|upgrade|ceo|tariff|investigation|halt)\b", re.I)
TOKEN = re.compile(r"[a-z']+")


class LexiconSentiment:
    name = "lexicon"

    def score(self, text: str) -> float:
        toks = TOKEN.findall(text.lower())
        pos = neg = 0.0
        for i, t in enumerate(toks):
            negated = any(w in NEGATORS for w in toks[max(0, i - 2):i])
            if t in POSITIVE:
                if negated:
                    neg += POSITIVE[t]
                else:
                    pos += POSITIVE[t]
            elif t in NEGATIVE:
                if negated:
                    pos += NEGATIVE[t] * 0.5   # "not a fraud" is weakly positive
                else:
                    neg += NEGATIVE[t]
        total = pos + neg
        if total == 0:
            return 0.0
        polarity = (pos - neg) / total            # direction, -1..1
        strength = 1 - math.exp(-total)           # more/stronger cue words -> more confident
        return float(max(-1.0, min(1.0, polarity * strength)))

    def score_batch(self, texts: list[str]) -> list[float]:
        return [self.score(t) for t in texts]


class FinBertSentiment:
    name = "finbert"

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        from transformers import pipeline  # optional dependency

        self.pipe = pipeline("text-classification", model=model_name, top_k=None, truncation=True)

    def score_batch(self, texts: list[str]) -> list[float]:
        out = []
        for res in self.pipe(texts, batch_size=16):
            probs = {d["label"].lower(): d["score"] for d in res}
            out.append(float(probs.get("positive", 0.0) - probs.get("negative", 0.0)))
        return out

    def score(self, text: str) -> float:
        return self.score_batch([text])[0]


def make_scorer(engine: str):
    if engine in ("auto", "finbert"):
        try:
            return FinBertSentiment()
        except Exception as e:  # transformers/torch not installed or model download blocked
            if engine == "finbert":
                raise
            log.info("FinBERT unavailable (%s); using lexicon sentiment", type(e).__name__)
    return LexiconSentiment()


class LLMEscalator:
    """Second-opinion scoring with Claude for ambiguous, high-impact headlines."""

    PROMPT = (
        "You score how a news headline is likely to move the stock price of {symbols} over the next few "
        "hours. Consider second-order effects (e.g. 'beats estimates but cuts guidance' is negative). "
        "Reply with JSON only: {{\"score\": <number from -1 to 1>, \"rationale\": \"<12 words max>\"}}.\n\n"
        "Headline: {headline}\nSummary: {summary}"
    )

    def __init__(self, cfg, api_key: str):
        import anthropic  # optional dependency

        self.cfg = cfg
        self.client = anthropic.Anthropic(api_key=api_key, timeout=cfg.timeout_s)
        self.calls: deque[float] = deque()

    def should_escalate(self, item: NewsItem, base: float) -> bool:
        return abs(base) < self.cfg.ambiguity_band and bool(HIGH_IMPACT.search(item.headline + " " + item.summary))

    def _rate_ok(self) -> bool:
        now = time.time()
        while self.calls and now - self.calls[0] > 3600:
            self.calls.popleft()
        return len(self.calls) < self.cfg.max_calls_per_hour

    def score(self, item: NewsItem) -> tuple[float, str] | None:
        if not self._rate_ok():
            return None
        self.calls.append(time.time())
        try:
            msg = self.client.messages.create(
                model=self.cfg.model, max_tokens=120,
                messages=[{"role": "user", "content": self.PROMPT.format(
                    symbols=", ".join(item.symbols) or "the company", headline=item.headline,
                    summary=item.summary[:600])}],
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            data = json.loads(text[text.find("{"): text.rfind("}") + 1])
            return float(max(-1.0, min(1.0, data["score"]))), str(data.get("rationale", ""))[:200]
        except Exception as e:
            log.warning("LLM escalation failed: %s", e)
            return None


class SentimentAnalyzer:
    def __init__(self, cfg: SentimentCfg, anthropic_key: str | None = None):
        self.cfg = cfg
        self.scorer = make_scorer(cfg.engine)
        self.llm = None
        if cfg.llm.enabled and anthropic_key:
            try:
                self.llm = LLMEscalator(cfg.llm, anthropic_key)
            except ImportError:
                log.warning("sentiment.llm.enabled but `anthropic` is not installed; skipping escalation")

    @property
    def engine(self) -> str:
        return self.scorer.name

    def score(self, item: NewsItem) -> SentimentScore:
        base = self.scorer.score(item.headline + ". " + item.summary)
        if self.llm and self.llm.should_escalate(item, base):
            res = self.llm.score(item)
            if res is not None:
                return SentimentScore(item.id, res[0], f"{self.engine}+llm", res[1])
        return SentimentScore(item.id, base, self.engine)

    def score_many(self, items: list[NewsItem]) -> list[SentimentScore]:
        """Batch path for history (no LLM: escalation is a live-trading feature)."""
        texts = [i.headline + ". " + i.summary for i in items]
        return [SentimentScore(i.id, s, self.engine) for i, s in zip(items, self.scorer.score_batch(texts))]


class SentimentState:
    """Per-symbol time-decayed sentiment.

    score(t) = sum_i s_i * w_i / max(1, sum_i w_i), with w_i = exp(-(t - t_i) / tau).
    One fresh strong headline counts in full; it fades over ~tau; several headlines average.
    Maintained recursively so it is O(1) per update and identical in backtest and live."""

    def __init__(self, decay_minutes: float):
        self.tau = decay_minutes * 60.0
        self._s: dict[str, float] = defaultdict(float)
        self._w: dict[str, float] = defaultdict(float)
        self._t: dict[str, datetime] = {}
        self.count: dict[str, int] = defaultdict(int)

    def _decay_to(self, sym: str, ts: datetime) -> None:
        last = self._t.get(sym)
        if last is not None and ts > last:
            f = math.exp(-(ts - last).total_seconds() / self.tau)
            self._s[sym] *= f
            self._w[sym] *= f
        if last is None or ts > last:
            self._t[sym] = ts

    def add(self, symbols: tuple[str, ...], ts: datetime, score: float) -> None:
        for sym in symbols:
            self._decay_to(sym, ts)
            self._s[sym] += score
            self._w[sym] += 1.0
            self.count[sym] += 1

    def score(self, sym: str, ts: datetime) -> float:
        last = self._t.get(sym)
        if last is None:
            return 0.0
        f = math.exp(-max(0.0, (ts - last).total_seconds()) / self.tau)
        s, w = self._s[sym] * f, self._w[sym] * f
        return s / max(1.0, w)

    def intensity(self, sym: str, ts: datetime) -> float:
        """Decayed headline count: how much news flow there is right now."""
        last = self._t.get(sym)
        if last is None:
            return 0.0
        return self._w[sym] * math.exp(-max(0.0, (ts - last).total_seconds()) / self.tau)
