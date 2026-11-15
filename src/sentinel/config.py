"""Typed configuration loaded from YAML, with secrets from the environment."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class DataCfg(BaseModel):
    feed: Literal["iex", "sip"] = "iex"
    history_days: int = 730
    db_path: str = "data/sentinel.db"


class LabelCfg(BaseModel):
    horizon_bars: int = 8
    tp_atr: float = 2.0
    sl_atr: float = 1.0

    @property
    def payoff_ratio(self) -> float:
        return self.tp_atr / self.sl_atr


class ModelCfg(BaseModel):
    path: str = "models/model_bundle.pkl"
    min_train_bars: int = 3000
    test_bars: int = 750
    embargo_bars: int = 8
    entry_threshold: float = 0.42
    min_edge: float = 0.05
    lgbm: dict = Field(default_factory=dict)


class RegimeCfg(BaseModel):
    n_states: int = 3
    block_high_vol: bool = True


class LLMCfg(BaseModel):
    enabled: bool = False
    model: str = "claude-haiku-5-5"
    ambiguity_band: float = 0.3
    max_calls_per_hour: int = 30
    timeout_s: float = 8.0


class SentimentCfg(BaseModel):
    engine: Literal["auto", "finbert", "lexicon"] = "auto"
    decay_minutes: float = 240
    veto_threshold: float = -0.5
    boost_threshold: float = 0.5
    boost_multiplier: float = 1.25
    llm: LLMCfg = Field(default_factory=LLMCfg)


class RiskCfg(BaseModel):
    risk_per_trade: float = 0.005
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.20
    max_gross_exposure: float = 1.0
    max_open_positions: int = 5
    daily_drawdown_halt: float = 0.02
    max_api_latency_ms: float = 2500
    latency_strikes: int = 3
    max_bar_delay_s: float = 120
    no_entry_minutes_after_open: int = 15
    no_entry_minutes_before_close: int = 30
    flatten_minutes_before_close: int = 10
    pdt_guard: bool = False   # legacy FINRA PDT limit; rule eliminated June 2026, brokers migrating
    kill_switch_file: str = "KILL"


class ExecutionCfg(BaseModel):
    slippage_bps: float = 2.0
    commission_per_share: float = 0.0
    flatten_on_shutdown: bool = False


class BacktestCfg(BaseModel):
    initial_capital: float = 100_000
    reports_dir: str = "reports"


class Config(BaseModel):
    mode: Literal["paper", "live"] = "paper"
    symbols: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "SPY"])
    timeframe_minutes: int = 15
    data: DataCfg = Field(default_factory=DataCfg)
    labels: LabelCfg = Field(default_factory=LabelCfg)
    model: ModelCfg = Field(default_factory=ModelCfg)
    regime: RegimeCfg = Field(default_factory=RegimeCfg)
    sentiment: SentimentCfg = Field(default_factory=SentimentCfg)
    risk: RiskCfg = Field(default_factory=RiskCfg)
    execution: ExecutionCfg = Field(default_factory=ExecutionCfg)
    backtest: BacktestCfg = Field(default_factory=BacktestCfg)

    @field_validator("timeframe_minutes")
    @classmethod
    def _tf_divides_30(cls, v: int) -> int:
        # Alpaca's N-minute bars are clock-aligned; requiring N | 30 keeps them aligned to the 9:30 open,
        # so historical bars and live-aggregated bars have identical boundaries.
        if v <= 0 or 30 % v != 0:
            raise ValueError("timeframe_minutes must divide 30 evenly (1, 2, 3, 5, 6, 10, 15 or 30)")
        return v

    @field_validator("symbols")
    @classmethod
    def _upper(cls, v: list[str]) -> list[str]:
        return [s.upper().strip() for s in v]


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no extra dependency). Existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def load_config(path: str | Path = "config/config.yaml", dotenv: str | Path = ".env") -> Config:
    _load_dotenv(Path(dotenv))
    p = Path(path)
    raw = yaml.safe_load(p.read_text()) if p.exists() else {}
    return Config(**(raw or {}))


class Secrets(BaseModel):
    alpaca_key: str | None = None
    alpaca_secret: str | None = None
    anthropic_key: str | None = None
    live_confirm: bool = False

    @classmethod
    def from_env(cls) -> "Secrets":
        return cls(
            alpaca_key=os.getenv("ALPACA_API_KEY") or None,
            alpaca_secret=os.getenv("ALPACA_SECRET_KEY") or None,
            anthropic_key=os.getenv("ANTHROPIC_API_KEY") or None,
            live_confirm=os.getenv("SENTINEL_LIVE_CONFIRM", "no").lower() == "yes",
        )

    def require_alpaca(self) -> tuple[str, str]:
        if not (self.alpaca_key and self.alpaca_secret):
            raise SystemExit("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set (copy .env.example to .env)")
        return self.alpaca_key, self.alpaca_secret
