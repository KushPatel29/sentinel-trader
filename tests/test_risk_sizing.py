from datetime import date

import pytest

from sentinel.config import Secrets
from sentinel.events import AccountState
from sentinel.live.runner import check_live_allowed
from sentinel.risk.officer import RiskOfficer
from sentinel.risk.sizing import kelly, position_size
from tests.conftest import ny

T = ny(2024, 3, 4, 11, 0)


def test_kelly():
    assert kelly(0.5, 2.0) == pytest.approx(0.25)
    assert kelly(0.2, 2.0) == 0.0


def test_volatility_targeting_scales_inversely_with_atr(cfg):
    cfg.risk.max_position_pct = 1.0
    a = position_size(100_000, 100, 1.0, 0.6, cfg)
    b = position_size(100_000, 100, 2.0, 0.6, cfg)
    assert a == 500                      # 0.5% of 100k = $500 risk / $1 stop
    assert b == 250
    # loss at the stop is the same fraction of equity either way
    assert a * 1.0 == pytest.approx(b * 2.0)


def test_sizing_caps(cfg):
    assert position_size(100_000, 100, 0.01, 0.6, cfg) == 200          # 20% notional cap
    assert position_size(100_000, 100, 1.0, 0.6, cfg, available_notional=5_000) == 50
    assert position_size(100_000, 100, 1.0, 0.2, cfg) == 0             # no Kelly edge -> no trade
    assert position_size(100_000, 100, float("nan"), 0.6, cfg) == 0


def _gate(r, **kw):
    args = dict(minutes_since_open=90, minutes_to_close=120, open_positions=0, gross_exposure=0,
                equity=100_000, new_notional=10_000, account=None)
    args.update(kw)
    return r.entry_check(**args)


def test_daily_drawdown_halts_until_next_session(cfg):
    r = RiskOfficer(cfg)
    r.new_session(date(2024, 3, 4), 100_000, T)
    assert r.on_equity(99_000, T) is False
    assert r.on_equity(97_900, T) is True              # -2.1%
    assert _gate(r)[0] is False
    r.new_session(date(2024, 3, 5), 97_900, T)
    assert _gate(r)[0] is True


def test_entry_gates(cfg):
    r = RiskOfficer(cfg)
    assert _gate(r)[0]
    assert not _gate(r, minutes_since_open=5)[0]
    assert not _gate(r, minutes_to_close=20)[0]
    assert not _gate(r, open_positions=cfg.risk.max_open_positions)[0]
    assert not _gate(r, gross_exposure=95_000)[0]
    cfg.risk.pdt_guard = True
    small = AccountState(equity=20_000, cash=20_000, buying_power=20_000, daytrade_count=3)
    assert "pattern-day-trader" in _gate(r, equity=20_000, new_notional=1_000, account=small)[1]


def test_latency_strikes_block_then_recover(cfg):
    r = RiskOfficer(cfg)
    for _ in range(cfg.risk.latency_strikes):
        r.on_latency(cfg.risk.max_api_latency_ms + 1, T)
    assert _gate(r)[1] == "broker latency"
    r.on_latency(50, T)
    assert _gate(r)[0]


def test_kill_switch(cfg, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = RiskOfficer(cfg)
    assert not r.kill_switch(T)
    (tmp_path / cfg.risk.kill_switch_file).touch()
    assert r.kill_switch(T)
    assert not _gate(r)[0]


def test_stale_bar(cfg):
    r = RiskOfficer(cfg)
    from datetime import timedelta
    assert not r.bar_is_stale(T, T + timedelta(seconds=10))
    assert r.bar_is_stale(T, T + timedelta(seconds=cfg.risk.max_bar_delay_s + 1))


def test_live_money_needs_both_confirmations(cfg):
    assert check_live_allowed(cfg, Secrets(), False) is True          # paper by default
    cfg.mode = "live"
    with pytest.raises(SystemExit):
        check_live_allowed(cfg, Secrets(live_confirm=True), False)
    with pytest.raises(SystemExit):
        check_live_allowed(cfg, Secrets(live_confirm=False), True)
    assert check_live_allowed(cfg, Secrets(live_confirm=True), True) is False
