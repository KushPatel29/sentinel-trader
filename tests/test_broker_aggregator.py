from datetime import timedelta

import pytest

from sentinel.execution.broker import SimBroker
from sentinel.live.aggregator import BarAggregator
from tests.conftest import bar, ny

T0 = ny(2024, 3, 4, 10, 0)
STEP = timedelta(minutes=15)


async def _open(cfg, close=100.0, tp=102.0, sl=99.0, qty=10):
    cfg.execution.slippage_bps = 0
    b = SimBroker(cfg, 10_000)
    b.process_bar(bar("X", T0, close, close, close, close))
    b.now = T0 + STEP
    ack = await b.submit_bracket("X", qty, tp, sl, "id-1")
    assert ack.status == "filled" and ack.filled_price == close
    return b


async def test_take_profit_fill_and_accounting(cfg):
    b = await _open(cfg)
    fills = b.process_bar(bar("X", T0 + STEP, 100, 102.5, 99.5, 101))
    assert fills[0].reason == "take profit" and fills[0].price == 102.0
    assert b.cash == pytest.approx(10_000 + 20)
    assert not b.positions


async def test_stop_wins_when_both_legs_touched(cfg):
    b = await _open(cfg)
    fills = b.process_bar(bar("X", T0 + STEP, 100, 103, 98, 101))
    assert fills[0].reason == "stop" and fills[0].price == 99.0


async def test_gap_through_stop_fills_at_open(cfg):
    b = await _open(cfg)
    fills = b.process_bar(bar("X", T0 + STEP, 97, 97.5, 96, 97))
    assert fills[0].reason == "stop (gap)" and fills[0].price == 97.0
    assert fills[0].pnl == pytest.approx(-30)


async def test_slippage_and_duplicate_order_ids(cfg):
    cfg.execution.slippage_bps = 10
    cfg.execution.commission_per_share = 0.01
    b = SimBroker(cfg, 10_000)
    b.process_bar(bar("X", T0, 100, 100, 100, 100))
    b.now = T0 + STEP
    ack = await b.submit_bracket("X", 10, 102, 99, "same")
    assert ack.filled_price == pytest.approx(100.1)
    assert (await b.submit_bracket("X", 10, 102, 99, "same")).status == "rejected_duplicate"
    assert b.cash == pytest.approx(10_000 - 1001 - 0.1)


def test_aggregator_builds_15m_bars():
    agg = BarAggregator(15)
    out = []
    for i in range(15):
        out += agg.add(bar("X", T0 + timedelta(minutes=i), 100 + i, 101 + i, 99 + i, 100.5 + i, minutes=1))
    assert len(out) == 1
    b = out[0]
    assert b.ts == T0 and b.minutes == 15
    assert (b.open, b.high, b.low, b.close, b.volume) == (100, 115, 99, 114.5, 15_000)
    assert agg.add(bar("X", T0 + timedelta(minutes=3), 1, 1, 1, 1, minutes=1)) == []   # late minute ignored


def test_aggregator_flushes_when_minutes_are_missing():
    agg = BarAggregator(15)
    agg.add(bar("X", T0, 100, 101, 99, 100, minutes=1))
    assert agg.flush_due(T0 + timedelta(minutes=10)) == []
    out = agg.flush_due(T0 + timedelta(minutes=15, seconds=30))
    assert len(out) == 1 and out[0].ts == T0


def test_aggregator_ignores_extended_hours():
    agg = BarAggregator(15)
    assert agg.add(bar("X", ny(2024, 3, 4, 8, 0), 1, 1, 1, 1, minutes=1)) == []
    assert not agg.partials
