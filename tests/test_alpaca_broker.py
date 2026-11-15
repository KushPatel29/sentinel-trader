"""AlpacaBroker against a fake TradingClient: checks the exact requests sent to Alpaca."""
from types import SimpleNamespace

from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce

from sentinel.execution.broker import AlpacaBroker


class FakeClient:
    def __init__(self):
        self.calls = []

    def submit_order(self, order_data):
        self.calls.append(("submit", order_data))
        return SimpleNamespace(id="abc", status=SimpleNamespace(value="accepted"))

    def get_orders(self, filter):
        self.calls.append(("get_orders", filter))
        return [SimpleNamespace(id="leg1"), SimpleNamespace(id="leg2")]

    def cancel_order_by_id(self, oid):
        self.calls.append(("cancel", oid))

    def close_position(self, sym):
        self.calls.append(("close", sym))


def _broker():
    b = AlpacaBroker.__new__(AlpacaBroker)
    b.client, b.paper, b.latency_cb = FakeClient(), True, None
    return b


async def test_bracket_order_request():
    b = _broker()
    lat = []
    b.latency_cb = lambda ms, what: lat.append(what)
    ack = await b.submit_bracket("AAPL", 12, 201.23456, 195.4321, "snt-AAPL-202406031000")
    req = b.client.calls[0][1]
    assert req.symbol == "AAPL" and req.qty == 12 and req.side == OrderSide.BUY
    assert req.order_class == OrderClass.BRACKET and req.time_in_force == TimeInForce.DAY
    assert req.take_profit.limit_price == 201.23 and req.stop_loss.stop_price == 195.43
    assert req.client_order_id == "snt-AAPL-202406031000"
    assert ack.status == "accepted" and ack.broker_id == "abc"
    assert lat == ["submit_order"]


async def test_close_cancels_bracket_legs_first():
    b = _broker()
    await b.close_position("AAPL", "test")
    kinds = [c[0] for c in b.client.calls]
    assert kinds == ["get_orders", "cancel", "cancel", "close"]


async def test_rejection_is_reported_not_raised():
    b = _broker()

    def boom(order_data):
        raise RuntimeError("insufficient buying power")
    b.client.submit_order = boom
    ack = await b.submit_bracket("AAPL", 1, 2, 1, "x")
    assert ack.status == "rejected" and "insufficient" in ack.extra["error"]
