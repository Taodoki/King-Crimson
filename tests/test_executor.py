"""Tests for the live signal executor — cash budget, T+1, fail-closed."""

import pandas as pd
import pytest

from quantbt.trader.executor import LiveExecutor
from quantbt.trader.tonghuashun import OrderResult, TonghuashunTrader


class StubTrader:
    """Minimal trader stub that can simulate read/order failures."""

    def __init__(self, fail_read: bool = False, fail_order: bool = False):
        self.fail_read = fail_read
        self.fail_order = fail_order
        self.orders: list[tuple] = []

    def get_positions(self):
        if self.fail_read:
            raise RuntimeError("cannot read positions")
        return []

    def buy(self, symbol, price, amount):
        if self.fail_order:
            return OrderResult(success=False, message="order rejected")
        self.orders.append(("buy", symbol, amount, price))
        return OrderResult(success=True, message="ok")

    def sell(self, symbol, price, amount):
        if self.fail_order:
            return OrderResult(success=False, message="order rejected")
        self.orders.append(("sell", symbol, amount, price))
        return OrderResult(success=True, message="ok")


def make_executor(trader, min_volume: int = 100) -> LiveExecutor:
    return LiveExecutor(trader=trader, max_positions=5, min_volume=min_volume)


def test_module_loads_without_easytrader():
    # Importing the module must not raise ImportError (easytrader removed).
    from quantbt.trader import tonghuashun  # noqa: F401

    trader = TonghuashunTrader()
    assert trader.get_positions() == []
    res = trader.buy("000001", 10.0, 100)
    assert res.success is True


def test_buy_capped_by_available_cash():
    trader = StubTrader()
    ex = make_executor(trader)
    target = pd.Series({"000001": 1.5, "000002": 0.8})  # weights sum > 1
    prices = pd.Series({"000001": 10.0, "000002": 20.0})

    result = ex.execute(
        target_weights=target, prices=prices, cash=10_000,
        positions={}, dry_run=True,
    )

    total_cost = sum(b["quantity"] * b["price"] for b in result["buys"])
    assert total_cost <= 10_000
    # 000001 is the higher weight and consumes all cash; 000002 gets no cash.
    assert result["buys"][0]["symbol"] == "000001"
    assert result["buys"][0]["quantity"] == 1000
    assert any(e["reason"] == "insufficient cash" for e in result["errors"])


def test_t_plus_1_blocks_selling_bought_today():
    trader = StubTrader()
    ex = make_executor(trader)
    target = pd.Series({"000001": 0.0})
    prices = pd.Series({"000001": 10.0})

    result = ex.execute(
        target_weights=target, prices=prices, cash=0,
        positions={"000001": 1000},
        bought_today={"000001": 1000},
        dry_run=True,
    )

    assert result["sells"] == []


def test_t_plus_1_caps_partial_sell():
    trader = StubTrader()
    ex = make_executor(trader)
    target = pd.Series({"000001": 0.0})
    prices = pd.Series({"000001": 10.0})

    result = ex.execute(
        target_weights=target, prices=prices, cash=0,
        positions={"000001": 1000},
        bought_today={"000001": 600},
        dry_run=True,
    )

    assert result["sells"] == [{"symbol": "000001", "quantity": 400, "price": 10.0}]


def test_position_read_failure_aborts_not_empty():
    trader = StubTrader(fail_read=True)
    ex = make_executor(trader)
    target = pd.Series({"000001": 0.0})
    prices = pd.Series({"000001": 10.0})

    with pytest.raises(RuntimeError):
        ex.execute(
            target_weights=target, prices=prices, cash=1_000,
            positions=None, dry_run=False,
        )


def test_failed_order_not_counted_as_success():
    trader = StubTrader(fail_order=True)
    ex = make_executor(trader)
    target = pd.Series({"000001": 1.0})
    prices = pd.Series({"000001": 10.0})

    result = ex.execute(
        target_weights=target, prices=prices, cash=10_000,
        positions={}, dry_run=False,
    )

    assert result["buys"] == []
    assert any(e["side"] == "buy" for e in result["errors"])
