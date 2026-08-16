"""Tests for broker (A-share trading simulation)."""

import pandas as pd
import numpy as np


def test_broker_buy():
    from quantbt.core.broker import Broker

    broker = Broker(commission_pct=0.0003, stamp_duty_pct=0.0005)
    target = pd.Series({"000001": 1000})
    current = pd.Series(dtype=np.float64)
    prices = pd.Series({"000001": 10.0})
    trades = broker.execute(target, current, prices)

    assert len(trades) == 1
    t = trades[0]
    assert t.side == "buy"
    assert t.quantity == 1000
    assert t.commission > 0
    assert t.stamp_duty == 0  # no stamp duty on buys


def test_broker_sell_stamp_duty():
    from quantbt.core.broker import Broker

    broker = Broker()
    target = pd.Series({"000001": 0})
    current = pd.Series({"000001": 1000})
    prices = pd.Series({"000001": 10.0})
    trades = broker.execute(target, current, prices)

    assert len(trades) == 1
    t = trades[0]
    assert t.side == "sell"
    assert t.stamp_duty > 0  # stamp duty on sells


def test_broker_volume_limit():
    from quantbt.core.broker import Broker

    broker = Broker(volume_limit_pct=0.01)
    target = pd.Series({"000001": 100000})
    current = pd.Series(dtype=np.float64)
    prices = pd.Series({"000001": 10.0})
    volumes = pd.Series({"000001": 100000})  # 1% = 1000 shares
    trades = broker.execute(target, current, prices, volumes)

    assert len(trades) == 1
    assert trades[0].quantity <= 1000


def test_broker_min_volume():
    from quantbt.core.broker import Broker

    broker = Broker(min_volume=100)
    target = pd.Series({"000001": 50})
    current = pd.Series(dtype=np.float64)
    prices = pd.Series({"000001": 10.0})
    trades = broker.execute(target, current, prices)

    assert len(trades) == 0  # 50 < 100, can't trade


def test_broker_no_trade_if_no_diff():
    from quantbt.core.broker import Broker

    broker = Broker()
    target = pd.Series({"000001": 100})
    current = pd.Series({"000001": 100})
    prices = pd.Series({"000001": 10.0})
    trades = broker.execute(target, current, prices)
    assert len(trades) == 0


def test_broker_cost_calculation():
    from quantbt.core.broker import Broker

    broker = Broker(commission_pct=0.0003, stamp_duty_pct=0.0005)
    cost = broker.calculate_trade_cost(10.0, 1000, "sell")
    assert cost["commission"] == 5.0  # max(10000 * 0.0003, 5.0)
    assert cost["stamp_duty"] == 5.0  # 10000 * 0.0005
    assert cost["total_cost"] == 20.0  # 5 + 5 + 10 (slippage)


def test_broker_min_commission_floor():
    from quantbt.core.broker import Broker

    broker = Broker(commission_pct=0.0003)
    # 1000 * 0.0003 = 0.3 yuan, below the 5 yuan floor.
    cost = broker.calculate_trade_cost(10.0, 100, "buy")
    assert cost["commission"] == 5.0


def test_broker_price_limit_up_blocks_buy():
    from quantbt.core.broker import Broker

    broker = Broker()
    target = pd.Series({"000001": 1000})
    current = pd.Series(dtype=np.float64)
    prices = pd.Series({"000001": 11.0})  # close at limit-up
    prev_close = pd.Series({"000001": 10.0})  # limit_up = 11.0
    trades = broker.execute(target, current, prices, prev_close=prev_close)

    assert len(trades) == 0  # 涨停买不进


def test_broker_price_limit_down_blocks_sell():
    from quantbt.core.broker import Broker

    broker = Broker()
    target = pd.Series({"000001": 0})
    current = pd.Series({"000001": 1000})
    prices = pd.Series({"000001": 9.0})  # close at limit-down
    prev_close = pd.Series({"000001": 10.0})  # limit_down = 9.0
    trades = broker.execute(target, current, prices, prev_close=prev_close)

    assert len(trades) == 0  # 跌停卖不出


def test_broker_price_limit_allows_within_band():
    from quantbt.core.broker import Broker

    broker = Broker()
    target = pd.Series({"000001": 1000})
    current = pd.Series(dtype=np.float64)
    prices = pd.Series({"000001": 10.5})
    prev_close = pd.Series({"000001": 10.0})
    trades = broker.execute(target, current, prices, prev_close=prev_close)

    assert len(trades) == 1
    assert trades[0].side == "buy"


def test_broker_t_plus_1_blocks_same_day_sell():
    from quantbt.core.broker import Broker

    broker = Broker()
    target = pd.Series({"000001": 0})
    current = pd.Series({"000001": 1000})
    prices = pd.Series({"000001": 10.0})
    # Bought today: nothing is available to sell.
    available = pd.Series({"000001": 0})
    trades = broker.execute(
        target, current, prices, available_shares=available
    )

    assert len(trades) == 0  # T+1: can't sell shares bought today


def test_broker_t_plus_1_caps_partial_sell():
    from quantbt.core.broker import Broker

    broker = Broker()
    target = pd.Series({"000001": 0})
    current = pd.Series({"000001": 1000})
    prices = pd.Series({"000001": 10.0})
    available = pd.Series({"000001": 400})  # only 400 sellable today
    trades = broker.execute(
        target, current, prices, available_shares=available
    )

    assert len(trades) == 1
    assert trades[0].quantity == 400


def test_broker_short_selling_false_rejects_negative_target():
    from quantbt.core.broker import Broker

    broker = Broker()  # short_selling=False by default
    target = pd.Series({"000001": -1000})
    current = pd.Series(dtype=np.float64)  # no position
    prices = pd.Series({"000001": 10.0})
    trades = broker.execute(target, current, prices)

    assert len(trades) == 0  # can't sell shares we don't hold


def test_broker_short_selling_false_caps_sell_at_held():
    from quantbt.core.broker import Broker

    broker = Broker()
    target = pd.Series({"000001": -2000})  # wants net short
    current = pd.Series({"000001": 1000})  # holds 1000
    prices = pd.Series({"000001": 10.0})
    trades = broker.execute(target, current, prices)

    assert len(trades) == 1
    assert trades[0].quantity == 1000  # flatten to 0, never negative


def test_broker_short_selling_true_opens_short():
    from quantbt.core.broker import Broker

    broker = Broker(short_selling=True)
    target = pd.Series({"000001": -1000})
    current = pd.Series(dtype=np.float64)
    prices = pd.Series({"000001": 10.0})
    trades = broker.execute(target, current, prices)

    assert len(trades) == 1
    assert trades[0].side == "sell"
    assert trades[0].quantity == 1000
