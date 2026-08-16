"""Regression tests for engine position-sizing and valuation fixes.

Each test pins one acceptance criterion from the engine rewrite:
    1. Position sizing uses total equity (cash + holdings value), not holdings value alone.
    2. Re-entering after a clear allocates the full equity.
    3. The equity curve is daily-frequency, not rebalance-frequency.
    4. Volume-constrained fills become the actual position for the next rebalance.
"""

import pandas as pd
import numpy as np

from quantbt.strategy.base import Strategy
from quantbt.core.engine import BacktestEngine


class ConstantWeight(Strategy):
    """Always target a fixed weight per symbol."""

    def __init__(self, weights: dict[str, float], name: str = "const"):
        super().__init__(name)
        self.weights = weights

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        prices = self._get_prices(data)
        sig = pd.DataFrame(0.0, index=data.index, columns=prices.columns)
        for sym, w in self.weights.items():
            if sym in sig.columns:
                sig[sym] = w
        return sig


class ScheduledWeight(Strategy):
    """Target a per-date weight map (0 elsewhere)."""

    def __init__(self, schedule: dict, name: str = "sched"):
        super().__init__(name)
        self.schedule = schedule

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        prices = self._get_prices(data)
        sig = pd.DataFrame(0.0, index=data.index, columns=prices.columns)
        for dt, wmap in self.schedule.items():
            if dt in sig.index:
                for sym, w in wmap.items():
                    if sym in sig.columns:
                        sig.loc[dt, sym] = w
        return sig


def make_single_stock_data(prices: pd.Series, volume: float = 1_000_000) -> pd.DataFrame:
    """MultiIndex OHLCV for a single ticker, constant volume."""
    ticker = "000001"
    fields = ["open", "high", "low", "close", "volume", "adj_close"]
    columns = pd.MultiIndex.from_product([[ticker], fields])
    vals = np.zeros((len(prices), len(fields)))
    p = prices.values
    vals[:, 0] = p
    vals[:, 1] = p
    vals[:, 2] = p
    vals[:, 3] = p
    vals[:, 4] = volume
    vals[:, 5] = p
    return pd.DataFrame(vals, index=prices.index, columns=columns)


def _engine(strategy, data, **kwargs):
    defaults = dict(
        initial_capital=1_000_000,
        commission=0.0,
        stamp_duty=0.0,
        slippage=0.0,
        rebalance_freq="daily",
    )
    defaults.update(kwargs)
    return BacktestEngine(strategy=strategy, data=data, **defaults)


def test_fifty_percent_position_doubles_to_1_5m():
    dates = pd.bdate_range("2021-01-01", periods=60)
    prices = pd.Series(100.0, index=dates)
    prices.iloc[30:] = 200.0

    engine = _engine(
        ConstantWeight({"000001": 0.5}),
        make_single_stock_data(prices),
    )
    engine.run()

    total = engine.portfolio.total_value
    holdings = engine.portfolio.holdings_series().get("000001", 0)
    cash = engine.portfolio.cash

    # 50% in a stock that doubled: 500k cash + 1M stock = 1.5M
    assert abs(total - 1_500_000) / 1_500_000 < 0.01
    # Position stays ~50% of total equity (not shrunk to 25%)
    assert abs(holdings * 200 - cash) < 50_000


def test_clear_then_reenter_uses_full_equity():
    dates = pd.bdate_range("2021-01-01", periods=30)
    prices = pd.Series(100.0, index=dates)
    prices.iloc[5:] = 200.0  # double, then hold

    schedule = {}
    for d in dates[:5]:
        schedule[d] = {"000001": 1.0}  # long at 100
    for d in dates[5:7]:
        schedule[d] = {"000001": 0.0}  # clear at 200 (2M cash)
    for d in dates[7:]:
        schedule[d] = {"000001": 1.0}  # re-enter with full equity

    engine = _engine(
        ScheduledWeight(schedule),
        make_single_stock_data(prices),
    )
    engine.run()

    holdings = engine.portfolio.holdings_series().get("000001", 0)
    # Full 2M re-invested at 200 -> ~10000 shares, not the 5000 from the fallback bug
    assert holdings >= 9500


def test_equity_curve_is_daily_frequency():
    dates = pd.bdate_range("2021-01-01", periods=504)
    prices = pd.Series(100.0, index=dates)

    engine = _engine(
        ConstantWeight({"000001": 0.5}),
        make_single_stock_data(prices),
        rebalance_freq="monthly",
    )
    result = engine.run()

    # 504 trading days -> 504 valuation points, not ~24 monthly points
    assert len(result.equity_curve) == 504


def test_volume_limit_uses_actual_holdings():
    # Two months -> two monthly rebalances.
    dates = pd.bdate_range("2021-01-01", "2021-02-28")
    prices = pd.Series(100.0, index=dates)

    engine = _engine(
        ConstantWeight({"000001": 0.5}),
        make_single_stock_data(prices, volume=100_000),  # 1% = 1000 shares max/trade
        rebalance_freq="monthly",
        volume_limit=0.01,
    )
    engine.run()

    holdings = engine.portfolio.holdings_series().get("000001", 0)
    # Two rebalances, each capped at 1000 shares -> 2000 actual shares.
    # The shadow-position bug would report 5000 (target) or stall at 1000.
    assert holdings == 2000


def test_short_selling_false_rejects_short_signal():
    dates = pd.bdate_range("2021-01-01", periods=30)
    prices = pd.Series(100.0, index=dates)

    engine = _engine(
        ConstantWeight({"000001": -1.0}),  # full short signal
        make_single_stock_data(prices),
    )
    engine.run()

    holdings = engine.portfolio.holdings_series()
    assert holdings.get("000001", 0) == 0  # never negative, never held
    assert abs(engine.portfolio.total_value - 1_000_000) < 1


def test_short_selling_true_tracks_negative_holdings():
    dates = pd.bdate_range("2021-01-01", periods=30)
    prices = pd.Series(100.0, index=dates)

    engine = _engine(
        ConstantWeight({"000001": -1.0}),
        make_single_stock_data(prices),
        short_selling=True,
    )
    engine.run()

    holdings = engine.portfolio.holdings_series()
    assert holdings.get("000001", 0) == -10000  # explicit short book
    # Short proceeds inflate cash; the negative book cancels it out.
    assert abs(engine.portfolio.total_value - 1_000_000) < 1
