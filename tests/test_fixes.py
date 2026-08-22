"""Regression tests for the re-review fix round.

Covers: cost-reserved full deployment (no negative cash), momentum
tie-break for max_positions, composite event_driven propagation and
monthly grid freezing, J-T skip_days/holding, executor rotation funding,
and the qfq/hfq return-identity invariant (synthetic, no network).
"""

import numpy as np
import pandas as pd
import pytest

from quantbt.core.engine import BacktestEngine
from quantbt.strategy.base import Strategy
from quantbt.strategy.momentum import TimeSeriesMomentum, CrossSectionalMomentum
from quantbt.strategy.mean_reversion import ZScoreMeanReversion
from quantbt.strategy.composite import CompositeStrategy
from quantbt.trader.executor import LiveExecutor
from quantbt.trader.tonghuashun import OrderResult


class FixedWeights(Strategy):
    """Signal a fixed weight per symbol on every trading day."""

    def __init__(self, weights: dict[str, float], name: str | None = None):
        super().__init__(name)
        self.weights = dict(weights)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        prices = self._get_prices(data)
        sig = pd.DataFrame(0.0, index=data.index, columns=prices.columns)
        for sym, w in self.weights.items():
            if sym in sig.columns:
                sig[sym] = w
        return sig


def _make_data(tickers, n=504, seed=42, volume_scale=5_000_000):
    """Synthetic MultiIndex OHLCV data (mirrors conftest conventions)."""
    np.random.seed(seed)
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    fields = ["open", "high", "low", "close", "volume", "adj_close"]
    columns = pd.MultiIndex.from_product([tickers, fields])
    vals = np.random.randn(n, len(tickers) * len(fields)).cumsum(0)
    for i, t in enumerate(tickers):
        base = 100.0 + i * 3.0
        vals[:, i * 6 + 0] += base
        vals[:, i * 6 + 1] += base + 1
        vals[:, i * 6 + 2] += base - 1
        vals[:, i * 6 + 3] += base
        vals[:, i * 6 + 4] = np.abs(vals[:, i * 6 + 4] * 100000 + volume_scale)
        vals[:, i * 6 + 5] = vals[:, i * 6 + 3]
    return pd.DataFrame(vals, index=dates, columns=columns)


def test_full_deployment_never_negative_cash():
    """Full-weight signals must not push cash negative via costs."""
    data = _make_data(["000001"], n=300, seed=11)
    engine = BacktestEngine(
        strategy=FixedWeights({"000001": 1.0}),
        data=data,
        initial_capital=1_000_000,
        rebalance_freq="monthly",
        max_positions=10,
    )
    result = engine.run()
    assert (result.equity_curve["cash"] >= -1e-6).all(), \
        f"negative cash: {result.equity_curve[result.equity_curve['cash'] < 0]['cash'].min()}"
    # ...and the strategy actually deployed capital.
    assert engine.portfolio.holdings_series()["000001"] > 0


def test_max_positions_tie_break_by_momentum():
    """0/1 signals tie on weight; the stronger momentum name must win."""
    n = 300
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    fields = ["open", "high", "low", "close", "volume", "adj_close"]
    columns = pd.MultiIndex.from_product([["A", "B"], fields])
    df = pd.DataFrame(index=dates, columns=columns, dtype=float)
    t = np.arange(n)
    for sym, price in [("A", 100.0 * 1.001 ** t), ("B", np.full(n, 100.0))]:
        for f in ("open", "high", "low", "close", "adj_close"):
            df[(sym, f)] = price
        df[(sym, "volume")] = 5_000_000.0

    engine = BacktestEngine(
        strategy=FixedWeights({"A": 1.0, "B": 1.0}),
        data=df,
        initial_capital=1_000_000,
        rebalance_freq="monthly",
        max_positions=1,
    )
    engine.run()
    hold = engine.portfolio.holdings_series()
    assert hold.get("A", 0) > 0
    assert hold.get("B", 0) == 0


def test_composite_propagates_event_driven():
    mom = TimeSeriesMomentum(lookback=20)
    mr = ZScoreMeanReversion(window=20)
    combo = CompositeStrategy([(mom, 0.5), (mr, 0.5)])
    assert combo.event_driven is True


def test_composite_grid_only_is_not_event_driven():
    a = TimeSeriesMomentum(lookback=20)
    b = CrossSectionalMomentum(lookback=60)
    combo = CompositeStrategy([(a, 0.5), (b, 0.5)])
    assert combo.event_driven is False


def test_composite_freezes_grid_signals_within_month():
    """Grid sub-strategies must not re-trade inside a calendar month."""
    data = _make_data(["A", "B"], n=126, seed=3)
    mom = TimeSeriesMomentum(lookback=5, vol_target=None)
    combo = CompositeStrategy([(mom, 1.0)])
    sig = combo.generate_signals(data)
    month_key = sig.index.to_period("M")
    for _, grp in sig.groupby(month_key):
        assert (grp.nunique() <= 1).all().all()


def test_cross_sectional_skip_days_shifts_signal():
    """skip_days must lag the raw ranking signal by exactly N days."""
    tickers = [f"S{i:02d}" for i in range(15)]
    data = _make_data(tickers, n=400, seed=7)
    base = CrossSectionalMomentum(lookback=60, skip_days=0, holding=1)
    lagged = CrossSectionalMomentum(lookback=60, skip_days=5, holding=1)
    s0 = base.generate_signals(data)
    s5 = lagged.generate_signals(data)
    expected = s0.shift(5).fillna(0.0)
    pd.testing.assert_frame_equal(s5, expected)


def test_cross_sectional_holding_freezes_signal():
    """holding must freeze each formed book for `holding` days."""
    tickers = [f"S{i:02d}" for i in range(15)]
    data = _make_data(tickers, n=400, seed=7)
    strat = CrossSectionalMomentum(lookback=60, skip_days=0, holding=10)
    sig = strat.generate_signals(data)
    for start in range(0, len(sig), 10):
        chunk = sig.iloc[start : start + 10]
        assert (chunk.nunique() <= 1).all().all()


class _StubTrader:
    def __init__(self):
        self.orders: list[tuple] = []

    def buy(self, symbol, price, amount):
        self.orders.append(("buy", symbol, amount, price))
        return OrderResult(success=True, message="ok")

    def sell(self, symbol, price, amount):
        self.orders.append(("sell", symbol, amount, price))
        return OrderResult(success=True, message="ok")


def test_executor_rotation_sell_proceeds_fund_buys():
    """Rotation with zero free cash: sell proceeds must fund the buy."""
    trader = _StubTrader()
    ex = LiveExecutor(trader=trader, max_positions=5, min_volume=100)
    target = pd.Series({"A": 0.0, "B": 1.0})
    prices = pd.Series({"A": 10.0, "B": 10.0})

    result = ex.execute(
        target_weights=target, prices=prices, cash=0.0,
        positions={"A": 1000}, dry_run=True,
    )

    assert result["sells"] == [{"symbol": "A", "quantity": 1000, "price": 10.0}]
    assert result["buys"], "rotation produced no buys — proceeds did not recycle"
    assert result["buys"][0]["symbol"] == "B"
    # Budget = 1000*10*(1 - 0.1% slippage) = 9990 -> 900 shares at 10 yuan.
    assert result["buys"][0]["quantity"] == 900
    assert not any(e["reason"] == "insufficient cash" for e in result["errors"])


def test_corporate_action_reinvestment_keeps_value_continuous():
    """Ex-dividend: shares re-invest the distribution, daily return = price move only."""
    # raw prices: 100, 101, 96 (ex-div: 5 yuan cash dividend), 97
    # adjusted (total return): 100, 101, 101, 101 * 97 / 96
    raw_p = [100.0, 101.0, 96.0, 97.0]
    adj_p = [100.0, 101.0, 101.0, 101.0 * 97.0 / 96.0]
    dates = pd.date_range("2021-01-01", periods=4, freq="B")
    fields = ["open", "high", "low", "close", "volume", "adj_close", "raw_close"]
    columns = pd.MultiIndex.from_product([["A"], fields])
    df = pd.DataFrame(index=dates, columns=columns, dtype=float)
    for f in ("open", "high", "low", "close", "adj_close"):
        df[("A", f)] = adj_p
    df[("A", "raw_close")] = raw_p
    df[("A", "volume")] = 5_000_000.0

    engine = BacktestEngine(
        strategy=FixedWeights({"A": 1.0}),
        data=df,
        initial_capital=1_000_000,
        rebalance_freq="monthly",
        max_positions=10,
    )
    result = engine.run()
    eq = result.equity_curve
    # Ex-date (day 2): value continuous, so the return is exactly 0.
    assert eq["returns"].iloc[2] == pytest.approx(0.0, abs=1e-9)
    # Holdings re-invested the 5-yuan dividend: 9900 * 101/96 shares.
    hold = engine.portfolio.holdings_series()
    assert hold["A"] == pytest.approx(9900.0 * 101.0 / 96.0, rel=1e-9)


def test_dual_price_matches_single_price_total_return():
    """Same series with/without raw_close must give the same equity curve."""
    raw_p = [100.0, 101.0, 96.0, 97.0]
    adj_p = [100.0, 101.0, 101.0, 101.0 * 97.0 / 96.0]
    dates = pd.date_range("2021-01-01", periods=4, freq="B")
    fields = ["open", "high", "low", "close", "volume", "adj_close", "raw_close"]
    columns = pd.MultiIndex.from_product([["A"], fields])
    df = pd.DataFrame(index=dates, columns=columns, dtype=float)
    for f in ("open", "high", "low", "close", "adj_close"):
        df[("A", f)] = adj_p
    df[("A", "raw_close")] = raw_p
    df[("A", "volume")] = 5_000_000.0

    df_single = df.drop(columns=[("A", "raw_close")])
    kwargs = dict(
        strategy=FixedWeights({"A": 1.0}),
        initial_capital=1_000_000,
        rebalance_freq="monthly",
        max_positions=10,
    )
    dual = BacktestEngine(data=df, **kwargs).run()
    single = BacktestEngine(data=df_single, **kwargs).run()
    pd.testing.assert_series_equal(
        dual.equity_curve["total"],
        single.equity_curve["total"],
        rtol=1e-9,
        check_names=False,
    )


def test_qfq_hfq_return_identity():
    """qfq and hfq differ by a constant factor: interval returns must match.

    Synthetic, network-free invariant anchoring the adjustment-policy
    change (qfq -> hfq): switching the anchor must never change any
    interval return.
    """
    n = 500
    raw = pd.Series(100.0 * 1.0003 ** np.arange(n))
    # Dividend events at t=200 and t=350 (1.2 yuan/share each).
    cum = np.ones(n)
    for ev in (200, 350):
        cum[ev:] *= raw.iloc[ev - 1] / (raw.iloc[ev - 1] - 1.2)
    hfq = raw * cum
    qfq = hfq / cum[-1]
    for start, end in [(0, 499), (100, 300), (199, 351), (201, 349)]:
        assert hfq.iloc[end] / hfq.iloc[start] == pytest.approx(
            qfq.iloc[end] / qfq.iloc[start]
        )


class _AlternatingWeights(Strategy):
    """Flip full weight between two names every month (forces sells).

    Every rebalance liquidates the previous month's position, so the
    sell leg (commission + stamp duty + slippage) is exercised monthly.
    """

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        prices = self._get_prices(data)
        sig = pd.DataFrame(0.0, index=data.index, columns=prices.columns)
        odd_month = data.index.month % 2 == 1
        sig["A"] = odd_month.astype(float)
        sig["B"] = (~odd_month).astype(float)
        return sig


def test_rotation_never_negative_cash():
    """Sell-leg costs must be funded: full monthly rotation never overdrafts."""
    data = _make_data(["A", "B"], n=504, seed=3)
    engine = BacktestEngine(
        strategy=_AlternatingWeights(),
        data=data,
        initial_capital=1_000_000,
        rebalance_freq="monthly",
        max_positions=10,
    )
    result = engine.run()
    cash = result.equity_curve["cash"]
    assert (cash >= -1e-6).all(), f"negative cash: min {cash.min():.2f}"
    # The rotation must actually trade both legs.
    assert (result.trades["side"] == "sell").any()
    assert (result.trades["side"] == "buy").any()


def test_partial_trim_never_negative_cash():
    """Trimming a position (target < current) also pays sell costs."""
    data = _make_data(["A", "B"], n=504, seed=5)

    class _TrimWeights(Strategy):
        def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
            prices = self._get_prices(data)
            sig = pd.DataFrame(0.0, index=data.index, columns=prices.columns)
            month = data.index.month
            sig["A"] = np.where(month % 3 == 0, 0.5, 1.0)
            sig["B"] = np.where(month % 3 == 0, 0.5, 0.0)
            return sig

    engine = BacktestEngine(
        strategy=_TrimWeights(),
        data=data,
        initial_capital=1_000_000,
        rebalance_freq="monthly",
        max_positions=10,
    )
    result = engine.run()
    cash = result.equity_curve["cash"]
    assert (cash >= -1e-6).all(), f"negative cash: min {cash.min():.2f}"
    assert (result.trades["side"] == "sell").any()


def test_cross_sectional_small_universe_warns():
    """A universe below the cross-sectional minimum must warn, not silently idle."""
    data = _make_data([f"S{i}" for i in range(4)], n=400, seed=7)
    strat = CrossSectionalMomentum(lookback=60, skip_days=0, holding=1)
    with pytest.warns(UserWarning, match="10"):
        sig = strat.generate_signals(data)
    # The signal stays all-zero (the strategy never ranks).
    assert (sig == 0.0).all().all()
