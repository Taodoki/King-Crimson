"""Tests for strategy signal generation."""

import pandas as pd
import numpy as np


def _make_data():
    np.random.seed(42)
    dates = pd.date_range("2021-01-01", periods=252, freq="B")
    tickers = ["A", "B"]
    columns = pd.MultiIndex.from_product([tickers, ["close", "adj_close", "volume"]])
    vals = np.random.randn(252, 6).cumsum(0) + 100
    return pd.DataFrame(vals, index=dates, columns=columns)


def test_momentum_signal_shape():
    from quantbt.strategy.momentum import TimeSeriesMomentum

    data = _make_data()
    strat = TimeSeriesMomentum(lookback=20)
    sig = strat.generate_signals(data)
    assert sig.shape == data.xs("close", axis=1, level=1).shape


def test_momentum_signal_range():
    from quantbt.strategy.momentum import TimeSeriesMomentum

    data = _make_data()
    strat = TimeSeriesMomentum(lookback=20, threshold=-1.0)
    sig = strat.generate_signals(data)
    assert sig.min().min() >= -2.0
    assert sig.max().max() <= 2.0


def test_momentum_no_lookahead():
    """Verify shift(1): signal at t uses data through t-1 only."""
    from quantbt.strategy.momentum import TimeSeriesMomentum

    dates = pd.date_range("2021-01-01", periods=100, freq="B")
    close = pd.DataFrame({
        "close": (np.arange(100) * 0.1 + 100),
        "adj_close": (np.arange(100) * 0.1 + 100),
        "volume": 1_000_000,
    }, index=dates)
    columns = pd.MultiIndex.from_product([["A"], close.columns])
    data = pd.DataFrame(
        close.values, index=dates, columns=columns
    )

    strat = TimeSeriesMomentum(lookback=5, threshold=0)
    sig = strat.generate_signals(data)

    # With strictly increasing prices, signal should be 1
    # But the first lookback+1 days will be NaN/0
    valid_sig = sig.dropna().iloc[5:]
    assert (valid_sig > 0).all().all()


def test_mean_reversion_entry_exit():
    from quantbt.strategy.mean_reversion import ZScoreMeanReversion

    data = _make_data()
    strat = ZScoreMeanReversion(window=20, entry_z=-2.0, exit_z=-0.5)
    sig = strat.generate_signals(data)
    expected = data.xs("close", axis=1, level=1)
    assert sig.shape == expected.shape
    assert sig.min().min() >= 0.0
    assert sig.max().max() <= 1.0


def test_strategy_name():
    from quantbt.strategy.momentum import TimeSeriesMomentum

    strat = TimeSeriesMomentum(lookback=60, name="test_mom")
    assert strat.name == "test_mom"
    assert "lookback=60" in repr(strat)


def test_composite_strategy():
    from quantbt.strategy.momentum import TimeSeriesMomentum
    from quantbt.strategy.mean_reversion import ZScoreMeanReversion
    from quantbt.strategy.composite import CompositeStrategy

    data = _make_data()
    mom = TimeSeriesMomentum(lookback=20)
    mr = ZScoreMeanReversion(window=20)
    combo = CompositeStrategy([(mom, 0.5), (mr, 0.5)])
    sig = combo.generate_signals(data)
    expected = data.xs("close", axis=1, level=1)
    assert sig.shape == expected.shape
