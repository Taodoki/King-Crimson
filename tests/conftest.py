"""Pytest fixtures for quantbt tests."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_returns() -> pd.Series:
    """Daily return series with known properties."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    # Generate returns with ~0.05% mean, ~1% std
    rets = np.random.randn(252) * 0.01 + 0.0005
    return pd.Series(rets, index=dates, name="returns")


@pytest.fixture
def sample_multiindex_data() -> pd.DataFrame:
    """MultiIndex OHLCV data for 3 stocks over 2 years."""
    np.random.seed(42)
    dates = pd.date_range("2021-01-01", periods=504, freq="B")
    tickers = ["000001", "000002", "000003"]
    fields = ["open", "high", "low", "close", "volume", "adj_close"]
    columns = pd.MultiIndex.from_product([tickers, fields])

    n = len(dates)
    k = len(tickers) * len(fields)
    vals = np.random.randn(n, k).cumsum(0)
    # Set prices around 100
    for i, ticker in enumerate(tickers):
        vals[:, i * 6] += 100
        vals[:, i * 6 + 1] += 101
        vals[:, i * 6 + 2] += 99
        vals[:, i * 6 + 3] += 100
        vals[:, i * 6 + 4] = np.abs(vals[:, i * 6 + 4] * 100000 + 5000000)
        vals[:, i * 6 + 5] = vals[:, i * 6 + 3]  # adj_close = close

    return pd.DataFrame(vals, index=dates, columns=columns)


@pytest.fixture
def sample_strategy():
    from quantbt.strategy.momentum import TimeSeriesMomentum
    return TimeSeriesMomentum(lookback=20, threshold=-1.0)
