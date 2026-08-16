"""Quickstart: 10-line backtest with momentum strategy."""

import pandas as pd
import numpy as np
from quantbt.api import Backtest
from quantbt.strategy import TimeSeriesMomentum

# Generate sample data (in real use, Backtest fetches from Baostock automatically)
np.random.seed(42)
dates = pd.date_range("2021-01-01", periods=504, freq="B")
tickers = ["000001", "000002", "000003"]
columns = pd.MultiIndex.from_product([tickers, ["open", "high", "low", "close", "volume", "adj_close"]])
vals = np.random.randn(504, 18).cumsum(0)
for i in range(3):
    vals[:, i * 6 + 3] += 100  # close
    vals[:, i * 6 + 5] = vals[:, i * 6 + 3]  # adj_close = close
    vals[:, i * 6 + 4] = np.abs(vals[:, i * 6 + 4] * 100000 + 5000000)
data = pd.DataFrame(vals, index=dates, columns=columns)

# The actual backtest — one line
result = Backtest(
    strategy=TimeSeriesMomentum(lookback=60, threshold=-1.0),
    data=data,
    initial_capital=1_000_000,
).run()

# Explore results
print(f"\nFinal portfolio value: {result.equity_curve['total'].iloc[-1]:,.2f}")
print(f"Total trades: {len(result.trades)}")
