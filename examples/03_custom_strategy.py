"""Custom strategy example — how to write your own strategy."""

import pandas as pd
import numpy as np
from quantbt.api import Backtest
from quantbt.strategy.base import Strategy


class BuyTheDip(Strategy):
    """Custom strategy: buy after N consecutive down days.

    Triggers when close[t-N:t] < close[t-N-1:t-1] for all N days
    (i.e., N consecutive down closes).
    """

    def __init__(self, down_days: int = 3, name: str | None = None):
        super().__init__(name)
        self.down_days = down_days

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        prices = self._get_prices(data)
        # Check for consecutive down days
        daily_ret = prices.pct_change().shift(1)
        # All of last N days had negative returns
        all_down = (daily_ret < 0).rolling(self.down_days).sum() >= self.down_days
        signal = all_down.astype(float)
        return signal


if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range("2021-01-01", periods=504, freq="B")
    tickers = ["000001"]
    columns = pd.MultiIndex.from_product([tickers, ["open", "high", "low", "close", "volume", "adj_close"]])
    vals = np.random.randn(504, 6).cumsum(0) + 100
    vals[:, 4] = np.abs(vals[:, 4] * 100000 + 5000000)
    data = pd.DataFrame(vals, index=dates, columns=columns)

    result = Backtest(
        strategy=BuyTheDip(down_days=3),
        data=data,
    ).run()
