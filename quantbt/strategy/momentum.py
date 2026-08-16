"""Momentum strategies for A-share market."""

import pandas as pd
import numpy as np

from quantbt.strategy.base import Strategy


class TimeSeriesMomentum(Strategy):
    """Time-series momentum: go long if past N-day return exceeds threshold.

    Includes optional volatility scaling (equal risk contribution).
    """

    def __init__(
        self,
        lookback: int = 60,
        threshold: float = 0.0,
        vol_target: float | None = 0.15,
        vol_window: int = 20,
        name: str | None = None,
    ):
        super().__init__(name)
        self.lookback = lookback
        self.threshold = threshold
        self.vol_target = vol_target
        self.vol_window = vol_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        prices = self._get_prices(data)

        # Signal at t uses past return through t-1 only
        past_return = prices.pct_change(self.lookback).shift(1)
        signal = pd.DataFrame(0.0, index=data.index, columns=prices.columns)
        signal[past_return > self.threshold] = 1.0

        if self.vol_target is not None:
            daily_ret = prices.pct_change()
            realized_vol = daily_ret.rolling(self.vol_window).std().shift(1)
            annualized_vol = realized_vol * np.sqrt(252)
            scaled = signal * (self.vol_target / annualized_vol.replace(0, np.nan))
            signal = scaled.clip(-2.0, 2.0)

        return signal


class CrossSectionalMomentum(Strategy):
    """Cross-sectional momentum: long top quantile, short bottom quantile.

    Reference: Jegadeesh & Titman (1993).
    For A-share retail context, long_only=True skips the short leg.
    """

    def __init__(
        self,
        lookback: int = 252,
        holding: int = 21,
        top_quantile: float = 0.2,
        bottom_quantile: float = 0.2,
        long_only: bool = True,
        skip_days: int = 5,
        name: str | None = None,
    ):
        super().__init__(name)
        self.lookback = lookback
        self.holding = holding
        self.top_quantile = top_quantile
        self.bottom_quantile = bottom_quantile
        self.long_only = long_only
        self.skip_days = skip_days

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        prices = self._get_prices(data)

        # Rank by past return through t-1
        past_return = prices.pct_change(self.lookback).shift(1)
        signal = pd.DataFrame(0.0, index=data.index, columns=prices.columns)

        for date_idx in data.index:
            row = past_return.loc[date_idx].dropna()
            if len(row) < 10:
                continue
            n_top = max(1, int(len(row) * self.top_quantile))
            n_bottom = max(1, int(len(row) * self.bottom_quantile))
            top = row.nlargest(n_top).index
            bottom = row.nsmallest(n_bottom).index
            signal.loc[date_idx, top] = 1.0 / n_top
            if not self.long_only:
                signal.loc[date_idx, bottom] = -1.0 / n_bottom

        return signal
