"""Momentum strategies for A-share market."""

import warnings

import pandas as pd
import numpy as np

from quantbt.strategy.base import Strategy

# Minimum universe size for a meaningful cross-sectional ranking. Below
# this the top/bottom quantile split degenerates, so the strategy stays
# flat (see the warning in CrossSectionalMomentum.generate_signals).
MIN_CROSS_SECTION = 10


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

    ``skip_days`` implements the J-T one-week skip between formation and
    holding (avoids short-term reversal and bid-ask bounce): the signal at
    date t reflects the ranking formed at t - skip_days. ``holding``
    freezes each formed portfolio for ``holding`` trading days instead of
    re-ranking every day.
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

        n_symbols = prices.shape[1]
        if n_symbols < MIN_CROSS_SECTION:
            # A below-minimum universe silently produces an all-zero
            # signal: the backtest then reports "zero trades, 100% cash",
            # which reads as "the strategy does not work" instead of
            # "the strategy never ran". Fail loud instead.
            warnings.warn(
                f"CrossSectionalMomentum: universe has {n_symbols} symbols, "
                f"fewer than the {MIN_CROSS_SECTION} required to form a "
                f"cross-section. All signals will be zero and the backtest "
                f"will hold 100% cash.",
                UserWarning,
                stacklevel=2,
            )

        # Rank by past return through t-1
        past_return = prices.pct_change(self.lookback).shift(1)
        signal = pd.DataFrame(0.0, index=data.index, columns=prices.columns)

        for date_idx in data.index:
            row = past_return.loc[date_idx].dropna()
            if len(row) < MIN_CROSS_SECTION:
                continue
            n_top = max(1, int(len(row) * self.top_quantile))
            n_bottom = max(1, int(len(row) * self.bottom_quantile))
            top = row.nlargest(n_top).index
            bottom = row.nsmallest(n_bottom).index
            signal.loc[date_idx, top] = 1.0 / n_top
            if not self.long_only:
                signal.loc[date_idx, bottom] = -1.0 / n_bottom

        # J-T skip: the signal at t reflects the ranking formed at
        # t - skip_days, so entries wait out the short-term reversal
        # window instead of buying at the formation price.
        if self.skip_days > 0:
            signal = signal.shift(self.skip_days).fillna(0.0)

        # Hold each formed book for `holding` trading days: sample the
        # signal on a fixed grid and forward-fill between formation
        # dates. Approximation: the grid aligns to the series start,
        # and the engine rebalance may still re-size frozen weights to
        # their targets when prices drift.
        if self.holding > 1:
            frozen = signal.iloc[:: self.holding]
            signal = frozen.reindex(signal.index, method="ffill").fillna(0.0)

        return signal
