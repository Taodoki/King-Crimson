"""Mean reversion strategies for A-share market."""

import pandas as pd
import numpy as np

from quantbt.strategy.base import Strategy


class ZScoreMeanReversion(Strategy):
    """Z-score mean reversion: go long when price is sufficiently below MA.

    Enters when z-score < entry_z (oversold), exits when z-score > exit_z.
    Designed for A-share where short selling is restricted for retail.

    This is a two-state machine (FLAT <-> LONG): entries, exits and the
    stop-loss fire on specific days, not on a fixed rebalance grid. The
    strategy therefore declares ``event_driven=True`` so the engine
    matches orders on the day the signal changes instead of waiting for
    the next monthly rebalance.
    """

    event_driven: bool = True

    def __init__(
        self,
        window: int = 20,
        entry_z: float = -2.0,
        exit_z: float = -0.5,
        stop_loss: float | None = 0.07,
        name: str | None = None,
    ):
        super().__init__(name)
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_loss = stop_loss

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        prices = self._get_prices(data)

        # Z-score using data through t-1 only
        ma = prices.rolling(self.window).mean().shift(1)
        std = prices.rolling(self.window).std(ddof=0).shift(1)
        z_score = (prices.shift(1) - ma) / std.replace(0, np.nan)

        entry_mask = z_score < self.entry_z
        exit_mask = z_score > self.exit_z

        # Mean reversion is a two-state machine (FLAT <-> LONG): entries and
        # exits are state transitions, not independent daily decisions. The
        # stop-loss price is a property of the open position, fixed once at
        # entry — never recomputed daily.
        signal = pd.DataFrame(0.0, index=data.index, columns=prices.columns)
        for col in prices.columns:
            col_entry = entry_mask[col].to_numpy()
            col_exit = exit_mask[col].to_numpy()
            col_price = prices[col].to_numpy()
            position = 0
            entry_price = np.nan
            col_signal = np.zeros(len(col_price))
            for t in range(len(col_price)):
                if position == 0 and col_entry[t]:
                    position = 1
                    entry_price = col_price[t]
                elif position == 1:
                    # 止损用 price[t]（t 日收盘）判断 t 日离场，属 close-close 约定，有意为之
                    stopped = (
                        self.stop_loss is not None
                        and col_price[t] / entry_price - 1 < -self.stop_loss
                    )
                    if col_exit[t] or stopped:
                        position = 0
                        entry_price = np.nan
                col_signal[t] = position
            signal[col] = col_signal

        return signal