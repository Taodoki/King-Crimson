"""Vectorized backtesting engine.

Two responsibilities, and only two:
    1. Execute trades on rebalance days.
    2. Mark the portfolio to market on every trading day.

Both derive from the single Portfolio state object — the engine keeps no
position book of its own.

Matching convention (T+1 fills):
    * A fill executes at day ``t``'s close.
    * Day ``t``'s NAV reflects the holdings held *during* day ``t`` (the
      book carried in from ``t-1``), marked at ``t``'s close.
    * A fill on day ``t`` takes effect on day ``t+1``.

Broker interface (per rebalance day):
    * ``prev_close`` — the previous trading day's close per symbol, used by the
      broker to enforce ±10% price limits.
    * ``available_shares`` — the shares sellable today per symbol, used by the
      broker to enforce T+1. Because fills take effect on ``t+1``, every share
      carried into day ``t`` was bought on or before ``t-1``, so this equals the
      current holdings; the gate exists in the broker regardless.
    * ``short_selling`` — forwarded to the broker; when False (default) the
      broker rejects sells that would open a short, so holdings stay non-negative.
"""

from datetime import date
import pandas as pd
import numpy as np

from quantbt.core.broker import Broker
from quantbt.core.portfolio import Portfolio
from quantbt.core.calendar import get_rebalance_dates
from quantbt.strategy.base import Strategy


class BacktestEngine:
    """Vectorized backtesting engine for A-shares."""

    def __init__(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        initial_capital: float = 1_000_000,
        commission: float = 0.0003,
        stamp_duty: float | None = None,
        slippage: float = 0.001,
        min_commission: float = 5.0,
        rebalance_freq: str = "monthly",
        max_positions: int | None = 10,
        short_selling: bool = False,
        volume_limit: float = 0.01,
    ):
        self.strategy = strategy
        self.data = data
        self.rebalance_freq = rebalance_freq
        self.max_positions = max_positions
        self.short_selling = short_selling

        # stamp_duty=None means "use the historical sell-side rate":
        # 0.1% before 2023-08-28, 0.05% after. An explicit value disables
        # the date-aware switch and pins the rate for the whole run.
        self.broker = Broker(
            commission_pct=commission,
            stamp_duty_pct=0.0005 if stamp_duty is None else stamp_duty,
            dynamic_stamp_duty=stamp_duty is None,
            slippage_pct=slippage,
            min_commission=min_commission,
            volume_limit_pct=volume_limit,
            short_selling=short_selling,
        )
        self.portfolio = Portfolio(initial_capital)

        self.config = {
            "initial_capital": initial_capital,
            "commission": commission,
            "stamp_duty": stamp_duty,
            "stamp_duty_dynamic": stamp_duty is None,
            "slippage": slippage,
            "min_commission": min_commission,
            "rebalance_freq": rebalance_freq,
            "max_positions": max_positions,
            "short_selling": short_selling,
            "volume_limit": volume_limit,
        }

    def run(self) -> "BacktestResult":
        result = BacktestResult()
        result.config = self.config
        result.strategy_name = self.strategy.name

        # Phase 1: Generate signals
        signals = self.strategy.generate_signals(self.data)

        # Phase 2: Trade + valuation over every trading day
        all_trading_dates = list(self.data.index)
        rebalance_set = set(
            get_rebalance_dates(pd.DatetimeIndex(self.data.index), self.rebalance_freq)
        )

        # Previous-day close, used by the broker for ±10% price limits.
        if isinstance(self.data.columns, pd.MultiIndex):
            close_all = self.data.xs("close", axis=1, level=1)
        else:
            close_all = self.data
        prev_close_all = close_all.shift(1)

        all_trades = []

        for dt in all_trading_dates:
            dt = pd.Timestamp(dt)
            prices = self._get_field_at_date(dt, "close")
            volumes = self._get_field_at_date(dt, "volume")
            trades = []

            is_rebalance = dt in rebalance_set
            # State-machine strategies (entry/exit/stop-loss) match orders on
            # every trading day: their signal only changes on transition days,
            # so daily matching adds no extra trades on flat days.
            if self.strategy.event_driven:
                is_rebalance = True

            if is_rebalance and not prices.empty:
                weights = signals.loc[dt].fillna(0.0)

                # The engine models a cash account, not a margin account:
                # signals are relative weights. A long leg summing above 1.0
                # (e.g. vol-targeted momentum) is scaled back to 1.0 instead
                # of borrowing free cash.
                long_mask = weights > 0
                if long_mask.any():
                    long_sum = float(weights[long_mask].sum())
                    if long_sum > 1.0:
                        weights = weights.copy()
                        weights[long_mask] = weights[long_mask] / long_sum

                # Size against today's equity marked at today's close (not
                # yesterday's), so the realized weights match the signal.
                total_equity = self.portfolio.value_at(prices)
                cur = self.portfolio.holdings_series()

                target_value = weights * total_equity
                # Floor to whole lots (100 shares) to avoid over-buying.
                target_shares = (target_value / prices.replace(0, np.nan)).fillna(0.0)
                target_shares = np.floor(target_shares / 100) * 100

                if self.max_positions:
                    # Rank by target WEIGHT, not by share count: share counts
                    # scale with 1/price, so ranking shares would favor cheap
                    # stocks over the strategy's actual preferences.
                    top = weights.abs().nlargest(self.max_positions)
                    target_shares = target_shares.where(
                        target_shares.index.isin(top.index), 0.0
                    )

                prev_close = (
                    prev_close_all.loc[dt] if dt in prev_close_all.index else None
                )
                trades = self.broker.execute(
                    target_positions=target_shares,
                    current_positions=cur,
                    prices=prices,
                    volumes=volumes,
                    date=dt,
                    prev_close=prev_close,
                    available_shares=cur,
                )

            # Mark to market every day; non-rebalance days pass trades=[].
            self.portfolio.update(dt, prices, trades)
            all_trades.extend(trades)

        result.equity_curve = self.portfolio.equity_curve
        result.trades = pd.DataFrame(
            [
                {
                    "date": t.date,
                    "symbol": t.symbol,
                    "side": t.side,
                    "quantity": t.quantity,
                    "price": t.price,
                    "total_cost": t.total_cost,
                }
                for t in all_trades
            ]
        )
        return result

    def _get_field_at_date(self, dt: date, field: str) -> pd.Series:
        row = self.data.loc[dt]
        if isinstance(self.data.columns, pd.MultiIndex):
            return row.xs(field, level=1)
        return row


class BacktestResult:
    """Container for backtest output."""

    def __init__(self):
        self.equity_curve: pd.DataFrame = pd.DataFrame()
        self.trades: pd.DataFrame = pd.DataFrame()
        self.config: dict = {}
        self.strategy_name: str = ""

    @property
    def portfolio_returns(self) -> pd.Series:
        if self.equity_curve.empty:
            return pd.Series(dtype=np.float64)
        return self.equity_curve.set_index("date")["returns"]

    def summary(self) -> None:
        """Print the performance summary table to the terminal."""
        from quantbt.utils.rich_utils import print_backtest_summary

        print_backtest_summary(self)

    def plot_equity(self, path: str = "equity_curve.html") -> str:
        """Save an interactive equity curve + drawdown chart to ``path``."""
        from quantbt.plot.equity import plot_equity_drawdown

        return plot_equity_drawdown(self.equity_curve, path)