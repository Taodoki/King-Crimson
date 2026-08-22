"""Vectorized backtesting engine.

Two responsibilities, and only two:
    1. Execute trades on rebalance days.
    2. Mark the portfolio to market on every trading day.

Both derive from the single Portfolio state object — the engine keeps no
position book of its own.

Dual price system:
    * Trading, valuation and price-limit prices are REAL (unadjusted)
      closes, so share counts, cash and volume limits match the actual
      market. Synthetic data without a ``raw_close`` column falls back
      to ``close`` (no corporate actions => the systems coincide).
    * Strategy signals use adjusted closes (total-return space).
    * On ex-dividend days the portfolio re-invests the distribution
      (shares *= adjusted/raw ratio change), keeping value continuous.

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

        if isinstance(self.data.columns, pd.MultiIndex):
            close_all = self.data.xs("close", axis=1, level=1)
            fields = self.data.columns.get_level_values(1)
            has_raw = "raw_close" in fields
        else:
            close_all = self.data
            has_raw = False

        # Trading, valuation and price-limit prices use REAL (unadjusted)
        # closes; synthetic data without raw_close falls back to close.
        if has_raw:
            raw_close_all = self.data.xs("raw_close", axis=1, level=1)
        else:
            raw_close_all = close_all

        # Previous-day real close, used by the broker for ±10% price limits.
        prev_close_all = raw_close_all.shift(1)

        # Trailing momentum used to break selection ties (see the
        # max_positions block). Falls back to shorter windows on short
        # histories. Computed on ADJUSTED closes (signal space).
        momentum_all = (
            close_all.pct_change(252)
            .fillna(close_all.pct_change(60))
            .fillna(0.0)
        )

        # Corporate-action factor: adjusted/raw price ratio. Constant
        # between ex-dates (float noise ~1e-16), jumps on ex-dates.
        # shares *= factor re-invests the distribution, keeping market
        # value continuous across the ex-date (total-return convention).
        ratio_all = close_all.ffill() / raw_close_all.replace(0, np.nan).ffill()
        factor_all = ratio_all / ratio_all.shift(1)
        factor_all = factor_all.where((factor_all - 1.0).abs() > 1e-6, 1.0)

        all_trades = []

        for dt in all_trading_dates:
            dt = pd.Timestamp(dt)
            prices = self._get_field_at_date(
                dt, "raw_close" if has_raw else "close"
            )
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
                cash_before = self.portfolio.cash

                prev_close = (
                    prev_close_all.loc[dt] if dt in prev_close_all.index else None
                )

                if self.max_positions:
                    # Rank by target WEIGHT, not by share count: share counts
                    # scale with 1/price, so ranking shares would favor cheap
                    # stocks over the strategy's actual preferences.
                    # Equal-weight signals (e.g. 0/1 momentum) tie on weight;
                    # break ties by trailing momentum so top-max selection is
                    # an explicit rule instead of silently following column
                    # order.
                    mom = momentum_all.loc[dt] if dt in momentum_all.index else 0.0
                    rank = pd.DataFrame(
                        {"weight": weights.abs(), "momentum": mom}
                    )
                    rank = rank.sort_values(
                        ["weight", "momentum"],
                        ascending=False,
                        na_position="last",
                    )
                    top_idx = rank.index[: self.max_positions]
                    weights = weights.where(weights.index.isin(top_idx), 0.0)

                # Size the buy leg against the cash this rebalance itself
                # raises, so the plan is affordable by construction. The
                # sell leg (liquidations and trims) pays commission +
                # stamp duty + slippage too; with long-only weights a
                # turnover formula over target weights alone never sees
                # it, so rotations used to leak sell costs into negative
                # cash. Two-phase sizing:
                #   1. Proportional shrink: scale the deployable by the
                #      affordability ratio of the planned trades.
                #   2. Lot-level safety net: 100-share lot flooring makes
                #      buy notional a step function of the deployable, so
                #      a proportional shrink can stall between two lots.
                #      Drop one lot at a time from the largest planned
                #      buy until the plan is exactly affordable. Each
                #      pass strictly reduces buy notional, so this
                #      terminates.
                deployable = total_equity
                target_shares: pd.Series | None = None
                for _ in range(6):
                    target_value = weights * deployable
                    # Floor to whole lots (100 shares) to avoid over-buying.
                    target_shares = (
                        target_value / prices.replace(0, np.nan)
                    ).fillna(0.0)
                    target_shares = np.floor(target_shares / 100) * 100

                    trades = self.broker.execute(
                        target_positions=target_shares,
                        current_positions=cur,
                        prices=prices,
                        volumes=volumes,
                        date=dt,
                        prev_close=prev_close,
                        available_shares=cur,
                    )
                    buy_notional = sum(
                        t.quantity * t.price for t in trades if t.side == "buy"
                    )
                    buy_costs = sum(
                        t.total_cost for t in trades if t.side == "buy"
                    )
                    sell_net = sum(
                        t.quantity * t.price - t.total_cost
                        for t in trades
                        if t.side == "sell"
                    )
                    available = cash_before + sell_net
                    if buy_notional + buy_costs <= available + 1e-6:
                        break
                    deployable *= (
                        max(available - buy_costs, 0.0) / buy_notional
                        if buy_notional > 0
                        else 0.0
                    )

                # Phase 2: exact affordability at lot granularity.
                # Cut one lot at a time from the largest planned buy
                # until the plan is affordable. Cutting the FILL itself
                # (not the target shares) guarantees progress: a buy
                # capped by the volume limit would not shrink if we only
                # lowered its target, which could loop forever. Each pass
                # reduces buy notional by >= one lot, so this terminates.
                while True:
                    buy_notional = sum(
                        t.quantity * t.price for t in trades if t.side == "buy"
                    )
                    buy_costs = sum(
                        t.total_cost for t in trades if t.side == "buy"
                    )
                    sell_net = sum(
                        t.quantity * t.price - t.total_cost
                        for t in trades
                        if t.side == "sell"
                    )
                    available = cash_before + sell_net
                    if buy_notional + buy_costs <= available + 1e-6:
                        break
                    buys = [t for t in trades if t.side == "buy"]
                    if not buys:
                        break  # nothing left to cut; plan already minimal
                    biggest = max(buys, key=lambda t: t.quantity * t.price)
                    qty = biggest.quantity - 100
                    if qty <= 0:
                        trades.remove(biggest)
                        continue
                    biggest.quantity = qty
                    cost = self.broker.calculate_trade_cost(
                        biggest.price, qty, "buy", biggest.date
                    )
                    biggest.commission = cost["commission"]
                    biggest.stamp_duty = cost["stamp_duty"]
                    biggest.slippage = cost["slippage"]
                    biggest.total_cost = cost["total_cost"]

            # Corporate-action factor for today (1.0 on ordinary days).
            adjust = factor_all.loc[dt] if dt in factor_all.index else None

            # Mark to market every day; non-rebalance days pass trades=[].
            self.portfolio.update(dt, prices, trades, adjust_factors=adjust)
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