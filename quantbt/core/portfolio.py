"""Portfolio tracking — NAV, cash, holdings over time.

The Portfolio is the single source of truth for position state during a
backtest. The engine keeps no position book of its own: it reads holdings
via :meth:`holdings_series` and equity via :attr:`total_value`.

Valuation convention (T+1 fills):
    * A fill executed at the close of day ``t`` takes effect on day ``t+1``.
    * Day ``t``'s NAV therefore reflects the holdings carried *into* day
      ``t``, marked to market at day ``t``'s close.
"""

from dataclasses import dataclass
from datetime import date
import pandas as pd

from quantbt.core.broker import Trade


@dataclass
class PortfolioSnapshot:
    date: date
    cash: float
    holdings: dict[str, int]
    prices: dict[str, float]
    total_value: float
    returns: float
    cumulative_returns: float


class Portfolio:
    """Tracks portfolio state over time. Records full equity curve."""

    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital
        self._cash: float = initial_capital
        # Holdings can be fractional: on ex-dividend days the distribution
        # is re-invested into the position, so share counts are a
        # total-return bookkeeping unit, not an integer share count.
        self._holdings: dict[str, float] = {}
        self._prices: dict[str, float] = {}
        self._history: list[PortfolioSnapshot] = []
        self._cum_ret: float = 1.0

    def holdings_series(self) -> pd.Series:
        """Current positions keyed by symbol (number of shares)."""
        return pd.Series(self._holdings, dtype=float)

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def holdings_value(self) -> float:
        """Market value of current holdings at the latest close."""
        return self._holdings_value()

    @property
    def total_value(self) -> float:
        """Cash + market value of current holdings at the latest close."""
        return self._cash + self._holdings_value()

    def _holdings_value(self) -> float:
        return self._value_with(self._prices)

    def _value_with(self, prices: dict[str, float]) -> float:
        total = 0.0
        for sym, qty in self._holdings.items():
            price = prices.get(sym, 0.0)
            if price is None or pd.isna(price):
                price = 0.0
            total += qty * price
        return total

    def _fill_prices(self, prices: pd.Series) -> dict[str, float]:
        """Merge today's closes over last-known prices.

        Suspended symbols (NaN close, missing rows) keep the previous
        close instead of being marked to zero - a one-day suspension must
        not erase the position's value.
        """
        merged = dict(self._prices)
        if prices is not None:
            for sym, p in prices.items():
                if p is None or pd.isna(p) or p <= 0:
                    continue
                merged[sym] = float(p)
        return merged

    def value_at(self, prices: pd.Series | None = None) -> float:
        """Cash + holdings marked with ``prices`` (typically today's close).

        NaN prices fall back to the last known close, so position sizing
        can use today's equity at today's prices instead of yesterday's.
        """
        marks = self._fill_prices(prices) if prices is not None else self._prices
        return self._cash + self._value_with(marks)

    def update(
        self,
        date: date,
        prices: pd.Series,
        trades: list[Trade],
        adjust_factors: pd.Series | None = None,
    ) -> PortfolioSnapshot:
        # 0. Corporate actions (ex-dividend days): re-invest the
        #    distribution BEFORE marking to market. shares *= factor keeps
        #    market value continuous across the ex-date, so the day's
        #    return reflects price moves only (total-return convention).
        if adjust_factors is not None:
            for sym, factor in adjust_factors.items():
                if factor is None or pd.isna(factor):
                    continue
                if abs(self._holdings.get(sym, 0.0)) > 0.0:
                    self._holdings[sym] *= float(factor)

        # 1. Mark to market the holdings carried into `date`, at `date` close.
        #    Suspended symbols (NaN close) carry their last known price forward.
        if prices is not None:
            self._prices = self._fill_prices(prices)

        total_value = self.total_value
        prev = self._get_prev_total()
        ret = total_value / prev - 1 if prev > 0 else 0.0
        self._cum_ret *= 1 + ret

        snapshot = PortfolioSnapshot(
            date=date,
            cash=round(self._cash, 2),
            holdings=self._holdings.copy(),
            prices=self._prices.copy(),
            total_value=round(total_value, 2),
            returns=ret,
            cumulative_returns=self._cum_ret,
        )
        self._history.append(snapshot)

        # 2. Apply trades — fills at `date` close, effective `date`+1.
        self._apply_trades(trades)
        return snapshot

    def _apply_trades(self, trades: list[Trade]) -> None:
        for t in trades:
            if t.side == "buy":
                self._cash -= t.quantity * t.price + t.total_cost
                self._holdings[t.symbol] = self._holdings.get(t.symbol, 0) + t.quantity
            else:
                self._cash += t.quantity * t.price - t.total_cost
                self._holdings[t.symbol] = self._holdings.get(t.symbol, 0) - t.quantity
                if abs(self._holdings[t.symbol]) < 1e-9:
                    del self._holdings[t.symbol]
                # A negative balance is kept: it is an explicit short book,
                # not a position to silently drop.

    def _get_prev_total(self) -> float:
        if not self._history:
            return self.initial_capital
        return self._history[-1].total_value

    @property
    def equity_curve(self) -> pd.DataFrame:
        records = []
        for s in self._history:
            records.append({
                "date": s.date,
                "cash": s.cash,
                "holdings_value": s.total_value - s.cash,
                "total": s.total_value,
                "returns": s.returns,
                "cumulative_returns": s.cumulative_returns,
            })
        return pd.DataFrame(records)