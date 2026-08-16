"""A-share broker — models the trading environment."""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd

# Sell-side stamp duty was cut from 0.1% to 0.05% on 2023-08-28.
STAMP_DUTY_CUT_DATE = date(2023, 8, 28)
STAMP_DUTY_BEFORE_CUT = 0.001
# ChiNext moved to the 20% price band with the registration reform
# (2020-08-24); STAR Market has used 20% since launch (2019-07-22).
CHINEXT_20PCT_DATE = date(2020, 8, 24)
STAR_MARKET_20PCT_DATE = date(2019, 7, 22)


@dataclass
class Trade:
    date: date
    symbol: str
    side: str  # "buy" or "sell"
    quantity: int
    price: float
    commission: float
    stamp_duty: float
    slippage: float
    total_cost: float


class Broker:
    """Simulates A-share trading with realistic cost and rule modeling.

    A-share specifics:
    - Commission: 0.025%–0.03% of turnover, min 5 yuan per order
    - Stamp duty: 0.05% on sell only (reformed 2024)
    - Slippage: configurable percentage of price
    - T+1 settlement: shares bought today cannot be sold today
    - Volume limit: max X% of daily volume per trade
    - Price limits: board-aware (main ±10%, ChiNext/STAR ±20% after their
      reform dates, BSE ±30%); ST ±5% left as a future extension (needs
      ST flags in the data)
    - Minimum trade unit: 100 shares (1手)
    - No short selling by default: ``short_selling=False`` forbids negative
      holdings; ``True`` explicitly models a short book.
    """

    def __init__(
        self,
        commission_pct: float = 0.0003,
        stamp_duty_pct: float | None = None,
        dynamic_stamp_duty: bool = False,
        slippage_pct: float = 0.001,
        volume_limit_pct: float = 0.01,
        min_volume: int = 100,
        min_commission: float = 5.0,
        price_limit: bool = True,
        t_plus_1: bool = True,
        short_selling: bool = False,
    ):
        self.commission_pct = commission_pct
        self.stamp_duty_pct = 0.0005 if stamp_duty_pct is None else stamp_duty_pct
        self.dynamic_stamp_duty = dynamic_stamp_duty
        self.slippage_pct = slippage_pct
        self.volume_limit_pct = volume_limit_pct
        self.min_volume = min_volume
        self.min_commission = min_commission
        self.price_limit = price_limit
        self.t_plus_1 = t_plus_1
        self.short_selling = short_selling

    @staticmethod
    def _as_date(trade_date) -> date | None:
        """Normalize pd.Timestamp / datetime to a plain date for comparisons."""
        if trade_date is None:
            return None
        # NOTE: pd.Timestamp subclasses datetime, which subclasses date —
        # check the concrete types first so Timestamp does not slip through.
        if isinstance(trade_date, (pd.Timestamp, datetime)):
            return trade_date.date()
        if isinstance(trade_date, date):
            return trade_date
        return pd.Timestamp(trade_date).date()

    def _stamp_duty_pct(self, trade_date: date | None) -> float:
        """Sell-side stamp duty rate effective on ``trade_date``.

        With ``dynamic_stamp_duty`` the pre-2023-08-28 rate (0.1%) applies
        automatically; an explicitly pinned ``stamp_duty_pct`` wins.
        """
        trade_date = self._as_date(trade_date)
        if self.dynamic_stamp_duty and trade_date is not None and trade_date < STAMP_DUTY_CUT_DATE:
            return STAMP_DUTY_BEFORE_CUT
        return self.stamp_duty_pct

    @staticmethod
    def _price_limit_pct(symbol: str, trade_date: date | None) -> float:
        """Price band for a symbol on a given date.

        Main board ±10%; ChiNext (300/301) ±20% since 2020-08-24;
        STAR (688/689) ±20%; Beijing Stock Exchange (4/8/92 prefixes)
        ±30%. ST ±5% is not modeled (no ST flag in the data).
        """
        trade_date = Broker._as_date(trade_date)
        clean = symbol.replace(".SH", "").replace(".SZ", "").strip()
        if clean.startswith(("300", "301")):
            if trade_date is None or trade_date >= CHINEXT_20PCT_DATE:
                return 0.20
            return 0.10
        if clean.startswith(("688", "689")):
            if trade_date is None or trade_date >= STAR_MARKET_20PCT_DATE:
                return 0.20
            return 0.10
        if clean.startswith(("4", "8", "92")):
            return 0.30
        return 0.10

    @staticmethod
    def _round_limit(prev_close: float, pct: float) -> float:
        """Exchange-style limit price: half-up rounding to 0.01 yuan."""
        base = Decimal(str(prev_close))
        multiplier = Decimal("1") + Decimal(str(pct))
        return float((base * multiplier).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

    def calculate_trade_cost(
        self,
        price: float,
        quantity: int,
        side: str,
        trade_date: date | None = None,
    ) -> dict:
        turnover = price * quantity
        commission = turnover * self.commission_pct
        # A-share convention: commission has a per-order floor (5 yuan),
        # but only when a commission is actually charged.
        if commission > 0:
            commission = max(commission, self.min_commission)
        stamp_duty = turnover * self._stamp_duty_pct(trade_date) if side == "sell" else 0.0
        slippage = price * quantity * self.slippage_pct
        total_cost = commission + stamp_duty + slippage
        return {
            "commission": round(commission, 2),
            "stamp_duty": round(stamp_duty, 2),
            "slippage": round(slippage, 2),
            "total_cost": round(total_cost, 2),
        }

    def execute(
        self,
        target_positions: pd.Series,
        current_positions: pd.Series,
        prices: pd.Series,
        volumes: pd.Series | None = None,
        date: date | None = None,
        prev_close: pd.Series | None = None,
        available_shares: pd.Series | None = None,
    ) -> list[Trade]:
        """Turn target positions into executable trades.

        The broker is the single gate for orders the market would not fill or
        the rules forbid. It rejects or reduces (rather than silently allowing):

        * **Price limits** — ``prev_close`` (previous day's close per symbol)
          gates against the board-specific band (main ±10%, ChiNext/STAR ±20%,
          BSE ±30%): a buy at/above limit-up is skipped, a sell at/below
          limit-down is skipped. ST (±5%) is a future extension.
        * **T+1** — ``available_shares`` (shares sellable today per symbol) caps
          the long-liquidation portion of a sell. Shares bought today cannot be
          sold today; shares sold to open/extend a short are not T+1-bound.
        * **Short selling** — with ``short_selling=False`` a sell is capped at
          the held quantity and selling an unheld position is rejected, so
          holdings can never go negative. ``short_selling=True`` lets a negative
          target explicitly open a short.
        * **Volume limit** — a single order is capped at ``volume_limit_pct`` of
          the day's volume.
        """
        trades: list[Trade] = []
        all_symbols = set(target_positions.index) | set(current_positions.index)

        for sym in all_symbols:
            target = int(target_positions.get(sym, 0))
            current = int(current_positions.get(sym, 0))
            diff = target - current

            if diff == 0:
                continue

            side = "buy" if diff > 0 else "sell"

            price = prices.get(sym)
            if price is None or pd.isna(price) or price <= 0:
                continue

            # Price limit: board-specific band vs previous close.
            if self.price_limit and prev_close is not None and sym in prev_close.index:
                prev = prev_close[sym]
                if prev is not None and not pd.isna(prev) and prev > 0:
                    band = self._price_limit_pct(sym, date)
                    limit_up = self._round_limit(prev, band)
                    limit_down = self._round_limit(prev, -band)
                    if side == "buy" and price >= limit_up:
                        continue  # limit-up: cannot buy
                    if side == "sell" and price <= limit_down:
                        continue  # limit-down: cannot sell

            # Intended quantity, with the short-sell guard applied.
            if side == "sell":
                qty = -diff
                if not self.short_selling:
                    if current <= 0:
                        continue  # selling an unheld position = opening a short
                    qty = min(qty, current)  # long-only: never go negative
            else:
                qty = diff

            # T+1: only the long-liquidation portion is bounded by what is
            # sellable today; a short leg is borrowed, not bought-today.
            if side == "sell" and self.t_plus_1:
                avail = current
                if available_shares is not None and sym in available_shares.index:
                    avail = int(available_shares[sym])
                long_book = max(0, current)
                liquidate = min(qty, long_book, max(0, avail))
                open_short = max(0, qty - long_book)
                qty = liquidate + open_short
                if qty <= 0:
                    continue

            # Round to whole lots (1手 = 100 shares). A-share exception: an
            # odd lot (< 100 shares) must be sold in one single order, so a
            # full liquidation keeps the odd lot instead of rounding it away.
            full_odd_lot_sell = (
                side == "sell" and qty == current and 0 < current < self.min_volume
            )
            if not full_odd_lot_sell:
                qty = (qty // self.min_volume) * self.min_volume
            if qty == 0:
                continue

            # Volume limit: single order capped at a fraction of daily volume.
            if volumes is not None and sym in volumes.index:
                vol = volumes[sym]
                if vol is None or pd.isna(vol) or vol <= 0:
                    continue  # no liquidity data: fail closed, never int(NaN)
                max_qty = int(vol * self.volume_limit_pct)
                if qty > max_qty:
                    qty = max_qty
                    # a capped order is no longer a full odd-lot liquidation
                    qty = (qty // self.min_volume) * self.min_volume
                if qty == 0:
                    continue

            cost = self.calculate_trade_cost(price, qty, side, trade_date=date)
            trades.append(
                Trade(
                    date=date,
                    symbol=sym,
                    side=side,
                    quantity=qty,
                    price=round(price, 2),
                    commission=cost["commission"],
                    stamp_duty=cost["stamp_duty"],
                    slippage=cost["slippage"],
                    total_cost=cost["total_cost"],
                )
            )

        return trades