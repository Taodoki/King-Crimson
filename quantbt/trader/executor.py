"""Signal executor — converts strategy signals into an order list.

实盘已移除真实券商下单；本模块只计算正确的调仓清单（买/卖），
由调用方决定是打印人工委托单还是记录日志。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from quantbt.trader.tonghuashun import TonghuashunTrader

logger = logging.getLogger("quantbt.trader")


class LiveExecutor:
    """Execute strategy signals against available cash and current positions.

    Workflow per rebalance:
    1. Read current positions (fail-closed: abort if unreadable)
    2. Get target positions from strategy weights
    3. Compute diff: what to buy / sell
    4. Sell first (proceeds fund buys), cap buys by available cash;
       enforce T+1 on sells
    5. Produce a human-executable order list
    """

    def __init__(
        self,
        trader: TonghuashunTrader,
        max_positions: int = 5,
        min_volume: int = 100,
        slippage: float = 0.001,
    ):
        self.trader = trader
        self.max_positions = max_positions
        self.min_volume = min_volume
        self.slippage = slippage

    def execute(
        self,
        target_weights: pd.Series,
        prices: pd.Series,
        cash: float,
        positions: dict[str, int] | None = None,
        bought_today: dict[str, int] | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Compute the rebalance order list.

        Args:
            target_weights: Strategy signal (fraction of cash per stock).
                May exceed 1.0 when the strategy uses leverage (vol_target).
            prices: Current market prices.
            cash: Available cash to deploy (NOT total assets).
            positions: Current holdings {symbol: quantity}. When None and not
                dry_run, positions are read from the trader and a read failure
                aborts (never treated as empty).
            bought_today: Shares bought today {symbol: quantity}, subject to
                T+1 (cannot be sold the same day).
            dry_run: If True, log orders without delegating to the trader.

        Returns:
            Dict with keys: buys, sells, errors, summary
        """
        if positions is None:
            if dry_run:
                positions = {}
            else:
                # fail-closed: a broker read failure must abort, not empty-out
                positions = self._read_positions()

        bought_today = bought_today or {}

        # Target shares: signal weight applied to TOTAL equity (cash +
        # current holdings marked at today's prices). Basing targets on
        # cash alone computes near-zero targets on rotation days, when
        # most equity sits in the positions about to be sold.
        def _px(sym: str) -> float:
            p = prices.get(sym)
            return 0.0 if p is None or pd.isna(p) or p <= 0 else float(p)

        holdings_value = float(sum(qty * _px(sym) for sym, qty in positions.items()))
        total_equity = cash + holdings_value
        target_value = target_weights * total_equity
        target_shares = (target_value / prices.replace(0, float("nan"))).fillna(0)
        target_shares = target_shares.astype(int)
        target_shares = (target_shares // self.min_volume) * self.min_volume

        # Limit to max_positions by target WEIGHT, not by share count:
        # share counts scale with 1/price, so ranking shares would favor
        # cheap stocks over the strategy's actual preferences.
        top = target_weights.abs().nlargest(self.max_positions)
        target_shares = target_shares.where(
            target_shares.index.isin(top.index), 0
        )

        # Determine trades
        all_symbols = set(target_shares.index) | set(positions.keys())
        buys: list[dict] = []
        sells: list[dict] = []
        errors: list[dict] = []

        buy_symbols = sorted(
            (s for s in all_symbols if target_shares.get(s, 0) - positions.get(s, 0) > 0),
            key=lambda s: target_weights.get(s, 0),
            reverse=True,
        )
        sell_symbols = sorted(
            (s for s in all_symbols if target_shares.get(s, 0) - positions.get(s, 0) < 0),
            key=lambda s: target_weights.get(s, 0),
        )

        remaining_cash = cash
        # Sells run first: on rotation days (sell A to buy B) the
        # proceeds must enter the buy budget, otherwise buying power is
        # understated and cash sits idle. Proceeds are net of estimated
        # slippage; commission and stamp duty settle at the broker.
        for sym in sell_symbols:
            diff = target_shares.get(sym, 0) - positions.get(sym, 0)
            current = positions.get(sym, 0)
            price = prices.get(sym)
            if price is None or pd.isna(price) or price <= 0:
                errors.append({"symbol": sym, "reason": "no price", "side": "sell"})
                continue

            # T+1: shares bought today cannot be sold today
            sellable = current - bought_today.get(sym, 0)
            qty = min(abs(diff), max(0, sellable))
            qty = (qty // self.min_volume) * self.min_volume
            if qty < self.min_volume:
                if abs(diff) >= self.min_volume:
                    errors.append({"symbol": sym, "reason": "T+1 locked", "side": "sell"})
                continue

            trade_info = {"symbol": sym, "quantity": qty, "price": price}
            if dry_run:
                logger.info("[DRY RUN] Sell %s: %d @ %.2f", sym, qty, price)
            else:
                result = self.trader.sell(sym, price, qty)
                if not result.success:
                    errors.append({"symbol": sym, "reason": result.message, "side": "sell"})
                    continue
                logger.info("Sell %s: %d @ %.2f — %s", sym, qty, price, result.message)
            sells.append(trade_info)
            remaining_cash += qty * price * (1 - self.slippage)

        for sym in buy_symbols:
            diff = target_shares.get(sym, 0) - positions.get(sym, 0)
            price = prices.get(sym)
            if price is None or pd.isna(price) or price <= 0:
                errors.append({"symbol": sym, "reason": "no price", "side": "buy"})
                continue

            affordable = int(remaining_cash // price // self.min_volume) * self.min_volume
            qty = min(diff, affordable)
            if qty < self.min_volume:
                errors.append({"symbol": sym, "reason": "insufficient cash", "side": "buy"})
                continue

            remaining_cash -= qty * price
            trade_info = {"symbol": sym, "quantity": qty, "price": price}
            if dry_run:
                logger.info("[DRY RUN] Buy %s: %d @ %.2f", sym, qty, price)
            else:
                result = self.trader.buy(sym, price, qty)
                if not result.success:
                    errors.append({"symbol": sym, "reason": result.message, "side": "buy"})
                    continue
                logger.info("Buy %s: %d @ %.2f — %s", sym, qty, price, result.message)
            buys.append(trade_info)

        return {
            "buys": buys,
            "sells": sells,
            "errors": errors,
            "summary": {
                "buy_count": len(buys),
                "sell_count": len(sells),
                "error_count": len(errors),
                "total_trades": len(buys) + len(sells),
            },
        }

    def _read_positions(self) -> dict[str, int]:
        """Read positions from the trader. A failure here propagates (abort)."""
        return {p.symbol: p.quantity for p in self.trader.get_positions()}
