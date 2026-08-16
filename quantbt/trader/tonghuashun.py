"""同花顺人工调仓清单生成器 — 不接入真实券商。

本模块不依赖任何券商下单接口（原 easytrader 依赖已移除）。
它把策略信号转成一份人工可执行的调仓清单（委托单），由使用者自行在
券商客户端手动下单。真实下单不在此发生。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("quantbt.trader")


@dataclass
class Balance:
    cash: float = 0.0
    market_value: float = 0.0
    total_assets: float = 0.0
    frozen_cash: float = 0.0


@dataclass
class Position:
    symbol: str
    name: str
    quantity: int
    available: int
    cost_price: float
    current_price: float
    market_value: float
    profit: float


@dataclass
class OrderResult:
    success: bool
    message: str
    order_id: str = ""


@dataclass
class OrderStatus:
    order_id: str
    symbol: str
    name: str
    side: str  # buy / sell
    price: float
    quantity: int
    status: str  # 已报 / 部成 / 已成 / 已撤
    filled_quantity: int = 0


class TonghuashunTrader:
    """人工委托单生成器 — 保留实盘接口形状，但不做真实下单。

    每个 buy/sell 方法只打印一份委托单，供人工在同花顺/券商客户端手动执行。
    """

    def __init__(self, exe_path: str = "", delay: float = 0.5):
        self.exe_path = exe_path
        self.delay = delay
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self, exe_path: str | None = None, auto: bool = False) -> None:
        """标记为已连接。实盘接口已移除，此方法只设置状态并提示人工下单。"""
        self._connected = True
        logger.warning(
            "实盘接口已移除：本模块只生成人工委托单，不接入真实券商。请手动下单。"
        )

    def get_balance(self) -> Balance:
        """无真实账户，返回全 0 余额。"""
        return Balance()

    def get_positions(self) -> list[Position]:
        """无真实账户，返回空持仓。"""
        return []

    def buy(self, symbol: str, price: float, amount: int) -> OrderResult:
        """生成买入委托单。"""
        return self._print_ticket("买入", symbol, price, amount)

    def sell(self, symbol: str, price: float, amount: int) -> OrderResult:
        """生成卖出委托单。"""
        return self._print_ticket("卖出", symbol, price, amount)

    def market_buy(self, symbol: str, amount: int) -> OrderResult:
        """生成市价买入委托单。"""
        return self._print_ticket("市价买入", symbol, None, amount)

    def market_sell(self, symbol: str, amount: int) -> OrderResult:
        """生成市价卖出委托单。"""
        return self._print_ticket("市价卖出", symbol, None, amount)

    def get_orders(self) -> list[OrderStatus]:
        """无真实账户，返回空委托列表。"""
        return []

    def get_today_trades(self) -> list[dict]:
        """无真实账户，返回空当日成交列表。"""
        return []

    def cancel_order(self, order_id: str) -> OrderResult:
        """生成撤单提示。"""
        logger.info("人工撤单: %s", order_id)
        return OrderResult(
            success=True,
            message="撤单提示已生成，请手动撤单",
            order_id=order_id,
        )

    def disconnect(self) -> None:
        self._connected = False
        logger.info("已断开（实盘接口未启用）")

    def _print_ticket(
        self, side: str, symbol: str, price: float | None, amount: int
    ) -> OrderResult:
        """打印一份人工可执行的委托单，失败时 fail-closed 返回 success=False。"""
        try:
            price_text = "市价" if price is None else f"{price:.2f}"
            logger.info(
                "[委托单] %s %s %d 股 @ %s", side, symbol, amount, price_text
            )
            return OrderResult(
                success=True,
                message=f"模拟委托单已生成，请手动{side}",
            )
        except Exception as e:
            logger.error("%s委托单生成失败: %s", side, e)
            return OrderResult(success=False, message=str(e))
