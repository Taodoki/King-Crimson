"""Daily scheduler — runs the full live trading loop once per trading day."""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from quantbt.strategy.base import Strategy
    from quantbt.trader.tonghuashun import TonghuashunTrader
    from quantbt.trader.executor import LiveExecutor

logger = logging.getLogger("quantbt.trader")


class DailyScheduler:
    """Orchestrates the daily live trading workflow.

    Daily flow:
    1. Fetch latest market data (AKShare)
    2. Generate strategy signals
    3. Connect to 同花顺
    4. Execute rebalance
    5. Log results
    6. Generate daily report
    """

    def __init__(
        self,
        strategy: Strategy,
        stock_pool: list[str],
        trader: TonghuashunTrader,
        executor: LiveExecutor,
        initial_capital: float = 1_000_000,
        rebalance_freq: str = "daily",
    ):
        self.strategy = strategy
        self.stock_pool = stock_pool
        self.trader = trader
        self.executor = executor
        self.config = {
            "initial_capital": initial_capital,
            "rebalance_freq": rebalance_freq,
        }
        self.history: list[dict] = []

    def run_once(
        self,
        today: date | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Execute one full daily cycle.

        Args:
            today: Trading date (default: today)
            dry_run: If True, log without placing real orders

        Returns:
            Dict with execution results
        """
        today = today or date.today()
        start = (today - timedelta(days=400)).isoformat()
        end = today.isoformat()

        # Step 1: Fetch data
        logger.info("Fetching data for %d stocks from AKShare...", len(self.stock_pool))
        data = self._fetch_data(start, end)
        if data is None or data.empty:
            logger.error("Failed to fetch market data")
            return {"error": "no_data", "date": today}

        # Step 2: Generate signals
        logger.info("Generating signals with %s...", self.strategy.name)
        signals = self.strategy.generate_signals(data)
        latest_signal = signals.iloc[-1].fillna(0)
        latest_prices = self._get_prices(data)

        if latest_prices is None:
            logger.error("Failed to get latest prices")
            return {"error": "no_prices", "date": today}

        # Step 3: Resolve available cash and current positions
        if dry_run:
            cash = self.config["initial_capital"]
            positions = {}
            bought_today = {}
            logger.info("Using initial capital %.2f as cash (dry_run)", cash)
        else:
            # 实盘已移除，trader 为 stub；此处保留 fail-closed 读取路径，
            # 未来接入真实券商时 get_positions 失败会直接中止而非按空仓继续。
            balance = self.trader.get_balance()
            cash = balance.cash
            positions = {p.symbol: p.quantity for p in self.trader.get_positions()}
            bought_today = {}
            logger.info("Available cash: %.2f, positions: %d", cash, len(positions))

        # Step 4: Execute trades
        logger.info("Executing trades (dry_run=%s)...", dry_run)

        result = self.executor.execute(
            target_weights=latest_signal,
            prices=latest_prices,
            cash=cash,
            positions=positions,
            bought_today=bought_today,
            dry_run=dry_run,
        )

        # Step 5: Record
        record = {
            "date": today,
            "strategy": self.strategy.name,
            "signals": latest_signal.to_dict(),
            "execution": result,
            "dry_run": dry_run,
        }
        self.history.append(record)

        # Step 6: Log summary
        summary = result["summary"]
        logger.info(
            "Execution complete — buys: %d, sells: %d, errors: %d",
            summary["buy_count"],
            summary["sell_count"],
            summary["error_count"],
        )

        # Disconnect 同花顺 to avoid keeping the window open
        if self.trader.connected:
            self.trader.disconnect()

        return record

    def run_month(
        self,
        dry_run: bool = False,
        max_days: int = 22,
    ) -> list[dict]:
        """Run daily for approximately one month.

        In persistent mode, sleeps until next trading day.
        For non-persistent (scheduled) mode, call run_once daily via task scheduler.
        """
        results = []
        for day in range(max_days):
            today = date.today() + timedelta(days=day)
            if today.weekday() >= 5:
                logger.info("%s is weekend, skipping", today)
                continue
            result = self.run_once(today=today, dry_run=dry_run)
            results.append(result)
            self._print_daily_report(result)

        return results

    def _fetch_data(self, start: str, end: str) -> pd.DataFrame | None:
        """Fetch OHLCV data for stock pool via Baostock."""
        try:
            from quantbt.data.baostock_source import BaostockSource

            source = BaostockSource()
            raw = source.fetch(self.stock_pool, start, end)
            if not raw:
                return None

            from quantbt.api import Backtest
            bt = Backtest(strategy=self.strategy, symbols=[])
            return bt._assemble_data(raw)
        except Exception as e:
            logger.error("Data fetch error: %s", e)
            return None

    def _get_prices(self, data: pd.DataFrame) -> pd.Series | None:
        """Extract latest close prices from MultiIndex DataFrame."""
        try:
            if isinstance(data.columns, pd.MultiIndex):
                return data.iloc[-1].xs("close", level=1)
            return data.iloc[-1]
        except Exception as e:
            logger.error("Price extraction error: %s", e)
            return None

    def _print_daily_report(self, record: dict) -> None:
        """Print a human-readable daily execution report."""
        dry = "[DRY RUN] " if record["dry_run"] else ""
        print(f"\n{'='*60}")
        print(f"{dry}Daily Report — {record['date']}")
        print(f"Strategy: {record['strategy']}")
        print(f"{'='*60}")

        signals = record.get("signals", {})
        active = {k: v for k, v in signals.items() if abs(v) > 0.01}
        if active:
            print(f"Signals ({len(active)} active):")
            for sym, w in sorted(active.items(), key=lambda x: -abs(x[1])):
                print(f"  {sym}: {w:.2%}")

        execution = record.get("execution", {})
        summary = execution.get("summary", {})
        print(f"\nTrades: {summary.get('buy_count', 0)} buys, "
              f"{summary.get('sell_count', 0)} sells, "
              f"{summary.get('error_count', 0)} errors")

        for side in ("buys", "sells"):
            for t in execution.get(side, []):
                print(f"  {side}: {t['symbol']} {t['quantity']} @ {t['price']:.2f}")

        for e in execution.get("errors", []):
            print(f"  ERROR: {e['symbol']} — {e.get('reason', 'unknown')}")


def run_daily_scheduled(config: dict, dry_run: bool = False):
    """Entry point for scheduled daily execution (Windows Task Scheduler).

    This function:
    1. Loads the strategy from config
    2. Connects to 同花顺
    3. Runs one daily cycle
    4. Disconnects

    Expected to be called by Windows Task Scheduler each trading day at ~9:25.
    """
    import sys
    from pathlib import Path

    # Setup logging
    log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"live_{date.today().isoformat()}.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Build strategy
    from quantbt.strategy.momentum import TimeSeriesMomentum
    from quantbt.strategy.mean_reversion import ZScoreMeanReversion

    strategy_map = {
        "momentum": TimeSeriesMomentum,
        "mean_reversion": ZScoreMeanReversion,
    }
    strategy_cls = strategy_map.get(config["strategy_name"])
    if strategy_cls is None:
        raise ValueError(f"Unknown strategy: {config['strategy_name']}")
    strategy = strategy_cls(**config["strategy_params"])

    # Build trader & executor
    trader = TonghuashunTrader(exe_path=config["ths_exe_path"])
    executor = LiveExecutor(
        trader=trader,
        max_positions=config["max_positions"],
    )

    # Run
    scheduler = DailyScheduler(
        strategy=strategy,
        stock_pool=config["stock_pool"],
        trader=trader,
        executor=executor,
        initial_capital=config["initial_capital"],
        rebalance_freq=config["rebalance_freq"],
    )
    return scheduler.run_once(dry_run=dry_run)
