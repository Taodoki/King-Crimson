"""Public API — Backtest class (fluent interface)."""

from datetime import date
import warnings

import pandas as pd

from quantbt.strategy.base import Strategy
from quantbt.core.engine import BacktestEngine, BacktestResult
from quantbt.utils.rich_utils import print_backtest_summary


class Backtest:
    """User-facing backtesting API.

    Two construction styles:

    1. Constructor (concise):
        result = Backtest(
            strategy=TimeSeriesMomentum(60),
            symbols=['000001', '000002'],
            start='2020-01-01', end='2023-12-31',
        ).run()

    2. Fluent (expressive):
        result = (Backtest()
                  .with_strategy(TimeSeriesMomentum(60))
                  .with_symbols(['000001', '000002'])
                  .with_date_range('2020-01-01', '2023-12-31')
                  .with_capital(1_000_000)
                  .run())
    """

    def __init__(
        self,
        strategy: Strategy | None = None,
        symbols: list[str] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
        data: pd.DataFrame | None = None,
        initial_capital: float = 1_000_000,
        commission: float = 0.0003,
        stamp_duty: float | None = None,
        slippage: float = 0.001,
        min_commission: float = 5.0,
        rebalance_freq: str = "monthly",
        max_positions: int | None = 10,
        short_selling: bool = False,
        benchmark: str | None = None,
    ):
        self.strategy = strategy
        self.symbols = symbols
        self.start = start
        self.end = end
        self.data = data
        self.benchmark = benchmark
        self.config = {
            "initial_capital": initial_capital,
            "commission": commission,
            "stamp_duty": stamp_duty,  # None = historical rate (0.1% -> 0.05% on 2023-08-28)
            "slippage": slippage,
            "min_commission": min_commission,
            "rebalance_freq": rebalance_freq,
            "max_positions": max_positions,
            "short_selling": short_selling,
        }

    def with_strategy(self, strategy: Strategy) -> "Backtest":
        self.strategy = strategy
        return self

    def with_symbols(self, symbols: list[str]) -> "Backtest":
        self.symbols = symbols
        return self

    def with_date_range(self, start, end) -> "Backtest":
        self.start = start
        self.end = end
        return self

    def with_capital(self, capital: float) -> "Backtest":
        self.config["initial_capital"] = capital
        return self

    def with_costs(self, **costs) -> "Backtest":
        self.config.update(costs)
        return self

    def with_data(self, data: pd.DataFrame) -> "Backtest":
        self.data = data
        return self

    def with_rebalance(self, freq: str) -> "Backtest":
        self.config["rebalance_freq"] = freq
        return self

    def run(self, verbose: bool = True) -> BacktestResult:
        if self.data is None and self.symbols is not None:
            if verbose:
                print("Fetching data from Baostock...")
            from quantbt.data.baostock_source import BaostockSource

            source = BaostockSource()
            raw = source.fetch(self.symbols, self.start, self.end)
            self.data = self._assemble_data(raw)

        if verbose:
            print(f"Running backtest: {self.strategy.name}")
            period = f"{self.start} → {self.end}" if self.start else "custom"
            print(f"Period: {period}")

        engine = BacktestEngine(
            strategy=self.strategy,
            data=self.data,
            **self.config,
        )
        result = engine.run()

        if self.benchmark is not None:
            self._attach_benchmark(result)

        if verbose:
            print_backtest_summary(result)

        return result

    def _attach_benchmark(self, result: BacktestResult) -> None:
        """Fetch the benchmark index and attach relative metrics to ``result``.

        Benchmark is an optional diagnostic: if the index cannot be resolved or
        fetched, a warning is emitted and the backtest is returned unchanged.
        """
        from quantbt.data.baostock_source import (
            resolve_benchmark_code,
            fetch_index_returns,
        )

        if result.equity_curve.empty:
            return

        start = pd.Timestamp(result.equity_curve["date"].iloc[0])
        end = pd.Timestamp(result.equity_curve["date"].iloc[-1])

        try:
            code = resolve_benchmark_code(self.benchmark)
            bench_ret = fetch_index_returns(code, start, end)
        except Exception as e:  # noqa: BLE001 - benchmark is best-effort
            warnings.warn(
                f"Benchmark {self.benchmark!r} unavailable: {e}. "
                "Running without benchmark."
            )
            return

        if bench_ret.empty:
            warnings.warn(
                f"No benchmark data for {self.benchmark!r}. "
                "Running without benchmark."
            )
            return

        _compute_benchmark(result, bench_ret, code)

    def _assemble_data(self, raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine individual symbol DataFrames into MultiIndex format."""
        if not raw:
            raise ValueError("No data returned from data source")

        tickers = list(raw.keys())
        dfs = []
        for sym in tickers:
            df = raw[sym].set_index("date")[["open", "high", "low", "close", "volume", "adj_close"]]
            dfs.append(df)

        combined = pd.concat(
            dfs, axis=1, keys=tickers, names=["ticker", "field"]
        ).dropna(how="all")
        return combined


def _compute_benchmark(
    result: BacktestResult,
    bench_ret: pd.Series,
    code: str,
) -> None:
    """Align benchmark daily returns to the equity curve and attach metrics.

    Pure function (no I/O) so it can be unit-tested with synthetic data.
    """
    from quantbt.attribution.returns import ReturnsDecomposition

    eq = result.equity_curve
    aligned = bench_ret.reindex(pd.DatetimeIndex(eq["date"])).fillna(0.0)
    eq["benchmark_returns"] = (1 + aligned).cumprod().to_numpy()

    port = result.portfolio_returns
    result.benchmark_name = code
    result.benchmark_excess_return = (
        eq["cumulative_returns"].iloc[-1] - eq["benchmark_returns"].iloc[-1]
    )
    stats = ReturnsDecomposition(
        port, benchmark_returns=aligned
    ).benchmark_relative_stats()
    result.benchmark_stats = stats
    result.benchmark_tracking_error = stats.get("tracking_error", float("nan"))