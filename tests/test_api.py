"""Tests for the public Backtest API surface and benchmark wiring."""

import pandas as pd
import pytest

from quantbt.core.engine import BacktestResult


def _make_result(periods: int = 30) -> BacktestResult:
    dates = pd.date_range("2023-01-01", periods=periods, freq="B")
    returns = pd.Series([0.0] + [0.02] * (periods - 1), index=dates)
    r = BacktestResult()
    r.strategy_name = "TestStrategy"
    r.trades = pd.DataFrame()
    r.equity_curve = pd.DataFrame({
        "date": dates,
        "returns": returns.values,
        "cumulative_returns": (1 + returns).cumprod().values,
    })
    return r


def test_backtestresult_summary_and_plot_equity(tmp_path):
    r = _make_result()

    r.summary()  # prints to terminal; must not raise

    path = r.plot_equity(str(tmp_path / "equity.html"))
    assert path.endswith("equity.html")
    assert (tmp_path / "equity.html").exists()


def test_compute_benchmark_alignment_and_metrics():
    from quantbt.api import _compute_benchmark

    r = _make_result()
    dates = r.equity_curve["date"]
    bench_ret = pd.Series([0.0] + [0.01] * (len(dates) - 1), index=dates)

    _compute_benchmark(r, bench_ret, "sh.000300")

    assert r.benchmark_name == "sh.000300"
    assert "benchmark_returns" in r.equity_curve.columns
    assert r.equity_curve["benchmark_returns"].iloc[0] == 1.0
    assert r.benchmark_excess_return > 0  # portfolio outpaces benchmark
    assert r.benchmark_tracking_error > 0


def test_resolve_benchmark_code():
    from quantbt.data.baostock_source import resolve_benchmark_code

    assert resolve_benchmark_code("csi300") == "sh.000300"
    assert resolve_benchmark_code("沪深300") == "sh.000300"
    assert resolve_benchmark_code("sh.000905") == "sh.000905"

    with pytest.raises(ValueError):
        resolve_benchmark_code("not-a-benchmark")
