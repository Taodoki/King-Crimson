"""Tests for backtesting engine."""

import pandas as pd
import numpy as np


def test_engine_returns_backtestresult(sample_multiindex_data, sample_strategy):
    from quantbt.core.engine import BacktestEngine, BacktestResult

    engine = BacktestEngine(
        strategy=sample_strategy,
        data=sample_multiindex_data,
        initial_capital=1_000_000,
        rebalance_freq="monthly",
    )
    result = engine.run()
    assert isinstance(result, BacktestResult)
    assert not result.equity_curve.empty
    assert "returns" in result.equity_curve.columns
    assert "cumulative_returns" in result.equity_curve.columns


def test_engine_produces_trades(sample_multiindex_data):
    from quantbt.strategy.momentum import TimeSeriesMomentum
    from quantbt.core.engine import BacktestEngine

    strategy = TimeSeriesMomentum(lookback=20, threshold=-1.0)
    engine = BacktestEngine(
        strategy=strategy,
        data=sample_multiindex_data,
        rebalance_freq="monthly",
    )
    result = engine.run()
    assert len(result.trades) > 0


def test_engine_tracks_portfolio_value(sample_multiindex_data, sample_strategy):
    from quantbt.core.engine import BacktestEngine

    engine = BacktestEngine(
        strategy=sample_strategy,
        data=sample_multiindex_data,
        initial_capital=1_000_000,
        rebalance_freq="monthly",
    )
    result = engine.run()
    equity = result.equity_curve
    assert not equity.empty
    assert "total" in equity.columns


def test_daily_rebalance(sample_multiindex_data, sample_strategy):
    from quantbt.core.engine import BacktestEngine

    engine = BacktestEngine(
        strategy=sample_strategy,
        data=sample_multiindex_data,
        rebalance_freq="daily",
    )
    result = engine.run()
    assert len(result.equity_curve) > 100


def test_config_preserved(sample_multiindex_data, sample_strategy):
    from quantbt.core.engine import BacktestEngine

    engine = BacktestEngine(
        strategy=sample_strategy,
        data=sample_multiindex_data,
        initial_capital=500_000,
        commission=0.0005,
    )
    result = engine.run()
    assert result.config["initial_capital"] == 500_000
    assert result.config["commission"] == 0.0005
