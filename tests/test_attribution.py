"""Tests for performance attribution."""

import pandas as pd
import numpy as np


def test_risk_analyzer_basic_metrics(sample_returns):
    from quantbt.attribution.risk import RiskAnalyzer

    ra = RiskAnalyzer(sample_returns)
    s = ra.summary()
    assert "total_return" in s
    assert "sharpe_ratio" in s
    assert "max_drawdown" in s
    assert s["total_return"] > -1.0  # sensible range
    assert s["win_rate"] > 0 and s["win_rate"] < 1


def test_risk_analyzer_sharpe():
    from quantbt.attribution.risk import RiskAnalyzer

    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    # Perfectly constant positive returns
    rets = pd.Series(np.full(252, 0.001), index=dates)
    ra = RiskAnalyzer(rets, risk_free_rate=0.0)
    assert ra.sharpe_ratio() > 0


def test_risk_analyzer_max_drawdown():
    from quantbt.attribution.risk import RiskAnalyzer

    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    # Drop 50% then recover
    rets = pd.Series(index=dates, dtype=np.float64)
    rets.iloc[:50] = -0.01
    rets.iloc[50:] = 0.01
    ra = RiskAnalyzer(rets)
    assert ra.max_drawdown() < 0


def test_monthly_returns(sample_returns):
    from quantbt.attribution.risk import RiskAnalyzer

    ra = RiskAnalyzer(sample_returns)
    monthly = ra.monthly_returns()
    assert isinstance(monthly, pd.DataFrame)
    assert monthly.shape[1] <= 12  # at most 12 months


def test_drawdown_table(sample_returns):
    from quantbt.attribution.risk import RiskAnalyzer

    ra = RiskAnalyzer(sample_returns)
    dd = ra.drawdown_table()
    assert isinstance(dd, pd.DataFrame)
    if not dd.empty:
        assert "max_drawdown" in dd.columns
        assert dd["max_drawdown"].min() <= 0


def test_returns_decomposition(sample_returns):
    from quantbt.attribution.returns import ReturnsDecomposition

    rd = ReturnsDecomposition(sample_returns)
    monthly = rd.monthly_returns()
    assert isinstance(monthly, pd.DataFrame)


def test_factor_analyzer():
    from quantbt.attribution.factor import FactorAnalyzer

    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    port = pd.Series(np.random.randn(252) * 0.01, index=dates)
    factors = pd.DataFrame({
        "market": np.random.randn(252) * 0.008,
        "size": np.random.randn(252) * 0.005,
    }, index=dates)

    fa = FactorAnalyzer(port, factors)
    result = fa.fit()
    assert "loadings" in result
    assert "rsquared" in result


def test_factor_rolling_fit_index_aligned():
    from quantbt.attribution.factor import FactorAnalyzer

    n = 400
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    port = pd.Series(np.random.randn(n) * 0.01, index=dates)
    factors = pd.DataFrame({
        "market": np.random.randn(n) * 0.008,
        "size": np.random.randn(n) * 0.005,
    }, index=dates)

    fa = FactorAnalyzer(port, factors)
    result = fa.rolling_fit(window=252, min_periods=60)

    assert len(result) == n - 60 + 1
    assert result.index[0] == dates[59]
    assert result.index[-1] == dates[-1]
