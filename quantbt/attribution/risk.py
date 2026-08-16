"""Comprehensive risk and return metrics for portfolio analysis."""

import pandas as pd
import numpy as np
from scipy import stats


class RiskAnalyzer:
    """Compute risk and return metrics from a portfolio return series.

    Includes:
    - Return metrics (total, annualized, rolling)
    - Risk metrics (volatility, Sharpe/Sortino/Calmar, VaR, CVaR)
    - Drawdown analysis (max, avg, duration)
    - Distribution analysis (skew, kurtosis, win rate)
    - Rolling metrics (Sharpe, volatility, VaR)
    """

    def __init__(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        trading_days: int = 252,
    ):
        self.returns = returns.dropna()
        self.rf = risk_free_rate
        self.trading_days = trading_days

    def summary(self) -> dict:
        """Return dict of all key metrics."""
        return {
            "total_return": self.total_return(),
            "annualized_return": self.annualized_return(),
            "annualized_volatility": self.annualized_volatility(),
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "calmar_ratio": self.calmar_ratio(),
            "max_drawdown": self.max_drawdown(),
            "max_drawdown_duration": self.max_drawdown_duration(),
            "var_95": self.var(0.95),
            "cvar_95": self.cvar(0.95),
            "win_rate": self.win_rate(),
            "profit_factor": self.profit_factor(),
            "skewness": self.skewness(),
            "kurtosis": self.kurtosis(),
        }

    def total_return(self) -> float:
        return (1 + self.returns).prod() - 1

    def annualized_return(self) -> float:
        n = len(self.returns)
        if n == 0:
            return 0.0
        total = 1 + self.total_return()
        if total <= 0:
            return float("nan")
        return total ** (self.trading_days / n) - 1

    def annualized_volatility(self) -> float:
        return self.returns.std() * np.sqrt(self.trading_days)

    def sharpe_ratio(self) -> float:
        excess = self.returns.mean() * self.trading_days - self.rf
        vol = self.annualized_volatility()
        if vol < 1e-12:
            return float("inf") if excess > 0 else (float("-inf") if excess < 0 else 0.0)
        return excess / vol

    def sortino_ratio(self) -> float:
        """Return / downside deviation, downside measured against daily rf.

        Downside deviation is sqrt(mean(min(r - MAR, 0)^2)) over ALL
        observations (positive days contribute 0), the standard Sortino
        definition. Using the std of negative returns only would measure
        dispersion around the loss mean instead of loss depth.
        """
        excess = self.returns.mean() * self.trading_days - self.rf
        mar_daily = self.rf / self.trading_days
        shortfall = (self.returns - mar_daily).clip(upper=0.0) ** 2
        downside_ann = np.sqrt(shortfall.mean()) * np.sqrt(self.trading_days)
        if downside_ann < 1e-12:
            if excess > 0:
                return float("inf")
            if excess < 0:
                return float("-inf")
            return 0.0
        return excess / downside_ann

    def calmar_ratio(self) -> float:
        mdd = abs(self.max_drawdown())
        return self.annualized_return() / mdd if mdd > 0 else 0.0

    def max_drawdown(self) -> float:
        cum = (1 + self.returns).cumprod()
        peak = cum.expanding().max()
        dd = (cum - peak) / peak
        return dd.min()

    def max_drawdown_duration(self) -> int:
        cum = (1 + self.returns).cumprod()
        peak = cum.expanding().max()
        dd = (cum - peak) / peak
        is_drawdown = dd < 0
        durations = []
        current = 0
        for val in is_drawdown:
            if val:
                current += 1
            else:
                if current > 0:
                    durations.append(current)
                current = 0
        if current > 0:
            durations.append(current)
        return max(durations) if durations else 0

    def var(self, confidence: float = 0.95) -> float:
        return np.percentile(self.returns, (1 - confidence) * 100)

    def cvar(self, confidence: float = 0.95) -> float:
        threshold = self.var(confidence)
        return self.returns[self.returns <= threshold].mean()

    def win_rate(self) -> float:
        return (self.returns > 0).sum() / len(self.returns)

    def profit_factor(self) -> float:
        gains = self.returns[self.returns > 0].sum()
        losses = abs(self.returns[self.returns < 0].sum())
        return gains / losses if losses > 0 else float("inf")

    def skewness(self) -> float:
        return float(self.returns.skew())

    def kurtosis(self) -> float:
        return float(self.returns.kurtosis())

    def rolling_sharpe(self, window: int = 252) -> pd.Series:
        roll = self.returns.rolling(window)
        excess = roll.mean() * self.trading_days - self.rf
        vol = roll.std() * np.sqrt(self.trading_days)
        return excess / vol.replace(0, np.nan)

    def rolling_volatility(self, window: int = 20) -> pd.Series:
        return self.returns.rolling(window).std() * np.sqrt(self.trading_days)

    def rolling_var(self, window: int = 60, confidence: float = 0.95) -> pd.Series:
        return self.returns.rolling(window).quantile(1 - confidence)

    def monthly_returns(self) -> pd.DataFrame:
        df = pd.DataFrame({"returns": self.returns})
        df["year"] = df.index.year
        df["month"] = df.index.month
        monthly = df.groupby(["year", "month"])["returns"].apply(
            lambda x: (1 + x).prod() - 1
        )
        return monthly.unstack(level="month")

    def drawdown_table(self) -> pd.DataFrame:
        """List all drawdown episodes with peak, trough, duration."""
        cum = (1 + self.returns).cumprod()
        running_max = cum.expanding().max()
        dd = (cum - running_max) / running_max

        episodes = []
        in_dd = False
        start = None

        for i in range(len(dd)):
            if dd.iloc[i] < -1e-6 and not in_dd:
                in_dd = True
                start = dd.index[i]
            elif dd.iloc[i] >= -1e-6 and in_dd:
                in_dd = False
                if start is not None:
                    trough = dd[start:dd.index[i]].min()
                    episodes.append({
                        "start": start,
                        "end": dd.index[i],
                        "max_drawdown": trough,
                        "duration": len(dd[start:dd.index[i]]),
                    })
        if in_dd:
            trough = dd.loc[start:].min()
            episodes.append({
                "start": start,
                "end": dd.index[-1],
                "max_drawdown": trough,
                "duration": len(dd.loc[start:]),
            })

        return pd.DataFrame(episodes).sort_values("max_drawdown")