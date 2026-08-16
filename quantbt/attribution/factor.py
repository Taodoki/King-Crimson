"""Factor exposure analysis using linear factor models."""

import pandas as pd
import numpy as np
import statsmodels.api as sm


class FactorAnalyzer:
    """Estimate factor exposures (betas) for the portfolio.

    Supports:
    - Custom factor returns (user-provided)
    - Market factor from universe (equal-weighted)
    - Simple Fama-French-style factor construction
    - OLS and rolling regression
    """

    def __init__(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame | None = None,
    ):
        self.portfolio_returns = portfolio_returns.dropna()
        self.factor_returns = factor_returns

    def fit(
        self,
        add_constant: bool = True,
    ) -> dict:
        """Estimate factor loadings via OLS regression.

        Returns
        -------
        dict with keys: 'loadings' (params), 'pvalues', 'rsquared',
                         'adjusted_rsquared', 'residuals'
        """
        if self.factor_returns is None:
            return {"error": "No factor returns provided"}

        aligned = pd.concat(
            [self.portfolio_returns, self.factor_returns], axis=1, join="inner"
        ).dropna()

        y = aligned.iloc[:, 0]
        X = aligned.iloc[:, 1:]
        if add_constant:
            X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()
        return {
            "loadings": model.params,
            "pvalues": model.pvalues,
            "rsquared": model.rsquared,
            "adjusted_rsquared": model.rsquared_adj,
            "residuals": model.resid,
            "nobs": model.nobs,
        }

    def rolling_fit(
        self,
        window: int = 252,
        min_periods: int = 60,
        add_constant: bool = True,
    ) -> pd.DataFrame:
        """Rolling-window factor regression.

        Returns
        -------
        pd.DataFrame of rolling betas with date index.
        """
        if self.factor_returns is None:
            return pd.DataFrame()

        aligned = pd.concat(
            [self.portfolio_returns, self.factor_returns], axis=1, join="inner"
        ).dropna()

        y = aligned.iloc[:, 0]
        X = aligned.iloc[:, 1:]
        if add_constant:
            X = sm.add_constant(X)

        results = []
        index = []
        for i in range(min_periods, len(y) + 1):
            yw = y.iloc[i - window : i] if i >= window else y.iloc[:i]
            Xw = X.iloc[i - window : i] if i >= window else X.iloc[:i]
            if len(yw) < min_periods:
                continue
            model = sm.OLS(yw, Xw).fit()
            results.append(model.params)
            index.append(y.index[i - 1])

        return pd.DataFrame(results, index=index)

    def build_market_factor(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Construct equal-weighted market factor from universe returns."""
        market = returns.mean(axis=1)
        return pd.DataFrame({"market": (1 + market).cumprod()})
