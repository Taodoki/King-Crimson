"""Returns decomposition and Brinson attribution."""

import pandas as pd
import numpy as np


class ReturnsDecomposition:
    """Decompose portfolio returns into component effects.

    Layers:
    1. Absolute decomposition: weighted contribution of each position
    2. Brinson-style (requires sector labels)
    3. Calendar returns (monthly matrix, annual breakdown)
    4. Rolling returns over various windows
    """

    def __init__(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series | None = None,
    ):
        self.returns = portfolio_returns.dropna()
        self.benchmark = (
            benchmark_returns.dropna() if benchmark_returns is not None else None
        )

    def absolute_decomposition(
        self,
        weights: pd.DataFrame,
        asset_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute contribution of each position per period.

        Parameters
        ----------
        weights : pd.DataFrame
            Date x asset weights (sum to 1 per date).
        asset_returns : pd.DataFrame
            Date x asset returns.

        Returns
        -------
        pd.DataFrame
            Contribution of each asset per date.
        """
        aligned_w = weights.reindex_like(asset_returns).fillna(0)
        return aligned_w * asset_returns

    def monthly_returns(self) -> pd.DataFrame:
        """Matrix: year x month -> portfolio return."""
        df = pd.DataFrame({"returns": self.returns})
        df["year"] = df.index.year
        df["month"] = df.index.month
        monthly = df.groupby(["year", "month"])["returns"].apply(
            lambda x: (1 + x).prod() - 1
        )
        return monthly.unstack(level="month")

    def annual_returns(self) -> pd.Series:
        df = pd.DataFrame({"returns": self.returns})
        df["year"] = df.index.year
        return df.groupby("year")["returns"].apply(lambda x: (1 + x).prod() - 1)

    def rolling_returns(self, window: int = 252) -> pd.Series:
        return (1 + self.returns).rolling(window).apply(
            lambda x: x.prod() - 1, raw=True
        )

    def benchmark_relative_stats(self) -> dict:
        """Return alpha, beta, and tracking error vs benchmark."""
        if self.benchmark is None:
            return {}
        aligned = pd.concat(
            [self.returns, self.benchmark], axis=1, join="inner"
        ).dropna()
        if len(aligned) < 10:
            return {}
        col = aligned.columns
        p_ret = aligned[col[0]]
        b_ret = aligned[col[1]]
        cov = np.cov(p_ret, b_ret)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
        alpha = (p_ret.mean() - beta * b_ret.mean()) * 252
        te = (p_ret - b_ret).std() * np.sqrt(252)
        return {
            "alpha": alpha,
            "beta": beta,
            "tracking_error": te,
            "correlation": p_ret.corr(b_ret),
        }
