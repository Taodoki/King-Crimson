"""Attribution report — aggregated analysis and terminal output."""

from quantbt.attribution.risk import RiskAnalyzer
from quantbt.attribution.returns import ReturnsDecomposition


class AttributionReport:
    """Aggregated attribution analysis for a backtest result."""

    def __init__(self, result):
        returns = result.portfolio_returns
        self.risk = RiskAnalyzer(returns)
        self.returns_decomp = ReturnsDecomposition(returns)
        self.trades = result.trades
        self.equity_curve = result.equity_curve

    def summary_dict(self) -> dict:
        return self.risk.summary()
