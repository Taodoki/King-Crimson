"""Compare momentum vs mean reversion strategies."""

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

from quantbt.api import Backtest
from quantbt.strategy import TimeSeriesMomentum, ZScoreMeanReversion

# Generate sample data
np.random.seed(42)
dates = pd.date_range("2021-01-01", periods=504, freq="B")
tickers = ["000001", "000002", "000003"]
columns = pd.MultiIndex.from_product([tickers, ["open", "high", "low", "close", "volume", "adj_close"]])
vals = np.random.randn(504, 18).cumsum(0)
for i in range(3):
    vals[:, i * 6 + 3] += 100
    vals[:, i * 6 + 5] = vals[:, i * 6 + 3]
    vals[:, i * 6 + 4] = np.abs(vals[:, i * 6 + 4] * 100000 + 5000000)
data = pd.DataFrame(vals, index=dates, columns=columns)

# Run momentum
print("Running momentum strategy...")
mom_result = Backtest(
    strategy=TimeSeriesMomentum(lookback=60, threshold=-1.0),
    data=data,
).run(verbose=False)

# Run mean reversion
print("Running mean reversion strategy...")
mr_result = Backtest(
    strategy=ZScoreMeanReversion(window=20, entry_z=-2.0),
    data=data,
).run(verbose=False)

# Compare
console = Console()
table = Table(title="Strategy Comparison", box=Table.rounded_box)
table.add_column("Metric", style="bold")
table.add_column("Momentum", justify="right")
table.add_column("Mean Reversion", justify="right")

from quantbt.attribution.risk import RiskAnalyzer
mom_ra = RiskAnalyzer(mom_result.portfolio_returns)
mr_ra = RiskAnalyzer(mr_result.portfolio_returns)

mom_s = mom_ra.summary()
mr_s = mr_ra.summary()

for key in ["total_return", "annualized_return", "sharpe_ratio", "max_drawdown", "win_rate"]:
    mom_val = f"{mom_s[key]:.4f}" if isinstance(mom_s[key], float) else str(mom_s[key])
    mr_val = f"{mr_s[key]:.4f}" if isinstance(mr_s[key], float) else str(mr_s[key])
    table.add_row(key, mom_val, mr_val)

console.print(table)
