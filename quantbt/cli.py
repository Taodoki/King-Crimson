"""CLI entry points for quantbt."""

import click
from rich.console import Console
from rich.table import Table
from rich import box


@click.group()
def cli():
    """quantbt — quantitative backtesting for A-shares."""
    pass


@cli.command()
@click.option("--strategy", "-s", required=True,
              type=click.Choice(["momentum", "mean_reversion"]))
@click.option("--lookback", default=60, type=int)
@click.option("--symbols", required=True, help="Comma-separated stock codes")
@click.option("--start", required=True, help="YYYY-MM-DD")
@click.option("--end", required=True, help="YYYY-MM-DD")
@click.option("--capital", default=1000000, type=float)
@click.option("--output", "-o", default=None, help="Output directory for charts")
def run(strategy, lookback, symbols, start, end, capital, output):
    """Run a backtest and print results."""
    from quantbt.api import Backtest
    from quantbt.strategy.momentum import TimeSeriesMomentum
    from quantbt.strategy.mean_reversion import ZScoreMeanReversion

    symbol_list = [s.strip() for s in symbols.split(",")]

    strategy_map = {
        "momentum": TimeSeriesMomentum(lookback=lookback),
        "mean_reversion": ZScoreMeanReversion(window=lookback),
    }

    result = Backtest(
        strategy=strategy_map[strategy],
        symbols=symbol_list,
        start=start,
        end=end,
        initial_capital=capital,
    ).run(verbose=True)

    if output:
        from pathlib import Path
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        from quantbt.plot.equity import (
            plot_equity_drawdown,
            plot_monthly_returns_heatmap,
        )
        plot_equity_drawdown(result.equity_curve, str(out_dir / "equity_curve.html"))
        from quantbt.attribution.risk import RiskAnalyzer
        ra = RiskAnalyzer(result.portfolio_returns)
        monthly = ra.monthly_returns()
        if not monthly.empty:
            plot_monthly_returns_heatmap(monthly, str(out_dir / "monthly_returns.html"))
        click.echo(f"Charts saved to {out_dir}")


@cli.command()
def list_strategies():
    """List all available built-in strategies."""
    console = Console()
    table = Table(title="Available Strategies", box=box.ROUNDED)
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Parameters", style="yellow")
    table.add_row(
        "TimeSeriesMomentum",
        "Go long when N-day return > threshold",
        "lookback, threshold, vol_target",
    )
    table.add_row(
        "CrossSectionalMomentum",
        "Long top quantile by past return",
        "lookback, top_quantile, long_only",
    )
    table.add_row(
        "ZScoreMeanReversion",
        "Buy when z-score below entry threshold",
        "window, entry_z, exit_z, stop_loss",
    )
    console.print(table)


if __name__ == "__main__":
    cli()
