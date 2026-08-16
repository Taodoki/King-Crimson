"""Rich table formatting for terminal output."""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

POSITIVE = "green"
NEGATIVE = "red"
NEUTRAL = "white"
HEADER = "cyan"
ACCENT = "yellow"


def metric_table(title: str, metrics: dict[str, tuple]) -> Table:
    """Build a two-column metric table.

    metrics format: {label: (value, is_good)}
    is_good: True = green, False = red, None = neutral
    """
    table = Table(title=title, box=box.ROUNDED, title_style="bold")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", justify="right")

    for label, (value, is_good) in metrics.items():
        color = POSITIVE if is_good else NEGATIVE if is_good is False else NEUTRAL
        value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
        table.add_row(label, f"[{color}]{value_str}[/]")

    return table


def performance_summary_table(risk) -> Table:
    """Build the main performance summary table from a RiskAnalyzer."""
    s = risk.summary()
    table = Table(title="Performance Summary", box=box.ROUNDED, title_style="bold")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", justify="right")

    rows = [
        ("Total Return", s["total_return"], s["total_return"] > 0),
        ("Annualized Return", s["annualized_return"], s["annualized_return"] > 0),
        ("Annualized Vol", s["annualized_volatility"], s["annualized_volatility"] < 0.3),
        ("Sharpe Ratio", s["sharpe_ratio"], s["sharpe_ratio"] > 1),
        ("Sortino Ratio", s["sortino_ratio"], s["sortino_ratio"] > 1),
        ("Calmar Ratio", s["calmar_ratio"], s["calmar_ratio"] > 1),
        ("Max Drawdown", s["max_drawdown"], s["max_drawdown"] > -0.2),
        ("VaR (95%)", s["var_95"], s["var_95"] > -0.02),
        ("CVaR (95%)", s["cvar_95"], s["cvar_95"] > -0.03),
        ("Win Rate", s["win_rate"], s["win_rate"] > 0.5),
        ("Profit Factor", s["profit_factor"], s["profit_factor"] > 1.5),
    ]

    for label, value, is_good in rows:
        color = POSITIVE if is_good else NEGATIVE if is_good is False else NEUTRAL
        value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
        table.add_row(label, f"[{color}]{value_str}[/]")

    return table


def trade_analysis_table(trades_df) -> Table:
    """Build a trade analysis table."""
    if trades_df.empty:
        table = Table(title="Trade Analysis", box=box.ROUNDED)
        table.add_column("Info", style="yellow")
        table.add_row("No trades executed")
        return table

    table = Table(title="Trade Summary", box=box.ROUNDED, title_style="bold")
    table.add_column("Metric", style="bold white")
    table.add_column("Value", justify="right")

    total_trades = len(trades_df)
    buys = len(trades_df[trades_df["side"] == "buy"])
    sells = len(trades_df[trades_df["side"] == "sell"])
    avg_cost = trades_df["total_cost"].mean()

    table.add_row("Total Trades", str(total_trades))
    table.add_row("Buys", str(buys))
    table.add_row("Sells", str(sells))
    table.add_row("Avg Cost/Trade", f"{avg_cost:.2f}")

    return table


def print_backtest_summary(result) -> None:
    """Print full backtest summary to terminal."""
    from quantbt.attribution.risk import RiskAnalyzer

    ra = RiskAnalyzer(result.portfolio_returns)

    console.print()
    console.print(
        Panel(
            f"[bold]Strategy:[/] {result.strategy_name}  |  "
            f"[bold]Trades:[/] {len(result.trades)}  |  "
            f"[bold]Period:[/] {result.equity_curve['date'].iloc[0].date()} → "
            f"{result.equity_curve['date'].iloc[-1].date()}",
            title="Backtest Complete",
            border_style="cyan",
        )
    )
    console.print(performance_summary_table(ra))
    console.print(trade_analysis_table(result.trades))

    if getattr(result, "benchmark_name", None):
        excess = getattr(result, "benchmark_excess_return", float("nan"))
        te = getattr(result, "benchmark_tracking_error", float("nan"))

        def _pct(v) -> str:
            return "n/a" if v != v else f"{v:.2%}"

        console.print(
            Panel(
                f"[bold]Benchmark:[/] {result.benchmark_name}  |  "
                f"[bold]Excess Return:[/] {_pct(excess)}  |  "
                f"[bold]Tracking Error:[/] {_pct(te)}",
                title="Benchmark Comparison",
                border_style="yellow",
            )
        )
    console.print()
