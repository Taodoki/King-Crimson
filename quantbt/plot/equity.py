"""Equity curve and drawdown visualization (Plotly HTML)."""

import pandas as pd
import numpy as np


def plot_equity_drawdown(equity_curve: pd.DataFrame, path: str = "equity_curve.html") -> str:
    """Create a two-panel figure: equity curve (top) + drawdown (bottom).

    Saves interactive HTML via Plotly.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
    )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=equity_curve["date"],
            y=equity_curve["cumulative_returns"],
            name="Portfolio",
            line=dict(color="#1f77b4", width=2),
        ),
        row=1, col=1,
    )

    # Benchmark (if available)
    if "benchmark_returns" in equity_curve.columns:
        fig.add_trace(
            go.Scatter(
                x=equity_curve["date"],
                y=equity_curve["benchmark_returns"],
                name="Benchmark",
                line=dict(color="#ff7f0e", width=1.5, dash="dash"),
            ),
            row=1, col=1,
        )

    # Horizontal line at 1.0
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", row=1, col=1)

    # Drawdown
    cum = equity_curve["cumulative_returns"]
    peak = cum.expanding().max()
    drawdown = (cum - peak) / peak

    fig.add_trace(
        go.Scatter(
            x=equity_curve["date"],
            y=drawdown * 100,
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="#d62728", width=1),
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

    fig.update_layout(
        title="Backtest Results — Equity Curve & Drawdown",
        template="plotly_white",
        height=600,
        showlegend=True,
    )
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    fig.write_html(path)
    return path


def plot_monthly_returns_heatmap(monthly_returns: pd.DataFrame, path: str = "monthly_returns.html") -> str:
    """Create a monthly returns heatmap."""
    import plotly.graph_objects as go

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig = go.Figure(data=go.Heatmap(
        z=monthly_returns.values * 100,
        x=[month_labels[m - 1] for m in monthly_returns.columns],
        y=monthly_returns.index.astype(str),
        colorscale="RdYlGn",
        zmid=0,
        text=monthly_returns.values.round(4).astype(str),
        texttemplate="%{text}%",
        hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        title="Monthly Returns (%)",
        template="plotly_white",
        height=400,
        xaxis_title="Month",
        yaxis_title="Year",
    )

    fig.write_html(path)
    return path
