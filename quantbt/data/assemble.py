"""Assemble per-symbol DataFrames into the engine's MultiIndex format."""

import numpy as np
import pandas as pd


def assemble_data(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine individual symbol DataFrames into MultiIndex columns.

    Parameters
    ----------
    raw : dict[str, pd.DataFrame]
        Per-symbol DataFrames as returned by a DataSource.fetch(), each
        with a ``date`` column plus OHLCV/adj_close/raw_close fields.

    Returns
    -------
    pd.DataFrame
        Single DataFrame indexed by date with MultiIndex columns
        (ticker, field). This is the format the backtest engine and
        strategies consume.

    Fields missing from a source (e.g. ``raw_close`` in older data) are
    filled with NaN; the engine falls back to ``close`` for trading.
    """
    if not raw:
        raise ValueError("No data returned from data source")

    fields = ["open", "high", "low", "close", "volume", "adj_close", "raw_close"]
    tickers = list(raw.keys())
    dfs = []
    for sym in tickers:
        df = raw[sym].set_index("date")
        for f in fields:
            if f not in df.columns:
                df[f] = np.nan
        dfs.append(df[fields])

    combined = pd.concat(
        dfs, axis=1, keys=tickers, names=["ticker", "field"]
    ).dropna(how="all")
    return combined
