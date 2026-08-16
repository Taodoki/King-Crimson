"""Baostock data source for A-share market data.

Baostock (http://baostock.com) is a free, registration-free A-share data source.
No token required. Date format: YYYY-MM-DD.
Stock code format: sh.600519 or sz.000001
"""

import pandas as pd
from datetime import date

from quantbt.data.source import DataSource


def _to_baostock_code(symbol: str) -> str:
    """Convert 000001 to sz.000001, 600519 to sh.600519."""
    clean = symbol.replace(".SH", "").replace(".SZ", "").strip()
    if clean.startswith(("600", "601", "603", "605", "688", "689")):
        return f"sh.{clean}"
    return f"sz.{clean}"


class BaostockSource(DataSource):
    """Fetch A-share daily data via baostock.

    Data includes: date, open, high, low, close, volume, adj_close.
    No registration or token needed.
    """

    def fetch(
        self,
        symbols: list[str],
        start: str | date,
        end: str | date,
    ) -> dict[str, pd.DataFrame]:
        import baostock as bs

        start_str = str(start) if isinstance(start, date) else start
        end_str = str(end) if isinstance(end, date) else end

        bs.login()
        result = {}

        for sym in symbols:
            bs_code = _to_baostock_code(sym)
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume",
                start_date=start_str,
                end_date=end_str,
                frequency="d",
                adjustflag="2",  # 1=不复权, 2=前复权, 3=后复权
            )

            rows = []
            while rs.next():
                row = rs.get_row_data()
                rows.append(row)

            if not rows:
                import warnings
                warnings.warn(f"No data returned for {sym}")
                continue

            df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
            df["date"] = pd.to_datetime(df["date"])
            # baostock volume is already in shares (unit-normalized with AKShare).
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.sort_values("date")
            df["adj_close"] = df["close"].values
            result[sym] = df

        bs.logout()
        return result


BENCHMARK_ALIASES = {
    "csi300": "sh.000300",
    "hs300": "sh.000300",
    "沪深300": "sh.000300",
    "000300": "sh.000300",
    "csi500": "sh.000905",
    "中证500": "sh.000905",
    "000905": "sh.000905",
    "上证指数": "sh.000001",
    "sse": "sh.000001",
    "创业板指": "sz.399006",
    "399006": "sz.399006",
}


def resolve_benchmark_code(benchmark: str) -> str:
    """Resolve a friendly benchmark name to a baostock index code."""
    key = benchmark.strip().lower()
    if key in BENCHMARK_ALIASES:
        return BENCHMARK_ALIASES[key]
    if key.startswith(("sh.", "sz.")):
        return key
    raise ValueError(
        f"Unknown benchmark {benchmark!r}; use a baostock code "
        "(e.g. 'sh.000300') or an alias like 'csi300'."
    )


def fetch_index_returns(
    code: str,
    start: str | date,
    end: str | date,
) -> pd.Series:
    """Fetch daily close-to-close returns for a benchmark index.

    ``code`` must already be a baostock index code (e.g. ``sh.000300``).
    Returns a Series of daily returns indexed by date.
    """
    import baostock as bs

    start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_str = pd.Timestamp(end).strftime("%Y-%m-%d")

    bs.login()
    try:
        rs = bs.query_history_k_data_plus(
            code,
            "date,close",
            start_date=start_str,
            end_date=end_str,
            frequency="d",
            adjustflag="3",
        )
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
    finally:
        bs.logout()

    if not rows:
        return pd.Series(dtype=float)

    df = pd.DataFrame(rows, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date")
    df = df.set_index("date")["close"]
    return df.pct_change().fillna(0.0)