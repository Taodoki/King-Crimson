"""AKShare data source for A-share market data."""

import time
import warnings

import pandas as pd
from datetime import date

from quantbt.data.source import DataSource

# Column mapping: AKShare Chinese names → English
COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "换手率": "turnover_rate",
}

REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

# 6-digit prefixes → exchange inference
SH_PREFIXES = ("600", "601", "603", "605", "688", "689")
SZ_PREFIXES = ("000", "001", "002", "003", "300", "301")


def _infer_exchange(symbol: str) -> str:
    """Infer exchange from A-share stock code prefix."""
    clean = symbol.replace(".SH", "").replace(".SZ", "").strip()
    if clean.startswith(SH_PREFIXES):
        return "SH"
    if clean.startswith(SZ_PREFIXES):
        return "SZ"
    raise ValueError(f"Cannot infer exchange for symbol: {symbol}")


class AKShareAStockSource(DataSource):
    """Fetch A-share daily data via AKShare with forward-adjusted prices.

    Uses ``ak.stock_zh_a_hist()`` with ``adjust="qfq"`` (前复权).
    """

    def __init__(self, delay: float = 0.5):
        self.delay = delay

    def fetch(
        self,
        symbols: list[str],
        start: str | date,
        end: str | date,
    ) -> dict[str, pd.DataFrame]:
        import akshare as ak

        start_str = str(start) if isinstance(start, date) else start
        end_str = str(end) if isinstance(end, date) else end

        result = {}
        for sym in symbols:
            try:
                exchange = _infer_exchange(sym)
                raw = ak.stock_zh_a_hist(
                    symbol=sym.replace(".SH", "").replace(".SZ", "").strip(),
                    period="daily",
                    start_date=start_str,
                    end_date=end_str,
                    adjust="qfq",
                )
            except Exception as e:
                warnings.warn(f"Failed to fetch {sym}: {e}")
                continue

            if raw is None or raw.empty:
                warnings.warn(f"No data returned for {sym}")
                continue

            df = raw.rename(columns=COLUMN_MAP)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

            # AKShare reports volume in 手 (lots); normalize to shares so the
            # broker's volume-limit math matches the baostock source (shares).
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * 100

            # AKShare with qfq returns adjusted prices → use close as adj_close
            df["adj_close"] = df["close"].values

            # Validate required columns
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                warnings.warn(f"{sym}: missing columns {missing}, skipping")
                continue

            result[sym] = df
            time.sleep(self.delay)

        return result