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

class AKShareAStockSource(DataSource):
    """Fetch A-share daily data via AKShare.

    Queries twice per symbol: ``adjust=""`` (不复权, for trading/valuation)
    and ``adjust="hfq"`` (后复权, for signals/total return). Columns:
    ``close`` = hfq close, ``raw_close`` = unadjusted close.
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
            code = sym.replace(".SH", "").replace(".SZ", "").strip()
            try:
                raw = ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start_str,
                    end_date=end_str,
                    adjust="",  # 不复权
                )
                hfq = ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start_str,
                    end_date=end_str,
                    adjust="hfq",  # 后复权
                )
            except Exception as e:
                warnings.warn(f"Failed to fetch {sym}: {e}")
                continue

            if raw is None or raw.empty or hfq is None or hfq.empty:
                warnings.warn(f"No data returned for {sym}")
                continue

            df = raw.rename(columns=COLUMN_MAP)
            df_hfq = hfq.rename(columns=COLUMN_MAP)[["date", "close"]].rename(
                columns={"close": "close_hfq"}
            )
            df = df.merge(df_hfq, on="date", how="outer").sort_values("date")
            df["date"] = pd.to_datetime(df["date"])

            # AKShare reports volume in 手 (lots); normalize to shares so the
            # broker's volume-limit math matches the baostock source (shares).
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * 100

            # close = 后复权(信号/总回报口径); raw_close = 不复权(撮合/估值口径)。
            df["raw_close"] = pd.to_numeric(df["close"], errors="coerce")
            df["close"] = pd.to_numeric(df["close_hfq"], errors="coerce")
            df = df.drop(columns=["close_hfq"])
            df["adj_close"] = df["close"].values
            for col in ["open", "high", "low", "close", "raw_close", "adj_close"]:
                df[col] = df[col].ffill()

            # Validate required columns
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                warnings.warn(f"{sym}: missing columns {missing}, skipping")
                continue

            result[sym] = df
            time.sleep(self.delay)

        return result