"""Abstract data source interface for price/volume data."""

from abc import ABC, abstractmethod
from datetime import date
import pandas as pd


class DataSource(ABC):
    """Abstract data source for OHLCV data.

    Each implementation must return a dict mapping symbol -> DataFrame.
    Every DataFrame must contain at minimum these columns:
        date, open, high, low, close, volume, adj_close
    """

    @abstractmethod
    def fetch(
        self,
        symbols: list[str],
        start: str | date,
        end: str | date,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV data for given symbols over date range."""
