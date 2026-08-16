"""Trading strategy base class.

A Strategy is a stateless transformation from price/volume data
into position targets (signals). Stateless means: given the same data,
generate_signals always returns the same result.

Signal convention:
    +1.0  = full allocation long
     0.0  = no position
    -1.0  = full allocation short (when short_selling=True)

Intermediate values represent fractional position sizes.

All signals MUST use shift(1) internally to prevent look-ahead bias:
    signal[t] uses data available at t-1, never t or later.
"""

from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    """Trading strategy base class. Subclasses must implement generate_signals()."""

    #: When True the engine matches orders on EVERY trading day instead of
    #: only on rebalance dates. State-machine strategies (entry/exit/
    #: stop-loss) need this: their signal only changes on transition days,
    #: so daily matching adds no extra trades on flat days.
    event_driven: bool = False

    def __init__(self, name: str | None = None):
        self._name = name

    @property
    def name(self) -> str:
        return self._name or self.__class__.__name__

    def _get_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract adjusted close prices from MultiIndex DataFrame.

        Column structure: (ticker, field), with field in
        {close, adj_close, open, high, low, volume}.
        """
        if isinstance(data.columns, pd.MultiIndex):
            if "adj_close" in data.columns.get_level_values(1):
                return data.xs("adj_close", axis=1, level=1)
            return data.xs("close", axis=1, level=1)
        return data

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from price/volume data.

        Parameters
        ----------
        data : pd.DataFrame
            MultiIndex columns (ticker, field) where field includes:
            close, open, high, low, volume, adj_close.
            Index is DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Signals aligned with data.index. Columns = tickers,
            values = position targets in [-1.0, 1.0].
        """
        ...

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self._params().items())
        return f"{self.name}({params})"

    def _params(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}