"""Composite strategy — combine multiple strategies with weights."""

import pandas as pd
from quantbt.strategy.base import Strategy


def _freeze_on_monthly_grid(signal: pd.DataFrame) -> pd.DataFrame:
    """Freeze a grid strategy's signal to the first trading day of each month.

    The first-day value is forward-filled across the month so a daily
    matching engine (triggered by an event-driven sibling) does not
    re-trade grid strategies between their natural monthly rebalance
    points.
    """
    month_key = signal.index.to_period("M")
    anchors = signal.groupby(month_key).head(1).index
    return signal.loc[anchors].reindex(signal.index, method="ffill").fillna(0.0)


class CompositeStrategy(Strategy):
    """Combine multiple strategies with configurable weights.

    The composite is event-driven when ANY sub-strategy is: the engine
    then matches orders daily, so state-machine exits/stop-losses fire
    on their transition days instead of waiting for the next rebalance.
    Signals of non-event-driven sub-strategies are frozen on a monthly
    grid so the daily matching does not re-trade them on days they
    would not have traded alone.

    Examples:
        combo = CompositeStrategy([
            (TimeSeriesMomentum(60), 0.5),
            (ZScoreMeanReversion(20), 0.5),
        ])
    """

    # Set per instance in __init__: True when any sub-strategy is
    # event-driven (see class docstring).
    event_driven: bool = False

    def __init__(
        self,
        strategies: list[tuple[Strategy, float]],
        name: str | None = None,
    ):
        super().__init__(name)
        total = sum(w for _, w in strategies)
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total}"
        self.sub_strategies = strategies
        self.event_driven = any(s.event_driven for s, _ in strategies)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        combined = None
        for strategy, weight in self.sub_strategies:
            s = strategy.generate_signals(data) * weight
            if not strategy.event_driven:
                s = _freeze_on_monthly_grid(s)
            combined = s if combined is None else combined + s
        if combined is None:
            return pd.DataFrame(
                0.0, index=data.index, columns=data.columns.levels[0]
            )
        return combined
