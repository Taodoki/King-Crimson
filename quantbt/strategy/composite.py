"""Composite strategy — combine multiple strategies with weights."""

import pandas as pd
from quantbt.strategy.base import Strategy


class CompositeStrategy(Strategy):
    """Combine multiple strategies with configurable weights.

    Examples:
        combo = CompositeStrategy([
            (TimeSeriesMomentum(60), 0.5),
            (ZScoreMeanReversion(20), 0.5),
        ])
    """

    def __init__(
        self,
        strategies: list[tuple[Strategy, float]],
        name: str | None = None,
    ):
        super().__init__(name)
        total = sum(w for _, w in strategies)
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total}"
        self.sub_strategies = strategies

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        combined = None
        for strategy, weight in self.sub_strategies:
            s = strategy.generate_signals(data) * weight
            combined = s if combined is None else combined + s
        return combined if combined is not None else pd.DataFrame(0.0, index=data.index, columns=data.columns.levels[0])
