from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cave.commitments.attention import AttentionState


class LearningRule(Protocol):
    def learning_rate(
        self,
        *,
        base_rate: float,
        attention: AttentionState,
        importance: float,
        surprise: float,
    ) -> float:
        ...


@dataclass(frozen=True)
class ImportanceWeightedLearningRule:
    surprise_gain: float = 0.0
    min_rate: float = 0.0
    max_rate: float = 1.0

    def __post_init__(self) -> None:
        if self.surprise_gain < 0.0:
            raise ValueError("surprise_gain must be non-negative")
        if self.max_rate < self.min_rate:
            raise ValueError("max_rate must be greater than or equal to min_rate")

    def learning_rate(
        self,
        *,
        base_rate: float,
        attention: AttentionState,
        importance: float,
        surprise: float,
    ) -> float:
        effective_rate = (
            base_rate
            * max(0.0, float(importance))
            * (1.0 + self.surprise_gain * max(0.0, float(surprise)))
        )
        return min(self.max_rate, max(self.min_rate, float(effective_rate)))


def default_learning_rule() -> ImportanceWeightedLearningRule:
    return ImportanceWeightedLearningRule()
