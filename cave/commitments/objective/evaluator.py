from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cave.commitments.affect import ValenceState
from cave.commitments.attention import AttentionState
from cave.commitments.objective.state import ObjectiveState
from cave.commitments.prediction import PredictionState


class ObjectiveEvaluator(Protocol):
    def evaluate(
        self,
        *,
        prediction: PredictionState,
        valence: ValenceState,
        attention: AttentionState,
        compression_cost: float = 0.0,
    ) -> ObjectiveState:
        ...


@dataclass(frozen=True)
class LinearObjectiveEvaluator:
    prediction_weight: float = 1.0
    pain_weight: float = 1.0
    pleasure_weight: float = 1.0
    attention_cost_weight: float = 0.0
    compression_cost_weight: float = 1.0

    def __post_init__(self) -> None:
        for name, value in {
            "prediction_weight": self.prediction_weight,
            "pain_weight": self.pain_weight,
            "pleasure_weight": self.pleasure_weight,
            "attention_cost_weight": self.attention_cost_weight,
            "compression_cost_weight": self.compression_cost_weight,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")

    def evaluate(
        self,
        *,
        prediction: PredictionState,
        valence: ValenceState,
        attention: AttentionState,
        compression_cost: float = 0.0,
    ) -> ObjectiveState:
        prediction_cost = self.prediction_weight * max(0.0, prediction.surprise)
        pain_cost = self.pain_weight * valence.pain
        pleasure_gain = self.pleasure_weight * valence.pleasure
        attention_cost = self.attention_cost_weight * attention.capacity
        compression = self.compression_cost_weight * max(0.0, compression_cost)
        utility = pleasure_gain - pain_cost - prediction_cost - attention_cost - compression
        return ObjectiveState(
            utility=utility,
            prediction_cost=prediction_cost,
            pain_cost=pain_cost,
            pleasure_gain=pleasure_gain,
            attention_cost=attention_cost,
            compression_cost=compression,
        )


def default_objective_evaluator() -> LinearObjectiveEvaluator:
    return LinearObjectiveEvaluator()
