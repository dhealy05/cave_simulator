from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from cave.observation.experience import Array
from cave.commitments.prediction.state import PredictionState
from cave.demonstrations.state import SubjectState


class Predictor(Protocol):
    def predict(self, state: SubjectState, vocabulary: list[str]) -> Array:
        ...

    def evaluate(
        self,
        expected_input: Array,
        actual_input: Array,
    ) -> PredictionState:
        ...


@dataclass(frozen=True)
class MemoryVectorPredictor:
    surprise_floor: float = 1e-12

    def predict(self, state: SubjectState, vocabulary: list[str]) -> Array:
        vector = state.memory.vector
        if vector.shape[0] != len(vocabulary):
            raise ValueError("memory vector length must match vocabulary length")
        return vector.copy()

    def evaluate(
        self,
        expected_input: Array,
        actual_input: Array,
    ) -> PredictionState:
        expected = np.asarray(expected_input, dtype=float)
        actual = np.asarray(actual_input, dtype=float)
        if expected.shape != actual.shape:
            raise ValueError(f"prediction shapes differ: {expected.shape} != {actual.shape}")
        error = actual - expected
        if error.size == 0:
            surprise = 0.0
        else:
            surprise = float(np.linalg.norm(error) / np.sqrt(error.size))
        if surprise < self.surprise_floor:
            surprise = 0.0
        return PredictionState(
            expected_input=expected.copy(),
            prediction_error=error,
            surprise=surprise,
        )


def default_predictor() -> MemoryVectorPredictor:
    return MemoryVectorPredictor()
