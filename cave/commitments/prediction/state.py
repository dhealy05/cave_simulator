from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cave.observation.experience import Array


@dataclass(frozen=True)
class PredictionState:
    expected_input: Array
    prediction_error: Array
    surprise: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected_input", np.array(self.expected_input, dtype=float))
        object.__setattr__(self, "prediction_error", np.array(self.prediction_error, dtype=float))
        object.__setattr__(self, "surprise", float(self.surprise))
