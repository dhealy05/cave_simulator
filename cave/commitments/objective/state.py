from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ObjectiveState:
    utility: float = 0.0
    prediction_cost: float = 0.0
    pain_cost: float = 0.0
    pleasure_gain: float = 0.0
    attention_cost: float = 0.0
    compression_cost: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "utility", float(self.utility))
        object.__setattr__(self, "prediction_cost", max(0.0, float(self.prediction_cost)))
        object.__setattr__(self, "pain_cost", max(0.0, float(self.pain_cost)))
        object.__setattr__(self, "pleasure_gain", max(0.0, float(self.pleasure_gain)))
        object.__setattr__(self, "attention_cost", max(0.0, float(self.attention_cost)))
        object.__setattr__(
            self,
            "compression_cost",
            max(0.0, float(self.compression_cost)),
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "utility": self.utility,
            "prediction_cost": self.prediction_cost,
            "pain_cost": self.pain_cost,
            "pleasure_gain": self.pleasure_gain,
            "attention_cost": self.attention_cost,
            "compression_cost": self.compression_cost,
        }
