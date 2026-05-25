from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from cave.observation.experience import Array, ExperienceObject


class Sensor(Protocol):
    channel: str

    def transduce(
        self,
        objects: list[ExperienceObject],
        vocabulary: list[str],
    ) -> Array:
        ...


@dataclass(frozen=True)
class SensorResponse:
    channel: str
    vector: Array

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector", np.array(self.vector, dtype=float))


@dataclass(frozen=True)
class FeatureSensor:
    modality: str = "visual"
    channel: str = "visual"
    gain: float = 1.0

    def transduce(
        self,
        objects: list[ExperienceObject],
        vocabulary: list[str],
    ) -> Array:
        vector = np.zeros(len(vocabulary), dtype=float)
        for obj in objects:
            if obj.modality != self.modality:
                continue
            vector += self.gain * obj.salience * obj.features.to_array(vocabulary)
        return vector


def visual_feature_sensor() -> FeatureSensor:
    return FeatureSensor(modality="visual", channel="visual")
