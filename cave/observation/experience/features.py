from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class FeatureVector:
    values: dict[str, float]

    def value(self, key: str, default: float = 0.0) -> float:
        return float(self.values.get(key, default))

    def to_array(self, vocabulary: list[str]) -> Array:
        return np.array([self.values.get(key, 0.0) for key in vocabulary], dtype=float)


@dataclass(frozen=True)
class FeatureProjection:
    name: str
    weights: dict[str, float]
    value_min: float = 0.0
    value_max: float = 1.0

    def __post_init__(self) -> None:
        if self.value_max <= self.value_min:
            raise ValueError("FeatureProjection.value_max must be greater than value_min")

    def raw_value(self, features: FeatureVector) -> float:
        return sum(features.value(key) * weight for key, weight in self.weights.items())

    def normalized_value(self, features: FeatureVector) -> float:
        value = (self.raw_value(features) - self.value_min) / (self.value_max - self.value_min)
        return min(1.0, max(0.0, float(value)))


FeatureAxis = str | FeatureProjection


def feature_axis_label(axis: FeatureAxis) -> str:
    if isinstance(axis, FeatureProjection):
        return axis.name
    return axis


def feature_axis_value(features: FeatureVector, axis: FeatureAxis) -> float:
    if isinstance(axis, FeatureProjection):
        return axis.normalized_value(features)
    return min(1.0, max(0.0, features.value(axis)))
