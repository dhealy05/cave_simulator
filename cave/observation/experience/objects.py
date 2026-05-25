from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np

from cave.observation.experience.features import Array, FeatureVector
from cave.observation.experience.presentations import Presentation, visual_presentation_from_features

if TYPE_CHECKING:
    from cave.commitments.attention.state import AttentionState


@dataclass(frozen=True)
class TemporalExtent:
    start: float
    end: float
    order_index: int

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("TemporalExtent.end must be greater than start")

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def center(self) -> float:
        return 0.5 * (self.start + self.end)

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end


@dataclass(frozen=True)
class ExperienceObject:
    id: str
    temporal_extent: TemporalExtent
    features: FeatureVector
    kind: str = "experience"
    presentation: Presentation | None = None
    salience: float = 1.0
    learning_weight: float = 1.0
    modality: str = "visual"
    metadata: dict[str, Any] = field(default_factory=dict)


def presentation_for_object(obj: ExperienceObject) -> Presentation:
    if obj.presentation is not None:
        return obj.presentation
    return visual_presentation_from_features(obj.features, obj.metadata)


@dataclass(frozen=True)
class InputSequence:
    objects: list[ExperienceObject]

    def __post_init__(self) -> None:
        ordered = sorted(
            self.objects,
            key=lambda obj: (obj.temporal_extent.start, obj.temporal_extent.order_index),
        )
        object.__setattr__(self, "objects", ordered)

    @property
    def duration(self) -> float:
        if not self.objects:
            return 0.0
        return max(obj.temporal_extent.end for obj in self.objects)

    def active_at(self, t: float) -> list[ExperienceObject]:
        return [obj for obj in self.objects if obj.temporal_extent.contains(t)]

    def past_before(self, t: float) -> list[ExperienceObject]:
        return [obj for obj in self.objects if obj.temporal_extent.end <= t]

    def future_after(self, t: float) -> list[ExperienceObject]:
        return [obj for obj in self.objects if obj.temporal_extent.start > t]

    def features_at(self, t: float, vocabulary: list[str]) -> Array:
        active = self.active_at(t)
        if not active:
            return np.zeros(len(vocabulary), dtype=float)
        vectors = [obj.features.to_array(vocabulary) * obj.salience for obj in active]
        return np.sum(vectors, axis=0)

    def channel_features_at(self, t: float, vocabulary: list[str]) -> dict[str, Array]:
        channels: dict[str, Array] = {}
        for obj in self.active_at(t):
            channel = obj.modality
            vector = obj.features.to_array(vocabulary) * obj.salience
            if channel not in channels:
                channels[channel] = np.zeros(len(vocabulary), dtype=float)
            channels[channel] += vector
        return channels

    def attended_features_at(
        self,
        t: float,
        vocabulary: list[str],
        attention: AttentionState,
    ) -> Array:
        attended = np.zeros(len(vocabulary), dtype=float)
        impact = attention.impact()
        for channel, vector in self.channel_features_at(t, vocabulary).items():
            attended += impact * attention.channel_weight(channel) * vector
        return attended
