from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from cave.commitments.attention.state import AttentionState, coerce_attention_state
from cave.observation.experience.features import Array
from cave.observation.experience.objects import ExperienceObject


@dataclass(frozen=True)
class MemoryParams:
    retention: float
    decay_tau: float
    max_age: float


@dataclass(frozen=True)
class MemoryItem:
    source: ExperienceObject
    ended_t: float
    strength: float

    def age(self, t: float) -> float:
        return max(0.0, t - self.ended_t)


@dataclass(frozen=True)
class MemoryAccumulator:
    source: ExperienceObject
    attention_total: float = 0.0
    samples: int = 0

    def add(self, attention: float) -> "MemoryAccumulator":
        return MemoryAccumulator(
            source=self.source,
            attention_total=self.attention_total + attention,
            samples=self.samples + 1,
        )

    @property
    def average_attention(self) -> float:
        if self.samples <= 0:
            return 0.0
        return self.attention_total / self.samples


@dataclass
class MemoryTrace:
    vector: Array
    items: list[MemoryItem] = field(default_factory=list)
    active: dict[str, MemoryAccumulator] = field(default_factory=dict)
    retention: float = 0.82
    decay_tau: float = 2.0
    max_age: float = 6.0
    expectation_vector: Array | None = None
    expectation_strength: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.retention <= 1.0:
            raise ValueError("retention must be between 0 and 1")
        if self.decay_tau <= 0.0:
            raise ValueError("decay_tau must be positive")
        self.vector = np.asarray(self.vector, dtype=float)
        if self.expectation_vector is None:
            self.expectation_vector = np.zeros_like(self.vector, dtype=float)
        else:
            self.expectation_vector = np.asarray(self.expectation_vector, dtype=float)
            if self.expectation_vector.shape != self.vector.shape:
                raise ValueError("expectation_vector must match vector shape")
        self.expectation_strength = min(
            1.0,
            max(0.0, float(self.expectation_strength)),
        )

    def update(
        self,
        t: float,
        u_t: Array,
        current_objects: list[ExperienceObject],
        current_attention: AttentionState | float | None = None,
        learning_rate: float | None = None,
        expected_input: Array | None = None,
    ) -> None:
        alpha = (1.0 - self.retention) if learning_rate is None else float(learning_rate)
        alpha = min(1.0, max(0.0, alpha))
        self.vector = self.vector + alpha * (u_t - self.vector)
        attention = coerce_attention_state(current_attention)
        if expected_input is not None:
            self._update_expectation(expected_input, attention)
        else:
            self.expectation_strength *= self.retention
        self._update_items(t, current_objects, attention)

    def snapshot(self) -> MemoryTrace:
        return MemoryTrace(
            vector=self.vector.copy(),
            expectation_vector=self.expectation_vector.copy(),
            expectation_strength=self.expectation_strength,
            items=list(self.items),
            active=dict(self.active),
            retention=self.retention,
            decay_tau=self.decay_tau,
            max_age=self.max_age,
        )

    def _update_items(
        self,
        t: float,
        current_objects: list[ExperienceObject],
        current_attention: AttentionState,
    ) -> None:
        current_by_id = {obj.id: obj for obj in current_objects}
        next_active: dict[str, MemoryAccumulator] = {}
        for obj in current_objects:
            accumulator = self.active.get(obj.id, MemoryAccumulator(source=obj))
            next_active[obj.id] = accumulator.add(
                current_attention.object_impact(obj) * obj.learning_weight
            )

        by_id = {item.source.id: item for item in self.items}
        for obj_id, accumulator in self.active.items():
            if obj_id in current_by_id:
                continue
            if accumulator.average_attention > 0.0:
                source = accumulator.source
                by_id[obj_id] = MemoryItem(
                    source=source,
                    ended_t=source.temporal_extent.end,
                    strength=accumulator.average_attention,
                )

        updated: list[MemoryItem] = []
        for item in by_id.values():
            strength = item.strength * memory_strength(item.age(t), self.decay_tau)
            if item.age(t) <= self.max_age and strength > 0.0:
                updated.append(
                    MemoryItem(
                        source=item.source,
                        ended_t=item.ended_t,
                        strength=strength,
                    )
                )

        self.active = next_active
        self.items = sorted(
            updated,
            key=lambda item: item.source.temporal_extent.order_index,
        )

    def _update_expectation(
        self,
        expected_input: Array,
        current_attention: AttentionState,
    ) -> None:
        expected = np.asarray(expected_input, dtype=float)
        if expected.shape != self.vector.shape:
            raise ValueError("expected_input must match memory vector shape")
        alpha = current_attention.internal_expectation_impact()
        alpha = min(1.0, max(0.0, float(alpha)))
        self.expectation_vector = (
            self.expectation_vector + alpha * (expected - self.expectation_vector)
        )
        self.expectation_strength = self.expectation_strength + alpha * (
            1.0 - self.expectation_strength
        )


def memory_strength(age: float, tau: float) -> float:
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    return math.exp(-age / tau)
