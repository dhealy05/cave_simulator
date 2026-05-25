from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from cave.commitments.attention.state import AttentionState
from cave.commitments.prediction.state import PredictionState

if TYPE_CHECKING:
    from cave.commitments.affect import ValenceState
    from cave.commitments.objective import ObjectiveState
    from cave.observation.sensing import SensorResponse


class AttentionPolicy(Protocol):
    def next_channel_weights(
        self,
        *,
        current_attention: AttentionState,
        sensor_responses: dict[str, SensorResponse],
        prediction: PredictionState,
        valence: ValenceState | None = None,
        objective: ObjectiveState | None = None,
    ) -> dict[str, float] | None:
        ...


@dataclass(frozen=True)
class FixedAttentionPolicy:
    def next_channel_weights(
        self,
        *,
        current_attention: AttentionState,
        sensor_responses: dict[str, SensorResponse],
        prediction: PredictionState,
        valence: ValenceState | None = None,
        objective: ObjectiveState | None = None,
    ) -> dict[str, float] | None:
        return None


@dataclass(frozen=True)
class SurpriseAdaptiveAttentionPolicy:
    learning_rate: float = 0.25
    surprise_gain: float = 0.0
    signal_floor: float = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 <= self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be between 0 and 1")
        if self.surprise_gain < 0.0:
            raise ValueError("surprise_gain must be non-negative")
        if self.signal_floor < 0.0:
            raise ValueError("signal_floor must be non-negative")

    def next_channel_weights(
        self,
        *,
        current_attention: AttentionState,
        sensor_responses: dict[str, SensorResponse],
        prediction: PredictionState,
        valence: ValenceState | None = None,
        objective: ObjectiveState | None = None,
    ) -> dict[str, float]:
        current = dict(current_attention.channel_weights)
        signals = {
            channel: _response_magnitude(response)
            for channel, response in sensor_responses.items()
        }
        active_signals = {
            channel: signal
            for channel, signal in signals.items()
            if signal > self.signal_floor
        }
        if not active_signals:
            return current

        channels = set(current) | set(active_signals)
        signal_total = sum(active_signals.values())
        target = {
            channel: active_signals.get(channel, 0.0) / signal_total
            for channel in channels
        }
        effective_rate = min(
            1.0,
            self.learning_rate * (1.0 + self.surprise_gain * prediction.surprise),
        )
        blended = {
            channel: (1.0 - effective_rate) * current.get(channel, 0.0)
            + effective_rate * target[channel]
            for channel in channels
        }
        return _normalize_weights(blended)


@dataclass(frozen=True)
class ObjectiveAdaptiveAttentionPolicy:
    learning_rate: float = 0.35
    signal_gain: float = 1.0
    pain_gain: float = 1.0
    pleasure_gain: float = 1.0
    surprise_gain: float = 0.0
    signal_floor: float = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 <= self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be between 0 and 1")
        for name, value in {
            "signal_gain": self.signal_gain,
            "pain_gain": self.pain_gain,
            "pleasure_gain": self.pleasure_gain,
            "surprise_gain": self.surprise_gain,
            "signal_floor": self.signal_floor,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")

    def next_channel_weights(
        self,
        *,
        current_attention: AttentionState,
        sensor_responses: dict[str, SensorResponse],
        prediction: PredictionState,
        valence: ValenceState | None = None,
        objective: ObjectiveState | None = None,
    ) -> dict[str, float]:
        current = dict(current_attention.channel_weights)
        signal_scores = {
            channel: _response_magnitude(response)
            for channel, response in sensor_responses.items()
        }
        channels = set(current) | set(signal_scores)
        if valence is not None:
            channels |= set(valence.channel_pain)
            channels |= set(valence.channel_pleasure)
        if not channels:
            return current

        scores = {}
        for channel in channels:
            signal = signal_scores.get(channel, 0.0)
            pain = 0.0 if valence is None else valence.channel_pain.get(channel, 0.0)
            pleasure = (
                0.0
                if valence is None
                else valence.channel_pleasure.get(channel, 0.0)
            )
            score = (
                self.signal_gain * signal
                + self.pain_gain * pain
                + self.pleasure_gain * pleasure
                + self.surprise_gain * max(0.0, prediction.surprise) * signal
            )
            if score > self.signal_floor:
                scores[channel] = score

        if not scores:
            return current

        target = _normalize_weights(scores)
        effective_rate = self.learning_rate
        blended = {
            channel: (1.0 - effective_rate) * current.get(channel, 0.0)
            + effective_rate * target.get(channel, 0.0)
            for channel in set(current) | set(target)
        }
        return _normalize_weights(blended)


def default_attention_policy() -> FixedAttentionPolicy:
    return FixedAttentionPolicy()


def _response_magnitude(response: SensorResponse) -> float:
    vector = np.asarray(response.vector, dtype=float)
    if vector.size == 0:
        return 0.0
    return float(np.linalg.norm(vector) / np.sqrt(vector.size))


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {
        str(channel): max(0.0, float(weight))
        for channel, weight in weights.items()
    }
    total = sum(cleaned.values())
    if total <= 0.0:
        return {"visual": 1.0}
    return {
        channel: weight / total
        for channel, weight in cleaned.items()
        if weight > 0.0
    }
