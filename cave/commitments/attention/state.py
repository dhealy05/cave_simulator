from __future__ import annotations

import math
from dataclasses import dataclass, field

from cave.observation.experience.objects import ExperienceObject


DEFAULT_ATTENTION_CHANNEL = "visual"
INTERNAL_EXPECTATION_CHANNEL = "internal_expectation"
DEFAULT_CHANNEL_WEIGHTS = {
    DEFAULT_ATTENTION_CHANNEL: 0.5,
    INTERNAL_EXPECTATION_CHANNEL: 0.5,
}


def attention_effect(attention: float, high_gamma: float = 2.0) -> float:
    attention = min(1.0, max(0.0, float(attention)))
    if attention <= 0.5:
        return attention
    high_band = (attention - 0.5) / 0.5
    return 0.5 + 0.5 * high_band**high_gamma


@dataclass(frozen=True)
class AttentionState:
    channel_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CHANNEL_WEIGHTS)
    )
    capacity: float = 1.0
    high_gamma: float = 2.0

    def __post_init__(self) -> None:
        if self.high_gamma <= 0.0:
            raise ValueError("AttentionState.high_gamma must be positive")
        capacity = min(1.0, max(0.0, float(self.capacity)))
        weights = {
            str(channel): max(0.0, float(weight))
            for channel, weight in self.channel_weights.items()
        }
        total = sum(weights.values())
        if total <= 0.0:
            weights = {DEFAULT_ATTENTION_CHANNEL: 1.0}
            total = 1.0
        normalized = {
            channel: weight / total
            for channel, weight in weights.items()
            if weight > 0.0
        }
        object.__setattr__(self, "capacity", capacity)
        object.__setattr__(self, "channel_weights", normalized)

    @property
    def scalar(self) -> float:
        return self.capacity

    def impact(self) -> float:
        return attention_effect(self.capacity, self.high_gamma)

    def channel_weight(self, channel: str) -> float:
        return float(self.channel_weights.get(channel, 0.0))

    def object_impact(self, obj: ExperienceObject) -> float:
        return self.impact() * self.channel_weight(obj.modality)

    def internal_expectation_impact(self) -> float:
        return self.impact() * self.channel_weight(INTERNAL_EXPECTATION_CHANNEL)


def coerce_attention_state(attention: AttentionState | float | None) -> AttentionState:
    if attention is None:
        return AttentionState()
    if isinstance(attention, AttentionState):
        return attention
    return AttentionState(capacity=float(attention))


@dataclass(frozen=True)
class AttentionChannelCurve:
    mode: str = "constant"
    level: float = 1.0
    amplitude: float = 0.0
    phase: float = 0.0
    cycles: float = 1.0

    def value_at(self, t: float, duration: float) -> float:
        if self.mode == "constant":
            value = self.level
        elif self.mode == "sine":
            span = max(duration, 1e-9)
            value = self.level + self.amplitude * math.sin(
                2.0 * math.pi * self.cycles * (t / span) + self.phase
            )
        else:
            raise ValueError(f"unsupported attention channel curve: {self.mode}")
        return min(1.0, max(0.0, float(value)))


@dataclass(frozen=True)
class AttentionProfile:
    mode: str = "constant"
    level: float = 1.0
    amplitude: float = 0.5
    phase: float = 0.0
    channel_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CHANNEL_WEIGHTS)
    )
    channel_curves: dict[str, AttentionChannelCurve] = field(default_factory=dict)

    def value_at(self, t: float, duration: float) -> float:
        if self.mode == "constant":
            value = self.level
        elif self.mode == "sine":
            span = max(duration, 1e-9)
            value = self.level + self.amplitude * math.sin(
                2.0 * math.pi * (t / span) + self.phase
            )
        else:
            raise ValueError(f"unsupported attention profile: {self.mode}")
        return min(1.0, max(0.0, float(value)))

    def state_at(self, t: float, duration: float) -> AttentionState:
        return AttentionState(
            channel_weights=self.channel_weights_at(t, duration),
            capacity=self.value_at(t, duration),
        )

    def channel_weights_at(self, t: float, duration: float) -> dict[str, float]:
        if not self.channel_curves:
            return dict(self.channel_weights)
        weights = dict(self.channel_weights)
        for channel, curve in self.channel_curves.items():
            weights[channel] = curve.value_at(t, duration)
        return weights


def balanced_attention_profile(*, level: float = 1.0) -> AttentionProfile:
    return AttentionProfile(
        mode="constant",
        level=level,
        channel_weights=dict(DEFAULT_CHANNEL_WEIGHTS),
    )


def zero_attention_profile() -> AttentionProfile:
    return AttentionProfile(mode="constant", level=0.0)


def external_only_attention_profile(
    *,
    level: float = 1.0,
    channel_weights: dict[str, float] | None = None,
) -> AttentionProfile:
    weights = (
        {DEFAULT_ATTENTION_CHANNEL: 1.0}
        if channel_weights is None
        else {
            str(channel): float(weight)
            for channel, weight in channel_weights.items()
            if channel != INTERNAL_EXPECTATION_CHANNEL
        }
    )
    if not weights:
        weights = {DEFAULT_ATTENTION_CHANNEL: 1.0}
    return AttentionProfile(mode="constant", level=level, channel_weights=weights)


def internal_only_attention_profile(*, level: float = 1.0) -> AttentionProfile:
    return AttentionProfile(
        mode="constant",
        level=level,
        channel_weights={INTERNAL_EXPECTATION_CHANNEL: 1.0},
    )


def legacy_visual_attention_profile(*, level: float = 1.0) -> AttentionProfile:
    return external_only_attention_profile(
        level=level,
        channel_weights={DEFAULT_ATTENTION_CHANNEL: 1.0},
    )
