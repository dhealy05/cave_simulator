from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cave.observation.experience.objects import ExperienceObject


@dataclass(frozen=True)
class ExperienceQualities:
    magnitude: float = 0.0
    novelty: float = 0.0
    variance: float = 0.0
    beauty: float = 0.0
    threat: float = 0.0
    pain: float = 0.0
    pleasure: float = 0.0
    ambiguity: float = 0.0
    personal_relevance: float = 0.0
    familiarity: float = 0.0
    overload: float = 0.0
    surprise: float = 0.0
    expectation_magnitude: float = 0.0

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ExperienceQualities:
        fields = cls.__dataclass_fields__
        return cls(
            **{
                key: _nonnegative_float(data.get(key, 0.0))
                for key in fields
            }
        )

    def to_metadata(self) -> dict[str, float]:
        return {
            key: float(getattr(self, key))
            for key in self.__dataclass_fields__
            if float(getattr(self, key)) != 0.0
        }


@dataclass(frozen=True)
class ResolvedExperienceEffects:
    salience: float
    learning_weight: float
    pain: float
    pleasure: float
    attention_pull: float

    def to_metadata(self) -> dict[str, float]:
        return {
            "salience": self.salience,
            "learning_weight": self.learning_weight,
            "pain": self.pain,
            "pleasure": self.pleasure,
            "attention_pull": self.attention_pull,
        }


@dataclass(frozen=True)
class ExperienceQualityResolver:
    base_salience: float = 0.1
    magnitude_salience_gain: float = 0.55
    novelty_salience_gain: float = 0.35
    variance_salience_gain: float = 0.4
    beauty_salience_gain: float = 0.2
    threat_salience_gain: float = 0.65
    pain_salience_gain: float = 0.75
    pleasure_salience_gain: float = 0.35
    surprise_salience_gain: float = 0.25
    expectation_salience_gain: float = 1.0
    base_learning_weight: float = 1.0
    novelty_learning_gain: float = 0.25
    variance_learning_gain: float = 0.35
    personal_relevance_learning_gain: float = 0.3
    surprise_learning_gain: float = 0.5
    pain_learning_gain: float = 0.25
    pleasure_learning_gain: float = 0.1
    beauty_pleasure_gain: float = 0.45
    novelty_pleasure_gain: float = 0.15
    threat_pain_gain: float = 0.65
    overload_pain_gain: float = 0.5
    ambiguity_pain_gain: float = 0.15

    def resolve(self, qualities: ExperienceQualities) -> ResolvedExperienceEffects:
        pleasure = _clamp01(
            qualities.pleasure
            + self.beauty_pleasure_gain * qualities.beauty
            + self.novelty_pleasure_gain * qualities.novelty
        )
        pain = _clamp01(
            qualities.pain
            + self.threat_pain_gain * qualities.threat
            + self.overload_pain_gain * qualities.overload * max(
                qualities.magnitude,
                qualities.expectation_magnitude,
            )
            + self.ambiguity_pain_gain * qualities.ambiguity
        )
        salience = _clamp01(
            self.base_salience
            + self.magnitude_salience_gain * qualities.magnitude
            + self.novelty_salience_gain * qualities.novelty
            + self.variance_salience_gain * qualities.variance
            + self.beauty_salience_gain * qualities.beauty
            + self.threat_salience_gain * qualities.threat
            + self.pain_salience_gain * pain
            + self.pleasure_salience_gain * pleasure
            + self.surprise_salience_gain * qualities.surprise
            + self.expectation_salience_gain * qualities.expectation_magnitude
        )
        learning_weight = max(
            0.0,
            float(
                self.base_learning_weight
                + self.novelty_learning_gain * qualities.novelty
                + self.variance_learning_gain * qualities.variance
                + self.personal_relevance_learning_gain * qualities.personal_relevance
                + self.surprise_learning_gain * qualities.surprise
                + self.pain_learning_gain * pain
                + self.pleasure_learning_gain * pleasure
            ),
        )
        return ResolvedExperienceEffects(
            salience=salience,
            learning_weight=learning_weight,
            pain=pain,
            pleasure=pleasure,
            attention_pull=salience,
        )


def resolve_experience_effects(
    qualities: ExperienceQualities | dict[str, Any],
    *,
    resolver: ExperienceQualityResolver | None = None,
) -> ResolvedExperienceEffects:
    resolved_qualities = (
        ExperienceQualities.from_mapping(qualities)
        if isinstance(qualities, dict)
        else qualities
    )
    resolved_resolver = resolver or ExperienceQualityResolver()
    return resolved_resolver.resolve(resolved_qualities)


def resolve_experience_object(
    obj: ExperienceObject,
    qualities: ExperienceQualities | dict[str, Any],
    *,
    resolver: ExperienceQualityResolver | None = None,
    salience: float | None = None,
    learning_weight: float | None = None,
) -> ExperienceObject:
    resolved_qualities = (
        ExperienceQualities.from_mapping(qualities)
        if isinstance(qualities, dict)
        else qualities
    )
    effects = resolve_experience_effects(resolved_qualities, resolver=resolver)
    metadata = dict(obj.metadata)
    metadata["qualities"] = resolved_qualities.to_metadata()
    metadata["resolved_effects"] = effects.to_metadata()
    affect = dict(_metadata_dict(metadata.get("affect", {})))
    affect.setdefault("pain", effects.pain)
    affect.setdefault("pleasure", effects.pleasure)
    metadata["affect"] = affect
    return ExperienceObject(
        id=obj.id,
        temporal_extent=obj.temporal_extent,
        features=obj.features,
        kind=obj.kind,
        presentation=obj.presentation,
        salience=effects.salience if salience is None else salience,
        learning_weight=(
            effects.learning_weight
            if learning_weight is None
            else learning_weight
        ),
        modality=obj.modality,
        metadata=metadata,
    )


def _metadata_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return dict(value)


def _nonnegative_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return max(0.0, float(value))


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
