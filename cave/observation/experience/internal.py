from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cave.observation.experience.authoring import (
    ExperienceQualities,
    ExperienceQualityResolver,
    resolve_experience_effects,
)
from cave.observation.experience.features import FeatureVector
from cave.observation.experience.objects import ExperienceObject, InputSequence, TemporalExtent


INTERNAL_EXPERIENCE_CHANNEL = "internal_expectation"


@dataclass(frozen=True)
class InternalExperienceGenerator:
    """Generate post-run internal experience objects from episode observations."""

    salience_floor: float = 1e-9
    expectation_gain: float = 1.0
    pain_gain: float = 0.75
    pleasure_gain: float = 0.35
    surprise_gain: float = 0.25
    learning_base: float = 1.0
    learning_surprise_gain: float = 0.5
    learning_pain_gain: float = 0.25
    learning_pleasure_gain: float = 0.1
    fallback_dt: float = 0.1
    id_prefix: str = "internal_expectation"

    def __post_init__(self) -> None:
        for name, value in {
            "salience_floor": self.salience_floor,
            "expectation_gain": self.expectation_gain,
            "pain_gain": self.pain_gain,
            "pleasure_gain": self.pleasure_gain,
            "surprise_gain": self.surprise_gain,
            "learning_base": self.learning_base,
            "learning_surprise_gain": self.learning_surprise_gain,
            "learning_pain_gain": self.learning_pain_gain,
            "learning_pleasure_gain": self.learning_pleasure_gain,
            "fallback_dt": self.fallback_dt,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")

    def generate(self, episode: Any) -> InputSequence:
        vocabulary = list(episode.vocabulary)
        observations = list(episode.observations)
        objects = [
            self._object_for_observation(
                observation,
                index=index,
                end_t=_observation_end_t(observations, index, self.fallback_dt),
                vocabulary=vocabulary,
            )
            for index, observation in enumerate(observations)
        ]
        return InputSequence([obj for obj in objects if obj is not None])

    def _object_for_observation(
        self,
        observation: Any,
        *,
        index: int,
        end_t: float,
        vocabulary: list[str],
    ) -> ExperienceObject | None:
        expected = np.asarray(observation.expected, dtype=float)
        if expected.ndim != 1 or expected.shape[0] != len(vocabulary):
            raise ValueError("observation.expected must be a vector matching episode vocabulary")

        magnitude = _vector_magnitude(expected)
        valence = _metadata_dict(observation.metadata.get("valence", {}))
        pain = _nonnegative_float(valence.get("pain", 0.0))
        pleasure = _nonnegative_float(valence.get("pleasure", 0.0))
        surprise = _nonnegative_float(getattr(observation, "surprise", 0.0))
        qualities = ExperienceQualities(
            expectation_magnitude=magnitude,
            pain=pain,
            pleasure=pleasure,
            surprise=surprise,
        )
        effects = resolve_experience_effects(
            qualities,
            resolver=self._resolver(),
        )
        if effects.salience <= self.salience_floor:
            return None

        features = FeatureVector(
            {
                key: float(value)
                for key, value in zip(vocabulary, expected, strict=True)
                if float(value) != 0.0
            }
        )
        return ExperienceObject(
            id=f"{self.id_prefix}:{index:04d}",
            temporal_extent=TemporalExtent(
                start=float(observation.t),
                end=end_t,
                order_index=index,
            ),
            features=features,
            kind="internal expectation",
            salience=effects.salience,
            learning_weight=effects.learning_weight,
            modality=INTERNAL_EXPERIENCE_CHANNEL,
            metadata={
                "source": "generated_internal_experience",
                "observation_index": index,
                "qualities": qualities.to_metadata(),
                "resolved_effects": effects.to_metadata(),
                "affect": {
                    "pain": effects.pain,
                    "pleasure": effects.pleasure,
                },
            },
        )

    def _resolver(self) -> ExperienceQualityResolver:
        return ExperienceQualityResolver(
            base_salience=0.0,
            expectation_salience_gain=self.expectation_gain,
            pain_salience_gain=self.pain_gain,
            pleasure_salience_gain=self.pleasure_gain,
            surprise_salience_gain=self.surprise_gain,
            base_learning_weight=self.learning_base,
            surprise_learning_gain=self.learning_surprise_gain,
            pain_learning_gain=self.learning_pain_gain,
            pleasure_learning_gain=self.learning_pleasure_gain,
            magnitude_salience_gain=0.0,
            novelty_salience_gain=0.0,
            variance_salience_gain=0.0,
            beauty_salience_gain=0.0,
            threat_salience_gain=0.0,
            novelty_learning_gain=0.0,
            variance_learning_gain=0.0,
            personal_relevance_learning_gain=0.0,
            beauty_pleasure_gain=0.0,
            novelty_pleasure_gain=0.0,
            threat_pain_gain=0.0,
            overload_pain_gain=0.0,
            ambiguity_pain_gain=0.0,
        )


def generate_internal_experiences(
    episode: Any,
    *,
    generator: InternalExperienceGenerator | None = None,
) -> InputSequence:
    resolved_generator = generator or InternalExperienceGenerator()
    return resolved_generator.generate(episode)


def _observation_end_t(
    observations: list[Any],
    index: int,
    fallback_dt: float,
) -> float:
    start_t = float(observations[index].t)
    if index + 1 < len(observations):
        end_t = float(observations[index + 1].t)
    elif index > 0:
        end_t = start_t + max(fallback_dt, start_t - float(observations[index - 1].t))
    else:
        end_t = start_t + fallback_dt
    if end_t <= start_t:
        end_t = start_t + max(fallback_dt, 1e-9)
    return end_t


def _vector_magnitude(vector: np.ndarray) -> float:
    if vector.size == 0:
        return 0.0
    return float(np.linalg.norm(vector) / np.sqrt(vector.size))


def _metadata_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return dict(value)


def _nonnegative_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return max(0.0, float(value))
