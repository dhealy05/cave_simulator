from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from typing import Any, Protocol

from cave.commitments.agency.state import ActionState
from cave.observation.experience import ExperienceObject


class ActionPolicy(Protocol):
    def choose_action(
        self,
        *,
        current_objects: list[ExperienceObject],
        vocabulary: list[str],
    ) -> ActionState:
        ...


@dataclass(frozen=True)
class FixedActionPolicy:
    def choose_action(
        self,
        *,
        current_objects: list[ExperienceObject],
        vocabulary: list[str],
    ) -> ActionState:
        return ActionState()


@dataclass(frozen=True)
class PreferenceProfile:
    feature_rewards: dict[str, float] = field(default_factory=dict)
    feature_aversions: dict[str, float] = field(default_factory=dict)
    novelty_reward: float = 0.0
    affect_pain_aversion: float = 1.0
    affect_pleasure_reward: float = 1.0
    action_cost: float = 0.0
    approach_gain: float = 0.6
    avoid_gain: float = 0.8
    threshold: float = 1e-12

    def __post_init__(self) -> None:
        if self.action_cost < 0.0:
            raise ValueError("action_cost must be non-negative")
        if self.approach_gain < 0.0:
            raise ValueError("approach_gain must be non-negative")
        if self.avoid_gain < 0.0:
            raise ValueError("avoid_gain must be non-negative")
        if self.threshold < 0.0:
            raise ValueError("threshold must be non-negative")


@dataclass(frozen=True)
class PreferenceActionPolicy:
    preferences: PreferenceProfile

    def choose_action(
        self,
        *,
        current_objects: list[ExperienceObject],
        vocabulary: list[str],
    ) -> ActionState:
        if not current_objects:
            return ActionState()

        scored = [
            (obj, _preference_score(obj, self.preferences))
            for obj in current_objects
        ]
        target, score = max(scored, key=lambda item: abs(item[1]))
        magnitude = abs(score)
        expected_delta = magnitude - self.preferences.action_cost
        if expected_delta <= self.preferences.threshold:
            return ActionState(
                components={
                    "best_score": score,
                    "action_cost": self.preferences.action_cost,
                },
            )

        if score >= 0.0:
            strength = min(1.0, magnitude)
            exposure = 1.0 + self.preferences.approach_gain * strength
            kind = "approach"
        else:
            strength = min(1.0, magnitude)
            exposure = max(0.0, 1.0 - self.preferences.avoid_gain * strength)
            kind = "avoid"

        return ActionState(
            kind=kind,
            target_id=target.id,
            target_channel=target.modality,
            strength=strength,
            expected_utility_delta=expected_delta,
            object_exposure={target.id: exposure},
            components={
                "preference_score": score,
                "action_cost": self.preferences.action_cost,
                "target_exposure": exposure,
            },
        )


def default_action_policy() -> FixedActionPolicy:
    return FixedActionPolicy()


def apply_action_exposure(
    current_objects: list[ExperienceObject],
    action: ActionState,
) -> list[ExperienceObject]:
    exposed = []
    for obj in current_objects:
        exposure = action.exposure_for(obj.id)
        if exposure == 1.0:
            exposed.append(obj)
            continue
        exposed.append(
            dataclass_replace(
                obj,
                salience=max(0.0, obj.salience * exposure),
            )
        )
    return exposed


def _preference_score(obj: ExperienceObject, preferences: PreferenceProfile) -> float:
    score = 0.0
    for feature, weight in preferences.feature_rewards.items():
        score += float(weight) * _feature_value(obj, feature)
    for feature, weight in preferences.feature_aversions.items():
        score -= float(weight) * _feature_value(obj, feature)
    score += preferences.novelty_reward * _feature_value(obj, "novelty")
    affect = _object_affect(obj.metadata.get("affect", {}))
    score += preferences.affect_pleasure_reward * affect["pleasure"]
    score -= preferences.affect_pain_aversion * affect["pain"]
    return score


def _feature_value(obj: ExperienceObject, feature: str) -> float:
    return float(obj.features.values.get(feature, 0.0))


def _object_affect(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {"pain": 0.0, "pleasure": 0.0}
    return {
        "pain": _nonnegative_float(value.get("pain", 0.0)),
        "pleasure": _nonnegative_float(value.get("pleasure", 0.0)),
    }


def _nonnegative_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return max(0.0, float(value))
