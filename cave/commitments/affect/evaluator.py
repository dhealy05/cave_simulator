from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from cave.commitments.affect.state import ValenceState
from cave.commitments.attention import AttentionState
from cave.observation.experience import ExperienceObject
from cave.commitments.prediction import PredictionState


class ValenceEvaluator(Protocol):
    def evaluate(
        self,
        *,
        current_objects: list[ExperienceObject],
        attention: AttentionState,
        prediction: PredictionState,
        previous: ValenceState | None = None,
    ) -> ValenceState:
        ...


@dataclass(frozen=True)
class MetadataValenceEvaluator:
    surprise_pain_gain: float = 0.0
    relief_pleasure_gain: float = 0.0
    metadata_key: str = "affect"

    def __post_init__(self) -> None:
        if self.surprise_pain_gain < 0.0:
            raise ValueError("surprise_pain_gain must be non-negative")
        if self.relief_pleasure_gain < 0.0:
            raise ValueError("relief_pleasure_gain must be non-negative")

    def evaluate(
        self,
        *,
        current_objects: list[ExperienceObject],
        attention: AttentionState,
        prediction: PredictionState,
        previous: ValenceState | None = None,
    ) -> ValenceState:
        object_pain = 0.0
        object_pleasure = 0.0
        channel_pain: dict[str, float] = {}
        channel_pleasure: dict[str, float] = {}

        for obj in current_objects:
            affect = _object_affect(obj.metadata.get(self.metadata_key, {}))
            impact = attention.object_impact(obj) * max(0.0, obj.salience)
            pain = impact * affect["pain"]
            pleasure = impact * affect["pleasure"]
            object_pain += pain
            object_pleasure += pleasure
            channel_pain[obj.modality] = channel_pain.get(obj.modality, 0.0) + pain
            channel_pleasure[obj.modality] = (
                channel_pleasure.get(obj.modality, 0.0) + pleasure
            )

        surprise_pain = self.surprise_pain_gain * max(0.0, prediction.surprise)
        raw_pain = object_pain + surprise_pain
        previous_pain = 0.0 if previous is None else previous.pain
        relief_pleasure = self.relief_pleasure_gain * max(0.0, previous_pain - raw_pain)
        pleasure = object_pleasure + relief_pleasure

        return ValenceState(
            pain=raw_pain,
            pleasure=pleasure,
            components={
                "object_pain": object_pain,
                "object_pleasure": object_pleasure,
                "surprise_pain": surprise_pain,
                "relief_pleasure": relief_pleasure,
            },
            channel_pain=channel_pain,
            channel_pleasure=channel_pleasure,
        )


def default_valence_evaluator() -> MetadataValenceEvaluator:
    return MetadataValenceEvaluator()


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
