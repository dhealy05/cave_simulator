from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from cave.commitments.attention import INTERNAL_EXPECTATION_CHANNEL, attention_effect
from cave.observation.experience import ExperienceObject, InputSequence, Presentation, presentation_for_object
from cave.demonstrations.simulation import ExperienceModel
from cave.demonstrations.simulation.state import SceneState


Array = np.ndarray


@dataclass(frozen=True)
class EpisodeInput:
    id: str
    kind: str
    start: float
    end: float
    order_index: int
    features: Array
    modality: str = "visual"
    salience: float = 1.0
    learning_weight: float = 1.0
    presentation: Presentation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("EpisodeInput.end must be greater than start")
        object.__setattr__(self, "features", np.asarray(self.features, dtype=float))

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def center(self) -> float:
        return 0.5 * (self.start + self.end)

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end


@dataclass(frozen=True)
class EpisodeObservation:
    t: float
    t_normalized: float
    expected: Array
    actual: Array
    memory_state: Array
    surprise: float
    learning_rate: float
    attention: float
    attention_weights: dict[str, float]
    active_inputs: list[str]
    input_features: dict[str, Array]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected", np.asarray(self.expected, dtype=float))
        object.__setattr__(self, "actual", np.asarray(self.actual, dtype=float))
        object.__setattr__(self, "memory_state", np.asarray(self.memory_state, dtype=float))
        object.__setattr__(
            self,
            "input_features",
            {
                key: np.asarray(value, dtype=float)
                for key, value in self.input_features.items()
            },
        )

    @property
    def error(self) -> Array:
        return self.actual - self.expected


@dataclass(frozen=True)
class Episode:
    source_name: str
    vocabulary: list[str]
    inputs: list[EpisodeInput]
    observations: list[EpisodeObservation]
    duration: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        inputs = sorted(self.inputs, key=lambda item: (item.start, item.order_index))
        observations = sorted(self.observations, key=lambda item: item.t)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "observations", observations)
        if self.duration < 0.0:
            raise ValueError("Episode.duration must be non-negative")

    def input_by_id(self) -> dict[str, EpisodeInput]:
        return {item.id: item for item in self.inputs}

    def active_at(self, t: float) -> list[EpisodeInput]:
        return [item for item in self.inputs if item.contains(t)]


class EpisodeProducer(Protocol):
    name: str

    def run(self, *args: Any, **kwargs: Any) -> Episode:
        ...


@dataclass(frozen=True)
class CaveProducer:
    model: ExperienceModel
    name: str = "cave"

    def run(
        self,
        *,
        start: float = 0.0,
        end: float | None = None,
        dt: float = 0.1,
    ) -> Episode:
        end = self.model.sequence.duration if end is None else end
        states = self.model.run(start=start, end=end, dt=dt)
        return episode_from_cave_states(
            self.name,
            self.model.sequence,
            self.model.vocabulary,
            states,
            metadata={
                "source": "cave.demonstrations.simulation",
                "adapter": "CaveProducer",
                "memory_decay_tau": self.model.subject_state.memory.decay_tau,
                "memory_max_age": self.model.subject_state.memory.max_age,
                "memory_retention": self.model.subject_state.memory.retention,
                "topology_params": self.model.params.topology,
                "attention_curve": [
                    {
                        "t": float(t),
                        "value": self.model.params.attention.value_at(
                            float(t),
                            self.model.sequence.duration,
                        ),
                    }
                    for t in np.linspace(0.0, self.model.sequence.duration, 240)
                ],
            },
        )


CaveEpisodeSource = CaveProducer
EpisodeSource = EpisodeProducer


def episode_from_cave_states(
    source_name: str,
    sequence: InputSequence,
    vocabulary: list[str],
    states: list[SceneState],
    metadata: dict[str, Any] | None = None,
) -> Episode:
    duration = sequence.duration
    inputs = [
        episode_input_from_object(obj, vocabulary)
        for obj in sequence.objects
    ]
    observations = [
        episode_observation_from_scene_state(state, duration)
        for state in states
    ]
    return Episode(
        source_name=source_name,
        vocabulary=list(vocabulary),
        inputs=inputs,
        observations=observations,
        duration=duration,
        metadata={"source": "cave.demonstrations.simulation"} if metadata is None else dict(metadata),
    )


def episode_input_from_object(
    obj: ExperienceObject,
    vocabulary: list[str],
) -> EpisodeInput:
    return EpisodeInput(
        id=obj.id,
        kind=obj.kind,
        start=obj.temporal_extent.start,
        end=obj.temporal_extent.end,
        order_index=obj.temporal_extent.order_index,
        features=obj.features.to_array(vocabulary),
        modality=obj.modality,
        salience=obj.salience,
        learning_weight=obj.learning_weight,
        presentation=presentation_for_object(obj),
        metadata=dict(obj.metadata),
    )


def episode_observation_from_scene_state(
    state: SceneState,
    duration: float,
) -> EpisodeObservation:
    input_features = {
        obj.id: obj.features.to_array(state.vocabulary)
        for obj in state.current_objects
    }
    attention_impact = attention_effect(state.attention)
    channel_impacts = {
        channel: attention_impact * weight
        for channel, weight in state.attention_state.channel_weights.items()
    }
    external_input_impact = sum(
        channel_impacts.get(channel, 0.0)
        for channel in state.sensor_responses
        if channel != INTERNAL_EXPECTATION_CHANNEL
    )
    return EpisodeObservation(
        t=state.t,
        t_normalized=0.0 if duration <= 0.0 else min(1.0, max(0.0, state.t / duration)),
        expected=state.prediction.expected_input.copy(),
        actual=state.input_vector.copy(),
        memory_state=state.subject_state.memory.vector.copy(),
        surprise=state.prediction.surprise,
        learning_rate=state.learning_rate,
        attention=state.attention,
        attention_weights={
            obj.id: state.attention_state.object_impact(obj) * obj.learning_weight
            for obj in state.current_objects
        },
        active_inputs=[obj.id for obj in state.current_objects],
        input_features=input_features,
        metadata={
            "attention_effect": attention_impact,
            "attention_channels": dict(state.attention_state.channel_weights),
            "attention_channel_impacts": channel_impacts,
            "effective_attention": {
                "external_input": external_input_impact,
                "internal_expectation": state.attention_state.internal_expectation_impact(),
            },
            "next_attention_channels": dict(state.next_attention_channel_weights),
            "sensor_channels": {
                channel: response.vector.copy()
                for channel, response in state.sensor_responses.items()
            },
            "learning_importance": state.learning_importance,
            "action": state.action.to_metadata(),
            "valence": state.valence.to_metadata(),
            "objective": state.objective.to_metadata(),
            "workspace": state.workspace.to_metadata(),
            "attended_input": state.attended_input_vector.copy(),
            "expectation_memory_state": state.subject_state.memory.expectation_vector.copy(),
            "expectation_memory_strength": state.subject_state.memory.expectation_strength,
            "workspace_input_mode": state.workspace_input_mode,
        },
    )
