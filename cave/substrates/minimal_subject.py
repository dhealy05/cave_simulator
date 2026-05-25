from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cave.observation.episodes import Episode, EpisodeInput, EpisodeObservation
from cave.observation.experience import ExperienceObject, InputSequence, presentation_for_object


Array = np.ndarray


@dataclass(frozen=True)
class MinimalSubjectConfig:
    workspace_capacity: int = 2
    memory_decay: float = 0.96
    similarity_gain: float = 6.0
    feature_priority_learning: float = 0.35
    memory_learning: float = 1.0
    memory_mode: str = "value"
    diagnostic_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.workspace_capacity <= 0:
            raise ValueError("workspace_capacity must be positive")
        if not 0.0 <= self.memory_decay <= 1.0:
            raise ValueError("memory_decay must be between 0 and 1")
        if self.similarity_gain < 0.0:
            raise ValueError("similarity_gain must be non-negative")
        if self.feature_priority_learning < 0.0:
            raise ValueError("feature_priority_learning must be non-negative")
        if self.memory_learning < 0.0:
            raise ValueError("memory_learning must be non-negative")
        if self.memory_mode not in {"value", "frequency"}:
            raise ValueError("memory_mode must be 'value' or 'frequency'")


@dataclass(frozen=True)
class AssociativeMemoryTrace:
    context: Array
    successor: Array
    value: float
    strength: float

    def decayed(self, decay: float) -> "AssociativeMemoryTrace":
        return AssociativeMemoryTrace(
            context=self.context.copy(),
            successor=self.successor.copy(),
            value=self.value,
            strength=self.strength * decay,
        )


@dataclass
class MinimalSubjectState:
    vocabulary: list[str]
    preference_vector: Array
    config: MinimalSubjectConfig
    feature_priority: Array = field(init=False)
    traces: list[AssociativeMemoryTrace] = field(default_factory=list)
    previous_workspace: Array | None = None

    def __post_init__(self) -> None:
        preference = np.asarray(self.preference_vector, dtype=float)
        if preference.shape != (len(self.vocabulary),):
            raise ValueError("preference vector length must match vocabulary")
        self.preference_vector = preference
        self.feature_priority = np.zeros(len(self.vocabulary), dtype=float)

    def expected_from_memory(self) -> Array:
        if self.previous_workspace is None or not self.traces:
            return np.zeros(len(self.vocabulary), dtype=float)
        query = self.previous_workspace
        weights = np.array(
            [
                trace.strength
                * np.exp(self.config.similarity_gain * _cosine_similarity(query, trace.context))
                for trace in self.traces
            ],
            dtype=float,
        )
        total = float(np.sum(weights))
        if total <= 1e-12:
            return np.zeros(len(self.vocabulary), dtype=float)
        successor = np.zeros(len(self.vocabulary), dtype=float)
        for weight, trace in zip(weights / total, self.traces):
            successor += float(weight) * trace.successor
        return successor

    def workspace(self, input_vector: Array) -> tuple[Array, dict[str, float]]:
        vector = np.asarray(input_vector, dtype=float)
        if vector.shape != (len(self.vocabulary),):
            raise ValueError("input vector length must match vocabulary")
        if vector.size == 0:
            return vector.copy(), {}

        priority = 1.0 + np.tanh(np.maximum(0.0, np.abs(self.feature_priority)))
        scores = np.abs(vector) * priority
        active = np.flatnonzero(np.abs(vector) > 1e-12)
        if active.size == 0:
            return np.zeros_like(vector), {}

        count = min(self.config.workspace_capacity, active.size)
        order = active[np.argsort(-scores[active], kind="stable")[:count]]
        workspace = np.zeros_like(vector, dtype=float)
        workspace[order] = vector[order] * priority[order]
        mass = float(np.sum(np.abs(workspace)))
        weights = {}
        if mass > 0.0:
            weights = {
                self.vocabulary[index]: float(abs(workspace[index]) / mass)
                for index in order
            }
        return workspace, weights

    def update(self, workspace: Array, value: float) -> None:
        self.traces = [
            trace.decayed(self.config.memory_decay)
            for trace in self.traces
            if trace.strength * self.config.memory_decay > 1e-12
        ]
        if self.previous_workspace is not None:
            if self.config.memory_mode == "frequency":
                importance = self.config.memory_learning
            else:
                importance = self.config.memory_learning * abs(float(value))
            if importance > 1e-12:
                self.traces.append(
                    AssociativeMemoryTrace(
                        context=self.previous_workspace.copy(),
                        successor=workspace.copy(),
                        value=float(value),
                        strength=importance,
                    )
                )
                self.feature_priority += (
                    self.config.feature_priority_learning
                    * importance
                    * np.abs(self.previous_workspace)
                )
        self.previous_workspace = workspace.copy()

    def memory_vector(self) -> Array:
        if not self.traces:
            return np.zeros(len(self.vocabulary), dtype=float)
        total = sum(trace.strength for trace in self.traces)
        if total <= 1e-12:
            return np.zeros(len(self.vocabulary), dtype=float)
        vector = np.zeros(len(self.vocabulary), dtype=float)
        for trace in self.traces:
            vector += (trace.strength / total) * trace.successor
        return vector

    def memory_geometry(self) -> dict[str, float]:
        if not self.traces:
            return {
                "trace_count": 0,
                "strength_total": 0.0,
                "value_separation": 0.0,
                "concentration": 0.0,
            }
        strengths = np.array([trace.strength for trace in self.traces], dtype=float)
        values = np.array([trace.value for trace in self.traces], dtype=float)
        total = float(np.sum(strengths))
        positive = float(np.sum(strengths[values > 0.0]))
        negative = float(np.sum(strengths[values < 0.0]))
        return {
            "trace_count": len(self.traces),
            "strength_total": total,
            "value_separation": abs(positive - negative) / max(total, 1e-12),
            "concentration": float(np.max(strengths) / max(total, 1e-12)),
        }


def run_minimal_subject(
    sequence: InputSequence,
    *,
    vocabulary: list[str],
    preference_vector: Array,
    config: MinimalSubjectConfig | None = None,
    source_name: str = "minimal-subject",
) -> Episode:
    config = config or MinimalSubjectConfig()
    state = MinimalSubjectState(
        vocabulary=list(vocabulary),
        preference_vector=np.asarray(preference_vector, dtype=float),
        config=config,
    )
    inputs = [
        EpisodeInput(
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
        for obj in sequence.objects
    ]
    observations: list[EpisodeObservation] = []
    for obj in sequence.objects:
        t = obj.temporal_extent.start
        expected = state.expected_from_memory()
        external = obj.features.to_array(vocabulary) * obj.salience
        workspace, workspace_weights = state.workspace(external)
        error = workspace - expected
        surprise = _normalized_norm(error)
        value = float(np.dot(workspace, state.preference_vector))
        utility = value - surprise
        attention_weight = _object_attention_weight(obj, vocabulary, workspace)
        metadata = {
            "minimal_subject": {
                "workspace_weights": workspace_weights,
                "feature_priority": state.feature_priority.copy(),
                "preference_value": value,
                "utility": utility,
                "diagnostic_attention": _diagnostic_attention(
                    workspace_weights,
                    config.diagnostic_features,
                ),
                "memory_geometry": state.memory_geometry(),
            },
            "valence": {
                "pain": max(0.0, -value),
                "pleasure": max(0.0, value),
                "net": value,
                "components": {"preference_value": value},
            },
            "objective": {
                "utility": utility,
                "prediction_cost": surprise,
                "pain_cost": max(0.0, -value),
                "pleasure_gain": max(0.0, value),
                "attention_cost": 0.0,
                "compression_cost": 0.0,
            },
        }
        observations.append(
            EpisodeObservation(
                t=t,
                t_normalized=0.0 if sequence.duration <= 0.0 else t / sequence.duration,
                expected=expected,
                actual=workspace,
                memory_state=state.memory_vector(),
                surprise=surprise,
                learning_rate=abs(value),
                attention=1.0,
                attention_weights={obj.id: attention_weight},
                active_inputs=[obj.id],
                input_features={obj.id: external},
                metadata=metadata,
            )
        )
        state.update(workspace, value)

    return Episode(
        source_name=source_name,
        vocabulary=list(vocabulary),
        inputs=inputs,
        observations=observations,
        duration=sequence.duration,
        metadata={
            "source": "minimal_subject",
            "adapter": "MinimalSubject",
            "config": config,
            "preference_vector": np.asarray(preference_vector, dtype=float),
        },
    )


def emergence_metrics(episode: Episode) -> dict[str, float]:
    active_observations = [
        obs
        for obs in episode.observations
        if _normalized_norm(obs.actual) > 1e-12
    ]
    outcome_observations = [
        obs
        for obs in active_observations
        if any(
            "preferred" in input_id or "threat" in input_id
            for input_id in obs.active_inputs
        )
    ]
    cue_observations = [
        obs
        for obs in active_observations
        if any("cue_" in input_id for input_id in obs.active_inputs)
    ]
    observations = outcome_observations or active_observations
    if not observations:
        return {
            "early_surprise": 0.0,
            "late_surprise": 0.0,
            "surprise_drop": 0.0,
            "early_skill": 0.0,
            "late_skill": 0.0,
            "skill_gain": 0.0,
            "early_diagnostic_attention": 0.0,
            "late_diagnostic_attention": 0.0,
            "diagnostic_attention_gain": 0.0,
            "late_memory_strength": 0.0,
            "late_value_separation": 0.0,
            "late_memory_concentration": 0.0,
            "utility_mean": 0.0,
        }
    surprise = np.array([obs.surprise for obs in observations], dtype=float)
    baseline = np.array([_normalized_norm(obs.actual) for obs in observations], dtype=float)
    skill = 1.0 - surprise / np.maximum(baseline, 1e-12)
    diagnostic_source = cue_observations or active_observations
    diagnostic = np.array(
        [
            float(
                obs.metadata.get("minimal_subject", {}).get(
                    "diagnostic_attention",
                    0.0,
                )
            )
            for obs in diagnostic_source
        ],
        dtype=float,
    )
    utility = np.array(
        [
            float(obs.metadata.get("minimal_subject", {}).get("utility", 0.0))
            for obs in observations
        ],
        dtype=float,
    )
    count = max(1, min(4, len(observations) // 3))
    diagnostic_count = max(1, min(4, diagnostic.size // 3))
    late_geometry = observations[-1].metadata.get("minimal_subject", {}).get(
        "memory_geometry",
        {},
    )
    return {
        "early_surprise": float(np.mean(surprise[:count])),
        "late_surprise": float(np.mean(surprise[-count:])),
        "surprise_drop": float(np.mean(surprise[:count]) - np.mean(surprise[-count:])),
        "early_skill": float(np.mean(skill[:count])),
        "late_skill": float(np.mean(skill[-count:])),
        "skill_gain": float(np.mean(skill[-count:]) - np.mean(skill[:count])),
        "early_diagnostic_attention": float(np.mean(diagnostic[:diagnostic_count])),
        "late_diagnostic_attention": float(np.mean(diagnostic[-diagnostic_count:])),
        "diagnostic_attention_gain": float(
            np.mean(diagnostic[-diagnostic_count:]) - np.mean(diagnostic[:diagnostic_count])
        ),
        "late_memory_strength": float(late_geometry.get("strength_total", 0.0)),
        "late_value_separation": float(late_geometry.get("value_separation", 0.0)),
        "late_memory_concentration": float(late_geometry.get("concentration", 0.0)),
        "utility_mean": float(np.mean(utility)),
    }


def pressure_role_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    baseline = metrics["minimal-preference"]
    no_memory = metrics["no-memory"]
    no_preference = metrics["no-preference"]
    no_bottleneck = metrics["no-bottleneck"]
    frequency = metrics["frequency-memory"]
    shuffled = metrics["shuffled"]
    return {
        "workspace_pressure_attention_like_selection": {
            "with_bottleneck_late_diagnostic_attention": baseline["late_diagnostic_attention"],
            "no_bottleneck_late_diagnostic_attention": no_bottleneck["late_diagnostic_attention"],
            "selection_margin": (
                baseline["late_diagnostic_attention"]
                - no_bottleneck["late_diagnostic_attention"]
            ),
            "claim_kind": "diagnostic_input_weight_concentration",
            "full_dynamic_attention_claimed": False,
            "internal_expectation_channel_claimed": False,
        },
        "temporal_recurrence_memory_like_trace": {
            "with_memory_strength": baseline["late_memory_strength"],
            "no_memory_strength": no_memory["late_memory_strength"],
            "memory_margin": (
                baseline["late_memory_strength"]
                - no_memory["late_memory_strength"]
            ),
        },
        "preference_pressure_value_shaped_memory": {
            "value_memory_strength": baseline["late_memory_strength"],
            "no_preference_strength": no_preference["late_memory_strength"],
            "frequency_memory_strength": frequency["late_memory_strength"],
            "value_strength_margin": (
                baseline["late_memory_strength"]
                - no_preference["late_memory_strength"]
            ),
            "value_memory_separation": baseline["late_value_separation"],
            "frequency_memory_separation": frequency["late_value_separation"],
            "value_separation_margin": (
                baseline["late_value_separation"]
                - frequency["late_value_separation"]
            ),
        },
        "delayed_consequence_prediction_like_readout": {
            "structured_late_skill": baseline["late_skill"],
            "no_memory_late_skill": no_memory["late_skill"],
            "shuffled_late_skill": shuffled["late_skill"],
            "readout_margin": baseline["late_skill"] - no_memory["late_skill"],
            "structure_margin": baseline["late_skill"] - shuffled["late_skill"],
        },
        "repeated_trajectory_topology_like_geometry": {
            "value_separation": baseline["late_value_separation"],
            "memory_concentration": baseline["late_memory_concentration"],
            "no_memory_value_separation": no_memory["late_value_separation"],
        },
    }


def _object_attention_weight(
    obj: ExperienceObject,
    vocabulary: list[str],
    workspace: Array,
) -> float:
    source = obj.features.to_array(vocabulary) * obj.salience
    source_mass = float(np.sum(np.abs(source)))
    if source_mass <= 1e-12:
        return 0.0
    return min(1.0, float(np.sum(np.abs(workspace)) / source_mass))


def _diagnostic_attention(
    workspace_weights: dict[str, float],
    diagnostic_features: tuple[str, ...],
) -> float:
    return sum(float(workspace_weights.get(feature, 0.0)) for feature in diagnostic_features)


def _cosine_similarity(a: Array, b: Array) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denominator)


def _normalized_norm(value: Array) -> float:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.linalg.norm(array.ravel()) / np.sqrt(array.size))
