from __future__ import annotations

import numpy as np

from cave.commitments.affect import ValenceState
from cave.commitments.attention import AttentionState
from cave.observation.experience import Array, ExperienceObject
from cave.commitments.learning import LearningRule
from cave.commitments.memory import MemoryTrace
from cave.commitments.objective import ObjectiveState
from cave.commitments.prediction import MemoryVectorPredictor, PredictionState
from cave.observation.sensing import SensorResponse, Sensorium
from cave.commitments.workspace import WorkspaceCompressor, WorkspaceState


def attention_gate(
    *,
    sensorium: Sensorium,
    sensor_responses: dict[str, SensorResponse],
    attention: AttentionState,
    vocabulary: list[str],
    gain: float = 1.0,
) -> Array:
    return float(gain) * sensorium.attended_input(sensor_responses, attention, vocabulary)


def workspace_block(
    *,
    compressor: WorkspaceCompressor,
    attended_input: Array,
    vocabulary: list[str],
) -> WorkspaceState:
    return compressor.compress(attended_input, vocabulary)


def state_input_from_workspace(
    *,
    attended_input: Array,
    workspace: WorkspaceState,
    mode: str,
    gain: float = 1.0,
) -> Array:
    if mode == "actual":
        return float(gain) * np.asarray(attended_input, dtype=float).copy()
    if mode == "workspace":
        return float(gain) * workspace.reconstructed.copy()
    raise ValueError("workspace input mode must be 'actual' or 'workspace'")


def expectation_readout(
    memory: MemoryTrace,
    vocabulary: list[str],
    *,
    attention: AttentionState,
    gain: float = 1.0,
) -> Array:
    vector = np.asarray(memory.vector, dtype=float)
    if vector.shape != (len(vocabulary),):
        raise ValueError("memory vector length must match vocabulary length")
    return float(gain) * attention.internal_expectation_impact() * vector.copy()


def error_surprise_block(
    *,
    expected_input: Array,
    actual_input: Array,
    surprise_gain: float = 1.0,
) -> PredictionState:
    prediction = MemoryVectorPredictor().evaluate(expected_input, actual_input)
    if surprise_gain == 1.0:
        return prediction
    return PredictionState(
        expected_input=prediction.expected_input,
        prediction_error=prediction.prediction_error,
        surprise=prediction.surprise * float(surprise_gain),
    )


def learning_importance(current_objects: list[ExperienceObject]) -> float:
    if not current_objects:
        return 1.0
    total_salience = sum(max(0.0, obj.salience) for obj in current_objects)
    if total_salience <= 0.0:
        return 1.0
    return sum(
        max(0.0, obj.learning_weight) * max(0.0, obj.salience)
        for obj in current_objects
    ) / total_salience


def learning_rate_block(
    *,
    learning_rule: LearningRule,
    base_rate: float,
    attention: AttentionState,
    importance: float,
    surprise: float,
    gain: float = 1.0,
) -> float:
    rate = learning_rule.learning_rate(
        base_rate=base_rate,
        attention=attention,
        importance=importance,
        surprise=surprise,
    )
    return min(1.0, max(0.0, float(gain) * rate))


def value_readout_metadata(
    *,
    valence: ValenceState,
    objective: ObjectiveState,
) -> dict[str, object]:
    return {
        "valence": valence.to_metadata(),
        "objective": objective.to_metadata(),
    }
