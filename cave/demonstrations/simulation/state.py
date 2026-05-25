from __future__ import annotations

from dataclasses import dataclass

from cave.commitments.agency import ActionState
from cave.commitments.attention import AttentionProfile, AttentionState
from cave.commitments.affect import ValenceState
from cave.observation.experience import Array, ExperienceObject, InputSequence
from cave.commitments.objective import ObjectiveState
from cave.commitments.prediction import PredictionState
from cave.observation.sensing import SensorResponse
from cave.demonstrations.state import SubjectState
from cave.commitments.workspace import WorkspaceState


@dataclass(frozen=True)
class SceneState:
    t: float
    sequence: InputSequence
    vocabulary: list[str]
    current_objects: list[ExperienceObject]
    action: ActionState
    attention: float
    attention_state: AttentionState
    next_attention_channel_weights: dict[str, float]
    attention_profile: AttentionProfile
    sensor_responses: dict[str, SensorResponse]
    attended_input_vector: Array
    input_vector: Array
    workspace: WorkspaceState
    workspace_input_mode: str
    prediction: PredictionState
    valence: ValenceState
    objective: ObjectiveState
    learning_rate: float
    learning_importance: float
    subject_state: SubjectState

    @property
    def topology(self):
        return self.subject_state.topology
