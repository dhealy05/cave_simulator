from __future__ import annotations

from dataclasses import dataclass

from cave.commitments.attention.state import AttentionState, coerce_attention_state
from cave.observation.experience import Array, ExperienceObject
from cave.commitments.memory import MemoryTrace
from cave.commitments.prediction.state import PredictionState
from cave.commitments.topology import SubjectiveTopologyParams, SubjectiveTopologyState


@dataclass
class SubjectState:
    memory: MemoryTrace
    topology: SubjectiveTopologyState

    @classmethod
    def initial(
        cls,
        memory: MemoryTrace,
        topology_params: SubjectiveTopologyParams,
    ) -> "SubjectState":
        return cls(
            memory=memory,
            topology=SubjectiveTopologyState.initial(
                feature_x=topology_params.feature_x,
                feature_y=topology_params.feature_y,
                bounds=topology_params.bounds,
                resolution=topology_params.resolution,
                prior=topology_params.prior,
            ),
        )

    def update(
        self,
        t: float,
        u_t: Array,
        current_objects: list[ExperienceObject],
        attention: AttentionState | float,
        topology_params: SubjectiveTopologyParams,
        *,
        vocabulary: list[str],
        prediction: PredictionState,
        learning_rate: float | None = None,
    ) -> None:
        attention = coerce_attention_state(attention)
        self.memory.update(
            t,
            u_t,
            current_objects,
            attention,
            learning_rate=learning_rate,
            expected_input=prediction.expected_input,
        )
        self.topology = self.topology.update(
            self.memory.snapshot(),
            current_objects,
            topology_params,
            current_attention=attention,
            vocabulary=vocabulary,
            expected_input=prediction.expected_input,
            actual_input=u_t,
            after_input=self.memory.vector,
            surprise=prediction.surprise,
            learning_rate=0.0 if learning_rate is None else learning_rate,
        )

    def snapshot(self) -> "SubjectState":
        correction = self.topology.correction
        return SubjectState(
            memory=self.memory.snapshot(),
            topology=SubjectiveTopologyState(
                density=self.topology.density.copy(),
                wells=list(self.topology.wells),
                expected_density=self.topology.expected_density.copy(),
                actual_density=self.topology.actual_density.copy(),
                correction=correction,
                bounds=self.topology.bounds,
                feature_x=self.topology.feature_x,
                feature_y=self.topology.feature_y,
                last_focus=(
                    None
                    if self.topology.last_focus is None
                    else self.topology.last_focus.copy()
                ),
            ),
        )
