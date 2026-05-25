from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cave.demonstrations.examples import DEFAULT_VOCABULARY, default_model_params
from cave.commitments.memory import MemoryTrace
from cave.observation.sensing import Sensorium, default_sensorium
from cave.demonstrations.simulation import ModelParams
from cave.demonstrations.state import SubjectState


@dataclass(frozen=True)
class SubjectProfile:
    id: str
    params: ModelParams
    initial_state: SubjectState
    vocabulary: list[str]
    sensorium: Sensorium

    def fresh_state(self) -> SubjectState:
        return self.initial_state.snapshot()


def make_subject(
    id: str,
    *,
    params: ModelParams | None = None,
    initial_state: SubjectState | None = None,
    vocabulary: list[str] | None = None,
    sensorium: Sensorium | None = None,
) -> SubjectProfile:
    vocabulary = list(DEFAULT_VOCABULARY if vocabulary is None else vocabulary)
    params = params or default_model_params()
    sensorium = sensorium or default_sensorium()
    if initial_state is None:
        trace = MemoryTrace(
            vector=np.zeros(len(vocabulary), dtype=float),
            retention=params.memory.retention,
            decay_tau=params.memory.decay_tau,
            max_age=params.memory.max_age,
        )
        initial_state = SubjectState.initial(trace, params.topology)
    return SubjectProfile(
        id=id,
        params=params,
        initial_state=initial_state.snapshot(),
        vocabulary=vocabulary,
        sensorium=sensorium,
    )


Subject = SubjectProfile
