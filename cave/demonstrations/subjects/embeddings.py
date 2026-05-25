from __future__ import annotations

import numpy as np

from cave.observation.structural import (
    actual_trajectory_embedding as episode_actual_trajectory_embedding,
    active_context_embedding as episode_active_context_embedding,
    expected_trajectory_embedding as episode_expected_trajectory_embedding,
    final_memory_embedding as episode_final_memory_embedding,
    memory_trajectory,
    memory_trajectory_embedding as episode_memory_trajectory_embedding,
    prediction_error_trajectory_embedding as episode_prediction_error_trajectory_embedding,
    surprise_learning_trajectory_embedding as episode_surprise_learning_trajectory_embedding,
    subjective_trajectory_embedding as episode_subjective_trajectory_embedding,
)
from cave.demonstrations.subjects.runs import SubjectRun


Array = np.ndarray


def final_memory_embedding(run: SubjectRun) -> Array:
    return episode_final_memory_embedding(run.episode)


def memory_trajectory_embedding(
    run: SubjectRun,
    *,
    samples: int = 128,
) -> Array:
    return episode_memory_trajectory_embedding(run.episode, samples=samples)


def attended_input_trajectory_embedding(
    run: SubjectRun,
    *,
    samples: int = 128,
) -> Array:
    return episode_actual_trajectory_embedding(run.episode, samples=samples)


def expected_input_trajectory_embedding(
    run: SubjectRun,
    *,
    samples: int = 128,
) -> Array:
    return episode_expected_trajectory_embedding(run.episode, samples=samples)


def prediction_error_trajectory_embedding(
    run: SubjectRun,
    *,
    samples: int = 128,
) -> Array:
    return episode_prediction_error_trajectory_embedding(run.episode, samples=samples)


def surprise_learning_trajectory_embedding(
    run: SubjectRun,
    *,
    samples: int = 128,
) -> Array:
    return episode_surprise_learning_trajectory_embedding(run.episode, samples=samples)


def active_context_embedding(
    run: SubjectRun,
    *,
    samples: int = 128,
) -> Array:
    return episode_active_context_embedding(run.episode, samples=samples)


def subjective_trajectory_embedding(
    run: SubjectRun,
    *,
    samples: int = 128,
) -> Array:
    return episode_subjective_trajectory_embedding(run.episode, samples=samples)


def state_effect_embedding(
    run: SubjectRun,
    *,
    samples: int = 128,
) -> Array:
    observed = _memory_trajectory(run)
    baseline = no_input_memory_baseline(run)
    return resample_trajectory(observed - baseline, samples=samples)


def internal_experience_embedding(
    run: SubjectRun,
    *,
    samples: int = 128,
) -> Array:
    return subjective_trajectory_embedding(run, samples=samples)


def experience_effect_embedding(
    run: SubjectRun,
    *,
    samples: int = 128,
) -> Array:
    return state_effect_embedding(run, samples=samples)


def no_input_memory_baseline(run: SubjectRun) -> Array:
    if not run.episode.observations:
        raise ValueError("run has no observations")
    retention = run.subject.initial_state.memory.retention
    vector = run.subject.initial_state.memory.vector.copy()
    baseline = []
    for _observation in run.episode.observations:
        vector = retention * vector
        baseline.append(vector.copy())
    return np.stack(baseline, axis=0)


def resample_trajectory(trajectory: Array, *, samples: int = 128) -> Array:
    from cave.observation.structural import (
        resample_trajectory as episode_resample_trajectory,
    )

    return episode_resample_trajectory(trajectory, samples=samples)


def _memory_trajectory(run: SubjectRun) -> Array:
    return memory_trajectory(run.episode)
