from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cave.observation.episodes import Episode


@dataclass(frozen=True)
class CaveNetComparison:
    ok: bool
    metrics: dict[str, float]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "metrics": dict(self.metrics),
            "errors": list(self.errors),
        }


def compare_cavenet_to_cave(
    cave_episode: Episode,
    cavenet_episode: Episode,
    *,
    tolerance: float = 1e-9,
) -> CaveNetComparison:
    if len(cave_episode.observations) != len(cavenet_episode.observations):
        return CaveNetComparison(
            ok=False,
            metrics={},
            errors=(
                "observation counts differ: "
                f"{len(cave_episode.observations)} != {len(cavenet_episode.observations)}",
            ),
        )

    actual_distances = []
    expected_distances = []
    error_distances = []
    surprise_distances = []
    memory_distances = []
    learning_distances = []
    errors = []
    for index, (cave_obs, cavenet_obs) in enumerate(
        zip(cave_episode.observations, cavenet_episode.observations)
    ):
        actual_distances.append(_distance(cave_obs.actual, cavenet_obs.actual))
        expected_distances.append(_distance(cave_obs.expected, cavenet_obs.expected))
        error_distances.append(_distance(cave_obs.error, cavenet_obs.error))
        surprise_distances.append(abs(cave_obs.surprise - cavenet_obs.surprise))
        memory_distances.append(_distance(cave_obs.memory_state, cavenet_obs.memory_state))
        learning_distances.append(abs(cave_obs.learning_rate - cavenet_obs.learning_rate))
        if abs(cave_obs.t - cavenet_obs.t) > tolerance:
            errors.append(f"frame {index}: times differ")
        if cave_obs.active_inputs != cavenet_obs.active_inputs:
            errors.append(f"frame {index}: active inputs differ")

    metrics = {
        "max_actual_distance": max(actual_distances, default=0.0),
        "max_expected_distance": max(expected_distances, default=0.0),
        "max_error_distance": max(error_distances, default=0.0),
        "max_surprise_distance": max(surprise_distances, default=0.0),
        "max_memory_distance": max(memory_distances, default=0.0),
        "max_learning_rate_distance": max(learning_distances, default=0.0),
    }
    for key, value in metrics.items():
        if value > tolerance:
            errors.append(f"{key} {value:g} exceeds tolerance {tolerance:g}")
    return CaveNetComparison(
        ok=not errors,
        metrics=metrics,
        errors=tuple(errors),
    )


def _distance(a, b) -> float:
    first = np.asarray(a, dtype=float)
    second = np.asarray(b, dtype=float)
    if first.shape != second.shape:
        return float("inf")
    if first.size == 0:
        return 0.0
    return float(np.linalg.norm((first - second).ravel()) / np.sqrt(first.size))
