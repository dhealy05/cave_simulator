from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from cave.demonstrations.subjects.distances import EmbeddingFn, pairwise_distance_matrix


RunT = TypeVar("RunT")


def threshold_clusters(
    runs: Sequence[RunT],
    embedding_fn: EmbeddingFn[RunT],
    *,
    threshold: float,
) -> list[list[int]]:
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    distances = pairwise_distance_matrix(runs, embedding_fn)
    seen: set[int] = set()
    clusters: list[list[int]] = []

    for start in range(len(runs)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        cluster: list[int] = []
        while stack:
            index = stack.pop()
            cluster.append(index)
            for candidate, distance in enumerate(distances[index]):
                if candidate not in seen and distance <= threshold:
                    seen.add(candidate)
                    stack.append(candidate)
        clusters.append(sorted(cluster))

    return clusters
