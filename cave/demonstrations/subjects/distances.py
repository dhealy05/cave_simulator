from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np


Array = np.ndarray
RunT = TypeVar("RunT")
EmbeddingFn = Callable[[RunT], Array]


def embedding_distance(a: Array, b: Array) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"embedding shapes differ: {a.shape} != {b.shape}")
    diff = a - b
    return float(np.linalg.norm(diff.ravel()) / np.sqrt(diff.size))


def pairwise_distance_matrix(
    runs: Sequence[RunT],
    embedding_fn: EmbeddingFn[RunT],
) -> Array:
    embeddings = [embedding_fn(run) for run in runs]
    count = len(embeddings)
    distances = np.zeros((count, count), dtype=float)
    for i in range(count):
        for j in range(i + 1, count):
            distance = embedding_distance(embeddings[i], embeddings[j])
            distances[i, j] = distance
            distances[j, i] = distance
    return distances


def nearest_neighbors(
    runs: Sequence[RunT],
    embedding_fn: EmbeddingFn[RunT],
    *,
    k: int = 1,
) -> list[list[tuple[int, float]]]:
    if k < 0:
        raise ValueError("k must be non-negative")
    distances = pairwise_distance_matrix(runs, embedding_fn)
    neighbors: list[list[tuple[int, float]]] = []
    for i, row in enumerate(distances):
        order = [index for index in np.argsort(row) if index != i]
        neighbors.append(
            [(index, float(row[index])) for index in order[:k]]
        )
    return neighbors
