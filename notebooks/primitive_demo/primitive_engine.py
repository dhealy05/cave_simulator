from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PrimitiveStep:
    t: int
    actual: float
    expected: float
    error: float
    surprise: float
    memory_previous: float
    memory: float


@dataclass(frozen=True)
class PrimitiveStep2D:
    t: int
    actual: tuple[float, float]
    expected: tuple[float, float]
    error: tuple[float, float]
    surprise: float
    memory_previous: tuple[float, float]
    memory: tuple[float, float]


@dataclass(frozen=True)
class PrimitiveVectorStep:
    t: int
    actual: tuple[float, ...]
    expected: tuple[float, ...]
    error: tuple[float, ...]
    surprise: float
    memory_previous: tuple[float, ...]
    memory: tuple[float, ...]


def rollout_vectors(
    inputs: Sequence[Sequence[float]],
    *,
    eta: float = 0.5,
    memory_initial: Sequence[float] | None = None,
) -> list[PrimitiveVectorStep]:
    """Run the primitive recurrence on arbitrary fixed-length vectors."""

    if not inputs:
        raise ValueError("rollout requires at least one input vector")
    actuals = [np.array(values, dtype=float) for values in inputs]
    dimensions = actuals[0].shape
    if len(dimensions) != 1:
        raise ValueError("input vectors must be one-dimensional")
    for actual in actuals:
        if actual.shape != dimensions:
            raise ValueError("all input vectors must have the same length")
    if memory_initial is None:
        memory_previous = np.zeros(dimensions, dtype=float)
    else:
        memory_previous = np.array(memory_initial, dtype=float)
        if memory_previous.shape != dimensions:
            raise ValueError("memory_initial must match input vector length")

    rows: list[PrimitiveVectorStep] = []
    for index, actual in enumerate(actuals, start=1):
        expected = memory_previous
        error = actual - expected
        memory = memory_previous + eta * error
        rows.append(
            PrimitiveVectorStep(
                t=index,
                actual=_tuple(actual),
                expected=_tuple(expected),
                error=_tuple(error),
                surprise=float(np.linalg.norm(error)),
                memory_previous=_tuple(memory_previous),
                memory=_tuple(memory),
            )
        )
        memory_previous = memory
    return rows


def rollout_1d(
    inputs: Sequence[float] = (0.10, 0.20, 0.25, 0.80, 0.75, 0.30),
    *,
    eta: float = 0.5,
    memory_initial: float = 0.0,
) -> list[PrimitiveStep]:
    """Run the primitive scalar recurrence."""

    vector_rows = rollout_vectors(
        [(value,) for value in inputs],
        eta=eta,
        memory_initial=(memory_initial,),
    )
    return [
        PrimitiveStep(
            t=row.t,
            actual=row.actual[0],
            expected=row.expected[0],
            error=row.error[0],
            surprise=abs(row.error[0]),
            memory_previous=row.memory_previous[0],
            memory=row.memory[0],
        )
        for row in vector_rows
    ]


def rollout_2d(
    inputs: Sequence[tuple[float, float]] = (
        (0.10, 0.18),
        (0.22, 0.25),
        (0.28, 0.30),
        (0.82, 0.74),
        (0.76, 0.69),
        (0.34, 0.35),
    ),
    *,
    eta: float = 0.5,
    memory_initial: tuple[float, float] = (0.0, 0.0),
) -> list[PrimitiveStep2D]:
    """Run the primitive recurrence in a two-feature plane."""

    vector_rows = rollout_vectors(inputs, eta=eta, memory_initial=memory_initial)
    return [
        PrimitiveStep2D(
            t=row.t,
            actual=(row.actual[0], row.actual[1]),
            expected=(row.expected[0], row.expected[1]),
            error=(row.error[0], row.error[1]),
            surprise=row.surprise,
            memory_previous=(row.memory_previous[0], row.memory_previous[1]),
            memory=(row.memory[0], row.memory[1]),
        )
        for row in vector_rows
    ]


def nearest_prototype(
    vector: Sequence[float],
    prototypes: Mapping[str, Sequence[float]],
) -> tuple[str, float]:
    """Return the nearest prototype label and Euclidean distance."""

    point = np.array(vector, dtype=float)
    best_label = ""
    best_distance = float("inf")
    for label, values in prototypes.items():
        distance = float(np.linalg.norm(point - np.array(values, dtype=float)))
        if distance < best_distance:
            best_label = label
            best_distance = distance
    if not best_label:
        raise ValueError("at least one prototype is required")
    return best_label, best_distance


def _tuple(values: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in values)
