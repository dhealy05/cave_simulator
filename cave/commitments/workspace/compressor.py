from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from cave.observation.experience import Array
from cave.commitments.workspace.state import WorkspaceState


class WorkspaceCompressor(Protocol):
    def compress(self, vector: Array, vocabulary: list[str]) -> WorkspaceState:
        ...


@dataclass(frozen=True)
class IdentityWorkspaceCompressor:
    def compress(self, vector: Array, vocabulary: list[str]) -> WorkspaceState:
        vector = np.asarray(vector, dtype=float)
        energy = _energy(vector)
        return WorkspaceState(
            represented=vector.copy(),
            reconstructed=vector.copy(),
            retained_energy=energy,
            dropped_energy=0.0,
            compression_cost=0.0,
            reconstruction_error=0.0,
            active_features=list(vocabulary[: vector.size]),
            method="identity",
        )


@dataclass(frozen=True)
class TopKWorkspaceCompressor:
    capacity: int

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")

    def compress(self, vector: Array, vocabulary: list[str]) -> WorkspaceState:
        vector = np.asarray(vector, dtype=float)
        if vector.size == 0:
            return WorkspaceState(
                represented=vector.copy(),
                reconstructed=vector.copy(),
                method="top_k",
            )

        count = min(self.capacity, vector.size)
        order = np.argsort(-np.abs(vector), kind="stable")[:count]
        order = np.sort(order)
        reconstructed = np.zeros_like(vector, dtype=float)
        reconstructed[order] = vector[order]
        represented = vector[order].copy()
        total_energy = _energy(vector)
        retained_energy = _energy(reconstructed)
        dropped_energy = max(0.0, total_energy - retained_energy)
        compression_cost = 0.0 if total_energy <= 1e-12 else dropped_energy / total_energy
        reconstruction_error = (
            0.0
            if vector.size == 0
            else float(np.linalg.norm(vector - reconstructed) / np.sqrt(vector.size))
        )
        return WorkspaceState(
            represented=represented,
            reconstructed=reconstructed,
            retained_energy=retained_energy,
            dropped_energy=dropped_energy,
            compression_cost=compression_cost,
            reconstruction_error=reconstruction_error,
            active_features=[
                vocabulary[index] if index < len(vocabulary) else str(index)
                for index in order
            ],
            method="top_k",
        )


def default_workspace_compressor() -> IdentityWorkspaceCompressor:
    return IdentityWorkspaceCompressor()


def _energy(vector: Array) -> float:
    vector = np.asarray(vector, dtype=float)
    return float(np.sum(vector * vector))
