from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cave.observation.experience import Array


@dataclass(frozen=True)
class WorkspaceState:
    represented: Array
    reconstructed: Array
    retained_energy: float = 0.0
    dropped_energy: float = 0.0
    compression_cost: float = 0.0
    reconstruction_error: float = 0.0
    active_features: list[str] = field(default_factory=list)
    method: str = "identity"

    def __post_init__(self) -> None:
        represented = np.asarray(self.represented, dtype=float)
        reconstructed = np.asarray(self.reconstructed, dtype=float)
        object.__setattr__(self, "represented", represented)
        object.__setattr__(self, "reconstructed", reconstructed)
        object.__setattr__(self, "retained_energy", max(0.0, float(self.retained_energy)))
        object.__setattr__(self, "dropped_energy", max(0.0, float(self.dropped_energy)))
        object.__setattr__(
            self,
            "compression_cost",
            max(0.0, float(self.compression_cost)),
        )
        object.__setattr__(
            self,
            "reconstruction_error",
            max(0.0, float(self.reconstruction_error)),
        )
        object.__setattr__(self, "active_features", [str(item) for item in self.active_features])
        object.__setattr__(self, "method", str(self.method))

    def to_metadata(self) -> dict[str, Any]:
        return {
            "represented": self.represented.copy(),
            "reconstructed": self.reconstructed.copy(),
            "retained_energy": self.retained_energy,
            "dropped_energy": self.dropped_energy,
            "compression_cost": self.compression_cost,
            "reconstruction_error": self.reconstruction_error,
            "active_features": list(self.active_features),
            "method": self.method,
        }
