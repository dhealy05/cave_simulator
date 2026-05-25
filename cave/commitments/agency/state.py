from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ActionState:
    kind: str = "maintain"
    target_id: str | None = None
    target_channel: str | None = None
    strength: float = 0.0
    expected_utility_delta: float = 0.0
    object_exposure: dict[str, float] = field(default_factory=dict)
    components: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(
            self,
            "target_id",
            None if self.target_id is None else str(self.target_id),
        )
        object.__setattr__(
            self,
            "target_channel",
            None if self.target_channel is None else str(self.target_channel),
        )
        object.__setattr__(self, "strength", max(0.0, float(self.strength)))
        object.__setattr__(
            self,
            "expected_utility_delta",
            float(self.expected_utility_delta),
        )
        object.__setattr__(
            self,
            "object_exposure",
            {
                str(key): max(0.0, float(value))
                for key, value in self.object_exposure.items()
            },
        )
        object.__setattr__(
            self,
            "components",
            {str(key): float(value) for key, value in self.components.items()},
        )

    def exposure_for(self, object_id: str) -> float:
        return float(self.object_exposure.get(object_id, 1.0))

    def to_metadata(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target_id": self.target_id,
            "target_channel": self.target_channel,
            "strength": self.strength,
            "expected_utility_delta": self.expected_utility_delta,
            "object_exposure": dict(self.object_exposure),
            "components": dict(self.components),
        }
