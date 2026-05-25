from __future__ import annotations

from dataclasses import dataclass

from cave.demonstrations.state import SubjectState


@dataclass(frozen=True)
class CaveNetReadout:
    block_names: tuple[str, ...]
    representation: str = "fixed_network_form"

    def to_metadata(self) -> dict[str, object]:
        return {
            "representation": self.representation,
            "blocks": list(self.block_names),
        }


@dataclass
class CaveNetState:
    subject_state: SubjectState
    adaptive_channel_weights: dict[str, float] | None = None
