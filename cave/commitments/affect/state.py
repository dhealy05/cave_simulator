from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ValenceState:
    pain: float = 0.0
    pleasure: float = 0.0
    net: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    channel_pain: dict[str, float] = field(default_factory=dict)
    channel_pleasure: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pain = max(0.0, float(self.pain))
        pleasure = max(0.0, float(self.pleasure))
        object.__setattr__(self, "pain", pain)
        object.__setattr__(self, "pleasure", pleasure)
        object.__setattr__(self, "net", float(pleasure - pain))
        object.__setattr__(
            self,
            "components",
            {str(key): float(value) for key, value in self.components.items()},
        )
        object.__setattr__(
            self,
            "channel_pain",
            {str(key): max(0.0, float(value)) for key, value in self.channel_pain.items()},
        )
        object.__setattr__(
            self,
            "channel_pleasure",
            {
                str(key): max(0.0, float(value))
                for key, value in self.channel_pleasure.items()
            },
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "pain": self.pain,
            "pleasure": self.pleasure,
            "net": self.net,
            "components": dict(self.components),
            "channel_pain": dict(self.channel_pain),
            "channel_pleasure": dict(self.channel_pleasure),
        }
