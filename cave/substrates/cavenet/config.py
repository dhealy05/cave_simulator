from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CaveNetConfig:
    attention_gain: float = 1.0
    state_input_gain: float = 1.0
    expectation_gain: float = 1.0
    surprise_gain: float = 1.0
    learning_rate_gain: float = 1.0
    topology_deposit_gain: float = 1.0
    topology_transition_gain: float = 1.0

    def __post_init__(self) -> None:
        for name, value in self.to_dict().items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, float]:
        return {
            "attention_gain": float(self.attention_gain),
            "state_input_gain": float(self.state_input_gain),
            "expectation_gain": float(self.expectation_gain),
            "surprise_gain": float(self.surprise_gain),
            "learning_rate_gain": float(self.learning_rate_gain),
            "topology_deposit_gain": float(self.topology_deposit_gain),
            "topology_transition_gain": float(self.topology_transition_gain),
        }

    def clipped(self, *, minimum: float = 0.0, maximum: float = 4.0) -> "CaveNetConfig":
        values = {
            key: min(maximum, max(minimum, value))
            for key, value in self.to_dict().items()
        }
        return replace(self, **values)


@dataclass(frozen=True)
class CaveNetAdaptationPolicy:
    enabled: bool = False
    surprise_threshold: float = 0.12
    utility_threshold: float = 0.0
    learning_gain_rate: float = 0.12
    attention_gain_rate: float = 0.08
    topology_gain_rate: float = 0.06
    decay_to_one_rate: float = 0.02
    min_gain: float = 0.0
    max_gain: float = 3.0

    def __post_init__(self) -> None:
        if self.surprise_threshold < 0.0:
            raise ValueError("surprise_threshold must be non-negative")
        for name, value in {
            "learning_gain_rate": self.learning_gain_rate,
            "attention_gain_rate": self.attention_gain_rate,
            "topology_gain_rate": self.topology_gain_rate,
            "decay_to_one_rate": self.decay_to_one_rate,
            "min_gain": self.min_gain,
            "max_gain": self.max_gain,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.max_gain < self.min_gain:
            raise ValueError("max_gain must be greater than or equal to min_gain")

    def adapt(
        self,
        config: CaveNetConfig,
        *,
        surprise: float,
        utility: float,
        compression_cost: float,
    ) -> CaveNetConfig:
        if not self.enabled:
            return config
        values = config.to_dict()
        pressure = max(0.0, float(surprise) - self.surprise_threshold)
        cost_pressure = max(0.0, -float(utility) - self.utility_threshold)
        compression_pressure = max(0.0, float(compression_cost))

        values["learning_rate_gain"] += self.learning_gain_rate * pressure
        values["attention_gain"] += self.attention_gain_rate * (
            pressure + compression_pressure
        )
        values["topology_deposit_gain"] += self.topology_gain_rate * (
            pressure + cost_pressure
        )
        values["topology_transition_gain"] += self.topology_gain_rate * pressure

        if pressure <= 0.0 and cost_pressure <= 0.0 and compression_pressure <= 0.0:
            for key, value in values.items():
                values[key] = value + self.decay_to_one_rate * (1.0 - value)

        adapted = replace(config, **values)
        return adapted.clipped(minimum=self.min_gain, maximum=self.max_gain)
