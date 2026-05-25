from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cave.substrates.cavenet.config import CaveNetConfig


@dataclass(frozen=True)
class CaveNetControllerAccess:
    pressure: bool = True
    workspace: bool = True
    memory: bool = True
    attention: bool = True
    topology: bool = True

    def to_dict(self) -> dict[str, bool]:
        return {
            "pressure": self.pressure,
            "workspace": self.workspace,
            "memory": self.memory,
            "attention": self.attention,
            "topology": self.topology,
        }


@dataclass(frozen=True)
class CaveNetControllerObservation:
    surprise: float
    utility: float
    compression_cost: float
    memory_norm: float
    attention: float
    external_attention: float
    internal_expectation_attention: float
    topology_mass: float

    def to_dict(self) -> dict[str, float]:
        return {
            "surprise": float(self.surprise),
            "utility": float(self.utility),
            "compression_cost": float(self.compression_cost),
            "memory_norm": float(self.memory_norm),
            "attention": float(self.attention),
            "attention_capacity": float(self.attention),
            "external_attention": float(self.external_attention),
            "internal_expectation_attention": float(
                self.internal_expectation_attention
            ),
            "topology_mass": float(self.topology_mass),
        }


@dataclass
class CaveNetControllerState:
    latent: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=float))
    step_count: int = 0
    readout_updates: int = 0
    initial_output_weights: np.ndarray | None = None
    history: list[dict[str, object]] = field(default_factory=list)


@dataclass
class CaveNetController:
    base_config: CaveNetConfig
    access: CaveNetControllerAccess = field(default_factory=CaveNetControllerAccess)
    state: CaveNetControllerState = field(default_factory=CaveNetControllerState)
    surprise_threshold: float = 0.08
    utility_threshold: float = 0.0
    latent_decay: float = 0.72
    latent_update_rate: float = 0.55
    topology_normalizer: float = 100.0
    min_gain: float = 0.0
    max_gain: float = 2.5
    readout_plasticity: bool = False
    readout_learning_rate: float = 0.0
    readout_weight_max: float = 2.0
    hidden_weights: np.ndarray = field(default_factory=lambda: np.eye(5, dtype=float))
    readout_plasticity_mask: np.ndarray = field(
        default_factory=lambda: np.zeros((7, 5), dtype=float)
    )
    output_weights: np.ndarray = field(default_factory=lambda: np.array(
        [
            [0.30, 0.42, 0.06, 0.22, 0.04],
            [0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00],
            [0.48, 0.04, 0.22, 0.02, 0.08],
            [0.34, 0.04, 0.18, 0.02, 0.25],
            [0.30, 0.02, 0.10, 0.02, 0.25],
        ],
        dtype=float,
    ))

    def __post_init__(self) -> None:
        self.state.latent = np.asarray(self.state.latent, dtype=float)
        self.hidden_weights = np.asarray(self.hidden_weights, dtype=float)
        self.output_weights = np.asarray(self.output_weights, dtype=float)
        if self.state.latent.shape != (5,):
            raise ValueError("controller latent must have shape (5,)")
        if self.hidden_weights.shape != (5, 5):
            raise ValueError("hidden_weights must have shape (5, 5)")
        if self.output_weights.shape != (7, 5):
            raise ValueError("output_weights must have shape (7, 5)")
        self.readout_plasticity_mask = np.asarray(
            self.readout_plasticity_mask,
            dtype=float,
        )
        if self.readout_plasticity_mask.shape != (7, 5):
            raise ValueError("readout_plasticity_mask must have shape (7, 5)")
        if self.state.initial_output_weights is None:
            self.state.initial_output_weights = self.output_weights.copy()
        else:
            self.state.initial_output_weights = np.asarray(
                self.state.initial_output_weights,
                dtype=float,
            )
            if self.state.initial_output_weights.shape != (7, 5):
                raise ValueError("initial_output_weights must have shape (7, 5)")
        if self.topology_normalizer <= 0.0:
            raise ValueError("topology_normalizer must be positive")
        if self.max_gain < self.min_gain:
            raise ValueError("max_gain must be greater than or equal to min_gain")
        if self.readout_learning_rate < 0.0:
            raise ValueError("readout_learning_rate must be non-negative")
        if self.readout_weight_max < 0.0:
            raise ValueError("readout_weight_max must be non-negative")

    def step(
        self,
        current: CaveNetConfig,
        observation: CaveNetControllerObservation,
    ) -> CaveNetConfig:
        del current
        inputs = self._input_vector(observation)
        hidden = np.maximum(0.0, self.hidden_weights @ inputs)
        self.state.latent = (
            self.latent_decay * self.state.latent
            + self.latent_update_rate * hidden
        )
        deltas = self.output_weights @ self.state.latent
        config = CaveNetConfig(
            attention_gain=self.base_config.attention_gain + float(deltas[0]),
            state_input_gain=self.base_config.state_input_gain + float(deltas[1]),
            expectation_gain=self.base_config.expectation_gain + float(deltas[2]),
            surprise_gain=self.base_config.surprise_gain + float(deltas[3]),
            learning_rate_gain=self.base_config.learning_rate_gain + float(deltas[4]),
            topology_deposit_gain=(
                self.base_config.topology_deposit_gain + float(deltas[5])
            ),
            topology_transition_gain=(
                self.base_config.topology_transition_gain + float(deltas[6])
            ),
        ).clipped(minimum=self.min_gain, maximum=self.max_gain)
        self._adapt_readout(observation)
        self.state.step_count += 1
        self.state.history.append(
            {
                "step": self.state.step_count,
                "inputs": inputs.tolist(),
                "latent": self.state.latent.tolist(),
                "observation": observation.to_dict(),
                "config": config.to_dict(),
                "output_weight_delta_norm": self.output_weight_delta_norm(),
            }
        )
        return config

    def to_metadata(self) -> dict[str, object]:
        return {
            "kind": "mlp_latent_gain_controller",
            "access": self.access.to_dict(),
            "base_config": self.base_config.to_dict(),
            "step_count": self.state.step_count,
            "readout_plasticity": self.readout_plasticity,
            "readout_updates": self.state.readout_updates,
            "latent": self.state.latent.tolist(),
            "output_weight_norm": _norm(self.output_weights),
            "output_weight_delta_norm": self.output_weight_delta_norm(),
            "history": list(self.state.history),
        }

    def output_weight_delta_norm(self) -> float:
        initial = self.state.initial_output_weights
        if initial is None:
            return 0.0
        return _norm(self.output_weights - initial)

    def _input_vector(self, observation: CaveNetControllerObservation) -> np.ndarray:
        pressure = max(0.0, float(observation.surprise) - self.surprise_threshold)
        pressure += max(0.0, -float(observation.utility) - self.utility_threshold)
        workspace = max(0.0, float(observation.compression_cost))
        memory = max(0.0, float(observation.memory_norm))
        attention = max(0.0, float(observation.attention))
        topology = max(0.0, float(observation.topology_mass) / self.topology_normalizer)
        values = np.array(
            [
                pressure if self.access.pressure else 0.0,
                workspace if self.access.workspace else 0.0,
                memory if self.access.memory else 0.0,
                attention if self.access.attention else 0.0,
                topology if self.access.topology else 0.0,
            ],
            dtype=float,
        )
        return np.clip(values, 0.0, 4.0)

    def _adapt_readout(self, observation: CaveNetControllerObservation) -> None:
        if not self.readout_plasticity or self.readout_learning_rate <= 0.0:
            return
        pressure = max(0.0, float(observation.surprise) - self.surprise_threshold)
        pressure += max(0.0, -float(observation.utility) - self.utility_threshold)
        pressure += max(0.0, float(observation.compression_cost))
        if pressure <= 0.0:
            return
        latent = np.maximum(0.0, self.state.latent)
        if not np.any(latent):
            return
        update = (
            self.readout_learning_rate
            * min(1.0, pressure)
            * self.readout_plasticity_mask
            * latent.reshape(1, -1)
        )
        self.output_weights = np.clip(
            self.output_weights + update,
            0.0,
            self.readout_weight_max,
        )
        self.state.readout_updates += 1


def _norm(value) -> float:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.linalg.norm(array.ravel()) / np.sqrt(array.size))
