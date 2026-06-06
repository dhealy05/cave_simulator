from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from cave.observation.episodes import Episode, EpisodeObservation


Array = np.ndarray


ENERGY_KEYS = (
    "rails_base",
    "world_update",
    "scheduler",
    "render",
    "subject_base",
    "sensing",
    "attention_update",
    "compression_compute",
    "prediction_compute",
    "loss_compute",
    "memory_update",
    "topology_update",
    "controller_update",
    "adaptive_extra",
)

RAILS_KEYS = ("rails_base", "world_update", "scheduler", "render")
SUBJECT_KEYS = tuple(key for key in ENERGY_KEYS if key not in RAILS_KEYS)


@dataclass(frozen=True)
class EnergyLedger:
    rails_base: float = 0.0
    world_update: float = 0.0
    scheduler: float = 0.0
    render: float = 0.0
    subject_base: float = 0.0
    sensing: float = 0.0
    attention_update: float = 0.0
    compression_compute: float = 0.0
    prediction_compute: float = 0.0
    loss_compute: float = 0.0
    memory_update: float = 0.0
    topology_update: float = 0.0
    controller_update: float = 0.0
    adaptive_extra: float = 0.0

    def __post_init__(self) -> None:
        for key in ENERGY_KEYS:
            value = max(0.0, float(getattr(self, key)))
            object.__setattr__(self, key, value)

    @property
    def rails_total(self) -> float:
        return float(sum(getattr(self, key) for key in RAILS_KEYS))

    @property
    def subject_total(self) -> float:
        return float(sum(getattr(self, key) for key in SUBJECT_KEYS))

    @property
    def total(self) -> float:
        return self.rails_total + self.subject_total

    def to_metadata(self) -> dict[str, float]:
        return {key: float(getattr(self, key)) for key in ENERGY_KEYS}

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "EnergyLedger":
        return cls(**{key: float(metadata.get(key, 0.0)) for key in ENERGY_KEYS})


def summarize_episode_energy(episode: Episode) -> dict[str, Any]:
    """Return compression, loss, and energy-coupling metrics for one episode.

    The function prefers explicit ``observation.metadata["energy"]`` ledgers but
    can infer a conservative ledger from ordinary Episode fields. The units are
    proxy units, not physical joules.
    """

    rows = [_observation_energy_row(obs) for obs in episode.observations]
    if not rows:
        return _empty_summary()

    total_input_energy = _sum(rows, "input_energy")
    retained_energy = _sum(rows, "retained_energy")
    dropped_energy = _sum(rows, "dropped_energy")
    rails_total = _sum(rows, "rails_total")
    subject_total = _sum(rows, "subject_total")
    adaptive_extra_total = _sum(rows, "adaptive_extra")
    losses = np.asarray([row["loss"] for row in rows], dtype=float)
    subject_energy = np.asarray([row["subject_total"] for row in rows], dtype=float)
    attention = np.asarray([row["attention"] for row in rows], dtype=float)

    dynamic_energy_coupling = _lagged_correlation(losses, subject_energy)
    hick_slope = _lagged_slope(losses, subject_energy)
    loss_to_attention_effect = _lagged_slope(losses, attention)
    compression_load = (
        0.0 if total_input_energy <= 1e-12 else dropped_energy / total_input_energy
    )
    rail_independence = (
        0.0
        if rails_total + subject_total <= 1e-12
        else subject_total / (rails_total + subject_total)
    )
    loss_presence = float(np.mean(losses))
    instantiation_proxy = _bounded_positive(compression_load) * _bounded_positive(
        loss_presence
    ) * _bounded_positive(dynamic_energy_coupling) * _bounded_positive(
        rail_independence
    )

    return {
        "compression": {
            "total_input_energy": total_input_energy,
            "retained_energy": retained_energy,
            "dropped_energy": dropped_energy,
            "compression_load": compression_load,
        },
        "loss": {
            "mean_loss_presence": loss_presence,
            "mean_surprise": float(np.mean([row["surprise"] for row in rows])),
            "mean_objective_pressure": float(
                np.mean([row["objective_pressure"] for row in rows])
            ),
        },
        "energy": {
            "rails_total": rails_total,
            "subject_total": subject_total,
            "adaptive_extra_total": adaptive_extra_total,
            "rail_independence": rail_independence,
        },
        "coupling": {
            "dynamic_energy_coupling": dynamic_energy_coupling,
            "hick_slope": hick_slope,
            "loss_to_attention_effect": loss_to_attention_effect,
        },
        "summary": {
            "instantiation_proxy": instantiation_proxy,
            "observation_count": len(rows),
        },
        "trace": rows,
    }


def _observation_energy_row(observation: EpisodeObservation) -> dict[str, float]:
    ledger = _ledger_for_observation(observation)
    input_energy = _input_energy(observation)
    retained_energy, dropped_energy = _compression_energy(observation, input_energy)
    objective = observation.metadata.get("objective", {})
    prediction_cost = _float_from_mapping(objective, "prediction_cost")
    compression_cost = _float_from_mapping(objective, "compression_cost")
    objective_pressure = max(0.0, prediction_cost) + max(0.0, compression_cost)
    surprise = max(0.0, float(observation.surprise))
    reconstruction_error = _float_from_mapping(
        observation.metadata.get("workspace", {}),
        "reconstruction_error",
    )
    loss_proxy = _float_from_mapping(observation.metadata, "energy_loss_proxy")
    loss = (
        max(surprise, objective_pressure, loss_proxy)
        + max(0.0, reconstruction_error)
    )
    return {
        "t": float(observation.t),
        "input_energy": input_energy,
        "retained_energy": retained_energy,
        "dropped_energy": dropped_energy,
        "loss": loss,
        "surprise": surprise,
        "objective_pressure": objective_pressure,
        "attention": float(observation.attention),
        "rails_total": ledger.rails_total,
        "subject_total": ledger.subject_total,
        "adaptive_extra": ledger.adaptive_extra,
    }


def _ledger_for_observation(observation: EpisodeObservation) -> EnergyLedger:
    energy = observation.metadata.get("energy")
    if isinstance(energy, Mapping):
        return EnergyLedger.from_metadata(energy)

    dimension = max(1, observation.actual.size)
    active_count = max(1, len(observation.active_inputs))
    workspace = observation.metadata.get("workspace", {})
    compression_compute = 0.0
    if isinstance(workspace, Mapping):
        compression_compute = dimension * (1.0 + _float_from_mapping(workspace, "compression_cost"))
    elif "minimal_subject" in observation.metadata:
        compression_compute = dimension
    return EnergyLedger(
        rails_base=1.0,
        scheduler=1.0,
        subject_base=1.0,
        sensing=float(dimension * active_count),
        attention_update=float(active_count),
        compression_compute=float(compression_compute),
        prediction_compute=float(dimension),
        loss_compute=1.0,
        memory_update=float(dimension),
    )


def _input_energy(observation: EpisodeObservation) -> float:
    attended = observation.metadata.get("attended_input")
    if attended is not None:
        return _energy(attended)
    if observation.input_features:
        total = np.zeros_like(observation.actual, dtype=float)
        for value in observation.input_features.values():
            array = np.asarray(value, dtype=float)
            if array.shape == total.shape:
                total += array
            else:
                return float(
                    sum(_energy(features) for features in observation.input_features.values())
                )
        return _energy(total)
    return _energy(observation.actual)


def _compression_energy(
    observation: EpisodeObservation,
    input_energy: float,
) -> tuple[float, float]:
    workspace = observation.metadata.get("workspace", {})
    if isinstance(workspace, Mapping) and "retained_energy" in workspace:
        retained = _float_from_mapping(workspace, "retained_energy")
        dropped = _float_from_mapping(workspace, "dropped_energy")
        return retained, dropped

    retained_attention_mass = _float_from_mapping(
        observation.metadata,
        "retained_attention_mass",
    )
    if 1e-12 < retained_attention_mass <= 1.0:
        # GPT-2 and conversation producers expose context-selection compression
        # as retained attention mass rather than a workspace vector.
        total_context_energy = input_energy / retained_attention_mass
        return input_energy, max(0.0, total_context_energy - input_energy)

    retained = _energy(observation.actual)
    dropped = max(0.0, input_energy - retained)
    return retained, dropped


def _energy(value: Any) -> float:
    array = np.asarray(value, dtype=float)
    return float(np.sum(array * array))


def _float_from_mapping(value: Any, key: str) -> float:
    if not isinstance(value, Mapping):
        return 0.0
    try:
        return float(value.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _sum(rows: list[dict[str, float]], key: str) -> float:
    return float(sum(row[key] for row in rows))


def _lagged_correlation(source: Array, target: Array) -> float:
    if source.size < 2 or target.size < 2:
        return 0.0
    x = np.asarray(source[:-1], dtype=float)
    y = np.asarray(target[1:], dtype=float)
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _lagged_slope(source: Array, target: Array) -> float:
    if source.size < 2 or target.size < 2:
        return 0.0
    x = np.asarray(source[:-1], dtype=float)
    y = np.asarray(target[1:], dtype=float)
    variance = float(np.var(x))
    if variance <= 1e-12:
        return 0.0
    return float(np.cov(x, y, bias=True)[0, 1] / variance)


def _bounded_positive(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _empty_summary() -> dict[str, Any]:
    return {
        "compression": {
            "total_input_energy": 0.0,
            "retained_energy": 0.0,
            "dropped_energy": 0.0,
            "compression_load": 0.0,
        },
        "loss": {
            "mean_loss_presence": 0.0,
            "mean_surprise": 0.0,
            "mean_objective_pressure": 0.0,
        },
        "energy": {
            "rails_total": 0.0,
            "subject_total": 0.0,
            "adaptive_extra_total": 0.0,
            "rail_independence": 0.0,
        },
        "coupling": {
            "dynamic_energy_coupling": 0.0,
            "hick_slope": 0.0,
            "loss_to_attention_effect": 0.0,
        },
        "summary": {
            "instantiation_proxy": 0.0,
            "observation_count": 0,
        },
        "trace": [],
    }
