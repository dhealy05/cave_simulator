from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from cave.observation.episodes import Episode, EpisodeObservation


Array = np.ndarray


OWNERSHIP_VALUES = ("subject", "rails", "amortized_training")


def summarize_episode_compression(episode: Episode) -> dict[str, Any]:
    """Return source/state pressure, distortion, work, and effect metrics.

    Producers can provide explicit ``observation.metadata["compression"]``
    values. When they do not, this summarizer infers a conservative row from the
    standard Episode fields. Units are proxy units; the goal is comparable cost
    accounting, not physical metrology.
    """

    rows = [_observation_compression_row(obs) for obs in episode.observations]
    if not rows:
        return _empty_summary()

    source_load = _sum(rows, "source_load")
    state_capacity = _sum(rows, "state_capacity")
    compression_ratio = _safe_ratio(source_load, state_capacity)
    distortion = float(np.mean([row["distortion"] for row in rows]))
    dropped_energy = _sum(rows, "dropped_energy")
    retained_energy = _sum(rows, "retained_energy")
    prediction_loss = float(np.mean([row["prediction_loss"] for row in rows]))
    update_work = _sum(rows, "update_work")
    energy_cost = _sum(rows, "energy_cost")
    subject_work = _sum(
        (row for row in rows if row["ownership_subject"] > 0.5),
        "update_work",
    )
    rails_work = _sum(
        (row for row in rows if row["ownership_rails"] > 0.5),
        "update_work",
    )
    amortized_work = _sum(
        (row for row in rows if row["ownership_amortized_training"] > 0.5),
        "update_work",
    )
    ownership_subject_fraction = _safe_ratio(subject_work, update_work)
    losses = np.asarray([row["prediction_loss"] for row in rows], dtype=float)
    works = np.asarray([row["update_work"] for row in rows], dtype=float)
    future_loss_improvement = _future_loss_improvement(losses)
    retained_predictive_info = _retained_predictive_info(rows, future_loss_improvement)
    loss_to_update_coupling = _correlation(losses, works)
    pressure_to_distortion_coupling = _correlation(
        np.asarray([row["compression_ratio"] for row in rows], dtype=float),
        np.asarray([row["distortion"] for row in rows], dtype=float),
    )

    return {
        "pressure": {
            "source_load": source_load,
            "state_capacity": state_capacity,
            "compression_ratio": compression_ratio,
            "mean_compression_ratio": float(
                np.mean([row["compression_ratio"] for row in rows])
            ),
        },
        "distortion": {
            "retained_energy": retained_energy,
            "dropped_energy": dropped_energy,
            "mean_distortion": distortion,
            "mean_prediction_loss": prediction_loss,
        },
        "work": {
            "update_work": update_work,
            "energy_cost": energy_cost,
            "ownership_subject_fraction": ownership_subject_fraction,
            "subject_work": subject_work,
            "rails_work": rails_work,
            "amortized_training_work": amortized_work,
        },
        "effect": {
            "future_loss_improvement": future_loss_improvement,
            "loss_to_update_coupling": loss_to_update_coupling,
            "pressure_to_distortion_coupling": pressure_to_distortion_coupling,
            "retained_predictive_info": retained_predictive_info,
        },
        "summary": {
            "paid_compression_proxy": (
                _bounded_positive(compression_ratio - 1.0)
                * _bounded_positive(ownership_subject_fraction)
                * _bounded_positive(loss_to_update_coupling)
                * _bounded_positive(future_loss_improvement)
            ),
            "observation_count": len(rows),
        },
        "trace": rows,
    }


def _observation_compression_row(observation: EpisodeObservation) -> dict[str, float]:
    explicit = observation.metadata.get("compression")
    if isinstance(explicit, Mapping):
        return _explicit_compression_row(observation, explicit)
    return _inferred_compression_row(observation)


def _explicit_compression_row(
    observation: EpisodeObservation,
    values: Mapping[str, Any],
) -> dict[str, float]:
    source_load = _positive_float(values, "source_load")
    state_capacity = _positive_float(values, "state_capacity")
    if source_load <= 0.0:
        source_load = _source_load(observation)
    if state_capacity <= 0.0:
        state_capacity = _state_capacity(observation)
    compression_ratio = _positive_float(values, "compression_ratio")
    if compression_ratio <= 0.0:
        compression_ratio = _safe_ratio(source_load, state_capacity)
    retained_energy = _positive_float(values, "retained_energy")
    dropped_energy = _positive_float(values, "dropped_energy")
    distortion = _positive_float(values, "distortion")
    prediction_loss = _positive_float(values, "prediction_loss")
    if prediction_loss <= 0.0:
        prediction_loss = max(0.0, float(observation.surprise))
    update_work = _positive_float(values, "update_work")
    energy_cost = _positive_float(values, "energy_cost")
    if energy_cost <= 0.0:
        energy_cost = update_work
    predictive_info = _positive_float(values, "predictive_info")
    ownership = str(values.get("ownership", "subject"))
    if ownership not in OWNERSHIP_VALUES:
        ownership = "subject"
    return {
        "t": float(observation.t),
        "source_load": source_load,
        "admitted_load": _positive_float(values, "admitted_load"),
        "state_capacity": state_capacity,
        "compression_ratio": compression_ratio,
        "retained_energy": retained_energy,
        "dropped_energy": dropped_energy,
        "distortion": distortion,
        "prediction_loss": prediction_loss,
        "update_work": update_work,
        "energy_cost": energy_cost,
        "predictive_info": predictive_info,
        "ownership_subject": 1.0 if ownership == "subject" else 0.0,
        "ownership_rails": 1.0 if ownership == "rails" else 0.0,
        "ownership_amortized_training": (
            1.0 if ownership == "amortized_training" else 0.0
        ),
    }


def _inferred_compression_row(observation: EpisodeObservation) -> dict[str, float]:
    source_load = _source_load(observation)
    state_capacity = _state_capacity(observation)
    retained_energy = _energy(observation.actual)
    source_energy = _input_energy(observation)
    dropped_energy = max(0.0, source_energy - retained_energy)
    memory_previous = observation.metadata.get("memory_previous")
    if memory_previous is None:
        update_work = _energy(observation.error) * float(observation.learning_rate)
    else:
        previous = np.asarray(memory_previous, dtype=float)
        update_work = _energy(observation.memory_state - previous)
    prediction_loss = max(0.0, float(observation.surprise))
    distortion = _safe_ratio(dropped_energy, source_energy)
    return {
        "t": float(observation.t),
        "source_load": source_load,
        "admitted_load": _state_capacity(observation),
        "state_capacity": state_capacity,
        "compression_ratio": _safe_ratio(source_load, state_capacity),
        "retained_energy": retained_energy,
        "dropped_energy": dropped_energy,
        "distortion": distortion,
        "prediction_loss": prediction_loss,
        "update_work": update_work,
        "energy_cost": update_work,
        "predictive_info": 0.0,
        "ownership_subject": 1.0,
        "ownership_rails": 0.0,
        "ownership_amortized_training": 0.0,
    }


def _source_load(observation: EpisodeObservation) -> float:
    source = observation.metadata.get("source_vector")
    if source is not None:
        return float(np.asarray(source, dtype=float).size)
    if observation.input_features:
        return float(sum(np.asarray(value, dtype=float).size for value in observation.input_features.values()))
    return float(observation.actual.size)


def _state_capacity(observation: EpisodeObservation) -> float:
    return float(max(1, observation.memory_state.size))


def _input_energy(observation: EpisodeObservation) -> float:
    if observation.input_features:
        return float(sum(_energy(value) for value in observation.input_features.values()))
    return _energy(observation.actual)


def _energy(value: Any) -> float:
    array = np.asarray(value, dtype=float)
    return float(np.sum(array * array))


def _positive_float(values: Mapping[str, Any], key: str) -> float:
    try:
        return max(0.0, float(values.get(key, 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _sum(rows: Any, key: str) -> float:
    return float(sum(row[key] for row in rows))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 1e-12:
        return 0.0
    return float(numerator / denominator)


def _correlation(source: Array, target: Array) -> float:
    if source.size < 2 or target.size < 2:
        return 0.0
    if float(np.std(source)) <= 1e-12 or float(np.std(target)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(source, target)[0, 1])


def _future_loss_improvement(losses: Array) -> float:
    if losses.size < 2:
        return 0.0
    deltas = losses[:-1] - losses[1:]
    return float(np.mean(np.maximum(0.0, deltas)))


def _retained_predictive_info(
    rows: list[dict[str, float]],
    future_loss_improvement: float,
) -> float:
    explicit = [row["predictive_info"] for row in rows if row["predictive_info"] > 0.0]
    if explicit:
        return float(np.mean(explicit))
    mean_loss = float(np.mean([row["prediction_loss"] for row in rows]))
    return _safe_ratio(future_loss_improvement, mean_loss)


def _bounded_positive(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _empty_summary() -> dict[str, Any]:
    return {
        "pressure": {
            "source_load": 0.0,
            "state_capacity": 0.0,
            "compression_ratio": 0.0,
            "mean_compression_ratio": 0.0,
        },
        "distortion": {
            "retained_energy": 0.0,
            "dropped_energy": 0.0,
            "mean_distortion": 0.0,
            "mean_prediction_loss": 0.0,
        },
        "work": {
            "update_work": 0.0,
            "energy_cost": 0.0,
            "ownership_subject_fraction": 0.0,
            "subject_work": 0.0,
            "rails_work": 0.0,
            "amortized_training_work": 0.0,
        },
        "effect": {
            "future_loss_improvement": 0.0,
            "loss_to_update_coupling": 0.0,
            "pressure_to_distortion_coupling": 0.0,
            "retained_predictive_info": 0.0,
        },
        "summary": {
            "paid_compression_proxy": 0.0,
            "observation_count": 0,
        },
        "trace": [],
    }
