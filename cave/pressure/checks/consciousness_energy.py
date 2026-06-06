from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from cave.observation.energy import EnergyLedger, summarize_episode_energy
from cave.observation.episodes import Episode, EpisodeInput, EpisodeObservation
from cave.observation.experience import (
    FeatureVector,
    InputSequence,
    ExperienceObject,
    TemporalExtent,
    presentation_for_object,
)
from cave.observation.projections import encode_value
from cave.observation.views import default_views
from cave.presentation.reports.specs import (
    ProducerReportSpec,
    ReportExtraAsset,
    ReportSection,
)


ENERGY_VOCABULARY = ["stable", "signal", "detail", "noise"]
EnergyVariant = Literal["dynamic", "fixed", "loss-ablation", "replay"]


@dataclass(frozen=True)
class DynamicComputeConfig:
    base_budget: float = 1.0
    max_budget: float = 3.0
    loss_gain: float = 2.4
    memory_rate: float = 0.45

    def __post_init__(self) -> None:
        if self.base_budget <= 0.0:
            raise ValueError("base_budget must be positive")
        if self.max_budget < self.base_budget:
            raise ValueError("max_budget must be >= base_budget")
        if self.loss_gain < 0.0:
            raise ValueError("loss_gain must be non-negative")
        if not 0.0 <= self.memory_rate <= 1.0:
            raise ValueError("memory_rate must be in [0, 1]")


def consciousness_energy_report_spec(
    *,
    dt: float = 1.0,
    fps: int = 4,
    include_assets: bool = True,
) -> ProducerReportSpec:
    del include_assets

    def build_episode() -> Episode:
        return build_consciousness_energy_episode("dynamic")

    return ProducerReportSpec(
        id="consciousness-energy",
        title="Consciousness Energy Estimator",
        episode_factory=build_episode,
        input_summary="dynamic-compute subject over alternating predictable and surprising inputs",
        description=(
            "A measurement probe for the simulation-versus-instantiation "
            "question. The dynamic variant lets loss buy future online subject "
            "work; controls remove or externalize that coupling."
        ),
        views=default_views(),
        extra_assets=(
            ReportExtraAsset(
                id="energy_metrics",
                title="Energy Estimator Metrics JSON",
                filename="consciousness_energy_metrics.json",
                writer=lambda episode, output: write_consciousness_energy_metrics_json(output),
            ),
        ),
        checks=(check_consciousness_energy,),
        frame_time=5.0,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "dynamic_compute_subject",
            "scenario": "consciousness_energy",
            "dt": dt,
            "fps": fps,
            "variants": ["dynamic", "fixed", "loss-ablation", "replay"],
        },
        sections=(
            ReportSection(
                title="Question",
                body=(
                    "Does internally generated loss change how much online "
                    "subject-side work the system spends on the next update?"
                ),
                asset_ids=("energy_metrics",),
            ),
            ReportSection(
                title="Boundary",
                body=(
                    "Energy values are deterministic proxy units, not joules. "
                    "The report checks a loss-coupled compression loop, not a "
                    "claim that the subject is conscious."
                ),
            ),
        ),
    )


def build_consciousness_energy_episode(
    variant: EnergyVariant = "dynamic",
    *,
    config: DynamicComputeConfig | None = None,
    replay_budgets: tuple[float, ...] | None = None,
) -> Episode:
    return run_dynamic_compute_subject(
        consciousness_energy_sequence(),
        variant=variant,
        config=config or DynamicComputeConfig(),
        replay_budgets=replay_budgets,
    )


def consciousness_energy_sequence(cycles: int = 6) -> InputSequence:
    objects: list[ExperienceObject] = []
    order = 0
    t = 0.0
    for cycle in range(cycles):
        objects.append(
            _event(
                f"stable_{cycle}",
                t,
                order,
                {"stable": 1.0, "signal": 0.35, "detail": 0.2, "noise": 0.15},
            )
        )
        t += 1.0
        order += 1
        objects.append(
            _event(
                f"detail_{cycle}",
                t,
                order,
                {"stable": 0.8, "detail": 0.55, "noise": 0.25, "signal": 0.15},
            )
        )
        t += 1.0
        order += 1
        objects.append(
            _event(
                f"surprise_{cycle}",
                t,
                order,
                {"signal": 1.0, "noise": 0.85, "detail": 0.4, "stable": 0.2},
            )
        )
        t += 1.0
        order += 1
    return InputSequence(objects)


def run_dynamic_compute_subject(
    sequence: InputSequence,
    *,
    variant: EnergyVariant,
    config: DynamicComputeConfig,
    replay_budgets: tuple[float, ...] | None = None,
) -> Episode:
    vocabulary = list(ENERGY_VOCABULARY)
    memory = np.zeros(len(vocabulary), dtype=float)
    previous_loss = 0.0
    previous_salience_pressure = 0.0
    observations: list[EpisodeObservation] = []
    budgets: list[float] = []
    inputs = [
        EpisodeInput(
            id=obj.id,
            kind=obj.kind,
            start=obj.temporal_extent.start,
            end=obj.temporal_extent.end,
            order_index=obj.temporal_extent.order_index,
            features=obj.features.to_array(vocabulary),
            modality=obj.modality,
            salience=obj.salience,
            learning_weight=obj.learning_weight,
            presentation=presentation_for_object(obj),
            metadata=dict(obj.metadata),
        )
        for obj in sequence.objects
    ]

    for index, obj in enumerate(sequence.objects):
        external = obj.features.to_array(vocabulary) * obj.salience
        budget = _budget_for_variant(
            variant,
            index=index,
            config=config,
            previous_loss=previous_loss,
            previous_salience_pressure=previous_salience_pressure,
            replay_budgets=replay_budgets,
        )
        capacity = max(1, min(len(vocabulary), int(np.ceil(budget))))
        actual, active_features, retained_energy, dropped_energy = _workspace(
            external,
            vocabulary,
            capacity,
        )
        error = actual - memory
        surprise = _normalized_norm(error)
        compression_cost = 0.0 if _energy(external) <= 1e-12 else dropped_energy / _energy(external)
        reconstruction_error = _normalized_norm(external - actual)
        utility = -(surprise + compression_cost)
        subject_budget = config.base_budget if variant == "replay" else budget
        external_budget = max(0.0, budget - config.base_budget) if variant == "replay" else 0.0
        ledger = _energy_ledger(
            dimension=len(vocabulary),
            budget=subject_budget,
            external_budget=external_budget,
            config=config,
        )
        observations.append(
            EpisodeObservation(
                t=obj.temporal_extent.start,
                t_normalized=(
                    0.0 if sequence.duration <= 0.0 else obj.temporal_extent.start / sequence.duration
                ),
                expected=memory.copy(),
                actual=actual.copy(),
                memory_state=memory.copy(),
                surprise=surprise,
                learning_rate=config.memory_rate,
                attention=min(1.0, budget / config.max_budget),
                attention_weights={obj.id: min(1.0, budget / config.max_budget)},
                active_inputs=[obj.id],
                input_features={obj.id: external.copy()},
                metadata={
                    "dynamic_compute": {
                        "variant": variant,
                        "budget": budget,
                        "capacity": capacity,
                        "active_features": active_features,
                        "previous_loss": previous_loss,
                        "previous_salience_pressure": previous_salience_pressure,
                    },
                    "workspace": {
                        "represented": actual[np.abs(actual) > 1e-12].copy(),
                        "reconstructed": actual.copy(),
                        "retained_energy": retained_energy,
                        "dropped_energy": dropped_energy,
                        "compression_cost": compression_cost,
                        "reconstruction_error": reconstruction_error,
                        "active_features": active_features,
                        "method": "dynamic_top_k",
                    },
                    "objective": {
                        "utility": utility,
                        "prediction_cost": surprise,
                        "pain_cost": 0.0,
                        "pleasure_gain": 0.0,
                        "attention_cost": 0.0,
                        "compression_cost": compression_cost,
                    },
                    "energy": ledger.to_metadata(),
                    "attended_input": external.copy(),
                },
            )
        )
        budgets.append(budget)
        memory = (1.0 - config.memory_rate) * memory + config.memory_rate * actual
        previous_loss = surprise + reconstruction_error
        previous_salience_pressure = max(0.0, obj.salience - 1.0)

    return Episode(
        source_name=f"dynamic-compute:{variant}",
        vocabulary=vocabulary,
        inputs=inputs,
        observations=observations,
        duration=sequence.duration,
        metadata={
            "source": "cave.pressure.checks.consciousness_energy",
            "adapter": "DynamicComputeSubject",
            "variant": variant,
            "budgets": tuple(budgets),
            "config": config,
        },
    )


def check_consciousness_energy(episode: Episode | None = None) -> dict[str, object]:
    del episode
    dynamic = build_consciousness_energy_episode("dynamic")
    replay_budgets = tuple(
        float(obs.metadata["dynamic_compute"]["budget"]) for obs in dynamic.observations
    )
    episodes = {
        "dynamic": dynamic,
        "fixed": build_consciousness_energy_episode("fixed"),
        "loss-ablation": build_consciousness_energy_episode("loss-ablation"),
        "replay": build_consciousness_energy_episode("replay", replay_budgets=replay_budgets),
    }
    metrics = {name: summarize_episode_energy(item) for name, item in episodes.items()}
    roles = _role_metrics(metrics)
    errors = []
    if roles["dynamic_energy_coupling"]["dynamic"] <= 0.5:
        errors.append("dynamic variant did not couple loss to next-step subject energy")
    if abs(roles["dynamic_energy_coupling"]["fixed"]) >= 0.05:
        errors.append("fixed variant unexpectedly coupled loss to subject energy")
    if abs(roles["dynamic_energy_coupling"]["loss-ablation"]) >= 0.05:
        errors.append("loss-ablation variant unexpectedly coupled loss to subject energy")
    if roles["rail_independence"]["dynamic_minus_replay"] <= 0.05:
        errors.append("replay control did not externalize adaptive budget to rails")
    if roles["instantiation_proxy"]["dynamic"] <= roles["instantiation_proxy"]["fixed"]:
        errors.append("dynamic variant did not exceed fixed instantiation proxy")

    return {
        "id": "consciousness_energy",
        "ok": not errors,
        "errors": errors,
        "metrics": _compact_metrics(metrics),
        "roles": roles,
    }


def write_consciousness_energy_metrics_json(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    result = check_consciousness_energy()
    output.write_text(json.dumps(encode_value(result), indent=2) + "\n", encoding="utf-8")


def _role_metrics(metrics: dict[str, dict[str, object]]) -> dict[str, dict[str, float]]:
    coupling = {
        name: float(summary["coupling"]["dynamic_energy_coupling"])  # type: ignore[index]
        for name, summary in metrics.items()
    }
    hick = {
        name: float(summary["coupling"]["hick_slope"])  # type: ignore[index]
        for name, summary in metrics.items()
    }
    rail = {
        name: float(summary["energy"]["rail_independence"])  # type: ignore[index]
        for name, summary in metrics.items()
    }
    proxy = {
        name: float(summary["summary"]["instantiation_proxy"])  # type: ignore[index]
        for name, summary in metrics.items()
    }
    return {
        "dynamic_energy_coupling": {
            **coupling,
            "dynamic_minus_fixed": coupling["dynamic"] - coupling["fixed"],
            "dynamic_minus_loss_ablation": coupling["dynamic"] - coupling["loss-ablation"],
        },
        "hick_style_compute_scaling": {
            **hick,
            "dynamic_minus_fixed": hick["dynamic"] - hick["fixed"],
        },
        "rail_independence": {
            **rail,
            "dynamic_minus_replay": rail["dynamic"] - rail["replay"],
        },
        "instantiation_proxy": {
            **proxy,
            "dynamic_minus_fixed": proxy["dynamic"] - proxy["fixed"],
            "dynamic_minus_replay": proxy["dynamic"] - proxy["replay"],
        },
    }


def _compact_metrics(metrics: dict[str, dict[str, object]]) -> dict[str, dict[str, float]]:
    compact: dict[str, dict[str, float]] = {}
    for name, summary in metrics.items():
        compact[name] = {
            "compression_load": float(summary["compression"]["compression_load"]),  # type: ignore[index]
            "mean_loss_presence": float(summary["loss"]["mean_loss_presence"]),  # type: ignore[index]
            "rails_total": float(summary["energy"]["rails_total"]),  # type: ignore[index]
            "subject_total": float(summary["energy"]["subject_total"]),  # type: ignore[index]
            "adaptive_extra_total": float(summary["energy"]["adaptive_extra_total"]),  # type: ignore[index]
            "rail_independence": float(summary["energy"]["rail_independence"]),  # type: ignore[index]
            "dynamic_energy_coupling": float(summary["coupling"]["dynamic_energy_coupling"]),  # type: ignore[index]
            "hick_slope": float(summary["coupling"]["hick_slope"]),  # type: ignore[index]
            "instantiation_proxy": float(summary["summary"]["instantiation_proxy"]),  # type: ignore[index]
        }
    return compact


def _budget_for_variant(
    variant: EnergyVariant,
    *,
    index: int,
    config: DynamicComputeConfig,
    previous_loss: float,
    previous_salience_pressure: float,
    replay_budgets: tuple[float, ...] | None,
) -> float:
    if variant == "dynamic":
        budget = config.base_budget + config.loss_gain * previous_loss
    elif variant == "fixed":
        budget = config.base_budget
    elif variant == "loss-ablation":
        budget = config.base_budget + config.loss_gain * previous_salience_pressure
    elif variant == "replay":
        if replay_budgets is None or index >= len(replay_budgets):
            raise ValueError("replay variant requires replay_budgets for every step")
        budget = replay_budgets[index]
    else:
        raise ValueError(f"unsupported energy variant: {variant}")
    return min(config.max_budget, max(config.base_budget, float(budget)))


def _workspace(
    external: np.ndarray,
    vocabulary: list[str],
    capacity: int,
) -> tuple[np.ndarray, list[str], float, float]:
    if external.size == 0:
        return external.copy(), [], 0.0, 0.0
    active = np.flatnonzero(np.abs(external) > 1e-12)
    if active.size == 0:
        return np.zeros_like(external), [], 0.0, 0.0
    order = active[np.argsort(-np.abs(external[active]), kind="stable")[:capacity]]
    order = np.sort(order)
    actual = np.zeros_like(external, dtype=float)
    actual[order] = external[order]
    retained = _energy(actual)
    dropped = max(0.0, _energy(external) - retained)
    return actual, [vocabulary[index] for index in order], retained, dropped


def _energy_ledger(
    *,
    dimension: int,
    budget: float,
    external_budget: float,
    config: DynamicComputeConfig,
) -> EnergyLedger:
    adaptive = max(0.0, budget - config.base_budget) * dimension
    external_adaptive = max(0.0, external_budget) * dimension
    return EnergyLedger(
        rails_base=1.0 + external_adaptive,
        scheduler=1.0,
        subject_base=1.0,
        sensing=float(dimension),
        attention_update=float(budget),
        compression_compute=float(dimension * budget),
        prediction_compute=float(dimension * budget),
        loss_compute=float(dimension),
        memory_update=float(dimension),
        adaptive_extra=float(adaptive),
    )


def _event(
    id: str,
    t: float,
    order: int,
    features: dict[str, float],
) -> ExperienceObject:
    return ExperienceObject(
        id=id,
        kind="energy_probe",
        temporal_extent=TemporalExtent(t, t + 0.85, order),
        features=FeatureVector(features),
        salience=1.0,
    )


def _energy(value: np.ndarray) -> float:
    return float(np.sum(np.asarray(value, dtype=float) ** 2))


def _normalized_norm(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.linalg.norm(array) / np.sqrt(array.size))
