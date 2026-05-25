from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cave.demonstrations.examples import model_for_sequence
from cave.demonstrations.scenarios._common import core_scenario_assets
from cave.observation.episodes import Episode
from cave.observation.projections import encode_value
from cave.observation.views import default_views
from cave.presentation.reports.specs import ProducerReportSpec, ReportExtraAsset, ReportSection
from cave.pressure.tests.common_behaviors import (
    COMMON_BEHAVIOR_VOCABULARY,
    common_behavior_params,
    common_behavior_sequence,
    expectation_repetition_metrics,
)
from cave.substrates.cavenet import CaveNet, CaveNetAdaptationPolicy, CaveNetConfig, CaveNetProducer
from cave.substrates.minimal_subject import MinimalSubjectConfig, run_minimal_subject


ROLE_RECOVERY_VARIANTS = (
    "cavenet-reference",
    "cavenet-no-expectation",
    "cavenet-weak-learning",
    "cavenet-adaptive-learning",
    "minimal-no-memory",
    "minimal-frequency-memory",
    "minimal-value-memory",
)


@dataclass(frozen=True)
class RecoveryRun:
    variant: str
    substrate: str
    episode: Episode
    metrics: dict[str, float | int | str]


def role_recovery_report_spec(
    *,
    dt: float = 1.0,
    fps: int = 4,
    include_assets: bool = True,
) -> ProducerReportSpec:
    def build_episode() -> Episode:
        return build_role_recovery_episode("cavenet-adaptive-learning", dt=dt)

    extra_assets = ()
    if include_assets:
        extra_assets = (
            ReportExtraAsset(
                id="role_recovery_metrics",
                title="Role Recovery Metrics JSON",
                filename="role_recovery_metrics.json",
                writer=lambda episode, output: write_role_recovery_metrics_json(
                    output,
                    dt=dt,
                ),
            ),
        )

    return ProducerReportSpec(
        id="role-recovery",
        title="Expectation Role Recovery",
        episode_factory=build_episode,
        input_summary="CaveNet and minimal_subject expectation recovery variants",
        description=(
            "Runs one repeated-input / violation probe through ablated and weak "
            "substrate variants. The question is whether expectation-like behavior "
            "recovers when memory or learning pressure is available."
        ),
        views=default_views(),
        view_assets=core_scenario_assets() if include_assets else (),
        extra_assets=extra_assets,
        checks=(lambda episode: check_role_recovery(dt=dt),),
        frame_time=2.0,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "role_recovery",
            "scenario": "expectation_role_recovery",
            "role": "expectation",
            "dt": dt,
            "fps": fps,
            "variants": list(ROLE_RECOVERY_VARIANTS),
        },
        sections=(
            ReportSection(
                title="Question",
                body=(
                    "The common behavior suite showed that all three substrates can "
                    "express expectation-like behavior. This report asks a stronger "
                    "question: when the role is weakened or removed, which substrate "
                    "pressures recover it?"
                ),
            ),
            ReportSection(
                title="Boundary",
                body=(
                    "CaveNet pressure currently adjusts learning, attention, and "
                    "topology gains. It does not adapt the expectation readout. A "
                    "zeroed expectation readout should therefore remain failed; weak "
                    "learning can recover partially when pressure increases learning."
                ),
                asset_ids=("role_recovery_metrics",),
            ),
        ),
    )


def build_role_recovery_episode(variant: str, *, dt: float = 1.0) -> Episode:
    sequence = common_behavior_sequence("expectation_repetition")
    if variant == "cavenet-reference":
        return _run_cavenet(
            variant,
            config=CaveNetConfig(),
            adaptation_policy=CaveNetAdaptationPolicy(enabled=False),
            dt=dt,
        )
    if variant == "cavenet-no-expectation":
        return _run_cavenet(
            variant,
            config=CaveNetConfig(expectation_gain=0.0),
            adaptation_policy=CaveNetAdaptationPolicy(enabled=False),
            dt=dt,
        )
    if variant == "cavenet-weak-learning":
        return _run_cavenet(
            variant,
            config=CaveNetConfig(learning_rate_gain=0.2),
            adaptation_policy=CaveNetAdaptationPolicy(enabled=False),
            dt=dt,
        )
    if variant == "cavenet-adaptive-learning":
        return _run_cavenet(
            variant,
            config=CaveNetConfig(
                attention_gain=0.8,
                learning_rate_gain=0.2,
                topology_deposit_gain=0.5,
                topology_transition_gain=0.5,
            ),
            adaptation_policy=CaveNetAdaptationPolicy(
                enabled=True,
                surprise_threshold=0.01,
                learning_gain_rate=1.0,
                attention_gain_rate=0.0,
                topology_gain_rate=0.0,
                max_gain=3.0,
            ),
            dt=dt,
        )
    if variant == "minimal-no-memory":
        return _run_minimal(
            variant,
            config=MinimalSubjectConfig(memory_learning=0.0, workspace_capacity=2),
        )
    if variant == "minimal-frequency-memory":
        return _run_minimal(
            variant,
            config=MinimalSubjectConfig(
                memory_learning=1.0,
                memory_mode="frequency",
                workspace_capacity=2,
            ),
        )
    if variant == "minimal-value-memory":
        return _run_minimal(
            variant,
            config=MinimalSubjectConfig(
                memory_learning=1.0,
                memory_mode="value",
                workspace_capacity=2,
            ),
        )
    raise ValueError(f"unsupported role recovery variant: {variant}")


def role_recovery_runs(*, dt: float = 1.0) -> tuple[RecoveryRun, ...]:
    runs: list[RecoveryRun] = []
    for variant in ROLE_RECOVERY_VARIANTS:
        episode = build_role_recovery_episode(variant, dt=dt)
        runs.append(
            RecoveryRun(
                variant=variant,
                substrate="cavenet" if variant.startswith("cavenet") else "minimal_subject",
                episode=episode,
                metrics=_recovery_metrics(episode),
            )
        )
    return tuple(runs)


def check_role_recovery(*, dt: float = 1.0) -> dict[str, object]:
    runs = role_recovery_runs(dt=dt)
    metrics = {run.variant: run.metrics for run in runs}
    roles = _role_metrics(metrics)
    errors: list[str] = []

    if not roles["cavenet_reference"]["surprise_drop"] > 0.0:
        errors.append("CaveNet reference did not show expectation")
    if not roles["cavenet_no_expectation"]["surprise_drop"] <= 1e-12:
        errors.append("zero expectation readout still showed expectation recovery")
    if not roles["cavenet_learning_recovery"]["adaptive_surprise_drop"] > roles["cavenet_learning_recovery"]["weak_surprise_drop"]:
        errors.append("CaveNet adaptive learning did not improve surprise drop")
    if not roles["cavenet_learning_recovery"]["adaptive_violation_margin"] > roles["cavenet_learning_recovery"]["weak_violation_margin"]:
        errors.append("CaveNet adaptive learning did not improve violation margin")
    if not roles["cavenet_learning_recovery"]["final_learning_gain"] > roles["cavenet_learning_recovery"]["initial_learning_gain"]:
        errors.append("CaveNet adaptive learning gain did not increase")
    if not roles["minimal_memory_recovery"]["no_memory_surprise_drop"] <= 1e-12:
        errors.append("minimal no-memory variant unexpectedly recovered expectation")
    if not roles["minimal_memory_recovery"]["frequency_surprise_drop"] > roles["minimal_memory_recovery"]["no_memory_surprise_drop"]:
        errors.append("minimal frequency memory did not recover expectation")
    if not roles["minimal_memory_recovery"]["value_surprise_drop"] > roles["minimal_memory_recovery"]["no_memory_surprise_drop"]:
        errors.append("minimal value memory did not recover expectation")
    if not roles["minimal_memory_recovery"]["value_violation_margin"] > roles["minimal_memory_recovery"]["no_memory_violation_margin"]:
        errors.append("minimal value memory did not recover violation sensitivity")

    return {
        "id": "role_recovery",
        "ok": not errors,
        "errors": errors,
        "metrics": metrics,
        "roles": roles,
    }


def write_role_recovery_metrics_json(output: Path, *, dt: float = 1.0) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    result = check_role_recovery(dt=dt)
    output.write_text(json.dumps(encode_value(result), indent=2) + "\n", encoding="utf-8")


def _role_metrics(metrics: dict[str, dict[str, float | int | str]]) -> dict[str, dict[str, float]]:
    return {
        "cavenet_reference": {
            "surprise_drop": float(metrics["cavenet-reference"]["surprise_drop"]),
            "violation_margin": float(metrics["cavenet-reference"]["violation_margin"]),
        },
        "cavenet_no_expectation": {
            "surprise_drop": float(metrics["cavenet-no-expectation"]["surprise_drop"]),
            "violation_margin": float(metrics["cavenet-no-expectation"]["violation_margin"]),
            "expectation_gain": float(metrics["cavenet-no-expectation"]["final_expectation_gain"]),
        },
        "cavenet_learning_recovery": {
            "weak_surprise_drop": float(metrics["cavenet-weak-learning"]["surprise_drop"]),
            "adaptive_surprise_drop": float(metrics["cavenet-adaptive-learning"]["surprise_drop"]),
            "surprise_drop_gain": (
                float(metrics["cavenet-adaptive-learning"]["surprise_drop"])
                - float(metrics["cavenet-weak-learning"]["surprise_drop"])
            ),
            "weak_violation_margin": float(metrics["cavenet-weak-learning"]["violation_margin"]),
            "adaptive_violation_margin": float(metrics["cavenet-adaptive-learning"]["violation_margin"]),
            "violation_margin_gain": (
                float(metrics["cavenet-adaptive-learning"]["violation_margin"])
                - float(metrics["cavenet-weak-learning"]["violation_margin"])
            ),
            "initial_learning_gain": float(metrics["cavenet-adaptive-learning"]["initial_learning_rate_gain"]),
            "final_learning_gain": float(metrics["cavenet-adaptive-learning"]["final_learning_rate_gain"]),
        },
        "minimal_memory_recovery": {
            "no_memory_surprise_drop": float(metrics["minimal-no-memory"]["surprise_drop"]),
            "frequency_surprise_drop": float(metrics["minimal-frequency-memory"]["surprise_drop"]),
            "value_surprise_drop": float(metrics["minimal-value-memory"]["surprise_drop"]),
            "no_memory_violation_margin": float(metrics["minimal-no-memory"]["violation_margin"]),
            "frequency_violation_margin": float(metrics["minimal-frequency-memory"]["violation_margin"]),
            "value_violation_margin": float(metrics["minimal-value-memory"]["violation_margin"]),
            "value_memory_strength": float(metrics["minimal-value-memory"]["memory_strength"]),
        },
    }


def _recovery_metrics(episode: Episode) -> dict[str, float | int | str]:
    metrics = dict(expectation_repetition_metrics(episode))
    metrics["memory_mass"] = _memory_mass(episode)
    metrics.update(_cavenet_config_metrics(episode))
    metrics.update(_minimal_memory_metrics(episode))
    return metrics


def _run_cavenet(
    source_name: str,
    *,
    config: CaveNetConfig,
    adaptation_policy: CaveNetAdaptationPolicy,
    dt: float,
) -> Episode:
    sequence = common_behavior_sequence("expectation_repetition")
    params = common_behavior_params("expectation_repetition")
    model = model_for_sequence(
        sequence,
        params=params,
        vocabulary=list(COMMON_BEHAVIOR_VOCABULARY),
    )
    cavenet = CaveNet.from_subject_state(
        sequence=model.sequence,
        subject_state=model.subject_state,
        params=model.params,
        vocabulary=model.vocabulary,
        sensorium=model.sensorium,
        config=config,
        adaptation_policy=adaptation_policy,
    )
    return CaveNetProducer(cavenet, name=f"role-recovery:{source_name}").run(dt=dt)


def _run_minimal(variant: str, *, config: MinimalSubjectConfig) -> Episode:
    return run_minimal_subject(
        common_behavior_sequence("expectation_repetition"),
        vocabulary=list(COMMON_BEHAVIOR_VOCABULARY),
        preference_vector=np.array([0.0, 1.0, 0.0, 0.0, 1.0, -1.0], dtype=float),
        config=config,
        source_name=f"role-recovery:{variant}",
    )


def _cavenet_config_metrics(episode: Episode) -> dict[str, float]:
    initial = episode.metadata.get("cavenet_initial_config")
    final = episode.metadata.get("cavenet_config")
    if not isinstance(initial, dict) or not isinstance(final, dict):
        return {}
    return {
        "initial_expectation_gain": float(initial.get("expectation_gain", 0.0)),
        "final_expectation_gain": float(final.get("expectation_gain", 0.0)),
        "initial_learning_rate_gain": float(initial.get("learning_rate_gain", 0.0)),
        "final_learning_rate_gain": float(final.get("learning_rate_gain", 0.0)),
    }


def _minimal_memory_metrics(episode: Episode) -> dict[str, float]:
    if episode.metadata.get("adapter") != "MinimalSubject" or not episode.observations:
        return {}
    geometry = (
        episode.observations[-1]
        .metadata.get("minimal_subject", {})
        .get("memory_geometry", {})
    )
    return {
        "memory_strength": float(geometry.get("strength_total", 0.0)),
        "value_separation": float(geometry.get("value_separation", 0.0)),
    }


def _memory_mass(episode: Episode) -> float:
    if not episode.observations:
        return 0.0
    memory = np.asarray(episode.observations[-1].memory_state, dtype=float)
    if memory.size == 0:
        return 0.0
    return float(np.linalg.norm(memory.ravel()) / np.sqrt(memory.size))
