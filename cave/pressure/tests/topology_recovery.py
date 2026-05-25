from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cave.demonstrations.examples import model_for_sequence
from cave.observation.episodes import Episode
from cave.observation.projections import encode_value
from cave.observation.views import default_views
from cave.presentation.reports.specs import ProducerReportSpec, ReportExtraAsset, ReportSection
from cave.pressure.tests.common_behaviors import (
    COMMON_BEHAVIOR_VOCABULARY,
    common_behavior_params,
    common_behavior_sequence,
)
from cave.substrates.cavenet import CaveNet, CaveNetAdaptationPolicy, CaveNetConfig, CaveNetProducer
from cave.substrates.minimal_subject import MinimalSubjectConfig, run_minimal_subject


TOPOLOGY_VARIANTS = (
    "cavenet-reference",
    "cavenet-no-topology",
    "cavenet-weak-topology",
    "cavenet-adaptive-topology",
    "minimal-no-memory",
    "minimal-value-memory",
)


@dataclass(frozen=True)
class TopologyRun:
    variant: str
    episode: Episode
    metrics: dict[str, float | int | str]


def topology_recovery_report_spec(
    *,
    dt: float = 1.0,
    fps: int = 4,
    include_assets: bool = True,
) -> ProducerReportSpec:
    def build_episode() -> Episode:
        return build_topology_recovery_episode("cavenet-adaptive-topology", dt=dt)

    extra_assets = ()
    if include_assets:
        extra_assets = (
            ReportExtraAsset(
                id="topology_recovery_metrics",
                title="Topology Recovery Metrics JSON",
                filename="topology_recovery_metrics.json",
                writer=lambda episode, output: write_topology_recovery_metrics_json(output, dt=dt),
            ),
        )

    return ProducerReportSpec(
        id="topology-recovery",
        title="Topology Recovery",
        episode_factory=build_episode,
        input_summary="CaveNet topology gain variants and minimal memory-geometry proxy",
        description=(
            "Tests whether topology-like geometry collapses when topology or memory "
            "is removed and partially recovers when topology gain or associative "
            "memory is available."
        ),
        views=default_views(),
        extra_assets=extra_assets,
        checks=(lambda episode: check_topology_recovery(dt=dt),),
        frame_time=2.0,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "topology_recovery",
            "scenario": "topology_recovery",
            "role": "topology",
            "variants": list(TOPOLOGY_VARIANTS),
        },
        sections=(
            ReportSection(
                title="Question",
                body=(
                    "Topology recovery is present when repeated experience produces "
                    "nonzero topology or memory geometry, and ablations collapse the "
                    "corresponding readout."
                ),
                asset_ids=("topology_recovery_metrics",),
            ),
        ),
    )


def build_topology_recovery_episode(variant: str, *, dt: float = 1.0) -> Episode:
    if variant == "cavenet-reference":
        return _run_cavenet(variant, CaveNetConfig(), CaveNetAdaptationPolicy(enabled=False), dt=dt)
    if variant == "cavenet-no-topology":
        return _run_cavenet(
            variant,
            CaveNetConfig(topology_deposit_gain=0.0, topology_transition_gain=0.0),
            CaveNetAdaptationPolicy(enabled=False),
            dt=dt,
        )
    if variant == "cavenet-weak-topology":
        return _run_cavenet(
            variant,
            CaveNetConfig(topology_deposit_gain=0.1, topology_transition_gain=0.1),
            CaveNetAdaptationPolicy(enabled=False),
            dt=dt,
        )
    if variant == "cavenet-adaptive-topology":
        return _run_cavenet(
            variant,
            CaveNetConfig(topology_deposit_gain=0.1, topology_transition_gain=0.1),
            CaveNetAdaptationPolicy(
                enabled=True,
                surprise_threshold=0.01,
                learning_gain_rate=0.0,
                attention_gain_rate=0.0,
                topology_gain_rate=1.0,
                max_gain=3.0,
            ),
            dt=dt,
        )
    if variant == "minimal-no-memory":
        return _run_minimal(variant, MinimalSubjectConfig(memory_learning=0.0, workspace_capacity=2))
    if variant == "minimal-value-memory":
        return _run_minimal(variant, MinimalSubjectConfig(memory_learning=1.0, memory_mode="value", workspace_capacity=2))
    raise ValueError(f"unsupported topology recovery variant: {variant}")


def topology_recovery_runs(*, dt: float = 1.0) -> tuple[TopologyRun, ...]:
    return tuple(
        TopologyRun(variant, episode, _topology_metrics(episode))
        for variant in TOPOLOGY_VARIANTS
        for episode in (build_topology_recovery_episode(variant, dt=dt),)
    )


def check_topology_recovery(*, dt: float = 1.0) -> dict[str, object]:
    runs = topology_recovery_runs(dt=dt)
    metrics = {run.variant: run.metrics for run in runs}
    roles = {
        "cavenet_topology_recovery": {
            "reference_topology_mass": metrics["cavenet-reference"]["topology_mass"],
            "no_topology_mass": metrics["cavenet-no-topology"]["topology_mass"],
            "weak_topology_mass": metrics["cavenet-weak-topology"]["topology_mass"],
            "adaptive_topology_mass": metrics["cavenet-adaptive-topology"]["topology_mass"],
            "topology_mass_gain": (
                float(metrics["cavenet-adaptive-topology"]["topology_mass"])
                - float(metrics["cavenet-weak-topology"]["topology_mass"])
            ),
            "final_topology_gain": metrics["cavenet-adaptive-topology"]["final_topology_deposit_gain"],
        },
        "minimal_geometry_proxy": {
            "no_memory_strength": metrics["minimal-no-memory"]["memory_strength"],
            "value_memory_strength": metrics["minimal-value-memory"]["memory_strength"],
            "value_separation": metrics["minimal-value-memory"]["value_separation"],
        },
    }
    errors: list[str] = []
    if not float(roles["cavenet_topology_recovery"]["reference_topology_mass"]) > 0.0:
        errors.append("reference topology did not accumulate mass")
    if not float(roles["cavenet_topology_recovery"]["no_topology_mass"]) <= 1e-12:
        errors.append("no-topology variant accumulated topology mass")
    if not float(roles["cavenet_topology_recovery"]["topology_mass_gain"]) > 0.0:
        errors.append("adaptive topology did not exceed weak topology")
    if not float(roles["minimal_geometry_proxy"]["value_memory_strength"]) > float(roles["minimal_geometry_proxy"]["no_memory_strength"]):
        errors.append("minimal memory geometry did not recover with value memory")
    return {
        "id": "topology_recovery",
        "ok": not errors,
        "errors": errors,
        "metrics": metrics,
        "roles": roles,
    }


def write_topology_recovery_metrics_json(output: Path, *, dt: float = 1.0) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(encode_value(check_topology_recovery(dt=dt)), indent=2) + "\n",
        encoding="utf-8",
    )


def _topology_metrics(episode: Episode) -> dict[str, float | int | str]:
    if episode.metadata.get("adapter") == "CaveNet":
        initial = episode.metadata.get("cavenet_initial_config", {})
        final = episode.metadata.get("cavenet_config", {})
        return {
            "topology_mass": float(episode.metadata.get("cavenet_final_topology_mass", 0.0)),
            "initial_topology_deposit_gain": float(initial.get("topology_deposit_gain", 0.0)),
            "final_topology_deposit_gain": float(final.get("topology_deposit_gain", 0.0)),
            "adapter": "CaveNet",
        }
    final_obs = episode.observations[-1]
    geometry = final_obs.metadata.get("minimal_subject", {}).get("memory_geometry", {})
    return {
        "memory_strength": float(geometry.get("strength_total", 0.0)),
        "value_separation": float(geometry.get("value_separation", 0.0)),
        "trace_count": int(geometry.get("trace_count", 0)),
        "adapter": str(episode.metadata.get("adapter", "")),
    }


def _run_cavenet(
    source_name: str,
    config: CaveNetConfig,
    adaptation_policy: CaveNetAdaptationPolicy,
    *,
    dt: float,
) -> Episode:
    model = model_for_sequence(
        common_behavior_sequence("expectation_repetition"),
        params=common_behavior_params("expectation_repetition"),
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
    return CaveNetProducer(cavenet, name=f"topology-recovery:{source_name}").run(dt=dt)


def _run_minimal(variant: str, config: MinimalSubjectConfig) -> Episode:
    return run_minimal_subject(
        common_behavior_sequence("expectation_repetition"),
        vocabulary=list(COMMON_BEHAVIOR_VOCABULARY),
        preference_vector=np.array([0.0, 1.0, 0.0, 0.0, 1.0, -1.0], dtype=float),
        config=config,
        source_name=f"topology-recovery:{variant}",
    )
