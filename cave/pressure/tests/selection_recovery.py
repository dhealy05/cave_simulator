from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path

import numpy as np

from cave.commitments.workspace import IdentityWorkspaceCompressor
from cave.demonstrations.examples import model_for_sequence
from cave.observation.episodes import Episode
from cave.observation.experience import ExperienceObject, FeatureVector, InputSequence, TemporalExtent
from cave.observation.projections import encode_value
from cave.observation.views import default_views
from cave.presentation.reports.specs import ProducerReportSpec, ReportExtraAsset, ReportSection
from cave.pressure.tests.common_behaviors import (
    COMMON_BEHAVIOR_VOCABULARY,
    common_behavior_params,
    common_behavior_sequence,
    workspace_selection_metrics,
)
from cave.substrates.cavenet import CaveNet, CaveNetAdaptationPolicy, CaveNetConfig, CaveNetProducer
from cave.substrates.minimal_subject import MinimalSubjectConfig, emergence_metrics, run_minimal_subject


SELECTION_RECOVERY_VARIANTS = (
    "cavenet-selection",
    "cavenet-no-bottleneck",
    "minimal-bottleneck",
    "minimal-no-bottleneck",
    "minimal-no-memory",
)


@dataclass(frozen=True)
class SelectionRun:
    variant: str
    substrate: str
    episode: Episode
    metrics: dict[str, float | int | str]


def selection_recovery_report_spec(
    *,
    dt: float = 1.0,
    fps: int = 4,
    include_assets: bool = True,
) -> ProducerReportSpec:
    def build_episode() -> Episode:
        return build_selection_recovery_episode("cavenet-selection", dt=dt)

    extra_assets = ()
    if include_assets:
        extra_assets = (
            ReportExtraAsset(
                id="selection_recovery_metrics",
                title="Selection Recovery Metrics JSON",
                filename="selection_recovery_metrics.json",
                writer=lambda episode, output: write_selection_recovery_metrics_json(
                    output,
                    dt=dt,
                ),
            ),
        )

    return ProducerReportSpec(
        id="selection-recovery",
        title="Selection Role Recovery",
        episode_factory=build_episode,
        input_summary="CaveNet and minimal_subject selection/bottleneck variants",
        description=(
            "Runs selection probes through bottlenecked and no-bottleneck variants. "
            "The question is whether a limited workspace preserves a smaller, more "
            "diagnostic state than an unconstrained pass-through variant."
        ),
        views=default_views(),
        view_assets=(),
        extra_assets=extra_assets,
        checks=(lambda episode: check_selection_recovery(dt=dt),),
        frame_time=0.0,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "selection_recovery",
            "scenario": "selection_role_recovery",
            "role": "selection",
            "dt": dt,
            "fps": fps,
            "variants": list(SELECTION_RECOVERY_VARIANTS),
        },
        sections=(
            ReportSection(
                title="Question",
                body=(
                    "Selection is treated as a role produced by bottleneck pressure: "
                    "the substrate must keep a smaller represented state while "
                    "dropping distractor mass. The minimal subject is also scored "
                    "for diagnostic workspace mass across a repeated cue/outcome "
                    "stream."
                ),
            ),
            ReportSection(
                title="Boundary",
                body=(
                    "This report does not claim that CaveNet learns a workspace "
                    "compressor. It shows that the common harness detects the role "
                    "and its no-bottleneck failure mode, while the minimal subject "
                    "expresses a comparable diagnostic-selection readout."
                ),
                asset_ids=("selection_recovery_metrics",),
            ),
        ),
    )


def build_selection_recovery_episode(variant: str, *, dt: float = 1.0) -> Episode:
    if variant == "cavenet-selection":
        return _run_cavenet(
            variant,
            no_bottleneck=False,
            dt=dt,
        )
    if variant == "cavenet-no-bottleneck":
        return _run_cavenet(
            variant,
            no_bottleneck=True,
            dt=dt,
        )
    if variant == "minimal-bottleneck":
        return _run_minimal(
            variant,
            config=MinimalSubjectConfig(
                workspace_capacity=2,
                diagnostic_features=("cue",),
            ),
        )
    if variant == "minimal-no-bottleneck":
        return _run_minimal(
            variant,
            config=MinimalSubjectConfig(
                workspace_capacity=len(COMMON_BEHAVIOR_VOCABULARY),
                diagnostic_features=("cue",),
            ),
        )
    if variant == "minimal-no-memory":
        return _run_minimal(
            variant,
            config=MinimalSubjectConfig(
                workspace_capacity=2,
                memory_learning=0.0,
                diagnostic_features=("cue",),
            ),
        )
    raise ValueError(f"unsupported selection recovery variant: {variant}")


def selection_recovery_runs(*, dt: float = 1.0) -> tuple[SelectionRun, ...]:
    runs: list[SelectionRun] = []
    for variant in SELECTION_RECOVERY_VARIANTS:
        episode = build_selection_recovery_episode(variant, dt=dt)
        runs.append(
            SelectionRun(
                variant=variant,
                substrate="cavenet" if variant.startswith("cavenet") else "minimal_subject",
                episode=episode,
                metrics=_selection_metrics(variant, episode),
            )
        )
    return tuple(runs)


def check_selection_recovery(*, dt: float = 1.0) -> dict[str, object]:
    runs = selection_recovery_runs(dt=dt)
    metrics = {run.variant: run.metrics for run in runs}
    roles = _role_metrics(metrics)
    errors: list[str] = []

    if not roles["cavenet_bottleneck"]["selected_active_feature_count"] <= 2:
        errors.append("CaveNet bottleneck did not keep the selected state within capacity")
    if not roles["cavenet_bottleneck"]["selected_dropped_mass"] > 0.0:
        errors.append("CaveNet bottleneck did not drop distractor mass")
    if not roles["cavenet_bottleneck"]["no_bottleneck_dropped_mass"] <= 1e-12:
        errors.append("CaveNet no-bottleneck unexpectedly dropped mass")
    if not roles["cavenet_bottleneck"]["selection_margin"] > 0.0:
        errors.append("CaveNet bottleneck did not separate from no-bottleneck")
    if not roles["minimal_diagnostic_selection"]["bottleneck_late_diagnostic_attention"] > roles["minimal_diagnostic_selection"]["no_bottleneck_late_diagnostic_attention"]:
        errors.append("minimal bottleneck did not produce stronger diagnostic selection")
    if not roles["minimal_diagnostic_selection"]["bottleneck_memory_strength"] > roles["minimal_diagnostic_selection"]["no_memory_strength"]:
        errors.append("minimal bottleneck variant did not retain selected traces")

    return {
        "id": "selection_recovery",
        "ok": not errors,
        "errors": errors,
        "metrics": metrics,
        "roles": roles,
    }


def write_selection_recovery_metrics_json(output: Path, *, dt: float = 1.0) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    result = check_selection_recovery(dt=dt)
    output.write_text(json.dumps(encode_value(result), indent=2) + "\n", encoding="utf-8")


def _role_metrics(metrics: dict[str, dict[str, float | int | str]]) -> dict[str, dict[str, float | int]]:
    return {
        "cavenet_bottleneck": {
            "selected_active_feature_count": int(metrics["cavenet-selection"]["active_feature_count"]),
            "no_bottleneck_active_feature_count": int(metrics["cavenet-no-bottleneck"]["active_feature_count"]),
            "selected_dropped_mass": float(metrics["cavenet-selection"]["dropped_mass"]),
            "no_bottleneck_dropped_mass": float(metrics["cavenet-no-bottleneck"]["dropped_mass"]),
            "selection_margin": (
                float(metrics["cavenet-selection"]["dropped_mass"])
                - float(metrics["cavenet-no-bottleneck"]["dropped_mass"])
            ),
        },
        "minimal_diagnostic_selection": {
            "bottleneck_late_diagnostic_attention": float(metrics["minimal-bottleneck"]["late_diagnostic_attention"]),
            "no_bottleneck_late_diagnostic_attention": float(metrics["minimal-no-bottleneck"]["late_diagnostic_attention"]),
            "no_memory_late_diagnostic_attention": float(metrics["minimal-no-memory"]["late_diagnostic_attention"]),
            "bottleneck_diagnostic_attention_gain": float(metrics["minimal-bottleneck"]["diagnostic_attention_gain"]),
            "no_memory_diagnostic_attention_gain": float(metrics["minimal-no-memory"]["diagnostic_attention_gain"]),
            "bottleneck_memory_strength": float(metrics["minimal-bottleneck"]["late_memory_strength"]),
            "no_memory_strength": float(metrics["minimal-no-memory"]["late_memory_strength"]),
            "selection_margin": (
                float(metrics["minimal-bottleneck"]["late_diagnostic_attention"])
                - float(metrics["minimal-no-bottleneck"]["late_diagnostic_attention"])
            ),
        },
    }


def _selection_metrics(variant: str, episode: Episode) -> dict[str, float | int | str]:
    if variant.startswith("cavenet"):
        return dict(workspace_selection_metrics(episode))
    metrics = emergence_metrics(episode)
    return {
        "early_diagnostic_attention": metrics["early_diagnostic_attention"],
        "late_diagnostic_attention": metrics["late_diagnostic_attention"],
        "diagnostic_attention_gain": metrics["diagnostic_attention_gain"],
        "late_memory_strength": metrics["late_memory_strength"],
        "adapter": str(episode.metadata.get("adapter", "")),
    }


def _run_cavenet(
    source_name: str,
    *,
    no_bottleneck: bool,
    dt: float,
) -> Episode:
    sequence = common_behavior_sequence("workspace_selection")
    params = common_behavior_params("workspace_selection")
    if no_bottleneck:
        params = dataclass_replace(
            params,
            workspace_compressor=IdentityWorkspaceCompressor(),
            workspace_input_mode="actual",
        )
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
        config=CaveNetConfig(),
        adaptation_policy=CaveNetAdaptationPolicy(enabled=False),
    )
    return CaveNetProducer(cavenet, name=f"selection-recovery:{source_name}").run(dt=dt)


def _run_minimal(variant: str, *, config: MinimalSubjectConfig) -> Episode:
    return run_minimal_subject(
        _minimal_selection_sequence(),
        vocabulary=list(COMMON_BEHAVIOR_VOCABULARY),
        preference_vector=np.array([0.0, 0.0, 0.0, 0.0, 1.0, -1.0], dtype=float),
        config=config,
        source_name=f"selection-recovery:{variant}",
    )


def _minimal_selection_sequence(cycles: int = 10) -> InputSequence:
    objects: list[ExperienceObject] = []
    t = 0.0
    order = 0
    for _ in range(cycles):
        for cue_id, cue_feature, outcome_id, outcome_feature in (
            ("cue_good", "expected", "good_outcome", "good"),
            ("cue_bad", "violation", "bad_outcome", "bad"),
        ):
            objects.append(
                _event(
                    cue_id,
                    t,
                    order,
                    {
                        "cue": 1.0,
                        cue_feature: 0.7,
                        "distractor": 0.8,
                    },
                )
            )
            t += 1.0
            order += 1
            objects.append(
                _event(
                    outcome_id,
                    t,
                    order,
                    {
                        outcome_feature: 1.0,
                        "distractor": 0.8,
                    },
                )
            )
            t += 1.0
            order += 1
    return InputSequence(objects)


def _event(
    id: str,
    start: float,
    order: int,
    features: dict[str, float],
) -> ExperienceObject:
    return ExperienceObject(
        id=f"{id}_{order}",
        temporal_extent=TemporalExtent(start=start, end=start + 1.0, order_index=order),
        features=FeatureVector(features),
        salience=1.0,
    )
