from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cave.substrates.minimal_subject import (
    MinimalSubjectConfig,
    emergence_metrics,
    pressure_role_metrics,
    run_minimal_subject,
)
from cave.observation.episodes import Episode
from cave.observation.experience import ExperienceObject, FeatureVector, InputSequence, TemporalExtent
from cave.observation.projections import encode_value
from cave.demonstrations.scenarios._common import core_scenario_assets
from cave.presentation.reports.specs import ProducerReportSpec, ReportExtraAsset, ReportSection
from cave.observation.views import default_views


PREFERENCE_VOCABULARY = [
    "cue_safe",
    "cue_threat",
    "preferred",
    "threat",
    "distractor_a",
    "distractor_b",
]


def preference_emergence_report_spec(
    *,
    dt: float = 0.1,
    fps: int = 12,
    include_assets: bool = True,
) -> ProducerReportSpec:
    def build_episode() -> Episode:
        return build_preference_emergence_episode("minimal-preference")

    view_assets = core_scenario_assets() if include_assets else ()
    extra_assets = ()
    if include_assets:
        extra_assets = (
            ReportExtraAsset(
                id="pressure_roles_json",
                title="Pressure Role Metrics JSON",
                filename="pressure_roles.json",
                writer=write_preference_pressure_roles_json,
            ),
            ReportExtraAsset(
                id="emergence_plot",
                title="Emergence Plot",
                filename="emergence.png",
                writer=save_preference_emergence_plot,
            ),
        )

    return ProducerReportSpec(
        id="preference-emergence",
        title="Cave Scenario: Preference-Driven Emergence",
        episode_factory=build_episode,
        input_summary="minimal subject with workspace, associative memory, and preference",
        description=(
            "A minimal adaptive subject receives a repeated cue/outcome stream. "
            "It has no dedicated predictor module: expectation-like readout comes "
            "from associative memory shaped by workspace and preference pressure."
        ),
        views=default_views(),
        view_assets=view_assets,
        extra_assets=extra_assets,
        frame_time=10.0,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "minimal_subject",
            "scenario": "preference_emergence",
            "dt": dt,
            "fps": fps,
            "vocabulary": list(PREFERENCE_VOCABULARY),
        },
        checks=(check_preference_emergence,),
        sections=(
            ReportSection(
                title="Claim",
                body=(
                    "The subject is given workspace, associative memory, and a "
                    "preference vector. It is not given a named transition predictor. "
                    "Cave-like roles are measured as readouts from the resulting "
                    "organization."
                ),
                asset_ids=("expectation_actual",),
            ),
            ReportSection(
                title="Ablations",
                body=(
                    "The report asks whether each pressure produces the expected "
                    "role: workspace pressure for diagnostic input selection, "
                    "temporal recurrence for memory-like trace, preference pressure "
                    "for value-shaped retention, delayed consequence for "
                    "prediction-like readout, and repeated trajectories for "
                    "topology-like geometry."
                ),
            ),
            ReportSection(
                title="Cave-Like Readouts",
                body=(
                    "The ablations remove memory, preference, workspace bottleneck, "
                    "or stable cue/outcome structure. A frequency-memory baseline "
                    "retains recurrence without value-shaped retention."
                ),
                asset_ids=("emergence_plot",),
            ),
        ),
    )


def build_preference_emergence_episode(variant: str) -> Episode:
    structured = variant != "shuffled"
    preference = preference_vector(enabled=variant != "no-preference")
    config = MinimalSubjectConfig(
        workspace_capacity=(
            len(PREFERENCE_VOCABULARY) if variant == "no-bottleneck" else 2
        ),
        memory_learning=0.0 if variant == "no-memory" else 1.0,
        memory_mode="frequency" if variant == "frequency-memory" else "value",
        diagnostic_features=("cue_safe", "cue_threat"),
    )
    return run_minimal_subject(
        preference_emergence_sequence(structured=structured),
        vocabulary=list(PREFERENCE_VOCABULARY),
        preference_vector=preference,
        config=config,
        source_name=f"minimal:{variant}",
    )


def preference_vector(*, enabled: bool = True) -> np.ndarray:
    if not enabled:
        return np.zeros(len(PREFERENCE_VOCABULARY), dtype=float)
    return np.array([0.0, 0.0, 1.0, -1.0, 0.0, 0.0], dtype=float)


def preference_emergence_sequence(
    *,
    structured: bool = True,
    cycles: int = 10,
) -> InputSequence:
    objects: list[ExperienceObject] = []
    order = 0
    t = 0.0
    broken = (
        ("cue_safe", "preferred"),
        ("cue_threat", "threat"),
        ("cue_safe", "threat"),
        ("cue_threat", "preferred"),
        ("cue_safe", "preferred"),
        ("cue_threat", "preferred"),
    )
    for cycle in range(cycles):
        pairs = (
            (("cue_safe", "preferred"), ("cue_threat", "threat"))
            if structured
            else (broken[cycle % len(broken)],)
        )
        for cue, outcome in pairs:
            objects.append(_event(cue, t, order))
            t += 1.0
            order += 1
            objects.append(_event(outcome, t, order))
            t += 1.0
            order += 1
    return InputSequence(objects)


def check_preference_emergence(episode: Episode) -> dict[str, object]:
    variants = {
        variant: build_preference_emergence_episode(variant)
        for variant in (
            "minimal-preference",
            "no-memory",
            "no-preference",
            "no-bottleneck",
            "frequency-memory",
            "shuffled",
        )
    }
    metrics = {
        variant: emergence_metrics(run)
        for variant, run in variants.items()
    }
    baseline = metrics["minimal-preference"]
    no_memory = metrics["no-memory"]
    no_preference = metrics["no-preference"]
    no_bottleneck = metrics["no-bottleneck"]
    frequency = metrics["frequency-memory"]
    shuffled = metrics["shuffled"]
    roles = pressure_role_metrics(metrics)

    errors = []
    if not roles["workspace_pressure_attention_like_selection"]["selection_margin"] > 0.04:
        errors.append("workspace pressure did not produce stronger diagnostic input selection")
    if bool(
        roles["workspace_pressure_attention_like_selection"][
            "full_dynamic_attention_claimed"
        ]
    ):
        errors.append("preference emergence should not claim full dynamic attention")
    if bool(
        roles["workspace_pressure_attention_like_selection"][
            "internal_expectation_channel_claimed"
        ]
    ):
        errors.append("preference emergence should not claim an internal expectation channel")
    if not roles["temporal_recurrence_memory_like_trace"]["memory_margin"] > 0.0:
        errors.append("temporal recurrence did not produce a memory-like trace")
    if not roles["preference_pressure_value_shaped_memory"]["value_strength_margin"] > 0.0:
        errors.append("preference pressure did not shape memory retention")
    if not roles["preference_pressure_value_shaped_memory"]["value_separation_margin"] > 0.0:
        errors.append("value-shaped memory did not separate valued traces beyond frequency memory")
    if not roles["delayed_consequence_prediction_like_readout"]["readout_margin"] > 0.25:
        errors.append("memory did not support prediction-like readout")
    if not roles["delayed_consequence_prediction_like_readout"]["structure_margin"] > 0.15:
        errors.append("stable structure did not improve prediction-like readout")
    if not roles["repeated_trajectory_topology_like_geometry"]["value_separation"] > 0.0:
        errors.append("repeated valued trajectories did not separate memory geometry")
    if not frequency["late_skill"] > no_memory["late_skill"]:
        errors.append("frequency memory did not retain recurrence better than no memory")
    if not baseline["utility_mean"] > no_preference["utility_mean"]:
        errors.append("preference pressure did not improve regulation relative to no preference")

    return {
        "id": "preference_emergence",
        "ok": not errors,
        "errors": errors,
        "metrics": metrics,
        "roles": roles,
    }


def write_preference_pressure_roles_json(episode: Episode, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(encode_value(pressure_role_metrics_for_report()), indent=2) + "\n",
        encoding="utf-8",
    )


def pressure_role_metrics_for_report() -> dict[str, object]:
    metrics = {
        variant: emergence_metrics(build_preference_emergence_episode(variant))
        for variant in (
            "minimal-preference",
            "no-memory",
            "no-preference",
            "no-bottleneck",
            "frequency-memory",
            "shuffled",
        )
    }
    return {
        "metrics": metrics,
        "roles": pressure_role_metrics(metrics),
    }


def save_preference_emergence_plot(episode: Episode, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    times = np.array([obs.t for obs in episode.observations], dtype=float)
    surprise = np.array([obs.surprise for obs in episode.observations], dtype=float)
    diagnostic = np.array(
        [
            obs.metadata.get("minimal_subject", {}).get("diagnostic_attention", 0.0)
            for obs in episode.observations
        ],
        dtype=float,
    )
    strength = np.array(
        [
            obs.metadata.get("minimal_subject", {})
            .get("memory_geometry", {})
            .get("strength_total", 0.0)
            for obs in episode.observations
        ],
        dtype=float,
    )
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.0), sharex=True)
    axes[0].plot(times, surprise, color="#b23a48")
    axes[0].set_ylabel("surprise")
    axes[1].plot(times, diagnostic, color="#2f6f8f")
    axes[1].set_ylabel("diagnostic mass")
    axes[2].plot(times, strength, color="#2e7d32")
    axes[2].set_ylabel("memory strength")
    axes[2].set_xlabel("time")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _event(name: str, start: float, order: int) -> ExperienceObject:
    features = {key: 0.0 for key in PREFERENCE_VOCABULARY}
    features[name] = 1.0
    features["distractor_a"] = 0.9
    features["distractor_b"] = 0.8
    return ExperienceObject(
        id=f"{name}_{order:03d}",
        temporal_extent=TemporalExtent(start=start, end=start + 1.0, order_index=order),
        features=FeatureVector(features),
        kind="experience",
        salience=1.0,
    )
