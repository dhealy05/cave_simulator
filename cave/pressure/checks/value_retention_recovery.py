from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cave.observation.episodes import Episode
from cave.observation.experience import ExperienceObject, FeatureVector, InputSequence, TemporalExtent
from cave.observation.projections import encode_value
from cave.observation.views import default_views
from cave.presentation.reports.specs import ProducerReportSpec, ReportExtraAsset, ReportSection
from cave.pressure.checks.common_behaviors import COMMON_BEHAVIOR_VOCABULARY
from cave.substrates.minimal_subject import MinimalSubjectConfig, run_minimal_subject


VALUE_RETENTION_VARIANTS = (
    "minimal-no-preference",
    "minimal-frequency-memory",
    "minimal-value-memory",
)


@dataclass(frozen=True)
class ValueRetentionRun:
    variant: str
    episode: Episode
    metrics: dict[str, float | int | str]


def value_retention_recovery_report_spec(
    *,
    dt: float = 1.0,
    fps: int = 4,
    include_assets: bool = True,
) -> ProducerReportSpec:
    def build_episode() -> Episode:
        return build_value_retention_episode("minimal-value-memory")

    extra_assets = ()
    if include_assets:
        extra_assets = (
            ReportExtraAsset(
                id="value_retention_metrics",
                title="Value Retention Metrics JSON",
                filename="value_retention_metrics.json",
                writer=lambda episode, output: write_value_retention_metrics_json(output),
            ),
        )

    return ProducerReportSpec(
        id="value-retention-recovery",
        title="Value Retention Recovery",
        episode_factory=build_episode,
        input_summary="minimal_subject value-memory, frequency-memory, and no-preference variants",
        description=(
            "Tests whether preference/value pressure shapes what the minimal "
            "subject retains, compared with frequency-only memory and no-preference "
            "controls."
        ),
        views=default_views(),
        extra_assets=extra_assets,
        checks=(lambda episode: check_value_retention_recovery(),),
        frame_time=8.0,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "value_retention_recovery",
            "scenario": "value_retention_recovery",
            "role": "value_retention",
            "variants": list(VALUE_RETENTION_VARIANTS),
        },
        sections=(
            ReportSection(
                title="Question",
                body=(
                    "Value retention is present when memory preferentially retains "
                    "valued successors instead of merely storing frequent neutral "
                    "state."
                ),
            ),
            ReportSection(
                title="Boundary",
                body=(
                    "This is a minimal-subject recovery result. It asks whether "
                    "associative memory becomes value-shaped under preference "
                    "pressure; it does not claim CaveNet has learned value retention."
                ),
                asset_ids=("value_retention_metrics",),
            ),
        ),
    )


def build_value_retention_episode(variant: str) -> Episode:
    if variant == "minimal-no-preference":
        preference = np.zeros(len(COMMON_BEHAVIOR_VOCABULARY), dtype=float)
        config = MinimalSubjectConfig(workspace_capacity=2, memory_mode="value")
    elif variant == "minimal-frequency-memory":
        preference = _preference_vector()
        config = MinimalSubjectConfig(workspace_capacity=2, memory_mode="frequency")
    elif variant == "minimal-value-memory":
        preference = _preference_vector()
        config = MinimalSubjectConfig(workspace_capacity=2, memory_mode="value")
    else:
        raise ValueError(f"unsupported value retention variant: {variant}")
    return run_minimal_subject(
        _value_retention_sequence(),
        vocabulary=list(COMMON_BEHAVIOR_VOCABULARY),
        preference_vector=preference,
        config=config,
        source_name=f"value-retention:{variant}",
    )


def value_retention_runs() -> tuple[ValueRetentionRun, ...]:
    runs = []
    for variant in VALUE_RETENTION_VARIANTS:
        episode = build_value_retention_episode(variant)
        runs.append(ValueRetentionRun(variant, episode, _value_metrics(episode)))
    return tuple(runs)


def check_value_retention_recovery() -> dict[str, object]:
    runs = value_retention_runs()
    metrics = {run.variant: run.metrics for run in runs}
    roles = {
        "value_shaped_retention": {
            "no_preference_memory_strength": metrics["minimal-no-preference"]["memory_strength"],
            "frequency_valued_focus": metrics["minimal-frequency-memory"]["valued_memory_focus"],
            "value_valued_focus": metrics["minimal-value-memory"]["valued_memory_focus"],
            "focus_margin": (
                float(metrics["minimal-value-memory"]["valued_memory_focus"])
                - float(metrics["minimal-frequency-memory"]["valued_memory_focus"])
            ),
            "frequency_value_separation": metrics["minimal-frequency-memory"]["value_separation"],
            "value_value_separation": metrics["minimal-value-memory"]["value_separation"],
            "value_memory_strength": metrics["minimal-value-memory"]["memory_strength"],
        }
    }
    errors: list[str] = []
    if not roles["value_shaped_retention"]["no_preference_memory_strength"] <= 1e-12:
        errors.append("no-preference variant retained value-shaped memory")
    if not roles["value_shaped_retention"]["value_valued_focus"] > roles["value_shaped_retention"]["frequency_valued_focus"]:
        errors.append("value memory did not focus retained state on valued features")
    if not roles["value_shaped_retention"]["focus_margin"] > 0.5:
        errors.append("value-memory focus margin was too small")
    if not roles["value_shaped_retention"]["value_memory_strength"] > 0.0:
        errors.append("value-memory variant did not retain traces")
    return {
        "id": "value_retention_recovery",
        "ok": not errors,
        "errors": errors,
        "metrics": metrics,
        "roles": roles,
    }


def write_value_retention_metrics_json(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(encode_value(check_value_retention_recovery()), indent=2) + "\n",
        encoding="utf-8",
    )


def _value_metrics(episode: Episode) -> dict[str, float | int | str]:
    final = episode.observations[-1]
    memory = np.asarray(final.memory_state, dtype=float)
    total = float(np.sum(np.abs(memory)))
    good_index = COMMON_BEHAVIOR_VOCABULARY.index("good")
    bad_index = COMMON_BEHAVIOR_VOCABULARY.index("bad")
    valued_focus = (
        0.0
        if total <= 1e-12
        else float((abs(memory[good_index]) + abs(memory[bad_index])) / total)
    )
    geometry = final.metadata.get("minimal_subject", {}).get("memory_geometry", {})
    return {
        "memory_strength": float(geometry.get("strength_total", 0.0)),
        "value_separation": float(geometry.get("value_separation", 0.0)),
        "valued_memory_focus": valued_focus,
        "trace_count": int(geometry.get("trace_count", 0)),
        "adapter": str(episode.metadata.get("adapter", "")),
    }


def _value_retention_sequence(cycles: int = 8) -> InputSequence:
    objects: list[ExperienceObject] = []
    t = 0.0
    order = 0
    for cycle in range(cycles):
        for outcome_id, outcome_features in (
            ("neutral", {"distractor": 1.0}),
            ("good", {"good": 1.0}),
            ("neutral_again", {"distractor": 1.0}),
            ("bad", {"bad": 1.0}),
        ):
            objects.append(_event(f"cue_{cycle}_{outcome_id}", t, order, {"cue": 1.0, "distractor": 0.8}))
            t += 1.0
            order += 1
            objects.append(_event(f"{outcome_id}_{cycle}", t, order, outcome_features))
            t += 1.0
            order += 1
    return InputSequence(objects)


def _event(id: str, start: float, order: int, features: dict[str, float]) -> ExperienceObject:
    return ExperienceObject(
        id=id,
        temporal_extent=TemporalExtent(start=start, end=start + 1.0, order_index=order),
        features=FeatureVector(features),
    )


def _preference_vector() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 0.0, 1.0, -1.0], dtype=float)
