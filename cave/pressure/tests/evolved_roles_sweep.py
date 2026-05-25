from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from cave.observation.episodes import Episode
from cave.observation.projections import encode_value
from cave.observation.views import default_views
from cave.presentation.reports.specs import ProducerReportSpec, ReportExtraAsset, ReportSection
from cave.pressure.tests.evolved_roles import (
    build_evolved_roles_episode,
    check_evolved_roles,
)


@dataclass(frozen=True)
class EvolvedRolesSweepRecord:
    seed: int
    strict_ok: bool
    cue_total_ratio: float
    cue_neutral_ratio: float
    cue_total_gain_over_nonrecurrent: float
    probe_accuracy: float
    value_signal_gain_over_reset: float
    value_signal_gain_over_shuffled: float
    good_exposure: float
    neutral_exposure: float
    bad_exposure: float
    exposure_contrast: float
    reset_exposure_contrast: float
    shuffled_exposure_contrast: float
    latent_value_signal: float
    topology_signal_gain_over_shuffled: float
    errors: tuple[str, ...]


def evolved_roles_sweep_report_spec(
    *,
    seed_start: int = 17,
    seed_count: int = 5,
    generations: int = 20,
    population_size: int = 24,
    world_count: int = 8,
    evaluation_cycles: int = 24,
    min_core_pass_rate: float = 0.8,
    min_selection_pass_rate: float = 0.6,
    dt: float = 1.0,
    fps: int = 4,
    include_assets: bool = True,
) -> ProducerReportSpec:
    def build_episode() -> Episode:
        return build_evolved_roles_episode(
            "evolved-recurrent",
            generations=generations,
            population_size=population_size,
            world_count=world_count,
            evaluation_cycles=evaluation_cycles,
            seed=seed_start,
        )

    extra_assets = ()
    if include_assets:
        extra_assets = (
            ReportExtraAsset(
                id="evolved_roles_sweep_metrics",
                title="Evolved Roles Sweep Metrics JSON",
                filename="evolved_roles_sweep_metrics.json",
                writer=lambda episode, output: write_evolved_roles_sweep_metrics_json(
                    output,
                    seed_start=seed_start,
                    seed_count=seed_count,
                    generations=generations,
                    population_size=population_size,
                    world_count=world_count,
                    evaluation_cycles=evaluation_cycles,
                    min_core_pass_rate=min_core_pass_rate,
                    min_selection_pass_rate=min_selection_pass_rate,
                ),
            ),
        )

    return ProducerReportSpec(
        id="evolved-roles-sweep",
        title="Evolved Subject: Role Sweep",
        episode_factory=build_episode,
        input_summary=f"{seed_count} evolution seeds x evolved-role readouts",
        description=(
            "Repeats the evolved role-emergence report across evolution seeds. "
            "The sweep reports cue-sensitive selection-like readout, "
            "value-retention, regulation, and latent-geometry readouts separately "
            "so weaker roles are not hidden behind the stronger "
            "expectation/regulation result."
        ),
        views=default_views(),
        view_assets=(),
        extra_assets=extra_assets,
        checks=(
            lambda episode: check_evolved_roles_sweep(
                seed_start=seed_start,
                seed_count=seed_count,
                generations=generations,
                population_size=population_size,
                world_count=world_count,
                evaluation_cycles=evaluation_cycles,
                min_core_pass_rate=min_core_pass_rate,
                min_selection_pass_rate=min_selection_pass_rate,
            ),
        ),
        frame_time=3.0,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "evolved_subject",
            "scenario": "evolved_roles_sweep",
            "seed_start": seed_start,
            "seed_count": seed_count,
            "generations": generations,
            "population_size": population_size,
            "world_count": world_count,
            "evaluation_cycles": evaluation_cycles,
            "min_core_pass_rate": min_core_pass_rate,
            "min_selection_pass_rate": min_selection_pass_rate,
            "dt": dt,
            "fps": fps,
        },
        sections=(
            ReportSection(
                title="Question",
                body=(
                    "Which evolved-role readouts survive variation in the "
                    "evolution seed, and which remain only suggestive?"
                ),
                asset_ids=("evolved_roles_sweep_metrics",),
            ),
            ReportSection(
                title="Interpretation",
                body=(
                    "The sweep intentionally separates selection from the core "
                    "expectation/regulation/value readouts. Selection is measured "
                    "as input-weight concentration, not as full dynamic attention "
                    "or a separate internal expectation channel."
                ),
            ),
        ),
    )


@lru_cache(maxsize=8)
def check_evolved_roles_sweep(
    *,
    seed_start: int = 17,
    seed_count: int = 5,
    generations: int = 20,
    population_size: int = 24,
    world_count: int = 8,
    evaluation_cycles: int = 24,
    min_core_pass_rate: float = 0.8,
    min_selection_pass_rate: float = 0.6,
) -> dict[str, object]:
    records = evolved_roles_sweep_records(
        seed_start=seed_start,
        seed_count=seed_count,
        generations=generations,
        population_size=population_size,
        world_count=world_count,
        evaluation_cycles=evaluation_cycles,
    )
    aggregate = _aggregate(
        records,
        min_core_pass_rate=min_core_pass_rate,
        min_selection_pass_rate=min_selection_pass_rate,
    )
    roles = {
        "selection_like_readout": {
            "cue_total_pass_count": aggregate["cue_total_pass_count"],
            "cue_neutral_pass_count": aggregate["cue_neutral_pass_count"],
            "cue_gain_positive_count": aggregate["cue_gain_positive_count"],
            "required_selection_pass_count": aggregate[
                "required_selection_pass_count"
            ],
            "median_cue_total_ratio": aggregate["median_cue_total_ratio"],
            "median_cue_neutral_ratio": aggregate["median_cue_neutral_ratio"],
            "median_cue_total_gain_over_nonrecurrent": aggregate[
                "median_cue_total_gain_over_nonrecurrent"
            ],
            "claim_kind": "cue_weight_concentration",
            "full_dynamic_attention_claimed": False,
            "internal_expectation_channel_claimed": False,
        },
        "attention_claim_boundary": {
            "selection_metric": "input_weight_concentration",
            "dynamic_attention_claimed": False,
            "internal_expectation_channel_claimed": False,
            "future_allocation_claimed": False,
        },
        "value_retention": {
            "probe_accuracy_pass_count": aggregate["probe_accuracy_pass_count"],
            "value_signal_reset_pass_count": aggregate[
                "value_signal_reset_pass_count"
            ],
            "value_signal_shuffled_pass_count": aggregate[
                "value_signal_shuffled_pass_count"
            ],
            "required_core_pass_count": aggregate["required_core_pass_count"],
            "median_probe_accuracy": aggregate["median_probe_accuracy"],
            "median_value_signal_gain_over_reset": aggregate[
                "median_value_signal_gain_over_reset"
            ],
            "median_value_signal_gain_over_shuffled": aggregate[
                "median_value_signal_gain_over_shuffled"
            ],
        },
        "exposure_regulation": {
            "exposure_order_pass_count": aggregate["exposure_order_pass_count"],
            "exposure_contrast_pass_count": aggregate["exposure_contrast_pass_count"],
            "required_core_pass_count": aggregate["required_core_pass_count"],
            "median_good_exposure": aggregate["median_good_exposure"],
            "median_neutral_exposure": aggregate["median_neutral_exposure"],
            "median_bad_exposure": aggregate["median_bad_exposure"],
            "median_exposure_contrast": aggregate["median_exposure_contrast"],
            "median_abs_reset_exposure_contrast": aggregate[
                "median_abs_reset_exposure_contrast"
            ],
            "median_abs_shuffled_exposure_contrast": aggregate[
                "median_abs_shuffled_exposure_contrast"
            ],
        },
        "latent_geometry": {
            "latent_value_signal_pass_count": aggregate[
                "latent_value_signal_pass_count"
            ],
            "topology_gain_shuffled_pass_count": aggregate[
                "topology_gain_shuffled_pass_count"
            ],
            "required_core_pass_count": aggregate["required_core_pass_count"],
            "median_latent_value_signal": aggregate["median_latent_value_signal"],
            "median_topology_signal_gain_over_shuffled": aggregate[
                "median_topology_signal_gain_over_shuffled"
            ],
        },
    }
    errors: list[str] = []
    core_required = int(aggregate["required_core_pass_count"])
    selection_required = int(aggregate["required_selection_pass_count"])
    if aggregate["probe_accuracy_pass_count"] < core_required:
        errors.append("too few seeds exposed future value in hidden state")
    if aggregate["value_signal_reset_pass_count"] < core_required:
        errors.append("too few seeds collapsed value signal under hidden reset")
    if aggregate["exposure_order_pass_count"] < core_required:
        errors.append("too few seeds ordered good, neutral, and bad exposure")
    if aggregate["exposure_contrast_pass_count"] < core_required:
        errors.append("too few seeds separated good and bad exposure")
    if aggregate["latent_value_signal_pass_count"] < core_required:
        errors.append("too few seeds formed separated latent value geometry")
    if aggregate["topology_gain_shuffled_pass_count"] < core_required:
        errors.append("too few seeds collapsed latent geometry under shuffle")
    if aggregate["cue_total_pass_count"] < selection_required:
        errors.append("too few seeds concentrated input sensitivity on cues")
    if aggregate["cue_neutral_pass_count"] < selection_required:
        errors.append("too few seeds weighted cues over neutral input")
    if bool(roles["attention_claim_boundary"]["dynamic_attention_claimed"]):
        errors.append("evolved-role sweep should not claim full dynamic attention")
    if bool(roles["attention_claim_boundary"]["internal_expectation_channel_claimed"]):
        errors.append("evolved-role sweep should not claim an internal expectation channel")

    return {
        "id": "evolved_roles_sweep",
        "ok": not errors,
        "errors": errors,
        "aggregate": aggregate,
        "roles": roles,
        "records": [record.__dict__ for record in records],
    }


def evolved_roles_sweep_records(
    *,
    seed_start: int,
    seed_count: int,
    generations: int,
    population_size: int,
    world_count: int,
    evaluation_cycles: int,
) -> tuple[EvolvedRolesSweepRecord, ...]:
    records: list[EvolvedRolesSweepRecord] = []
    for offset in range(seed_count):
        seed = seed_start + offset
        check = check_evolved_roles(
            generations=generations,
            population_size=population_size,
            world_count=world_count,
            evaluation_cycles=evaluation_cycles,
            seed=seed,
        )
        roles = check["roles"]  # type: ignore[index]
        selection = roles["selection_under_bottleneck"]
        retention = roles["value_shaped_retention"]
        regulation = roles["exposure_regulation"]
        topology = roles["latent_topology"]
        records.append(
            EvolvedRolesSweepRecord(
                seed=seed,
                strict_ok=bool(check["ok"]),
                cue_total_ratio=float(selection["evolved_cue_total_ratio"]),
                cue_neutral_ratio=float(selection["evolved_cue_neutral_ratio"]),
                cue_total_gain_over_nonrecurrent=float(
                    selection["cue_total_gain_over_nonrecurrent"]
                ),
                probe_accuracy=float(retention["evolved_probe_accuracy"]),
                value_signal_gain_over_reset=float(
                    retention["signal_gain_over_reset"]
                ),
                value_signal_gain_over_shuffled=float(
                    retention["signal_gain_over_shuffled"]
                ),
                good_exposure=float(regulation["evolved_good_exposure"]),
                neutral_exposure=float(regulation["evolved_neutral_exposure"]),
                bad_exposure=float(regulation["evolved_bad_exposure"]),
                exposure_contrast=float(regulation["evolved_exposure_contrast"]),
                reset_exposure_contrast=float(regulation["reset_exposure_contrast"]),
                shuffled_exposure_contrast=float(
                    regulation["shuffled_exposure_contrast"]
                ),
                latent_value_signal=float(topology["evolved_latent_value_signal"]),
                topology_signal_gain_over_shuffled=float(
                    topology["topology_signal_gain_over_shuffled"]
                ),
                errors=tuple(str(error) for error in check.get("errors", ())),
            )
        )
    return tuple(records)


def write_evolved_roles_sweep_metrics_json(
    output: Path,
    *,
    seed_start: int = 17,
    seed_count: int = 5,
    generations: int = 20,
    population_size: int = 24,
    world_count: int = 8,
    evaluation_cycles: int = 24,
    min_core_pass_rate: float = 0.8,
    min_selection_pass_rate: float = 0.6,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            encode_value(
                check_evolved_roles_sweep(
                    seed_start=seed_start,
                    seed_count=seed_count,
                    generations=generations,
                    population_size=population_size,
                    world_count=world_count,
                    evaluation_cycles=evaluation_cycles,
                    min_core_pass_rate=min_core_pass_rate,
                    min_selection_pass_rate=min_selection_pass_rate,
                )
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _aggregate(
    records: tuple[EvolvedRolesSweepRecord, ...],
    *,
    min_core_pass_rate: float,
    min_selection_pass_rate: float,
) -> dict[str, float | int]:
    seed_count = len(records)
    core_required = int(np.ceil(seed_count * min_core_pass_rate)) if seed_count else 0
    selection_required = (
        int(np.ceil(seed_count * min_selection_pass_rate)) if seed_count else 0
    )
    return {
        "seed_count": seed_count,
        "required_core_pass_count": core_required,
        "required_selection_pass_count": selection_required,
        "strict_pass_count": _count(records, lambda record: record.strict_ok),
        "strict_pass_rate": _rate(records, lambda record: record.strict_ok),
        "cue_total_pass_count": _count(
            records,
            lambda record: record.cue_total_ratio > 0.4,
        ),
        "cue_neutral_pass_count": _count(
            records,
            lambda record: record.cue_neutral_ratio > 0.6,
        ),
        "cue_gain_positive_count": _count(
            records,
            lambda record: record.cue_total_gain_over_nonrecurrent > 0.0,
        ),
        "probe_accuracy_pass_count": _count(
            records,
            lambda record: record.probe_accuracy >= 0.8,
        ),
        "value_signal_reset_pass_count": _count(
            records,
            lambda record: record.value_signal_gain_over_reset > 2.0,
        ),
        "value_signal_shuffled_pass_count": _count(
            records,
            lambda record: record.value_signal_gain_over_shuffled > 2.0,
        ),
        "exposure_order_pass_count": _count(
            records,
            lambda record: record.good_exposure
            > record.neutral_exposure
            > record.bad_exposure,
        ),
        "exposure_contrast_pass_count": _count(
            records,
            lambda record: record.exposure_contrast > 0.5,
        ),
        "latent_value_signal_pass_count": _count(
            records,
            lambda record: record.latent_value_signal > 5.0,
        ),
        "topology_gain_shuffled_pass_count": _count(
            records,
            lambda record: record.topology_signal_gain_over_shuffled > 2.0,
        ),
        "median_cue_total_ratio": _median(
            records,
            lambda record: record.cue_total_ratio,
        ),
        "median_cue_neutral_ratio": _median(
            records,
            lambda record: record.cue_neutral_ratio,
        ),
        "median_cue_total_gain_over_nonrecurrent": _median(
            records,
            lambda record: record.cue_total_gain_over_nonrecurrent,
        ),
        "median_probe_accuracy": _median(records, lambda record: record.probe_accuracy),
        "median_value_signal_gain_over_reset": _median(
            records,
            lambda record: record.value_signal_gain_over_reset,
        ),
        "median_value_signal_gain_over_shuffled": _median(
            records,
            lambda record: record.value_signal_gain_over_shuffled,
        ),
        "median_good_exposure": _median(records, lambda record: record.good_exposure),
        "median_neutral_exposure": _median(
            records,
            lambda record: record.neutral_exposure,
        ),
        "median_bad_exposure": _median(records, lambda record: record.bad_exposure),
        "median_exposure_contrast": _median(
            records,
            lambda record: record.exposure_contrast,
        ),
        "median_abs_reset_exposure_contrast": _median(
            records,
            lambda record: abs(record.reset_exposure_contrast),
        ),
        "median_abs_shuffled_exposure_contrast": _median(
            records,
            lambda record: abs(record.shuffled_exposure_contrast),
        ),
        "median_latent_value_signal": _median(
            records,
            lambda record: record.latent_value_signal,
        ),
        "median_topology_signal_gain_over_shuffled": _median(
            records,
            lambda record: record.topology_signal_gain_over_shuffled,
        ),
    }


def _count(records, predicate) -> int:
    return sum(1 for record in records if predicate(record))


def _rate(records, predicate) -> float:
    if not records:
        return 0.0
    return float(_count(records, predicate) / len(records))


def _median(records, getter) -> float:
    if not records:
        return 0.0
    return float(np.median([float(getter(record)) for record in records]))
