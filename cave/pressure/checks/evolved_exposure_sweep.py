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
from cave.pressure.checks.evolved_exposure import (
    build_evolved_exposure_episode,
    check_evolved_exposure,
)


@dataclass(frozen=True)
class EvolvedExposureSweepRecord:
    seed: int
    ok: bool
    utility_gain_over_random: float
    utility_gain_over_nonrecurrent: float
    evolved_exposure_contrast: float
    reset_exposure_contrast: float
    shuffled_exposure_contrast: float
    evolved_probe_accuracy: float
    shuffled_probe_accuracy: float
    errors: tuple[str, ...]


def evolved_exposure_sweep_report_spec(
    *,
    seed_start: int = 17,
    seed_count: int = 5,
    generations: int = 20,
    population_size: int = 24,
    world_count: int = 8,
    evaluation_cycles: int = 16,
    min_pass_rate: float = 0.8,
    dt: float = 1.0,
    fps: int = 4,
    include_assets: bool = True,
) -> ProducerReportSpec:
    def build_episode() -> Episode:
        return build_evolved_exposure_episode(
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
                id="evolved_exposure_sweep_metrics",
                title="Evolved Exposure Sweep Metrics JSON",
                filename="evolved_exposure_sweep_metrics.json",
                writer=lambda episode, output: write_evolved_exposure_sweep_metrics_json(
                    output,
                    seed_start=seed_start,
                    seed_count=seed_count,
                    generations=generations,
                    population_size=population_size,
                    world_count=world_count,
                    evaluation_cycles=evaluation_cycles,
                    min_pass_rate=min_pass_rate,
                ),
            ),
        )

    return ProducerReportSpec(
        id="evolved-exposure-sweep",
        title="Evolved Subject: Exposure Sweep",
        episode_factory=build_episode,
        input_summary=(
            f"{seed_count} evolution seeds x delayed cue/outcome exposure controls"
        ),
        description=(
            "Repeats the evolved exposure-control experiment across evolution "
            "seeds. The report asks whether delayed value pressure reliably "
            "selects expectation-like hidden state and exposure regulation, "
            "instead of only producing a single successful demonstration."
        ),
        views=default_views(),
        view_assets=(),
        extra_assets=extra_assets,
        checks=(
            lambda episode: check_evolved_exposure_sweep(
                seed_start=seed_start,
                seed_count=seed_count,
                generations=generations,
                population_size=population_size,
                world_count=world_count,
                evaluation_cycles=evaluation_cycles,
                min_pass_rate=min_pass_rate,
            ),
        ),
        frame_time=2.0,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "evolved_subject",
            "scenario": "evolved_exposure_sweep",
            "seed_start": seed_start,
            "seed_count": seed_count,
            "generations": generations,
            "population_size": population_size,
            "world_count": world_count,
            "evaluation_cycles": evaluation_cycles,
            "min_pass_rate": min_pass_rate,
            "dt": dt,
            "fps": fps,
        },
        sections=(
            ReportSection(
                title="Question",
                body=(
                    "Does the evolved exposure result survive variation in the "
                    "evolution seed?"
                ),
                asset_ids=("evolved_exposure_sweep_metrics",),
            ),
            ReportSection(
                title="Interpretation",
                body=(
                    "The sweep is evaluated by pass rates and medians. A single "
                    "seed can be noisy, especially for shuffled temporal controls; "
                    "the aggregate asks whether the pressure-role relationship is "
                    "stable enough to treat as a result rather than an anecdote."
                ),
            ),
        ),
    )


@lru_cache(maxsize=8)
def check_evolved_exposure_sweep(
    *,
    seed_start: int = 17,
    seed_count: int = 5,
    generations: int = 20,
    population_size: int = 24,
    world_count: int = 8,
    evaluation_cycles: int = 16,
    min_pass_rate: float = 0.8,
) -> dict[str, object]:
    records = evolved_exposure_sweep_records(
        seed_start=seed_start,
        seed_count=seed_count,
        generations=generations,
        population_size=population_size,
        world_count=world_count,
        evaluation_cycles=evaluation_cycles,
    )
    aggregate = _aggregate(records, min_pass_rate=min_pass_rate)
    roles = {
        "robust_expectation_regulation": {
            "seed_count": seed_count,
            "strict_pass_count": aggregate["strict_pass_count"],
            "strict_pass_rate": aggregate["strict_pass_rate"],
            "required_pass_count": aggregate["required_pass_count"],
            "utility_gain_over_random_pass_count": aggregate[
                "utility_gain_over_random_pass_count"
            ],
            "utility_gain_over_nonrecurrent_pass_count": aggregate[
                "utility_gain_over_nonrecurrent_pass_count"
            ],
            "exposure_contrast_pass_count": aggregate[
                "exposure_contrast_pass_count"
            ],
            "probe_accuracy_pass_count": aggregate["probe_accuracy_pass_count"],
            "mean_utility_gain_over_random": aggregate[
                "mean_utility_gain_over_random"
            ],
            "mean_utility_gain_over_nonrecurrent": aggregate[
                "mean_utility_gain_over_nonrecurrent"
            ],
            "median_evolved_exposure_contrast": aggregate[
                "median_evolved_exposure_contrast"
            ],
            "median_evolved_probe_accuracy": aggregate[
                "median_evolved_probe_accuracy"
            ],
        },
        "control_collapse": {
            "median_abs_reset_exposure_contrast": aggregate[
                "median_abs_reset_exposure_contrast"
            ],
            "median_abs_shuffled_exposure_contrast": aggregate[
                "median_abs_shuffled_exposure_contrast"
            ],
            "median_abs_shuffled_minus_evolved_contrast": (
                aggregate["median_abs_shuffled_exposure_contrast"]
                - aggregate["median_evolved_exposure_contrast"]
            ),
            "median_shuffled_probe_accuracy": aggregate[
                "median_shuffled_probe_accuracy"
            ],
        },
    }
    errors: list[str] = []
    required = int(aggregate["required_pass_count"])
    if aggregate["utility_gain_over_random_pass_count"] < required:
        errors.append("too few seeds beat random recurrent control")
    if aggregate["utility_gain_over_nonrecurrent_pass_count"] < required:
        errors.append("too few seeds beat non-recurrent control")
    if aggregate["exposure_contrast_pass_count"] < required:
        errors.append("too few seeds separated good and bad exposure")
    if aggregate["probe_accuracy_pass_count"] < required:
        errors.append("too few seeds exposed future outcome in hidden state")
    if not aggregate["median_abs_reset_exposure_contrast"] < 0.1:
        errors.append("hidden-reset median exposure contrast did not collapse")
    if not aggregate["median_abs_shuffled_exposure_contrast"] < 0.2:
        errors.append("shuffled temporal median exposure contrast did not collapse")
    if not (
        aggregate["median_abs_shuffled_exposure_contrast"]
        < aggregate["median_evolved_exposure_contrast"]
    ):
        errors.append("shuffled contrast did not fall below evolved contrast")

    return {
        "id": "evolved_exposure_sweep",
        "ok": not errors,
        "errors": errors,
        "aggregate": aggregate,
        "roles": roles,
        "records": [record.__dict__ for record in records],
    }


def evolved_exposure_sweep_records(
    *,
    seed_start: int,
    seed_count: int,
    generations: int,
    population_size: int,
    world_count: int,
    evaluation_cycles: int,
) -> tuple[EvolvedExposureSweepRecord, ...]:
    records: list[EvolvedExposureSweepRecord] = []
    for offset in range(seed_count):
        seed = seed_start + offset
        check = check_evolved_exposure(
            generations=generations,
            population_size=population_size,
            world_count=world_count,
            evaluation_cycles=evaluation_cycles,
            seed=seed,
        )
        exposure = check["roles"]["exposure_regulation"]  # type: ignore[index]
        probe = check["roles"]["latent_expectation_probe"]  # type: ignore[index]
        records.append(
            EvolvedExposureSweepRecord(
                seed=seed,
                ok=bool(check["ok"]),
                utility_gain_over_random=float(
                    exposure["utility_gain_over_random"]  # type: ignore[index]
                ),
                utility_gain_over_nonrecurrent=float(
                    exposure["utility_gain_over_nonrecurrent"]  # type: ignore[index]
                ),
                evolved_exposure_contrast=float(
                    exposure["evolved_exposure_contrast"]  # type: ignore[index]
                ),
                reset_exposure_contrast=float(
                    exposure["reset_exposure_contrast"]  # type: ignore[index]
                ),
                shuffled_exposure_contrast=float(
                    exposure["shuffled_exposure_contrast"]  # type: ignore[index]
                ),
                evolved_probe_accuracy=float(
                    probe["evolved_probe_accuracy"]  # type: ignore[index]
                ),
                shuffled_probe_accuracy=float(
                    probe["shuffled_probe_accuracy"]  # type: ignore[index]
                ),
                errors=tuple(str(error) for error in check.get("errors", ())),
            )
        )
    return tuple(records)


def write_evolved_exposure_sweep_metrics_json(
    output: Path,
    *,
    seed_start: int = 17,
    seed_count: int = 5,
    generations: int = 20,
    population_size: int = 24,
    world_count: int = 8,
    evaluation_cycles: int = 16,
    min_pass_rate: float = 0.8,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            encode_value(
                check_evolved_exposure_sweep(
                    seed_start=seed_start,
                    seed_count=seed_count,
                    generations=generations,
                    population_size=population_size,
                    world_count=world_count,
                    evaluation_cycles=evaluation_cycles,
                    min_pass_rate=min_pass_rate,
                )
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _aggregate(
    records: tuple[EvolvedExposureSweepRecord, ...],
    *,
    min_pass_rate: float,
) -> dict[str, float | int]:
    seed_count = len(records)
    required = int(np.ceil(seed_count * min_pass_rate)) if seed_count else 0
    return {
        "seed_count": seed_count,
        "required_pass_count": required,
        "min_pass_rate": float(min_pass_rate),
        "strict_pass_count": _count(records, lambda record: record.ok),
        "strict_pass_rate": _rate(records, lambda record: record.ok),
        "utility_gain_over_random_pass_count": _count(
            records,
            lambda record: record.utility_gain_over_random > 1.0,
        ),
        "utility_gain_over_nonrecurrent_pass_count": _count(
            records,
            lambda record: record.utility_gain_over_nonrecurrent > 1.0,
        ),
        "exposure_contrast_pass_count": _count(
            records,
            lambda record: record.evolved_exposure_contrast > 0.5,
        ),
        "probe_accuracy_pass_count": _count(
            records,
            lambda record: record.evolved_probe_accuracy >= 0.8,
        ),
        "reset_collapse_pass_count": _count(
            records,
            lambda record: abs(record.reset_exposure_contrast) < 0.1,
        ),
        "shuffled_collapse_pass_count": _count(
            records,
            lambda record: abs(record.shuffled_exposure_contrast) < 0.2,
        ),
        "mean_utility_gain_over_random": _mean(
            records,
            lambda record: record.utility_gain_over_random,
        ),
        "mean_utility_gain_over_nonrecurrent": _mean(
            records,
            lambda record: record.utility_gain_over_nonrecurrent,
        ),
        "median_evolved_exposure_contrast": _median(
            records,
            lambda record: record.evolved_exposure_contrast,
        ),
        "median_abs_reset_exposure_contrast": _median(
            records,
            lambda record: abs(record.reset_exposure_contrast),
        ),
        "median_abs_shuffled_exposure_contrast": _median(
            records,
            lambda record: abs(record.shuffled_exposure_contrast),
        ),
        "median_evolved_probe_accuracy": _median(
            records,
            lambda record: record.evolved_probe_accuracy,
        ),
        "median_shuffled_probe_accuracy": _median(
            records,
            lambda record: record.shuffled_probe_accuracy,
        ),
        "min_evolved_probe_accuracy": _min(
            records,
            lambda record: record.evolved_probe_accuracy,
        ),
        "min_evolved_exposure_contrast": _min(
            records,
            lambda record: record.evolved_exposure_contrast,
        ),
    }


def _count(records, predicate) -> int:
    return sum(1 for record in records if predicate(record))


def _rate(records, predicate) -> float:
    if not records:
        return 0.0
    return float(_count(records, predicate) / len(records))


def _mean(records, getter) -> float:
    if not records:
        return 0.0
    return float(np.mean([float(getter(record)) for record in records]))


def _median(records, getter) -> float:
    if not records:
        return 0.0
    return float(np.median([float(getter(record)) for record in records]))


def _min(records, getter) -> float:
    if not records:
        return 0.0
    return float(np.min([float(getter(record)) for record in records]))
