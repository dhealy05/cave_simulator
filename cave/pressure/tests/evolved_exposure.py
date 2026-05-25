from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from cave.observation.episodes import Episode
from cave.observation.projections import encode_value
from cave.observation.views import ObserverView, SubjectSurfaceView, default_views
from cave.presentation.renderers.matplotlib_renderer.iris import save_iris_expression_animation
from cave.presentation.renderers.matplotlib_renderer.observer_grid import save_observer_comparison_animation
from cave.presentation.reports.specs import ProducerReportSpec, ReportExtraAsset, ReportSection, ReportViewAsset
from cave.substrates.evolved_subject import (
    EvolvedSubject,
    EvolvedSubjectConfig,
    cached_evolution_result,
    cached_nonrecurrent_evolution_result,
    evolved_behavior_metrics,
    evolved_episode_from_run,
    exposure_control_sequence,
    genome_size,
    latent_future_outcome_accuracy,
    run_evolved_subject,
)


EVOLVED_EXPOSURE_VARIANTS = (
    "evolved-recurrent",
    "random-recurrent",
    "non-recurrent",
    "hidden-reset",
    "shuffled-temporal",
)


@dataclass(frozen=True)
class EvolvedExposureRun:
    variant: str
    episode: Episode
    metrics: dict[str, float | int | str]


def evolved_exposure_report_spec(
    *,
    generations: int = 30,
    population_size: int = 32,
    world_count: int = 12,
    evaluation_cycles: int = 20,
    seed: int = 17,
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
            seed=seed,
        )

    view_assets = ()
    extra_assets = ()
    if include_assets:
        view_assets = (
            ReportViewAsset(
                id="subject_surface",
                title="Subject Surface",
                views=[SubjectSurfaceView()],
                filename="subject_surface.gif",
            ),
            ReportViewAsset(
                id="observer",
                title="Observer",
                views=[ObserverView()],
                filename="observer.gif",
            ),
        )
        extra_assets = (
            ReportExtraAsset(
                id="evolved_exposure_metrics",
                title="Evolved Exposure Metrics JSON",
                filename="evolved_exposure_metrics.json",
                writer=lambda episode, output: write_evolved_exposure_metrics_json(
                    output,
                    generations=generations,
                    population_size=population_size,
                    world_count=world_count,
                    evaluation_cycles=evaluation_cycles,
                    seed=seed,
                ),
            ),
            ReportExtraAsset(
                id="observer_controls",
                title="Observer Control Grid",
                filename="observer_controls.gif",
                writer=lambda episode, output: write_evolved_exposure_observer_controls(
                    output,
                    generations=generations,
                    population_size=population_size,
                    world_count=world_count,
                    evaluation_cycles=evaluation_cycles,
                    seed=seed,
                    fps=fps,
                ),
            ),
            ReportExtraAsset(
                id="iris_expression",
                title="Iris Diaphragm Expression",
                filename="iris_expression.gif",
                writer=lambda episode, output: save_iris_expression_animation(
                    episode,
                    output,
                    fps=fps,
                ),
            ),
        )

    return ProducerReportSpec(
        id="evolved-exposure",
        title="Evolved Subject: Exposure Control",
        episode_factory=build_episode,
        input_summary="evolved recurrent subject in delayed cue/outcome world",
        description=(
            "Evolves a small recurrent controller for exposure regulation in a "
            "delayed cue/outcome world. The subject is selected for utility, not "
            "prediction loss. Expectation is measured by behavior and latent-state "
            "future-outcome decodability."
        ),
        views=default_views(),
        view_assets=view_assets,
        extra_assets=extra_assets,
        checks=(
            lambda episode: check_evolved_exposure(
                generations=generations,
                population_size=population_size,
                world_count=world_count,
                evaluation_cycles=evaluation_cycles,
                seed=seed,
            ),
        ),
        frame_time=2.0,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "evolved_subject",
            "scenario": "evolved_exposure",
            "generations": generations,
            "population_size": population_size,
            "world_count": world_count,
            "evaluation_cycles": evaluation_cycles,
            "seed": seed,
            "variants": list(EVOLVED_EXPOSURE_VARIANTS),
        },
        sections=(
            ReportSection(
                title="Question",
                body=(
                    "Does delayed value consequence plus exposure control select for "
                    "a recurrent latent state that behaves like expectation?"
                ),
                asset_ids=(
                    "iris_expression",
                    "subject_surface",
                    "observer",
                    "observer_controls",
                    "evolved_exposure_metrics",
                ),
            ),
            ReportSection(
                title="Boundary",
                body=(
                    "The evolved subject has no native Cave expectation variable, "
                    "memory trace, or prediction loss. Future-outcome decoding is a "
                    "probe over hidden state, not a native readout."
                ),
            ),
        ),
    )


def build_evolved_exposure_episode(
    variant: str,
    *,
    generations: int = 30,
    population_size: int = 32,
    world_count: int = 12,
    evaluation_cycles: int = 20,
    seed: int = 17,
) -> Episode:
    result = cached_evolution_result(
        seed=seed,
        generations=generations,
        population_size=population_size,
        world_count=world_count,
    )
    nonrecurrent = cached_nonrecurrent_evolution_result(
        seed=seed + 1,
        generations=generations,
        population_size=population_size,
        world_count=world_count,
    )
    structured = exposure_control_sequence(
        cycles=evaluation_cycles,
        seed=seed + 4000,
        structured=True,
    )
    shuffled = exposure_control_sequence(
        cycles=evaluation_cycles,
        seed=seed + 4000,
        structured=False,
    )
    if variant == "evolved-recurrent":
        run = run_evolved_subject(result.subject, structured)
        metadata = _metadata(result, variant)
    elif variant == "random-recurrent":
        random_subject = EvolvedSubject(
            genome=np.zeros(genome_size(result.subject_config), dtype=float),
            config=result.subject_config,
        )
        run = run_evolved_subject(random_subject, structured)
        metadata = _metadata(result, variant)
    elif variant == "non-recurrent":
        run = run_evolved_subject(nonrecurrent.subject, structured)
        metadata = _metadata(nonrecurrent, variant)
    elif variant == "hidden-reset":
        run = run_evolved_subject(result.subject, structured, reset_each_step=True)
        metadata = _metadata(result, variant)
    elif variant == "shuffled-temporal":
        run = run_evolved_subject(result.subject, shuffled)
        metadata = _metadata(result, variant)
    else:
        raise ValueError(f"unsupported evolved exposure variant: {variant}")
    return evolved_episode_from_run(
        run,
        source_name=f"evolved:{variant}",
        metadata=metadata,
    )


def evolved_exposure_runs(
    *,
    generations: int = 30,
    population_size: int = 32,
    world_count: int = 12,
    evaluation_cycles: int = 20,
    seed: int = 17,
) -> tuple[EvolvedExposureRun, ...]:
    return tuple(
        EvolvedExposureRun(
            variant,
            episode,
            _metrics(episode),
        )
        for variant in EVOLVED_EXPOSURE_VARIANTS
        for episode in (
            build_evolved_exposure_episode(
                variant,
                generations=generations,
                population_size=population_size,
                world_count=world_count,
                evaluation_cycles=evaluation_cycles,
                seed=seed,
            ),
        )
    )


@lru_cache(maxsize=8)
def check_evolved_exposure(
    *,
    generations: int = 30,
    population_size: int = 32,
    world_count: int = 12,
    evaluation_cycles: int = 20,
    seed: int = 17,
) -> dict[str, object]:
    runs = evolved_exposure_runs(
        generations=generations,
        population_size=population_size,
        world_count=world_count,
        evaluation_cycles=evaluation_cycles,
        seed=seed,
    )
    metrics = {run.variant: run.metrics for run in runs}
    evolved = metrics["evolved-recurrent"]
    random = metrics["random-recurrent"]
    nonrecurrent = metrics["non-recurrent"]
    reset = metrics["hidden-reset"]
    shuffled = metrics["shuffled-temporal"]
    roles = {
        "exposure_regulation": {
            "evolved_utility": evolved["utility"],
            "random_utility": random["utility"],
            "nonrecurrent_utility": nonrecurrent["utility"],
            "reset_utility": reset["utility"],
            "shuffled_utility": shuffled["utility"],
            "utility_gain_over_random": float(evolved["utility"]) - float(random["utility"]),
            "utility_gain_over_nonrecurrent": float(evolved["utility"]) - float(nonrecurrent["utility"]),
            "evolved_exposure_contrast": evolved["exposure_contrast"],
            "reset_exposure_contrast": reset["exposure_contrast"],
            "shuffled_exposure_contrast": shuffled["exposure_contrast"],
        },
        "latent_expectation_probe": {
            "evolved_probe_accuracy": evolved["probe_accuracy"],
            "random_probe_accuracy": random["probe_accuracy"],
            "nonrecurrent_probe_accuracy": nonrecurrent["probe_accuracy"],
            "reset_probe_accuracy": reset["probe_accuracy"],
            "shuffled_probe_accuracy": shuffled["probe_accuracy"],
        },
    }
    errors: list[str] = []
    if not roles["exposure_regulation"]["utility_gain_over_random"] > 1.0:
        errors.append("evolved recurrent subject did not beat random recurrent control")
    if not roles["exposure_regulation"]["utility_gain_over_nonrecurrent"] > 1.0:
        errors.append("evolved recurrent subject did not beat non-recurrent control")
    if not float(roles["exposure_regulation"]["evolved_exposure_contrast"]) > 0.5:
        errors.append("evolved recurrent subject did not separate good and bad exposure")
    if not float(roles["latent_expectation_probe"]["evolved_probe_accuracy"]) >= 0.8:
        errors.append("future outcome was not decodable from evolved hidden state")
    if not float(roles["exposure_regulation"]["reset_exposure_contrast"]) < 0.1:
        errors.append("hidden reset did not collapse exposure contrast")
    if not float(roles["exposure_regulation"]["shuffled_exposure_contrast"]) < 0.2:
        errors.append("shuffled temporal control retained strong exposure contrast")
    return {
        "id": "evolved_exposure",
        "ok": not errors,
        "errors": errors,
        "metrics": metrics,
        "roles": roles,
    }


def write_evolved_exposure_metrics_json(
    output: Path,
    *,
    generations: int = 30,
    population_size: int = 32,
    world_count: int = 12,
    evaluation_cycles: int = 20,
    seed: int = 17,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            encode_value(
                check_evolved_exposure(
                    generations=generations,
                    population_size=population_size,
                    world_count=world_count,
                    evaluation_cycles=evaluation_cycles,
                    seed=seed,
                )
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_evolved_exposure_observer_controls(
    output: Path,
    *,
    generations: int = 30,
    population_size: int = 32,
    world_count: int = 12,
    evaluation_cycles: int = 20,
    seed: int = 17,
    fps: int = 4,
) -> None:
    variants = {
        "random": "random-recurrent",
        "non-recurrent": "non-recurrent",
        "reset": "hidden-reset",
        "evolved": "evolved-recurrent",
    }
    episodes = {
        label: build_evolved_exposure_episode(
            variant,
            generations=generations,
            population_size=population_size,
            world_count=world_count,
            evaluation_cycles=evaluation_cycles,
            seed=seed,
        )
        for label, variant in variants.items()
    }
    save_observer_comparison_animation(episodes, output, fps=fps)


def _metrics(episode: Episode) -> dict[str, float | int | str]:
    behavior = evolved_behavior_metrics(episode)
    return {
        **behavior,
        "probe_accuracy": latent_future_outcome_accuracy(episode),
        "adapter": str(episode.metadata.get("adapter", "")),
        "final_training_fitness": float(episode.metadata.get("final_training_fitness", 0.0)),
    }


def _metadata(result, variant: str) -> dict[str, object]:
    return {
        "variant": variant,
        "fitness_history": list(result.fitness_history),
        "initial_training_fitness": result.fitness_history[0],
        "final_training_fitness": result.fitness_history[-1],
        "random_baseline_fitness": result.random_baseline_fitness,
        "evolution_config": result.config,
        "subject_config": result.subject_config,
    }
