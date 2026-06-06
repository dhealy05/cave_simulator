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
from cave.substrates.evolved_subject import (
    EVOLVED_VOCABULARY,
    EvolvedSubject,
    cached_evolution_result,
    cached_nonrecurrent_evolution_result,
    decode_genome,
    evolved_behavior_metrics,
    evolved_episode_from_run,
    exposure_control_sequence,
    genome_size,
    latent_future_outcome_accuracy,
    run_evolved_subject,
)


EVOLVED_ROLE_VARIANTS = (
    "evolved-recurrent",
    "random-recurrent",
    "non-recurrent",
    "hidden-reset",
    "shuffled-temporal",
)


@dataclass(frozen=True)
class EvolvedRoleRun:
    variant: str
    episode: Episode
    metrics: dict[str, float | int | str]


def evolved_roles_report_spec(
    *,
    generations: int = 30,
    population_size: int = 32,
    world_count: int = 12,
    evaluation_cycles: int = 32,
    seed: int = 17,
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
            seed=seed,
        )

    view_assets = ()
    extra_assets = ()
    if include_assets:
        extra_assets = (
            ReportExtraAsset(
                id="evolved_roles_metrics",
                title="Evolved Role Metrics JSON",
                filename="evolved_roles_metrics.json",
                writer=lambda episode, output: write_evolved_roles_metrics_json(
                    output,
                    generations=generations,
                    population_size=population_size,
                    world_count=world_count,
                    evaluation_cycles=evaluation_cycles,
                    seed=seed,
                ),
            ),
        )

    return ProducerReportSpec(
        id="evolved-roles",
        title="Evolved Subject: Role Emergence",
        episode_factory=build_episode,
        input_summary="evolved recurrent subject over repeated valued cue/outcome trajectories",
        description=(
            "Uses one evolved recurrent subject and matched controls to ask whether "
            "cue-sensitive selection-like readout, value-shaped retention, "
            "regulation, and latent topology signals emerge under delayed value "
            "pressure."
        ),
        views=default_views(),
        view_assets=view_assets,
        extra_assets=extra_assets,
        checks=(
            lambda episode: check_evolved_roles(
                generations=generations,
                population_size=population_size,
                world_count=world_count,
                evaluation_cycles=evaluation_cycles,
                seed=seed,
            ),
        ),
        frame_time=3.0,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "evolved_subject",
            "scenario": "evolved_roles",
            "generations": generations,
            "population_size": population_size,
            "world_count": world_count,
            "evaluation_cycles": evaluation_cycles,
            "seed": seed,
            "variants": list(EVOLVED_ROLE_VARIANTS),
        },
        sections=(
            ReportSection(
                title="Question",
                body=(
                    "Can a weak recurrent subject selected only for exposure utility "
                    "show early cue sensitivity, value-retention, regulation, "
                    "and topology signals over repeated valued trajectories?"
                ),
                asset_ids=(
                    "evolved_roles_metrics",
                ),
            ),
            ReportSection(
                title="Boundary",
                body=(
                    "These are emergence probes over a compact controller. The "
                    "selection result is measured as input-weight concentration "
                    "on diagnostic cues, not as full dynamic attention or as a "
                    "separate internal expectation channel. Retention and topology "
                    "are latent-state probes, and regulation is measured behaviorally."
                ),
            ),
        ),
    )


def build_evolved_roles_episode(
    variant: str,
    *,
    generations: int = 30,
    population_size: int = 32,
    world_count: int = 12,
    evaluation_cycles: int = 32,
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
        seed=seed + 5000,
        structured=True,
    )
    shuffled = exposure_control_sequence(
        cycles=evaluation_cycles,
        seed=seed + 5000,
        structured=False,
    )
    if variant == "evolved-recurrent":
        subject = result.subject
        run = run_evolved_subject(subject, structured)
        metadata = _metadata(result, variant)
    elif variant == "random-recurrent":
        subject = EvolvedSubject(
            genome=np.zeros(genome_size(result.subject_config), dtype=float),
            config=result.subject_config,
        )
        run = run_evolved_subject(subject, structured)
        metadata = _metadata(result, variant)
    elif variant == "non-recurrent":
        subject = nonrecurrent.subject
        run = run_evolved_subject(subject, structured)
        metadata = _metadata(nonrecurrent, variant)
    elif variant == "hidden-reset":
        subject = result.subject
        run = run_evolved_subject(subject, structured, reset_each_step=True)
        metadata = _metadata(result, variant)
    elif variant == "shuffled-temporal":
        subject = result.subject
        run = run_evolved_subject(subject, shuffled)
        metadata = _metadata(result, variant)
    else:
        raise ValueError(f"unsupported evolved role variant: {variant}")
    return evolved_episode_from_run(
        run,
        source_name=f"evolved-roles:{variant}",
        metadata=metadata,
    )


def evolved_role_runs(
    *,
    generations: int = 30,
    population_size: int = 32,
    world_count: int = 12,
    evaluation_cycles: int = 32,
    seed: int = 17,
) -> tuple[EvolvedRoleRun, ...]:
    return tuple(
        EvolvedRoleRun(variant, episode, _metrics(episode))
        for variant in EVOLVED_ROLE_VARIANTS
        for episode in (
            build_evolved_roles_episode(
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
def check_evolved_roles(
    *,
    generations: int = 30,
    population_size: int = 32,
    world_count: int = 12,
    evaluation_cycles: int = 32,
    seed: int = 17,
) -> dict[str, object]:
    runs = evolved_role_runs(
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
        "selection_under_bottleneck": {
            "evolved_cue_total_ratio": evolved["cue_total_ratio"],
            "evolved_cue_neutral_ratio": evolved["cue_neutral_ratio"],
            "nonrecurrent_cue_total_ratio": nonrecurrent["cue_total_ratio"],
            "random_cue_total_ratio": random["cue_total_ratio"],
            "cue_total_gain_over_nonrecurrent": (
                float(evolved["cue_total_ratio"]) - float(nonrecurrent["cue_total_ratio"])
            ),
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
        "value_shaped_retention": {
            "evolved_probe_accuracy": evolved["probe_accuracy"],
            "reset_probe_accuracy": reset["probe_accuracy"],
            "shuffled_probe_accuracy": shuffled["probe_accuracy"],
            "evolved_latent_value_signal": evolved["latent_value_signal"],
            "reset_latent_value_signal": reset["latent_value_signal"],
            "shuffled_latent_value_signal": shuffled["latent_value_signal"],
            "signal_gain_over_reset": (
                float(evolved["latent_value_signal"]) - float(reset["latent_value_signal"])
            ),
            "signal_gain_over_shuffled": (
                float(evolved["latent_value_signal"]) - float(shuffled["latent_value_signal"])
            ),
        },
        "exposure_regulation": {
            "evolved_good_exposure": evolved["good_exposure"],
            "evolved_neutral_exposure": evolved["neutral_exposure"],
            "evolved_bad_exposure": evolved["bad_exposure"],
            "evolved_exposure_contrast": evolved["exposure_contrast"],
            "reset_exposure_contrast": reset["exposure_contrast"],
            "shuffled_exposure_contrast": shuffled["exposure_contrast"],
            "evolved_utility": evolved["utility"],
            "random_utility": random["utility"],
            "nonrecurrent_utility": nonrecurrent["utility"],
        },
        "latent_topology": {
            "evolved_latent_value_separation": evolved["latent_value_separation"],
            "evolved_latent_within_class_distance": evolved["latent_within_class_distance"],
            "evolved_latent_value_signal": evolved["latent_value_signal"],
            "reset_latent_value_signal": reset["latent_value_signal"],
            "shuffled_latent_value_signal": shuffled["latent_value_signal"],
            "topology_signal_gain_over_reset": (
                float(evolved["latent_value_signal"]) - float(reset["latent_value_signal"])
            ),
            "topology_signal_gain_over_shuffled": (
                float(evolved["latent_value_signal"]) - float(shuffled["latent_value_signal"])
            ),
        },
    }
    errors: list[str] = []
    if not float(roles["selection_under_bottleneck"]["evolved_cue_total_ratio"]) > 0.5:
        errors.append("evolved subject did not concentrate enough input sensitivity on cues")
    if not float(roles["selection_under_bottleneck"]["evolved_cue_neutral_ratio"]) > 0.7:
        errors.append("evolved subject did not weight cues over neutral input")
    if not float(roles["selection_under_bottleneck"]["cue_total_gain_over_nonrecurrent"]) > 0.1:
        errors.append("recurrent cue sensitivity did not exceed non-recurrent control")
    if bool(roles["attention_claim_boundary"]["dynamic_attention_claimed"]):
        errors.append("evolved-role report should not claim full dynamic attention")
    if bool(roles["attention_claim_boundary"]["internal_expectation_channel_claimed"]):
        errors.append("evolved-role report should not claim an internal expectation channel")
    if not float(roles["value_shaped_retention"]["evolved_probe_accuracy"]) >= 0.8:
        errors.append("future outcome was not decodable from evolved hidden state")
    if not float(roles["value_shaped_retention"]["signal_gain_over_reset"]) > 2.0:
        errors.append("hidden reset did not collapse value-retention signal")
    if not float(roles["value_shaped_retention"]["signal_gain_over_shuffled"]) > 2.0:
        errors.append("shuffled temporal control retained value-retention signal")
    if not float(roles["exposure_regulation"]["evolved_exposure_contrast"]) > 0.5:
        errors.append("evolved subject did not regulate good versus bad exposure")
    if not (
        float(roles["exposure_regulation"]["evolved_good_exposure"])
        > float(roles["exposure_regulation"]["evolved_neutral_exposure"])
        > float(roles["exposure_regulation"]["evolved_bad_exposure"])
    ):
        errors.append("exposure did not separate good, neutral, and bad contexts")
    if not abs(float(roles["exposure_regulation"]["reset_exposure_contrast"])) < 0.1:
        errors.append("hidden reset retained exposure contrast")
    if not abs(float(roles["exposure_regulation"]["shuffled_exposure_contrast"])) < 0.2:
        errors.append("shuffled temporal control retained strong exposure contrast")
    if not float(roles["latent_topology"]["evolved_latent_value_signal"]) > 5.0:
        errors.append("evolved latent geometry did not separate valued trajectories")
    if not float(roles["latent_topology"]["topology_signal_gain_over_shuffled"]) > 2.0:
        errors.append("shuffled temporal control retained topology-like value signal")
    return {
        "id": "evolved_roles",
        "ok": not errors,
        "errors": errors,
        "metrics": metrics,
        "roles": roles,
    }


def write_evolved_roles_metrics_json(
    output: Path,
    *,
    generations: int = 30,
    population_size: int = 32,
    world_count: int = 12,
    evaluation_cycles: int = 32,
    seed: int = 17,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            encode_value(
                check_evolved_roles(
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


def _metrics(episode: Episode) -> dict[str, float | int | str]:
    behavior = evolved_behavior_metrics(episode)
    selection = _selection_metrics(episode)
    topology = _latent_value_topology_metrics(episode)
    return {
        **behavior,
        **selection,
        **topology,
        "neutral_exposure": _neutral_exposure(episode),
        "probe_accuracy": latent_future_outcome_accuracy(episode),
        "adapter": str(episode.metadata.get("adapter", "")),
        "variant": str(episode.metadata.get("variant", "")),
        "final_training_fitness": float(episode.metadata.get("final_training_fitness", 0.0)),
    }


def _selection_metrics(episode: Episode) -> dict[str, float]:
    result = cached_evolution_result(
        seed=_evolution_config_int(episode, "seed", 17),
        generations=_evolution_config_int(episode, "generations", 30),
        population_size=_evolution_config_int(episode, "population_size", 32),
        world_count=_evolution_config_int(episode, "world_count", 12),
    )
    subject = result.subject
    if episode.metadata.get("variant") == "random-recurrent":
        subject = EvolvedSubject(
            genome=np.zeros(genome_size(result.subject_config), dtype=float),
            config=result.subject_config,
        )
    elif episode.metadata.get("variant") == "non-recurrent":
        nonrecurrent = cached_nonrecurrent_evolution_result(
            seed=_evolution_config_int(episode, "seed", 18),
            generations=_evolution_config_int(episode, "generations", 30),
            population_size=_evolution_config_int(episode, "population_size", 32),
            world_count=_evolution_config_int(episode, "world_count", 12),
        )
        subject = nonrecurrent.subject
    weights = decode_genome(subject.genome, subject.config)
    columns = np.linalg.norm(weights["w_x"], axis=0)
    cue_mass = float(columns[EVOLVED_VOCABULARY.index("cue_good")] + columns[EVOLVED_VOCABULARY.index("cue_bad")])
    outcome_mass = float(columns[EVOLVED_VOCABULARY.index("good")] + columns[EVOLVED_VOCABULARY.index("bad")])
    neutral_mass = float(columns[EVOLVED_VOCABULARY.index("neutral")])
    total = cue_mass + outcome_mass + neutral_mass
    return {
        "cue_input_mass": cue_mass,
        "outcome_input_mass": outcome_mass,
        "neutral_input_mass": neutral_mass,
        "cue_total_ratio": 0.0 if total <= 1e-12 else cue_mass / total,
        "cue_neutral_ratio": 0.0 if cue_mass + neutral_mass <= 1e-12 else cue_mass / (cue_mass + neutral_mass),
        "cue_outcome_ratio": 0.0 if cue_mass + outcome_mass <= 1e-12 else cue_mass / (cue_mass + outcome_mass),
    }


def _evolution_config_int(episode: Episode, field: str, default: int) -> int:
    config = episode.metadata.get("evolution_config", {})
    if isinstance(config, dict):
        return int(config.get(field, default))
    return int(getattr(config, field, default))


def _latent_value_topology_metrics(episode: Episode) -> dict[str, float]:
    hidden: list[np.ndarray] = []
    labels: list[int] = []
    for obs in episode.observations:
        meta = obs.metadata.get("evolved_subject", {})
        outcome_value = float(meta.get("outcome_value", 0.0))
        if outcome_value == 0.0:
            continue
        hidden.append(np.asarray(obs.memory_state, dtype=float))
        labels.append(1 if outcome_value > 0.0 else 0)
    if len(set(labels)) < 2 or len(labels) < 4:
        return {
            "latent_value_separation": 0.0,
            "latent_within_class_distance": 0.0,
            "latent_value_signal": 0.0,
        }
    vectors = np.stack(hidden, axis=0)
    labels_array = np.asarray(labels, dtype=int)
    positive = vectors[labels_array == 1]
    negative = vectors[labels_array == 0]
    positive_center = np.mean(positive, axis=0)
    negative_center = np.mean(negative, axis=0)
    separation = float(np.linalg.norm(positive_center - negative_center))
    positive_within = float(np.mean(np.linalg.norm(positive - positive_center, axis=1)))
    negative_within = float(np.mean(np.linalg.norm(negative - negative_center, axis=1)))
    within = 0.5 * (positive_within + negative_within)
    return {
        "latent_value_separation": separation,
        "latent_within_class_distance": within,
        "latent_value_signal": separation / (within + 1e-9),
    }


def _neutral_exposure(episode: Episode) -> float:
    values = [
        obs.attention
        for obs in episode.observations
        if float(obs.metadata.get("evolved_subject", {}).get("outcome_value", 0.0)) == 0.0
    ]
    return float(np.mean(values)) if values else 0.0


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
