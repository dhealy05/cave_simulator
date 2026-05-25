from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

import numpy as np

from cave.substrates.cavenet import (
    CaveNet,
    CaveNetAdaptationPolicy,
    CaveNetConfig,
    CaveNetProducer,
)
from cave.observation.episodes import Episode
from cave.demonstrations.examples import demo_model, model_for_sequence, random_experience_sequence
from cave.observation.experience import InputSequence
from cave.presentation.reports.specs import ProducerReportSpec, ReportSection
from cave.commitments.topology import SubjectiveTopologyParams, SubjectiveTopologyPrior
from cave.observation.views import default_views


@dataclass(frozen=True)
class PressureRun:
    id: str
    episode: Episode


def cavenet_pressure_report_spec(
    *,
    dt: float = 0.2,
    fps: int = 8,
    include_assets: bool = True,
) -> ProducerReportSpec:
    def build_episode() -> Episode:
        return build_pressure_episode("adaptive", dt=dt)

    return ProducerReportSpec(
        id="cavenet-pressure",
        title="CaveNet: Pressure-Shaped Parameters",
        episode_factory=build_episode,
        input_summary=f"demo sequence through adaptive CaveNet(dt={dt})",
        description=(
            "Starts CaveNet with weakened architectural gains, then lets a simple "
            "pressure policy adjust selected gains from surprise, utility, and "
            "compression pressure. This is the first developmental CaveNet step."
        ),
        views=default_views(),
        view_assets=(),
        frame_time=2.0,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "cavenet",
            "scenario": "cavenet_pressure",
            "dt": dt,
            "fps": fps,
        },
        checks=(lambda episode: check_cavenet_pressure(dt=dt),),
        sections=(
            ReportSection(
                title="Pressure",
                body=(
                    "The adaptive run begins with weak attention, learning, and "
                    "topology gains. Surprise and negative utility increase the "
                    "corresponding gains over time."
                ),
            ),
            ReportSection(
                title="Question",
                body=(
                    "This does not prove emergence. It asks whether pressure can "
                    "move CaveNet parameters toward more functional Cave-like "
                    "regimes than a fixed weak baseline."
                ),
            ),
        ),
    )


def cavenet_pressure_population_report_spec(
    *,
    sequence_count: int = 6,
    event_count: int = 5,
    seed: int = 31,
    dt: float = 0.2,
    fps: int = 8,
    include_assets: bool = True,
) -> ProducerReportSpec:
    def build_episode() -> Episode:
        sequence = random_experience_sequence(count=event_count, seed=seed)
        return build_pressure_episode("adaptive", dt=dt, sequence=sequence)

    return ProducerReportSpec(
        id="cavenet-pressure-population",
        title="CaveNet: Pressure Across Generated Worlds",
        episode_factory=build_episode,
        input_summary=(
            f"{sequence_count} generated sequences x weak/adaptive/reference CaveNets"
        ),
        description=(
            "Runs the same pressure rule across a small population of generated "
            "experience sequences. The question is whether adaptive CaveNet "
            "moves toward the fixed reference more reliably than a fixed weak "
            "architecture, not whether one selected scene behaves well."
        ),
        views=default_views(),
        view_assets=(),
        frame_time=2.0,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "cavenet",
            "scenario": "cavenet_pressure_population",
            "sequence_count": sequence_count,
            "event_count": event_count,
            "seed": seed,
            "dt": dt,
            "fps": fps,
        },
        checks=(
            lambda episode: check_cavenet_pressure_population(
                sequence_count=sequence_count,
                event_count=event_count,
                seed=seed,
                dt=dt,
            ),
        ),
        sections=(
            ReportSection(
                title="Population Test",
                body=(
                    "Each generated world is run three ways: fixed weak, adaptive "
                    "from the same weak start, and fixed reference. The aggregate "
                    "metric is distance to the reference trajectory."
                ),
            ),
            ReportSection(
                title="Interpretation",
                body=(
                    "A positive result means the pressure rule generalizes across "
                    "several inputs. It still does not show open-ended emergence; "
                    "it shows an architectural parameterization can be shaped "
                    "toward Cave-like function by local pressure signals."
                ),
            ),
        ),
    )


def build_pressure_episode(
    variant: str,
    *,
    dt: float,
    sequence: InputSequence | None = None,
) -> Episode:
    if variant == "fixed-weak":
        return _run_cavenet(
            source_name="cavenet:fixed-weak",
            config=_weak_config(),
            adaptation_policy=CaveNetAdaptationPolicy(enabled=False),
            dt=dt,
            sequence=sequence,
        )
    if variant == "adaptive":
        return _run_cavenet(
            source_name="cavenet:adaptive",
            config=_weak_config(),
            adaptation_policy=CaveNetAdaptationPolicy(
                enabled=True,
                surprise_threshold=0.08,
                learning_gain_rate=0.35,
                attention_gain_rate=0.2,
                topology_gain_rate=0.18,
                max_gain=2.5,
            ),
            dt=dt,
            sequence=sequence,
        )
    if variant == "fixed-reference":
        return _run_cavenet(
            source_name="cavenet:fixed-reference",
            config=CaveNetConfig(),
            adaptation_policy=CaveNetAdaptationPolicy(enabled=False),
            dt=dt,
            sequence=sequence,
        )
    raise ValueError(f"unsupported pressure variant: {variant}")


def check_cavenet_pressure(*, dt: float) -> dict[str, object]:
    runs = {
        variant: build_pressure_episode(variant, dt=dt)
        for variant in ("fixed-weak", "adaptive", "fixed-reference")
    }
    metrics = {
        variant: _pressure_metrics(episode)
        for variant, episode in runs.items()
    }
    closeness = {
        "fixed-weak": _trajectory_distance(
            runs["fixed-weak"],
            runs["fixed-reference"],
        ),
        "adaptive": _trajectory_distance(
            runs["adaptive"],
            runs["fixed-reference"],
        ),
    }
    roles = {
        "parameter_development": {
            "initial_learning_gain": metrics["adaptive"]["initial_learning_rate_gain"],
            "final_learning_gain": metrics["adaptive"]["final_learning_rate_gain"],
            "initial_attention_gain": metrics["adaptive"]["initial_attention_gain"],
            "final_attention_gain": metrics["adaptive"]["final_attention_gain"],
            "initial_external_attention_gain": metrics["adaptive"][
                "initial_attention_gain"
            ],
            "final_external_attention_gain": metrics["adaptive"][
                "final_attention_gain"
            ],
            "initial_topology_gain": metrics["adaptive"]["initial_topology_deposit_gain"],
            "final_topology_gain": metrics["adaptive"]["final_topology_deposit_gain"],
        },
        "functional_recovery": {
            "fixed_weak_memory_mass": metrics["fixed-weak"]["memory_mass"],
            "adaptive_memory_mass": metrics["adaptive"]["memory_mass"],
            "reference_memory_mass": metrics["fixed-reference"]["memory_mass"],
            "fixed_weak_topology": metrics["fixed-weak"]["topology_mass"],
            "adaptive_topology": metrics["adaptive"]["topology_mass"],
            "reference_topology": metrics["fixed-reference"]["topology_mass"],
        },
        "pressure_response": {
            "adaptive_config_delta": metrics["adaptive"]["config_delta"],
            "fixed_weak_config_delta": metrics["fixed-weak"]["config_delta"],
        },
        "reference_closeness": {
            "fixed_weak_distance": closeness["fixed-weak"]["combined_distance"],
            "adaptive_distance": closeness["adaptive"]["combined_distance"],
            "distance_improvement": (
                closeness["fixed-weak"]["combined_distance"]
                - closeness["adaptive"]["combined_distance"]
            ),
            "fixed_weak_memory_distance": closeness["fixed-weak"]["memory_distance"],
            "adaptive_memory_distance": closeness["adaptive"]["memory_distance"],
            "fixed_weak_actual_distance": closeness["fixed-weak"]["actual_distance"],
            "adaptive_actual_distance": closeness["adaptive"]["actual_distance"],
        },
    }
    errors = []
    if not roles["parameter_development"]["final_learning_gain"] > roles["parameter_development"]["initial_learning_gain"]:
        errors.append("learning gain did not increase under pressure")
    if not roles["parameter_development"]["final_attention_gain"] > roles["parameter_development"]["initial_attention_gain"]:
        errors.append("external attention gain did not increase under pressure")
    if not roles["parameter_development"]["final_topology_gain"] > roles["parameter_development"]["initial_topology_gain"]:
        errors.append("topology gain did not increase under pressure")
    if not roles["functional_recovery"]["adaptive_memory_mass"] > roles["functional_recovery"]["fixed_weak_memory_mass"]:
        errors.append("adaptive run did not recover more memory than fixed weak run")
    if not roles["functional_recovery"]["adaptive_topology"] > roles["functional_recovery"]["fixed_weak_topology"]:
        errors.append("adaptive run did not recover more topology than fixed weak run")
    if not roles["pressure_response"]["adaptive_config_delta"] > roles["pressure_response"]["fixed_weak_config_delta"]:
        errors.append("adaptive config did not move more than fixed weak config")
    if not roles["reference_closeness"]["distance_improvement"] > 0.0:
        errors.append("adaptive run did not move closer to reference trajectory")
    if not roles["reference_closeness"]["adaptive_memory_distance"] < roles["reference_closeness"]["fixed_weak_memory_distance"]:
        errors.append("adaptive run did not move memory closer to reference")

    return {
        "id": "cavenet_pressure",
        "ok": not errors,
        "errors": errors,
        "metrics": metrics,
        "closeness": closeness,
        "roles": roles,
    }


def pressure_population_records(
    *,
    sequence_count: int,
    event_count: int,
    seed: int,
    dt: float,
) -> list[dict[str, object]]:
    records = []
    for index in range(sequence_count):
        sequence_seed = seed + index
        sequence = random_experience_sequence(count=event_count, seed=sequence_seed)
        runs = {
            variant: build_pressure_episode(variant, dt=dt, sequence=sequence)
            for variant in ("fixed-weak", "adaptive", "fixed-reference")
        }
        weak_distance = _trajectory_distance(
            runs["fixed-weak"],
            runs["fixed-reference"],
        )
        adaptive_distance = _trajectory_distance(
            runs["adaptive"],
            runs["fixed-reference"],
        )
        metrics = {
            variant: _pressure_metrics(episode)
            for variant, episode in runs.items()
        }
        records.append(
            {
                "sequence_id": f"Q{index}",
                "seed": sequence_seed,
                "fixed_weak_distance": weak_distance,
                "adaptive_distance": adaptive_distance,
                "distance_improvement": (
                    weak_distance["combined_distance"]
                    - adaptive_distance["combined_distance"]
                ),
                "memory_distance_improvement": (
                    weak_distance["memory_distance"]
                    - adaptive_distance["memory_distance"]
                ),
                "actual_distance_improvement": (
                    weak_distance["actual_distance"]
                    - adaptive_distance["actual_distance"]
                ),
                "fixed_weak_metrics": metrics["fixed-weak"],
                "adaptive_metrics": metrics["adaptive"],
                "reference_metrics": metrics["fixed-reference"],
            }
        )
    return records


def check_cavenet_pressure_population(
    *,
    sequence_count: int,
    event_count: int,
    seed: int,
    dt: float,
) -> dict[str, object]:
    records = pressure_population_records(
        sequence_count=sequence_count,
        event_count=event_count,
        seed=seed,
        dt=dt,
    )
    improved = [
        record
        for record in records
        if float(record["distance_improvement"]) > 0.0
    ]
    memory_improved = [
        record
        for record in records
        if float(record["memory_distance_improvement"]) > 0.0
    ]
    aggregate = {
        "sequence_count": sequence_count,
        "event_count": event_count,
        "seed": seed,
        "improved_sequence_count": len(improved),
        "memory_improved_sequence_count": len(memory_improved),
        "mean_fixed_weak_distance": _mean_record_distance(
            records,
            "fixed_weak_distance",
            "combined_distance",
        ),
        "mean_adaptive_distance": _mean_record_distance(
            records,
            "adaptive_distance",
            "combined_distance",
        ),
        "mean_distance_improvement": _mean(records, "distance_improvement"),
        "mean_memory_distance_improvement": _mean(
            records,
            "memory_distance_improvement",
        ),
        "mean_actual_distance_improvement": _mean(
            records,
            "actual_distance_improvement",
        ),
        "mean_fixed_weak_memory_mass": _mean_record_metric(
            records,
            "fixed_weak_metrics",
            "memory_mass",
        ),
        "mean_adaptive_memory_mass": _mean_record_metric(
            records,
            "adaptive_metrics",
            "memory_mass",
        ),
        "mean_fixed_weak_topology": _mean_record_metric(
            records,
            "fixed_weak_metrics",
            "topology_mass",
        ),
        "mean_adaptive_topology": _mean_record_metric(
            records,
            "adaptive_metrics",
            "topology_mass",
        ),
        "mean_adaptive_config_delta": _mean_record_metric(
            records,
            "adaptive_metrics",
            "config_delta",
        ),
    }
    roles = {
        "population_recovery": {
            "mean_distance_improvement": aggregate["mean_distance_improvement"],
            "improved_sequence_count": aggregate["improved_sequence_count"],
            "sequence_count": sequence_count,
            "mean_memory_distance_improvement": aggregate[
                "mean_memory_distance_improvement"
            ],
            "memory_improved_sequence_count": aggregate[
                "memory_improved_sequence_count"
            ],
        },
        "functional_recovery": {
            "mean_fixed_weak_memory_mass": aggregate["mean_fixed_weak_memory_mass"],
            "mean_adaptive_memory_mass": aggregate["mean_adaptive_memory_mass"],
            "mean_fixed_weak_topology": aggregate["mean_fixed_weak_topology"],
            "mean_adaptive_topology": aggregate["mean_adaptive_topology"],
        },
        "pressure_response": {
            "mean_adaptive_config_delta": aggregate["mean_adaptive_config_delta"],
        },
    }
    minimum_improved = max(1, int(np.ceil(sequence_count * 0.6)))
    errors = []
    if not aggregate["mean_distance_improvement"] > 0.0:
        errors.append("adaptive population did not move closer to reference on average")
    if not aggregate["improved_sequence_count"] >= minimum_improved:
        errors.append("adaptive run did not improve enough generated sequences")
    if not aggregate["mean_memory_distance_improvement"] > 0.0:
        errors.append("adaptive memory trajectory did not move closer on average")
    if not aggregate["mean_adaptive_memory_mass"] > aggregate["mean_fixed_weak_memory_mass"]:
        errors.append("adaptive population did not recover more memory mass")
    if not aggregate["mean_adaptive_topology"] > aggregate["mean_fixed_weak_topology"]:
        errors.append("adaptive population did not recover more topology mass")
    if not aggregate["mean_adaptive_config_delta"] > 0.0:
        errors.append("adaptive population gains did not move")

    return {
        "id": "cavenet_pressure_population",
        "ok": not errors,
        "errors": errors,
        "aggregate": aggregate,
        "roles": roles,
        "records": records,
    }


def _weak_config() -> CaveNetConfig:
    return CaveNetConfig(
        attention_gain=0.45,
        learning_rate_gain=0.2,
        topology_deposit_gain=0.2,
        topology_transition_gain=0.2,
    )


def _run_cavenet(
    *,
    source_name: str,
    config: CaveNetConfig,
    adaptation_policy: CaveNetAdaptationPolicy,
    dt: float,
    sequence: InputSequence | None = None,
) -> Episode:
    model = _base_model(sequence=sequence)
    cavenet = CaveNet.from_subject_state(
        sequence=model.sequence,
        subject_state=model.subject_state,
        params=model.params,
        vocabulary=model.vocabulary,
        sensorium=model.sensorium,
        config=config,
        adaptation_policy=adaptation_policy,
    )
    return CaveNetProducer(cavenet, name=source_name).run(dt=dt)


def _pressure_metrics(episode: Episode) -> dict[str, float]:
    initial = episode.metadata.get("cavenet_initial_config", {})
    final = episode.metadata.get("cavenet_config", {})
    return {
        "initial_attention_gain": float(initial.get("attention_gain", 0.0)),
        "final_attention_gain": float(final.get("attention_gain", 0.0)),
        "initial_learning_rate_gain": float(initial.get("learning_rate_gain", 0.0)),
        "final_learning_rate_gain": float(final.get("learning_rate_gain", 0.0)),
        "initial_topology_deposit_gain": float(
            initial.get("topology_deposit_gain", 0.0)
        ),
        "final_topology_deposit_gain": float(final.get("topology_deposit_gain", 0.0)),
        "config_delta": _config_delta(initial, final),
        "actual_mass": sum(_norm(obs.actual) for obs in episode.observations),
        "surprise_total": sum(obs.surprise for obs in episode.observations),
        "memory_mass": sum(_norm(obs.memory_state) for obs in episode.observations),
        "final_memory_mass": _norm(episode.observations[-1].memory_state),
        "topology_mass": float(episode.metadata.get("cavenet_final_topology_mass", 0.0)),
    }


def _config_delta(initial: dict[str, float], final: dict[str, float]) -> float:
    keys = sorted(set(initial) | set(final))
    if not keys:
        return 0.0
    return float(
        np.linalg.norm(
            np.array(
                [
                    float(final.get(key, 0.0)) - float(initial.get(key, 0.0))
                    for key in keys
                ],
                dtype=float,
            )
        )
        / np.sqrt(len(keys))
    )


def _trajectory_distance(episode: Episode, reference: Episode) -> dict[str, float]:
    actual = _stack_distance(
        [obs.actual for obs in episode.observations],
        [obs.actual for obs in reference.observations],
    )
    expected = _stack_distance(
        [obs.expected for obs in episode.observations],
        [obs.expected for obs in reference.observations],
    )
    memory = _stack_distance(
        [obs.memory_state for obs in episode.observations],
        [obs.memory_state for obs in reference.observations],
    )
    surprise = _stack_distance(
        [[obs.surprise] for obs in episode.observations],
        [[obs.surprise] for obs in reference.observations],
    )
    combined = float(np.mean([actual, expected, memory, surprise]))
    return {
        "actual_distance": actual,
        "expected_distance": expected,
        "memory_distance": memory,
        "surprise_distance": surprise,
        "combined_distance": combined,
    }


def _stack_distance(values, reference_values) -> float:
    first = np.asarray(values, dtype=float)
    second = np.asarray(reference_values, dtype=float)
    if first.shape != second.shape:
        return float("inf")
    if first.size == 0:
        return 0.0
    return float(np.linalg.norm((first - second).ravel()) / np.sqrt(first.size))


def _base_model(sequence: InputSequence | None = None):
    model = demo_model(seed=1) if sequence is None else model_for_sequence(sequence)
    params = replace(
        model.params,
        topology=SubjectiveTopologyParams(prior=SubjectiveTopologyPrior()),
    )
    return model_for_sequence(
        model.sequence,
        params=params,
        vocabulary=model.vocabulary,
    )


def _mean(records: list[dict[str, object]], key: str) -> float:
    if not records:
        return 0.0
    return float(np.mean([float(record[key]) for record in records]))


def _mean_record_distance(
    records: list[dict[str, object]],
    record_key: str,
    distance_key: str,
) -> float:
    if not records:
        return 0.0
    return float(
        np.mean(
            [
                float(record[record_key][distance_key])  # type: ignore[index]
                for record in records
            ]
        )
    )


def _mean_record_metric(
    records: list[dict[str, object]],
    record_key: str,
    metric_key: str,
) -> float:
    if not records:
        return 0.0
    return float(
        np.mean(
            [
                float(record[record_key][metric_key])  # type: ignore[index]
                for record in records
            ]
        )
    )


def _norm(value) -> float:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.linalg.norm(array.ravel()) / np.sqrt(array.size))
