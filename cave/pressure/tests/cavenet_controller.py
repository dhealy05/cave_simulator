from __future__ import annotations

from dataclasses import replace

import numpy as np

from cave.substrates.cavenet import (
    CaveNet,
    CaveNetAdaptationPolicy,
    CaveNetConfig,
    CaveNetController,
    CaveNetControllerAccess,
    CaveNetProducer,
)
from cave.observation.episodes import Episode
from cave.demonstrations.examples import demo_model, model_for_sequence, random_experience_sequence
from cave.observation.experience import InputSequence
from cave.pressure.tests.cavenet_pressure import (
    _pressure_metrics,
    _trajectory_distance,
    _weak_config,
)
from cave.presentation.reports.specs import ProducerReportSpec, ReportSection
from cave.commitments.topology import SubjectiveTopologyParams, SubjectiveTopologyPrior
from cave.observation.views import default_views


def cavenet_controller_report_spec(
    *,
    dt: float = 0.2,
    fps: int = 8,
    include_assets: bool = True,
) -> ProducerReportSpec:
    def build_episode() -> Episode:
        return build_controller_episode("controller-full", dt=dt)

    return ProducerReportSpec(
        id="cavenet-controller",
        title="CaveNet: Latent Gain Controller",
        episode_factory=build_episode,
        input_summary=f"demo sequence through controlled CaveNet(dt={dt})",
        description=(
            "Replaces direct named-gain adaptation with a small latent controller. "
            "The controller reads pressure, workspace, memory, attention "
            "capacity, and topology signals, while recording effective external "
            "and internal expectation attention for audit. It then produces the "
            "CaveNet gains through an MLP-style readout."
        ),
        views=default_views(),
        view_assets=(),
        frame_time=2.0,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "cavenet",
            "scenario": "cavenet_controller",
            "dt": dt,
            "fps": fps,
        },
        checks=(lambda episode: check_cavenet_controller(dt=dt),),
        sections=(
            ReportSection(
                title="Controller",
                body=(
                    "The controller has its own latent state. Local signals update "
                    "that state, and the latent state produces the next step's "
                    "CaveNet gains."
                ),
            ),
            ReportSection(
                title="Ablations",
                body=(
                    "Input-access ablations remove pressure, workspace, memory, "
                    "attention capacity, or topology signals from the controller "
                    "while leaving the CaveNet substrate otherwise unchanged."
                ),
            ),
        ),
    )


def cavenet_controller_population_report_spec(
    *,
    sequence_count: int = 6,
    event_count: int = 5,
    seed: int = 61,
    dt: float = 0.2,
    fps: int = 8,
    include_assets: bool = True,
) -> ProducerReportSpec:
    def build_episode() -> Episode:
        sequence = random_experience_sequence(count=event_count, seed=seed)
        return build_controller_episode("controller-full", dt=dt, sequence=sequence)

    return ProducerReportSpec(
        id="cavenet-controller-population",
        title="CaveNet: Controller Across Generated Worlds",
        episode_factory=build_episode,
        input_summary=(
            f"{sequence_count} generated sequences x controlled CaveNet variants"
        ),
        description=(
            "Runs the latent CaveNet controller across generated experience "
            "sequences. This asks whether controller-produced gains generalize "
            "beyond the authored demo."
        ),
        views=default_views(),
        view_assets=(),
        frame_time=2.0,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "cavenet",
            "scenario": "cavenet_controller_population",
            "sequence_count": sequence_count,
            "event_count": event_count,
            "seed": seed,
            "dt": dt,
            "fps": fps,
        },
        checks=(
            lambda episode: check_cavenet_controller_population(
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
                    "Each generated world is run through fixed weak CaveNet, "
                    "full controller CaveNet, controller input ablations, and "
                    "fixed reference CaveNet."
                ),
            ),
            ReportSection(
                title="Question",
                body=(
                    "The test asks whether a latent controller with ablatable "
                    "access to the substrate reliably produces more Cave-like "
                    "function than a fixed weak architecture."
                ),
            ),
        ),
    )


def cavenet_controller_learning_report_spec(
    *,
    sequence_count: int = 6,
    event_count: int = 5,
    seed: int = 91,
    dt: float = 0.2,
    fps: int = 8,
    include_assets: bool = True,
) -> ProducerReportSpec:
    def build_episode() -> Episode:
        sequence = random_experience_sequence(count=event_count, seed=seed)
        return build_controller_episode("controller-learning", dt=dt, sequence=sequence)

    return ProducerReportSpec(
        id="cavenet-controller-learning",
        title="CaveNet: Plastic Controller Readout",
        episode_factory=build_episode,
        input_summary=(
            f"{sequence_count} generated sequences x static/plastic controllers"
        ),
        description=(
            "Starts a CaveNet controller with a weak gain readout, then lets "
            "local pressure and latent activity strengthen the readout weights. "
            "This asks whether the controller can develop a more Cave-like gain "
            "profile instead of receiving the full authored readout upfront."
        ),
        views=default_views(),
        view_assets=(),
        frame_time=2.0,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "cavenet",
            "scenario": "cavenet_controller_learning",
            "sequence_count": sequence_count,
            "event_count": event_count,
            "seed": seed,
            "dt": dt,
            "fps": fps,
        },
        checks=(
            lambda episode: check_cavenet_controller_learning(
                sequence_count=sequence_count,
                event_count=event_count,
                seed=seed,
                dt=dt,
            ),
        ),
        sections=(
            ReportSection(
                title="Plastic Readout",
                body=(
                    "The learning controller begins with a weak readout from "
                    "latent controller state to CaveNet gains. Pressure-gated "
                    "Hebbian updates strengthen the readout during the run."
                ),
            ),
            ReportSection(
                title="Comparison",
                body=(
                    "The check compares fixed weak CaveNet, static weak "
                    "controller, plastic weak controller, and fixed reference "
                    "CaveNet across generated worlds."
                ),
            ),
        ),
    )


def build_controller_episode(
    variant: str,
    *,
    dt: float,
    sequence: InputSequence | None = None,
) -> Episode:
    if variant == "fixed-weak":
        return _run_cavenet(
            source_name="cavenet-controller:fixed-weak",
            config=_weak_config(),
            controller=None,
            dt=dt,
            sequence=sequence,
        )
    if variant == "fixed-reference":
        return _run_cavenet(
            source_name="cavenet-controller:fixed-reference",
            config=CaveNetConfig(),
            controller=None,
            dt=dt,
            sequence=sequence,
        )
    if variant == "controller-static-weak":
        return _run_cavenet(
            source_name="cavenet-controller:controller-static-weak",
            config=_weak_config(),
            controller=_weak_readout_controller(plastic=False),
            dt=dt,
            sequence=sequence,
        )
    if variant == "controller-learning":
        return _run_cavenet(
            source_name="cavenet-controller:controller-learning",
            config=_weak_config(),
            controller=_weak_readout_controller(plastic=True),
            dt=dt,
            sequence=sequence,
        )
    access = _controller_access_for_variant(variant)
    return _run_cavenet(
        source_name=f"cavenet-controller:{variant}",
        config=_weak_config(),
        controller=CaveNetController(base_config=_weak_config(), access=access),
        dt=dt,
        sequence=sequence,
    )


def check_cavenet_controller(*, dt: float) -> dict[str, object]:
    variants = (
        "fixed-weak",
        "controller-full",
        "no-pressure-input",
        "no-workspace-input",
        "no-memory-input",
        "no-attention-input",
        "no-topology-input",
        "fixed-reference",
    )
    runs = {variant: build_controller_episode(variant, dt=dt) for variant in variants}
    metrics = {variant: _controller_metrics(episode) for variant, episode in runs.items()}
    closeness = {
        "fixed-weak": _trajectory_distance(
            runs["fixed-weak"],
            runs["fixed-reference"],
        ),
        "controller-full": _trajectory_distance(
            runs["controller-full"],
            runs["fixed-reference"],
        ),
    }
    roles = {
        "controlled_recovery": {
            "fixed_weak_distance": closeness["fixed-weak"]["combined_distance"],
            "controller_distance": closeness["controller-full"]["combined_distance"],
            "distance_improvement": (
                closeness["fixed-weak"]["combined_distance"]
                - closeness["controller-full"]["combined_distance"]
            ),
            "fixed_weak_memory_mass": metrics["fixed-weak"]["memory_mass"],
            "controller_memory_mass": metrics["controller-full"]["memory_mass"],
            "fixed_weak_topology": metrics["fixed-weak"]["topology_mass"],
            "controller_topology": metrics["controller-full"]["topology_mass"],
        },
        "controller_state": {
            "latent_norm": metrics["controller-full"]["controller_latent_norm"],
            "config_delta": metrics["controller-full"]["config_delta"],
            "mean_attention_capacity": metrics["controller-full"][
                "mean_controller_attention_capacity"
            ],
            "mean_external_attention": metrics["controller-full"][
                "mean_controller_external_attention"
            ],
            "mean_internal_expectation_attention": metrics["controller-full"][
                "mean_controller_internal_expectation_attention"
            ],
            "pressureless_latent_norm": metrics["no-pressure-input"][
                "controller_latent_norm"
            ],
        },
        "input_ablation_effects": {
            "full_attention_gain": metrics["controller-full"][
                "final_attention_gain"
            ],
            "full_external_attention_gain": metrics["controller-full"][
                "final_attention_gain"
            ],
            "no_workspace_attention_gain": metrics["no-workspace-input"][
                "final_attention_gain"
            ],
            "workspace_attention_gain_delta": (
                metrics["controller-full"]["final_attention_gain"]
                - metrics["no-workspace-input"]["final_attention_gain"]
            ),
            "no_attention_attention_gain": metrics["no-attention-input"][
                "final_attention_gain"
            ],
            "no_attention_capacity_external_attention_gain": metrics[
                "no-attention-input"
            ]["final_attention_gain"],
            "full_learning_gain": metrics["controller-full"][
                "final_learning_rate_gain"
            ],
            "no_memory_learning_gain": metrics["no-memory-input"][
                "final_learning_rate_gain"
            ],
            "full_topology_gain": metrics["controller-full"][
                "final_topology_deposit_gain"
            ],
            "no_topology_topology_gain": metrics["no-topology-input"][
                "final_topology_deposit_gain"
            ],
        },
    }
    errors = []
    if not roles["controlled_recovery"]["distance_improvement"] > 0.0:
        errors.append("controller did not move closer to reference than fixed weak")
    if not roles["controlled_recovery"]["controller_memory_mass"] > roles["controlled_recovery"]["fixed_weak_memory_mass"]:
        errors.append("controller did not recover more memory than fixed weak")
    if not roles["controlled_recovery"]["controller_topology"] > roles["controlled_recovery"]["fixed_weak_topology"]:
        errors.append("controller did not recover more topology than fixed weak")
    if not roles["controller_state"]["latent_norm"] > 0.0:
        errors.append("controller latent state did not move")
    if not roles["controller_state"]["config_delta"] > 0.0:
        errors.append("controller did not produce gain movement")
    if not roles["input_ablation_effects"]["full_attention_gain"] > roles["input_ablation_effects"]["no_attention_attention_gain"]:
        errors.append("attention-capacity ablation did not reduce controller external attention gain")
    if not roles["input_ablation_effects"]["full_learning_gain"] > roles["input_ablation_effects"]["no_memory_learning_gain"]:
        errors.append("memory ablation did not reduce controller learning gain")
    if not roles["input_ablation_effects"]["full_topology_gain"] > roles["input_ablation_effects"]["no_topology_topology_gain"]:
        errors.append("topology ablation did not reduce controller topology gain")

    return {
        "id": "cavenet_controller",
        "ok": not errors,
        "errors": errors,
        "metrics": metrics,
        "closeness": closeness,
        "roles": roles,
    }


def controller_population_records(
    *,
    sequence_count: int,
    event_count: int,
    seed: int,
    dt: float,
) -> list[dict[str, object]]:
    records = []
    variants = (
        "fixed-weak",
        "controller-full",
        "no-pressure-input",
        "no-memory-input",
        "no-attention-input",
        "no-topology-input",
        "fixed-reference",
    )
    for index in range(sequence_count):
        sequence_seed = seed + index
        sequence = random_experience_sequence(count=event_count, seed=sequence_seed)
        runs = {
            variant: build_controller_episode(variant, dt=dt, sequence=sequence)
            for variant in variants
        }
        weak_distance = _trajectory_distance(
            runs["fixed-weak"],
            runs["fixed-reference"],
        )
        controller_distance = _trajectory_distance(
            runs["controller-full"],
            runs["fixed-reference"],
        )
        metrics = {
            variant: _controller_metrics(episode)
            for variant, episode in runs.items()
        }
        records.append(
            {
                "sequence_id": f"Q{index}",
                "seed": sequence_seed,
                "fixed_weak_distance": weak_distance,
                "controller_distance": controller_distance,
                "distance_improvement": (
                    weak_distance["combined_distance"]
                    - controller_distance["combined_distance"]
                ),
                "memory_distance_improvement": (
                    weak_distance["memory_distance"]
                    - controller_distance["memory_distance"]
                ),
                "actual_distance_improvement": (
                    weak_distance["actual_distance"]
                    - controller_distance["actual_distance"]
                ),
                "metrics": metrics,
            }
        )
    return records


def check_cavenet_controller_population(
    *,
    sequence_count: int,
    event_count: int,
    seed: int,
    dt: float,
) -> dict[str, object]:
    records = controller_population_records(
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
    aggregate = {
        "sequence_count": sequence_count,
        "event_count": event_count,
        "seed": seed,
        "improved_sequence_count": len(improved),
        "mean_fixed_weak_distance": _mean_record_distance(
            records,
            "fixed_weak_distance",
            "combined_distance",
        ),
        "mean_controller_distance": _mean_record_distance(
            records,
            "controller_distance",
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
        "mean_fixed_weak_memory_mass": _mean_variant_metric(
            records,
            "fixed-weak",
            "memory_mass",
        ),
        "mean_controller_memory_mass": _mean_variant_metric(
            records,
            "controller-full",
            "memory_mass",
        ),
        "mean_fixed_weak_topology": _mean_variant_metric(
            records,
            "fixed-weak",
            "topology_mass",
        ),
        "mean_controller_topology": _mean_variant_metric(
            records,
            "controller-full",
            "topology_mass",
        ),
        "mean_controller_latent_norm": _mean_variant_metric(
            records,
            "controller-full",
            "controller_latent_norm",
        ),
        "mean_pressureless_latent_norm": _mean_variant_metric(
            records,
            "no-pressure-input",
            "controller_latent_norm",
        ),
        "mean_controller_config_delta": _mean_variant_metric(
            records,
            "controller-full",
            "config_delta",
        ),
        "mean_controller_attention_capacity": _mean_variant_metric(
            records,
            "controller-full",
            "mean_controller_attention_capacity",
        ),
        "mean_controller_external_attention": _mean_variant_metric(
            records,
            "controller-full",
            "mean_controller_external_attention",
        ),
        "mean_controller_internal_expectation_attention": _mean_variant_metric(
            records,
            "controller-full",
            "mean_controller_internal_expectation_attention",
        ),
        "mean_attention_gain_drop_without_attention": _mean_variant_delta(
            records,
            "controller-full",
            "no-attention-input",
            "final_attention_gain",
        ),
        "mean_external_attention_gain_drop_without_attention_capacity": _mean_variant_delta(
            records,
            "controller-full",
            "no-attention-input",
            "final_attention_gain",
        ),
        "mean_learning_gain_drop_without_memory": _mean_variant_delta(
            records,
            "controller-full",
            "no-memory-input",
            "final_learning_rate_gain",
        ),
        "mean_topology_gain_drop_without_topology": _mean_variant_delta(
            records,
            "controller-full",
            "no-topology-input",
            "final_topology_deposit_gain",
        ),
    }
    roles = {
        "population_recovery": {
            "mean_fixed_weak_distance": aggregate["mean_fixed_weak_distance"],
            "mean_controller_distance": aggregate["mean_controller_distance"],
            "mean_distance_improvement": aggregate["mean_distance_improvement"],
            "improved_sequence_count": aggregate["improved_sequence_count"],
            "sequence_count": sequence_count,
            "mean_memory_distance_improvement": aggregate[
                "mean_memory_distance_improvement"
            ],
        },
        "functional_recovery": {
            "mean_fixed_weak_memory_mass": aggregate["mean_fixed_weak_memory_mass"],
            "mean_controller_memory_mass": aggregate["mean_controller_memory_mass"],
            "mean_fixed_weak_topology": aggregate["mean_fixed_weak_topology"],
            "mean_controller_topology": aggregate["mean_controller_topology"],
        },
        "controller_state": {
            "mean_controller_latent_norm": aggregate[
                "mean_controller_latent_norm"
            ],
            "mean_pressureless_latent_norm": aggregate[
                "mean_pressureless_latent_norm"
            ],
            "mean_controller_config_delta": aggregate[
                "mean_controller_config_delta"
            ],
            "mean_attention_capacity": aggregate[
                "mean_controller_attention_capacity"
            ],
            "mean_external_attention": aggregate[
                "mean_controller_external_attention"
            ],
            "mean_internal_expectation_attention": aggregate[
                "mean_controller_internal_expectation_attention"
            ],
        },
        "input_ablation_effects": {
            "mean_attention_gain_drop_without_attention": aggregate[
                "mean_attention_gain_drop_without_attention"
            ],
            "mean_external_attention_gain_drop_without_attention_capacity": aggregate[
                "mean_external_attention_gain_drop_without_attention_capacity"
            ],
            "mean_learning_gain_drop_without_memory": aggregate[
                "mean_learning_gain_drop_without_memory"
            ],
            "mean_topology_gain_drop_without_topology": aggregate[
                "mean_topology_gain_drop_without_topology"
            ],
        },
    }
    minimum_improved = max(1, int(np.ceil(sequence_count * 0.6)))
    errors = []
    if not aggregate["mean_distance_improvement"] > 0.0:
        errors.append("controller population did not move closer on average")
    if not aggregate["improved_sequence_count"] >= minimum_improved:
        errors.append("controller did not improve enough generated sequences")
    if not aggregate["mean_memory_distance_improvement"] > 0.0:
        errors.append("controller memory trajectory did not move closer on average")
    if not aggregate["mean_controller_memory_mass"] > aggregate["mean_fixed_weak_memory_mass"]:
        errors.append("controller population did not recover more memory mass")
    if not aggregate["mean_controller_topology"] > aggregate["mean_fixed_weak_topology"]:
        errors.append("controller population did not recover more topology mass")
    if not aggregate["mean_controller_latent_norm"] > 0.0:
        errors.append("controller latent state did not move across population")
    if not aggregate["mean_controller_config_delta"] > 0.0:
        errors.append("controller did not produce gain movement across population")
    if not aggregate["mean_controller_latent_norm"] > aggregate["mean_pressureless_latent_norm"]:
        errors.append("pressure access did not increase controller latent movement")
    if not aggregate["mean_attention_gain_drop_without_attention"] > 0.0:
        errors.append("attention-capacity ablation did not reduce external attention gain on average")
    if not aggregate["mean_learning_gain_drop_without_memory"] > 0.0:
        errors.append("memory ablation did not reduce learning gain on average")
    if not aggregate["mean_topology_gain_drop_without_topology"] > 0.0:
        errors.append("topology ablation did not reduce topology gain on average")

    return {
        "id": "cavenet_controller_population",
        "ok": not errors,
        "errors": errors,
        "aggregate": aggregate,
        "roles": roles,
        "records": records,
    }


def controller_learning_records(
    *,
    sequence_count: int,
    event_count: int,
    seed: int,
    dt: float,
) -> list[dict[str, object]]:
    records = []
    variants = (
        "fixed-weak",
        "controller-static-weak",
        "controller-learning",
        "fixed-reference",
    )
    for index in range(sequence_count):
        sequence_seed = seed + index
        sequence = random_experience_sequence(count=event_count, seed=sequence_seed)
        runs = {
            variant: build_controller_episode(variant, dt=dt, sequence=sequence)
            for variant in variants
        }
        static_distance = _trajectory_distance(
            runs["controller-static-weak"],
            runs["fixed-reference"],
        )
        learning_distance = _trajectory_distance(
            runs["controller-learning"],
            runs["fixed-reference"],
        )
        fixed_weak_distance = _trajectory_distance(
            runs["fixed-weak"],
            runs["fixed-reference"],
        )
        metrics = {
            variant: _controller_metrics(episode)
            for variant, episode in runs.items()
        }
        records.append(
            {
                "sequence_id": f"Q{index}",
                "seed": sequence_seed,
                "fixed_weak_distance": fixed_weak_distance,
                "static_distance": static_distance,
                "learning_distance": learning_distance,
                "distance_improvement_over_static": (
                    static_distance["combined_distance"]
                    - learning_distance["combined_distance"]
                ),
                "distance_improvement_over_fixed_weak": (
                    fixed_weak_distance["combined_distance"]
                    - learning_distance["combined_distance"]
                ),
                "memory_distance_improvement_over_static": (
                    static_distance["memory_distance"]
                    - learning_distance["memory_distance"]
                ),
                "metrics": metrics,
            }
        )
    return records


def check_cavenet_controller_learning(
    *,
    sequence_count: int,
    event_count: int,
    seed: int,
    dt: float,
) -> dict[str, object]:
    records = controller_learning_records(
        sequence_count=sequence_count,
        event_count=event_count,
        seed=seed,
        dt=dt,
    )
    improved_over_static = [
        record
        for record in records
        if float(record["distance_improvement_over_static"]) > 0.0
    ]
    aggregate = {
        "sequence_count": sequence_count,
        "event_count": event_count,
        "seed": seed,
        "improved_over_static_count": len(improved_over_static),
        "mean_static_distance": _mean_record_distance(
            records,
            "static_distance",
            "combined_distance",
        ),
        "mean_learning_distance": _mean_record_distance(
            records,
            "learning_distance",
            "combined_distance",
        ),
        "mean_fixed_weak_distance": _mean_record_distance(
            records,
            "fixed_weak_distance",
            "combined_distance",
        ),
        "mean_distance_improvement_over_static": _mean(
            records,
            "distance_improvement_over_static",
        ),
        "mean_distance_improvement_over_fixed_weak": _mean(
            records,
            "distance_improvement_over_fixed_weak",
        ),
        "mean_memory_distance_improvement_over_static": _mean(
            records,
            "memory_distance_improvement_over_static",
        ),
        "mean_static_memory_mass": _mean_variant_metric(
            records,
            "controller-static-weak",
            "memory_mass",
        ),
        "mean_learning_memory_mass": _mean_variant_metric(
            records,
            "controller-learning",
            "memory_mass",
        ),
        "mean_static_topology": _mean_variant_metric(
            records,
            "controller-static-weak",
            "topology_mass",
        ),
        "mean_learning_topology": _mean_variant_metric(
            records,
            "controller-learning",
            "topology_mass",
        ),
        "mean_static_config_delta": _mean_variant_metric(
            records,
            "controller-static-weak",
            "config_delta",
        ),
        "mean_learning_config_delta": _mean_variant_metric(
            records,
            "controller-learning",
            "config_delta",
        ),
        "mean_learning_readout_delta_norm": _mean_variant_metric(
            records,
            "controller-learning",
            "controller_output_weight_delta_norm",
        ),
        "mean_learning_readout_updates": _mean_variant_metric(
            records,
            "controller-learning",
            "controller_readout_updates",
        ),
    }
    roles = {
        "learned_recovery": {
            "mean_static_distance": aggregate["mean_static_distance"],
            "mean_learning_distance": aggregate["mean_learning_distance"],
            "mean_distance_improvement_over_static": aggregate[
                "mean_distance_improvement_over_static"
            ],
            "mean_distance_improvement_over_fixed_weak": aggregate[
                "mean_distance_improvement_over_fixed_weak"
            ],
            "improved_over_static_count": aggregate["improved_over_static_count"],
            "sequence_count": sequence_count,
        },
        "functional_recovery": {
            "mean_static_memory_mass": aggregate["mean_static_memory_mass"],
            "mean_learning_memory_mass": aggregate["mean_learning_memory_mass"],
            "mean_static_topology": aggregate["mean_static_topology"],
            "mean_learning_topology": aggregate["mean_learning_topology"],
            "mean_memory_distance_improvement_over_static": aggregate[
                "mean_memory_distance_improvement_over_static"
            ],
        },
        "readout_learning": {
            "mean_static_config_delta": aggregate["mean_static_config_delta"],
            "mean_learning_config_delta": aggregate["mean_learning_config_delta"],
            "mean_learning_readout_delta_norm": aggregate[
                "mean_learning_readout_delta_norm"
            ],
            "mean_learning_readout_updates": aggregate[
                "mean_learning_readout_updates"
            ],
        },
    }
    minimum_improved = max(1, int(np.ceil(sequence_count * 0.6)))
    errors = []
    if not aggregate["mean_distance_improvement_over_static"] > 0.0:
        errors.append("plastic controller did not improve over static controller")
    if not aggregate["improved_over_static_count"] >= minimum_improved:
        errors.append("plastic controller did not improve enough generated sequences")
    if not aggregate["mean_distance_improvement_over_fixed_weak"] > 0.0:
        errors.append("plastic controller did not improve over fixed weak CaveNet")
    if not aggregate["mean_memory_distance_improvement_over_static"] > 0.0:
        errors.append("plastic controller memory trajectory did not improve")
    if not aggregate["mean_learning_memory_mass"] > aggregate["mean_static_memory_mass"]:
        errors.append("plastic controller did not recover more memory mass")
    if not aggregate["mean_learning_topology"] > aggregate["mean_static_topology"]:
        errors.append("plastic controller did not recover more topology")
    if not aggregate["mean_learning_config_delta"] > aggregate["mean_static_config_delta"]:
        errors.append("plastic controller did not produce more gain movement")
    if not aggregate["mean_learning_readout_delta_norm"] > 0.0:
        errors.append("plastic controller readout weights did not move")
    if not aggregate["mean_learning_readout_updates"] > 0.0:
        errors.append("plastic controller did not record readout updates")

    return {
        "id": "cavenet_controller_learning",
        "ok": not errors,
        "errors": errors,
        "aggregate": aggregate,
        "roles": roles,
        "records": records,
    }


def _controller_access_for_variant(variant: str) -> CaveNetControllerAccess:
    if variant == "controller-full":
        return CaveNetControllerAccess()
    if variant == "no-pressure-input":
        return CaveNetControllerAccess(pressure=False)
    if variant == "no-workspace-input":
        return CaveNetControllerAccess(workspace=False)
    if variant == "no-memory-input":
        return CaveNetControllerAccess(memory=False)
    if variant == "no-attention-input":
        return CaveNetControllerAccess(attention=False)
    if variant == "no-topology-input":
        return CaveNetControllerAccess(topology=False)
    raise ValueError(f"unsupported controller variant: {variant}")


def _weak_readout_controller(*, plastic: bool) -> CaveNetController:
    template = CaveNetController(base_config=_weak_config())
    output_weights = 0.18 * template.output_weights
    return CaveNetController(
        base_config=_weak_config(),
        output_weights=output_weights,
        readout_plasticity=plastic,
        readout_learning_rate=0.04 if plastic else 0.0,
        readout_plasticity_mask=_plastic_readout_mask(),
    )


def _plastic_readout_mask() -> np.ndarray:
    mask = np.zeros((7, 5), dtype=float)
    mask[0, :] = 1.0
    mask[4, :] = 1.0
    mask[5, :] = 1.0
    mask[6, :] = 1.0
    return mask


def _run_cavenet(
    *,
    source_name: str,
    config: CaveNetConfig,
    controller: CaveNetController | None,
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
        adaptation_policy=CaveNetAdaptationPolicy(enabled=False),
        controller=controller,
    )
    return CaveNetProducer(cavenet, name=source_name).run(dt=dt)


def _controller_metrics(episode: Episode) -> dict[str, float]:
    metrics = _pressure_metrics(episode)
    controller = episode.metadata.get("cavenet_controller") or {}
    latent = np.asarray(controller.get("latent", []), dtype=float)
    metrics["controller_latent_norm"] = _norm(latent)
    metrics["controller_step_count"] = float(controller.get("step_count", 0.0))
    metrics["controller_output_weight_norm"] = float(
        controller.get("output_weight_norm", 0.0)
    )
    metrics["controller_output_weight_delta_norm"] = float(
        controller.get("output_weight_delta_norm", 0.0)
    )
    metrics["controller_readout_updates"] = float(
        controller.get("readout_updates", 0.0)
    )
    history = episode.metadata.get("cavenet_config_history") or ()
    observations = [
        item.get("controller_observation", {})
        for item in history
        if isinstance(item, dict)
    ]
    if observations:
        metrics["mean_controller_attention_capacity"] = float(
            np.mean(
                [
                    float(obs.get("attention_capacity", obs.get("attention", 0.0)))
                    for obs in observations
                ]
            )
        )
        metrics["mean_controller_external_attention"] = float(
            np.mean(
                [
                    float(obs.get("external_attention", 0.0))
                    for obs in observations
                ]
            )
        )
        metrics["mean_controller_internal_expectation_attention"] = float(
            np.mean(
                [
                    float(obs.get("internal_expectation_attention", 0.0))
                    for obs in observations
                ]
            )
        )
    else:
        metrics["mean_controller_attention_capacity"] = 0.0
        metrics["mean_controller_external_attention"] = 0.0
        metrics["mean_controller_internal_expectation_attention"] = 0.0
    return metrics


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


def _norm(value) -> float:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.linalg.norm(array.ravel()) / np.sqrt(array.size))


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


def _mean_variant_metric(
    records: list[dict[str, object]],
    variant: str,
    metric_key: str,
) -> float:
    if not records:
        return 0.0
    return float(
        np.mean(
            [
                float(record["metrics"][variant][metric_key])  # type: ignore[index]
                for record in records
            ]
        )
    )


def _mean_variant_delta(
    records: list[dict[str, object]],
    first_variant: str,
    second_variant: str,
    metric_key: str,
) -> float:
    if not records:
        return 0.0
    return float(
        np.mean(
            [
                (
                    float(record["metrics"][first_variant][metric_key])  # type: ignore[index]
                    - float(record["metrics"][second_variant][metric_key])  # type: ignore[index]
                )
                for record in records
            ]
        )
    )
