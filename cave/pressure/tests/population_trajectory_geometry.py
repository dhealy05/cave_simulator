from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from cave.commitments.attention import (
    external_only_attention_profile,
    internal_only_attention_profile,
    zero_attention_profile,
)
from cave.commitments.learning import ImportanceWeightedLearningRule
from cave.demonstrations.examples import random_experience_sequence
from cave.demonstrations.reports.cave_matrices import (
    MatrixVariant,
    ZeroPredictor,
    initial_condition_subjects,
    initial_conditions_base_params,
    run_matrix_cell,
)
from cave.demonstrations.subjects import (
    active_context_embedding,
    attended_input_trajectory_embedding,
    memory_trajectory_embedding,
    state_effect_embedding,
    subjective_trajectory_embedding,
)
from cave.observation.episodes import Episode
from cave.observation.projections import encode_value
from cave.observation.views import default_views
from cave.observation.population_metrics import population_geometry_metrics
from cave.presentation.reports.specs import (
    MatrixReportSpec,
    MatrixRunRecord,
    ProducerReportSpec,
    ReportExtraAsset,
    ReportSection,
)


@dataclass(frozen=True)
class PopulationTrajectorySweepRecord:
    seed: int
    ok: bool
    baseline_subjective_accuracy: float
    baseline_subjective_chance: float
    baseline_subjective_lift: float
    baseline_subjective_margin: float
    baseline_subjective_permutation_p: float
    zero_attention_subjective_accuracy: float
    zero_attention_actual_accuracy: float
    external_only_subjective_accuracy: float
    external_only_actual_accuracy: float
    internal_only_subjective_accuracy: float
    internal_only_actual_accuracy: float
    no_memory_observed_memory_accuracy: float
    no_memory_actual_accuracy: float
    zero_predictor_subjective_accuracy: float
    errors: tuple[str, ...]


def population_trajectory_geometry_report_spec(
    *,
    treatment_count: int = 4,
    start_count: int = 16,
    event_count: int = 5,
    seed: int = 701,
    dt: float = 0.2,
    end: float = 3.0,
    samples: int = 48,
    permutations: int = 199,
    min_treatment_accuracy_lift: float = 0.20,
    max_permutation_p: float = 0.05,
) -> MatrixReportSpec:
    def build_records():
        return controlled_population_records(
            treatment_count=treatment_count,
            start_count=start_count,
            event_count=event_count,
            seed=seed,
            dt=dt,
            end=end,
        )

    def check(records):
        return check_population_trajectory_geometry(
            records,
            samples=samples,
            permutations=permutations,
            seed=seed,
            min_treatment_accuracy_lift=min_treatment_accuracy_lift,
            max_permutation_p=max_permutation_p,
        )

    return MatrixReportSpec(
        id="population-trajectory-geometry",
        title="Population Trajectory Geometry: Treatments x Starts x Controls",
        run_factory=build_records,
        description=(
            "Crosses generated treatment sequences with matched initial "
            "subjective states and mechanism controls, then tests whether "
            "treatment identity is recoverable from subjective trajectory "
            "geometry rather than only visible in a qualitative population plot."
        ),
        sections=(
            ReportSection(
                title="Design",
                body=(
                    "Each treatment is a generated experience sequence. Each "
                    "start condition is an initial memory state. Conditions "
                    "include baseline, zero capacity, external-only attention, "
                    "internal-only attention, no memory update, and zero "
                    "prediction. This makes sensory access and expectation "
                    "readout separate control contrasts."
                ),
            ),
            ReportSection(
                title="Primary Evidence",
                body=(
                    "The primary check uses the subjective-trajectory embedding. "
                    "It compares same-treatment and different-treatment distances, "
                    "decodes treatment labels while holding out start conditions, "
                    "and evaluates the treatment margin against a permutation null "
                    "that shuffles treatment labels within matched starts. The "
                    "same metrics are also reported per condition and embedding."
                ),
            ),
            ReportSection(
                title="Interpretation Boundary",
                body=(
                    "This report does not claim open-ended emergence. It tests a "
                    "population prerequisite for Paper 2: matched subjects should "
                    "follow measurably different internal paths under different "
                    "treatments, and that structure should be recoverable by a "
                    "simple out-of-start classifier."
                ),
            ),
        ),
        checks=(check,),
        samples=samples,
        cluster_thresholds={
            "state_effect": 0.02,
            "observed_memory": 0.02,
            "subjective_trajectory": 0.02,
            "active_context": 0.02,
        },
        config={
            "matrix": "population_trajectory_geometry",
            "treatment_count": treatment_count,
            "start_count": start_count,
            "event_count": event_count,
            "seed": seed,
            "dt": dt,
            "end": end,
            "samples": samples,
            "permutations": permutations,
            "min_treatment_accuracy_lift": min_treatment_accuracy_lift,
            "max_permutation_p": max_permutation_p,
        },
    )


def population_trajectory_geometry_sweep_report_spec(
    *,
    seed_start: int = 701,
    seed_count: int = 5,
    treatment_count: int = 4,
    start_count: int = 12,
    event_count: int = 5,
    dt: float = 0.2,
    fps: int = 8,
    end: float = 3.0,
    samples: int = 32,
    permutations: int = 49,
    min_pass_rate: float = 0.8,
    min_treatment_accuracy_lift: float = 0.20,
    max_permutation_p: float = 0.08,
    include_assets: bool = True,
) -> ProducerReportSpec:
    def build_episode() -> Episode:
        records = controlled_population_records(
            treatment_count=treatment_count,
            start_count=start_count,
            event_count=event_count,
            seed=seed_start,
            dt=dt,
            end=end,
        )
        for record in records:
            if record.variant_id == "baseline":
                return record.run.episode
        return records[0].run.episode

    extra_assets = ()
    if include_assets:
        extra_assets = (
            ReportExtraAsset(
                id="population_trajectory_geometry_sweep_metrics",
                title="Population Trajectory Geometry Sweep Metrics JSON",
                filename="population_trajectory_geometry_sweep_metrics.json",
                writer=lambda episode, output: write_population_trajectory_geometry_sweep_metrics_json(
                    output,
                    seed_start=seed_start,
                    seed_count=seed_count,
                    treatment_count=treatment_count,
                    start_count=start_count,
                    event_count=event_count,
                    dt=dt,
                    end=end,
                    samples=samples,
                    permutations=permutations,
                    min_pass_rate=min_pass_rate,
                    min_treatment_accuracy_lift=min_treatment_accuracy_lift,
                    max_permutation_p=max_permutation_p,
                ),
            ),
        )

    return ProducerReportSpec(
        id="population-trajectory-geometry-sweep",
        title="Population Trajectory Geometry: Seed Sweep",
        episode_factory=build_episode,
        input_summary=(
            f"{seed_count} seeds x {treatment_count} treatments x "
            f"{start_count} starts x 6 conditions"
        ),
        description=(
            "Repeats the controlled treatment-geometry report across generated "
            "seeds. The sweep asks whether treatment decoding, permutation-null "
            "separability, zero-capacity collapse, external/internal attention "
            "contrasts, and no-memory observed-memory collapse are stable "
            "rather than one favorable generated matrix."
        ),
        views=default_views(),
        view_assets=(),
        extra_assets=extra_assets,
        checks=(
            lambda episode: check_population_trajectory_geometry_sweep(
                seed_start=seed_start,
                seed_count=seed_count,
                treatment_count=treatment_count,
                start_count=start_count,
                event_count=event_count,
                dt=dt,
                end=end,
                samples=samples,
                permutations=permutations,
                min_pass_rate=min_pass_rate,
                min_treatment_accuracy_lift=min_treatment_accuracy_lift,
                max_permutation_p=max_permutation_p,
            ),
        ),
        frame_time=2.0,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "cave",
            "scenario": "population_trajectory_geometry_sweep",
            "seed_start": seed_start,
            "seed_count": seed_count,
            "treatment_count": treatment_count,
            "start_count": start_count,
            "event_count": event_count,
            "dt": dt,
            "end": end,
            "samples": samples,
            "permutations": permutations,
            "min_pass_rate": min_pass_rate,
            "min_treatment_accuracy_lift": min_treatment_accuracy_lift,
            "max_permutation_p": max_permutation_p,
        },
        sections=(
            ReportSection(
                title="Question",
                body=(
                    "Does the controlled treatment-geometry result survive "
                    "variation in the generated treatment set?"
                ),
                asset_ids=("population_trajectory_geometry_sweep_metrics",),
            ),
            ReportSection(
                title="Interpretation",
                body=(
                    "The sweep reports pass counts and medians. Baseline should "
                    "decode treatment above chance and beat the permutation null. "
                    "Zero capacity should collapse treatment access. External-only "
                    "attention should preserve current actual-input decoding, while "
                    "internal-only attention should collapse current actual-input "
                    "decoding. No-memory should collapse observed-memory treatment "
                    "decoding while leaving current actual-input decoding intact."
                ),
            ),
        ),
    )


def check_population_trajectory_geometry(
    records,
    *,
    samples: int = 48,
    permutations: int = 199,
    seed: int = 0,
    min_treatment_accuracy_lift: float = 0.20,
    max_permutation_p: float = 0.05,
) -> dict[str, object]:
    embeddings = {
        "state_effect": lambda record: state_effect_embedding(record.run, samples=samples),
        "observed_memory": lambda record: memory_trajectory_embedding(record.run, samples=samples),
        "subjective_trajectory": lambda record: subjective_trajectory_embedding(record.run, samples=samples),
        "actual_input": lambda record: attended_input_trajectory_embedding(record.run, samples=samples),
        "active_context": lambda record: active_context_embedding(record.run, samples=samples),
    }
    metrics = population_geometry_metrics(
        records,
        embeddings,
        label_fn=_record_factor_id,
        permutations=permutations,
        seed=seed,
    )
    by_condition = _condition_geometry_metrics(
        records,
        embeddings=embeddings,
        samples=samples,
        permutations=permutations,
        seed=seed,
    )
    baseline = by_condition["baseline"]
    subjective = baseline["embedding_metrics"]["subjective_trajectory"]
    treatment = subjective["treatment"]
    decoding = subjective["treatment_decoding"]
    null = subjective["permutation_null"]
    zero_attention = by_condition.get("zero-attention")
    external_only = by_condition.get("external-only-attention")
    internal_only = by_condition.get("internal-only-attention")
    no_memory = by_condition.get("no-memory")

    errors = []
    if metrics["treatment_count"] < 2:
        errors.append("population trajectory geometry requires at least two treatments")
    if metrics["start_count"] < 2:
        errors.append("population trajectory geometry requires at least two starts")
    if treatment["margin"] <= 0.0:
        errors.append("subjective trajectories are not farther between treatments than within treatments")
    if decoding["lift"] < min_treatment_accuracy_lift:
        errors.append(
            "subjective trajectory treatment decoding did not clear the configured lift threshold"
        )
    if null["p_value"] > max_permutation_p:
        errors.append("subjective trajectory treatment margin did not beat the permutation null")
    if zero_attention is None:
        errors.append("missing zero-attention control condition")
    else:
        zero_actual = zero_attention["embedding_metrics"]["actual_input"]["treatment_decoding"]
        if zero_actual["accuracy"] > zero_actual["chance"]:
            errors.append("zero-attention actual input preserved treatment decoding above chance")
    if external_only is None:
        errors.append("missing external-only attention control condition")
    else:
        external_actual = external_only["embedding_metrics"]["actual_input"]["treatment_decoding"]
        if external_actual["lift"] < min_treatment_accuracy_lift:
            errors.append("external-only attention did not preserve actual-input treatment decoding")
    if internal_only is None:
        errors.append("missing internal-only attention control condition")
    else:
        internal_actual = internal_only["embedding_metrics"]["actual_input"]["treatment_decoding"]
        if internal_actual["accuracy"] > internal_actual["chance"]:
            errors.append("internal-only attention preserved actual-input treatment decoding above chance")
    if no_memory is None:
        errors.append("missing no-memory control condition")

    return {
        "id": "population_trajectory_geometry",
        "ok": not errors,
        "errors": errors,
        "metrics": {
            "overall": _compact_metrics(metrics),
            "by_condition": {
                condition: _compact_metrics(condition_metrics)
                for condition, condition_metrics in by_condition.items()
            },
        },
        "roles": {
            "treatment_separability": {
                "subjective_within_treatment_distance": treatment["within_mean"],
                "subjective_between_treatment_distance": treatment["between_mean"],
                "subjective_treatment_margin": treatment["margin"],
                "subjective_treatment_ratio": treatment["ratio"],
                "subjective_permutation_p": null["p_value"],
            },
            "matched_start_deformation": {
                "matched_start_treatment_distance": subjective[
                    "matched_start_treatment_distance"
                ]["mean"],
                "same_treatment_start_distance": subjective[
                    "same_treatment_start_distance"
                ]["mean"],
            },
            "treatment_decoding": {
                "accuracy": decoding["accuracy"],
                "chance": decoding["chance"],
                "lift": decoding["lift"],
                "held_out_factor": decoding["held_out_factor"],
            },
            "control_contrasts": _control_contrast_roles(by_condition),
        },
    }


@lru_cache(maxsize=8)
def check_population_trajectory_geometry_sweep(
    *,
    seed_start: int = 701,
    seed_count: int = 5,
    treatment_count: int = 4,
    start_count: int = 12,
    event_count: int = 5,
    dt: float = 0.2,
    end: float = 3.0,
    samples: int = 32,
    permutations: int = 49,
    min_pass_rate: float = 0.8,
    min_treatment_accuracy_lift: float = 0.20,
    max_permutation_p: float = 0.08,
) -> dict[str, object]:
    records = population_trajectory_geometry_sweep_records(
        seed_start=seed_start,
        seed_count=seed_count,
        treatment_count=treatment_count,
        start_count=start_count,
        event_count=event_count,
        dt=dt,
        end=end,
        samples=samples,
        permutations=permutations,
        min_treatment_accuracy_lift=min_treatment_accuracy_lift,
        max_permutation_p=max_permutation_p,
    )
    aggregate = _sweep_aggregate(
        records,
        min_pass_rate=min_pass_rate,
        min_treatment_accuracy_lift=min_treatment_accuracy_lift,
        max_permutation_p=max_permutation_p,
    )
    roles = {
        "robust_treatment_recovery": {
            "seed_count": seed_count,
            "required_pass_count": aggregate["required_pass_count"],
            "strict_pass_count": aggregate["strict_pass_count"],
            "strict_pass_rate": aggregate["strict_pass_rate"],
            "baseline_decoding_pass_count": aggregate[
                "baseline_decoding_pass_count"
            ],
            "baseline_permutation_pass_count": aggregate[
                "baseline_permutation_pass_count"
            ],
            "median_baseline_subjective_accuracy": aggregate[
                "median_baseline_subjective_accuracy"
            ],
            "median_baseline_subjective_lift": aggregate[
                "median_baseline_subjective_lift"
            ],
            "median_baseline_subjective_margin": aggregate[
                "median_baseline_subjective_margin"
            ],
            "median_baseline_subjective_permutation_p": aggregate[
                "median_baseline_subjective_permutation_p"
            ],
        },
        "control_collapse": {
            "zero_attention_subjective_collapse_count": aggregate[
                "zero_attention_subjective_collapse_count"
            ],
            "zero_attention_actual_collapse_count": aggregate[
                "zero_attention_actual_collapse_count"
            ],
            "external_only_actual_preserved_count": aggregate[
                "external_only_actual_preserved_count"
            ],
            "internal_only_actual_collapse_count": aggregate[
                "internal_only_actual_collapse_count"
            ],
            "internal_only_subjective_collapse_count": aggregate[
                "internal_only_subjective_collapse_count"
            ],
            "no_memory_observed_memory_collapse_count": aggregate[
                "no_memory_observed_memory_collapse_count"
            ],
            "no_memory_actual_preserved_count": aggregate[
                "no_memory_actual_preserved_count"
            ],
            "zero_predictor_subjective_preserved_count": aggregate[
                "zero_predictor_subjective_preserved_count"
            ],
            "median_zero_attention_subjective_accuracy": aggregate[
                "median_zero_attention_subjective_accuracy"
            ],
            "median_external_only_subjective_accuracy": aggregate[
                "median_external_only_subjective_accuracy"
            ],
            "median_external_only_actual_accuracy": aggregate[
                "median_external_only_actual_accuracy"
            ],
            "median_internal_only_subjective_accuracy": aggregate[
                "median_internal_only_subjective_accuracy"
            ],
            "median_internal_only_actual_accuracy": aggregate[
                "median_internal_only_actual_accuracy"
            ],
            "median_no_memory_observed_memory_accuracy": aggregate[
                "median_no_memory_observed_memory_accuracy"
            ],
            "median_zero_predictor_subjective_accuracy": aggregate[
                "median_zero_predictor_subjective_accuracy"
            ],
        },
    }
    required = int(aggregate["required_pass_count"])
    errors: list[str] = []
    if aggregate["baseline_decoding_pass_count"] < required:
        errors.append("too few seeds decoded treatment above chance in baseline")
    if aggregate["baseline_permutation_pass_count"] < required:
        errors.append("too few seeds beat the baseline permutation null")
    if aggregate["zero_attention_subjective_collapse_count"] < required:
        errors.append("too few seeds collapsed zero-attention subjective decoding")
    if aggregate["zero_attention_actual_collapse_count"] < required:
        errors.append("too few seeds collapsed zero-attention actual decoding")
    if aggregate["external_only_actual_preserved_count"] < required:
        errors.append("too few seeds preserved external-only actual-input treatment decoding")
    if aggregate["internal_only_actual_collapse_count"] < required:
        errors.append("too few seeds collapsed internal-only actual-input treatment decoding")
    if aggregate["internal_only_subjective_collapse_count"] < required:
        errors.append("too few seeds collapsed internal-only subjective treatment decoding")
    if aggregate["no_memory_observed_memory_collapse_count"] < required:
        errors.append("too few seeds collapsed no-memory observed-memory decoding")
    if aggregate["no_memory_actual_preserved_count"] < required:
        errors.append("too few seeds preserved no-memory actual-input treatment decoding")
    if aggregate["zero_predictor_subjective_preserved_count"] < required:
        errors.append("too few seeds preserved zero-predictor subjective treatment decoding")

    return {
        "id": "population_trajectory_geometry_sweep",
        "ok": not errors,
        "errors": errors,
        "aggregate": aggregate,
        "roles": roles,
        "records": [record.__dict__ for record in records],
    }


def population_trajectory_geometry_sweep_records(
    *,
    seed_start: int,
    seed_count: int,
    treatment_count: int,
    start_count: int,
    event_count: int,
    dt: float,
    end: float,
    samples: int,
    permutations: int,
    min_treatment_accuracy_lift: float,
    max_permutation_p: float,
) -> tuple[PopulationTrajectorySweepRecord, ...]:
    records = []
    for offset in range(seed_count):
        seed = seed_start + offset
        spec = population_trajectory_geometry_report_spec(
            treatment_count=treatment_count,
            start_count=start_count,
            event_count=event_count,
            seed=seed,
            dt=dt,
            end=end,
            samples=samples,
            permutations=permutations,
            min_treatment_accuracy_lift=min_treatment_accuracy_lift,
            max_permutation_p=max_permutation_p,
        )
        check = check_population_trajectory_geometry(
            list(spec.run_factory()),
            samples=samples,
            permutations=permutations,
            seed=seed,
            min_treatment_accuracy_lift=min_treatment_accuracy_lift,
            max_permutation_p=max_permutation_p,
        )
        roles = check["roles"]
        recovery = roles["treatment_decoding"]
        separability = roles["treatment_separability"]
        controls = roles["control_contrasts"]
        baseline = controls["baseline"]
        zero_attention = controls["zero-attention"]
        external_only = controls["external-only-attention"]
        internal_only = controls["internal-only-attention"]
        no_memory = controls["no-memory"]
        zero_predictor = controls["zero-predictor"]
        records.append(
            PopulationTrajectorySweepRecord(
                seed=seed,
                ok=bool(check["ok"]),
                baseline_subjective_accuracy=float(recovery["accuracy"]),
                baseline_subjective_chance=float(recovery["chance"]),
                baseline_subjective_lift=float(recovery["lift"]),
                baseline_subjective_margin=float(
                    separability["subjective_treatment_margin"]
                ),
                baseline_subjective_permutation_p=float(
                    separability["subjective_permutation_p"]
                ),
                zero_attention_subjective_accuracy=float(
                    zero_attention["subjective_decoding_accuracy"]
                ),
                zero_attention_actual_accuracy=float(
                    zero_attention["actual_decoding_accuracy"]
                ),
                external_only_subjective_accuracy=float(
                    external_only["subjective_decoding_accuracy"]
                ),
                external_only_actual_accuracy=float(
                    external_only["actual_decoding_accuracy"]
                ),
                internal_only_subjective_accuracy=float(
                    internal_only["subjective_decoding_accuracy"]
                ),
                internal_only_actual_accuracy=float(
                    internal_only["actual_decoding_accuracy"]
                ),
                no_memory_observed_memory_accuracy=float(
                    no_memory["observed_memory_decoding_accuracy"]
                ),
                no_memory_actual_accuracy=float(no_memory["actual_decoding_accuracy"]),
                zero_predictor_subjective_accuracy=float(
                    zero_predictor["subjective_decoding_accuracy"]
                ),
                errors=tuple(str(error) for error in check.get("errors", ())),
            )
        )
    return tuple(records)


def write_population_trajectory_geometry_sweep_metrics_json(
    output: Path,
    *,
    seed_start: int = 701,
    seed_count: int = 5,
    treatment_count: int = 4,
    start_count: int = 12,
    event_count: int = 5,
    dt: float = 0.2,
    end: float = 3.0,
    samples: int = 32,
    permutations: int = 49,
    min_pass_rate: float = 0.8,
    min_treatment_accuracy_lift: float = 0.20,
    max_permutation_p: float = 0.08,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            encode_value(
                check_population_trajectory_geometry_sweep(
                    seed_start=seed_start,
                    seed_count=seed_count,
                    treatment_count=treatment_count,
                    start_count=start_count,
                    event_count=event_count,
                    dt=dt,
                    end=end,
                    samples=samples,
                    permutations=permutations,
                    min_pass_rate=min_pass_rate,
                    min_treatment_accuracy_lift=min_treatment_accuracy_lift,
                    max_permutation_p=max_permutation_p,
                )
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def controlled_population_records(
    *,
    treatment_count: int,
    start_count: int,
    event_count: int,
    seed: int,
    dt: float,
    end: float,
):
    if treatment_count <= 0:
        raise ValueError("treatment_count must be positive")
    if start_count <= 0:
        raise ValueError("start_count must be positive")
    subjects = initial_condition_subjects(condition_count=start_count)
    variants = controlled_population_variants()
    records = []
    for treatment_index in range(treatment_count):
        sequence_id = f"Q{treatment_index:03d}"
        sequence = random_experience_sequence(
            count=event_count,
            seed=seed + treatment_index,
        )
        for subject in subjects:
            for variant in variants:
                records.append(
                    run_matrix_cell(
                        sequence_id=sequence_id,
                        sequence=sequence,
                        subject=subject,
                        variant=variant,
                        dt=dt,
                        end=end,
                    )
                )
    return records


def controlled_population_variants() -> tuple[MatrixVariant, ...]:
    base = initial_conditions_base_params()
    no_memory_params = replace(
        base,
        learning_rule=ImportanceWeightedLearningRule(min_rate=0.0, max_rate=0.0),
    )
    return (
        MatrixVariant("baseline", "Baseline", base, comparison_role="baseline"),
        MatrixVariant(
            "zero-attention",
            "Zero attention",
            replace(base, attention=zero_attention_profile()),
            comparison_role="negative_control",
            control_for="baseline",
        ),
        MatrixVariant(
            "external-only-attention",
            "External-only attention",
            replace(base, attention=external_only_attention_profile()),
            comparison_role="attention_control",
            control_for="baseline",
        ),
        MatrixVariant(
            "internal-only-attention",
            "Internal-only attention",
            replace(base, attention=internal_only_attention_profile()),
            comparison_role="attention_control",
            control_for="baseline",
        ),
        MatrixVariant(
            "no-memory",
            "No memory update",
            no_memory_params,
            comparison_role="capacity_control",
            control_for="baseline",
        ),
        MatrixVariant(
            "zero-predictor",
            "Zero predictor",
            base,
            predictor_factory=ZeroPredictor,
            comparison_role="mechanism_control",
            control_for="baseline",
        ),
    )


def _record_factor_id(record: MatrixRunRecord, factor: str) -> str | None:
    return record.factor_id(factor)


def _condition_geometry_metrics(
    records,
    *,
    embeddings,
    samples: int,
    permutations: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    del samples
    by_condition = {}
    for record in records:
        by_condition.setdefault(record.variant_id, []).append(record)
    return {
        condition: population_geometry_metrics(
            condition_records,
            embeddings,
            label_fn=_record_factor_id,
            permutations=permutations,
            seed=seed,
        )
        for condition, condition_records in sorted(by_condition.items())
    }


def _control_contrast_roles(by_condition: dict[str, dict[str, object]]) -> dict[str, object]:
    roles = {}
    for condition, metrics in sorted(by_condition.items()):
        embedding_metrics = metrics["embedding_metrics"]
        subjective = embedding_metrics["subjective_trajectory"]
        actual = embedding_metrics["actual_input"]
        observed = embedding_metrics["observed_memory"]
        roles[condition] = {
            "subjective_treatment_margin": subjective["treatment"]["margin"],
            "subjective_decoding_accuracy": subjective["treatment_decoding"]["accuracy"],
            "subjective_decoding_lift": subjective["treatment_decoding"]["lift"],
            "subjective_permutation_p": subjective["permutation_null"]["p_value"],
            "actual_decoding_accuracy": actual["treatment_decoding"]["accuracy"],
            "actual_decoding_lift": actual["treatment_decoding"]["lift"],
            "observed_memory_decoding_accuracy": observed["treatment_decoding"]["accuracy"],
            "observed_memory_decoding_lift": observed["treatment_decoding"]["lift"],
        }
    baseline = roles.get("baseline")
    if baseline is not None:
        for condition, values in roles.items():
            if condition == "baseline":
                continue
            values["subjective_lift_delta_from_baseline"] = (
                baseline["subjective_decoding_lift"] - values["subjective_decoding_lift"]
            )
            values["actual_lift_delta_from_baseline"] = (
                baseline["actual_decoding_lift"] - values["actual_decoding_lift"]
            )
            values["observed_memory_lift_delta_from_baseline"] = (
                baseline["observed_memory_decoding_lift"]
                - values["observed_memory_decoding_lift"]
            )
    return roles


def _compact_metrics(metrics: dict[str, object]) -> dict[str, object]:
    embedding_metrics = metrics["embedding_metrics"]
    compact = {
        "run_count": metrics["run_count"],
        "treatment_count": metrics["treatment_count"],
        "start_count": metrics["start_count"],
        "permutations": metrics["permutations"],
    }
    for name in (
        "state_effect",
        "observed_memory",
        "subjective_trajectory",
        "actual_input",
        "active_context",
    ):
        values = embedding_metrics[name]
        compact[name] = {
            "treatment_margin": values["treatment"]["margin"],
            "treatment_ratio": values["treatment"]["ratio"],
            "treatment_decoding_accuracy": values["treatment_decoding"]["accuracy"],
            "treatment_decoding_lift": values["treatment_decoding"]["lift"],
            "permutation_p": values["permutation_null"]["p_value"],
            "matched_start_treatment_distance": values[
                "matched_start_treatment_distance"
            ]["mean"],
            "same_treatment_start_distance": values[
                "same_treatment_start_distance"
            ]["mean"],
        }
    return compact


def _sweep_aggregate(
    records: tuple[PopulationTrajectorySweepRecord, ...],
    *,
    min_pass_rate: float,
    min_treatment_accuracy_lift: float,
    max_permutation_p: float,
) -> dict[str, object]:
    seed_count = len(records)
    required = int(np.ceil(seed_count * min_pass_rate))

    def median(field: str) -> float:
        return float(np.median([getattr(record, field) for record in records]))

    baseline_decoding_pass_count = sum(
        record.baseline_subjective_lift >= min_treatment_accuracy_lift
        for record in records
    )
    baseline_permutation_pass_count = sum(
        record.baseline_subjective_permutation_p <= max_permutation_p
        for record in records
    )
    zero_attention_subjective_collapse_count = sum(
        record.zero_attention_subjective_accuracy <= record.baseline_subjective_chance
        for record in records
    )
    zero_attention_actual_collapse_count = sum(
        record.zero_attention_actual_accuracy <= record.baseline_subjective_chance
        for record in records
    )
    external_only_actual_preserved_count = sum(
        (record.external_only_actual_accuracy - record.baseline_subjective_chance)
        >= min_treatment_accuracy_lift
        for record in records
    )
    internal_only_actual_collapse_count = sum(
        record.internal_only_actual_accuracy <= record.baseline_subjective_chance
        for record in records
    )
    internal_only_subjective_collapse_count = sum(
        record.internal_only_subjective_accuracy <= record.baseline_subjective_chance
        for record in records
    )
    no_memory_observed_memory_collapse_count = sum(
        record.no_memory_observed_memory_accuracy <= record.baseline_subjective_chance
        for record in records
    )
    no_memory_actual_preserved_count = sum(
        (record.no_memory_actual_accuracy - record.baseline_subjective_chance)
        >= min_treatment_accuracy_lift
        for record in records
    )
    zero_predictor_subjective_preserved_count = sum(
        (record.zero_predictor_subjective_accuracy - record.baseline_subjective_chance)
        >= min_treatment_accuracy_lift
        for record in records
    )
    strict_pass_count = sum(record.ok for record in records)
    return {
        "seed_count": seed_count,
        "required_pass_count": required,
        "strict_pass_count": strict_pass_count,
        "strict_pass_rate": float(strict_pass_count / seed_count) if seed_count else 0.0,
        "baseline_decoding_pass_count": baseline_decoding_pass_count,
        "baseline_permutation_pass_count": baseline_permutation_pass_count,
        "zero_attention_subjective_collapse_count": zero_attention_subjective_collapse_count,
        "zero_attention_actual_collapse_count": zero_attention_actual_collapse_count,
        "external_only_actual_preserved_count": external_only_actual_preserved_count,
        "internal_only_actual_collapse_count": internal_only_actual_collapse_count,
        "internal_only_subjective_collapse_count": internal_only_subjective_collapse_count,
        "no_memory_observed_memory_collapse_count": no_memory_observed_memory_collapse_count,
        "no_memory_actual_preserved_count": no_memory_actual_preserved_count,
        "zero_predictor_subjective_preserved_count": zero_predictor_subjective_preserved_count,
        "median_baseline_subjective_accuracy": median("baseline_subjective_accuracy"),
        "median_baseline_subjective_lift": median("baseline_subjective_lift"),
        "median_baseline_subjective_margin": median("baseline_subjective_margin"),
        "median_baseline_subjective_permutation_p": median(
            "baseline_subjective_permutation_p"
        ),
        "median_zero_attention_subjective_accuracy": median(
            "zero_attention_subjective_accuracy"
        ),
        "median_zero_attention_actual_accuracy": median(
            "zero_attention_actual_accuracy"
        ),
        "median_external_only_subjective_accuracy": median(
            "external_only_subjective_accuracy"
        ),
        "median_external_only_actual_accuracy": median(
            "external_only_actual_accuracy"
        ),
        "median_internal_only_subjective_accuracy": median(
            "internal_only_subjective_accuracy"
        ),
        "median_internal_only_actual_accuracy": median(
            "internal_only_actual_accuracy"
        ),
        "median_no_memory_observed_memory_accuracy": median(
            "no_memory_observed_memory_accuracy"
        ),
        "median_no_memory_actual_accuracy": median("no_memory_actual_accuracy"),
        "median_zero_predictor_subjective_accuracy": median(
            "zero_predictor_subjective_accuracy"
        ),
    }
