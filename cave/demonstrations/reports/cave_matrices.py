from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np

from cave.commitments.attention import (
    balanced_attention_profile,
    external_only_attention_profile,
    zero_attention_profile,
)
from cave.observation.episodes import episode_from_cave_states
from cave.demonstrations.examples import DEFAULT_VOCABULARY, default_model_params, random_experience_sequence
from cave.commitments.learning import ImportanceWeightedLearningRule
from cave.commitments.memory import MemoryTrace
from cave.commitments.prediction import MemoryVectorPredictor
from cave.presentation.reports.matrix import write_matrix_report
from cave.presentation.reports.specs import MatrixReportSpec, MatrixRunRecord, ReportSection
from cave.observation.population import factor_level
from cave.observation.sensing import default_sensorium
from cave.demonstrations.simulation import ExperienceModel, ModelParams
from cave.demonstrations.state import SubjectState
from cave.demonstrations.subjects import Subject, SubjectRun, embedding_distance
from cave.demonstrations.subjects import (
    state_effect_embedding,
    subjective_trajectory_embedding,
    memory_trajectory_embedding,
)
from cave.commitments.topology import SubjectiveTopologyParams, SubjectiveTopologyPrior


@dataclass(frozen=True)
class MatrixVariant:
    id: str
    label: str
    params: ModelParams
    predictor_factory: Callable[[], object] | None = None
    comparison_role: str = "treatment"
    control_for: str | None = None


class ZeroPredictor:
    def predict(self, state: SubjectState, vocabulary: list[str]):
        return np.zeros(len(vocabulary), dtype=float)

    def evaluate(self, expected_input, actual_input):
        return MemoryVectorPredictor().evaluate(expected_input, actual_input)


def subject_ablation_matrix_report_spec(
    *,
    event_count: int = 5,
    seed: int = 101,
    dt: float = 0.2,
    end: float = 3.0,
    samples: int = 48,
) -> MatrixReportSpec:
    def build_records():
        sequence = random_experience_sequence(count=event_count, seed=seed)
        subjects = subject_ablation_subjects()
        variants = subject_ablation_variants()
        records = []
        for subject in subjects:
            for variant in variants:
                records.append(
                    run_matrix_cell(
                        sequence_id="Q0",
                        sequence=sequence,
                        subject=subject,
                        variant=variant,
                        dt=dt,
                        end=end,
                    )
                )
        return records

    return MatrixReportSpec(
        id="subject-ablation",
        title="Cave Matrix: Subjects x Mechanism Variants",
        run_factory=build_records,
        description=(
            "Runs one generated experience sequence through a small population "
            "of subjects crossed with native Cave mechanism variants. This "
            "turns ablation into a population of outputs rather than a single "
            "before/after comparison."
        ),
        sections=(
            ReportSection(
                title="Matrix Shape",
                body=(
                    "This report uses one sequence, two subject initial states, "
                    "and five variants: baseline, zero attention, external "
                    "split attention, surprise-weighted learning, and zero "
                    "prediction."
                ),
            ),
            ReportSection(
                title="Reading The Dashboard",
                body=(
                    "The dashboard compares state effect, observed memory, "
                    "and subjective trajectory distances across every matrix cell. "
                    "Rows with the same subject but different variants show "
                    "mechanism effects; rows with the same variant but different "
                    "subjects show subject effects."
                ),
            ),
            ReportSection(
                title="Checks",
                body=(
                    "The checks assert expected qualitative geometry: zero "
                    "attention collapses state effect across subjects, "
                    "baseline diverges from zero attention, prior state changes "
                    "observed memory while state effect remains isolated, "
                    "and zero prediction changes "
                    "subjective trajectory geometry."
                ),
            ),
        ),
        checks=(check_subject_ablation_matrix,),
        samples=samples,
        cluster_thresholds={
            "state_effect": 1e-12,
            "observed_memory": 1e-12,
            "subjective_trajectory": 1e-12,
        },
        config={
            "matrix": "subject_ablation",
            "sequence_count": 1,
            "event_count": event_count,
            "seed": seed,
            "dt": dt,
            "end": end,
            "samples": samples,
        },
    )


def population_clusters_matrix_report_spec(
    *,
    sequence_count: int = 6,
    event_count: int = 5,
    seed: int = 101,
    dt: float = 0.2,
    end: float = 3.0,
    samples: int = 48,
) -> MatrixReportSpec:
    def build_records():
        subjects = subject_ablation_subjects()
        variants = subject_ablation_variants()
        records = []
        for sequence_index in range(sequence_count):
            sequence_id = f"Q{sequence_index:03d}"
            sequence = random_experience_sequence(
                count=event_count,
                seed=seed + sequence_index,
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

    return MatrixReportSpec(
        id="population-clusters",
        title="Cave Matrix: Population Clusters",
        run_factory=build_records,
        description=(
            "Runs multiple generated sequences through subject and mechanism "
            "variant axes, then writes cluster artifacts for each embedding. "
            "This report is for inspecting the output population rather than a "
            "single ablation pair."
        ),
        sections=(
            ReportSection(
                title="Population Shape",
                body=(
                    "The matrix crosses generated sequences with subject initial "
                    "states and Cave mechanism variants. The output is a "
                    "population of episodes, not one canonical run."
                ),
            ),
            ReportSection(
                title="Cluster Artifacts",
                body=(
                    "The `clusters/` folder records threshold clusters for "
                    "state effect, observed memory, and subjective trajectory. "
                    "Those files are the primary output for this report."
                ),
            ),
            ReportSection(
                title="Expected Structure",
                body=(
                    "Zero-attention runs should collapse in experience-effect "
                    "space across sequences and subjects. Observed-memory and "
                    "internal-prediction embeddings should retain more structure."
                ),
            ),
        ),
        checks=(check_population_clusters_matrix,),
        samples=samples,
        cluster_thresholds={
            "state_effect": 1e-12,
            "observed_memory": 0.02,
            "subjective_trajectory": 0.02,
        },
        config={
            "matrix": "population_clusters",
            "sequence_count": sequence_count,
            "event_count": event_count,
            "seed": seed,
            "dt": dt,
            "end": end,
            "samples": samples,
        },
    )


def initial_conditions_matrix_report_spec(
    *,
    condition_count: int = 16,
    treatment_count: int = 1,
    event_count: int = 5,
    seed: int = 101,
    dt: float = 0.2,
    end: float = 3.0,
    samples: int = 48,
) -> MatrixReportSpec:
    if treatment_count <= 0:
        raise ValueError("treatment_count must be positive")

    def build_records():
        subjects = initial_condition_subjects(condition_count=condition_count)
        variant = MatrixVariant(
            "baseline",
            "Baseline",
            initial_conditions_base_params(),
            comparison_role="baseline",
        )
        records = []
        for treatment_index in range(treatment_count):
            sequence_id = "Q0" if treatment_count == 1 else f"Q{treatment_index:03d}"
            sequence = random_experience_sequence(
                count=event_count,
                seed=seed + treatment_index,
            )
            for subject in subjects:
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

    title = (
        "Cave Matrix: Initial Conditions x Treatments"
        if treatment_count > 1
        else "Cave Matrix: Same Treatment, Many Initial Conditions"
    )

    return MatrixReportSpec(
        id="initial-conditions",
        title=title,
        run_factory=build_records,
        description=(
            "Runs generated experience sequences through many subjects that "
            "share the same mechanism but start from different memory-state "
            "positions. With multiple treatments, each start condition is "
            "crossed with each generated sequence."
        ),
        sections=(
            ReportSection(
                title="Population Shape",
                body=(
                    "The condition axis is held fixed at baseline. The population "
                    "varies subject starting memory state and can cross those "
                    "starts with multiple generated treatment sequences."
                ),
            ),
            ReportSection(
                title="Reading The Subjective Trajectory",
                body=(
                    "The topology subjective-trajectory view projects each "
                    "subject's memory state onto the shared topology axes. This "
                    "is the subject-side state trajectory, not accumulated "
                    "topology density."
                ),
            ),
        ),
        checks=(check_initial_conditions_matrix,),
        samples=samples,
        cluster_thresholds={
            "state_effect": 1e-12,
            "observed_memory": 0.02,
            "subjective_trajectory": 0.02,
            "active_context": 0.02,
        },
        config={
            "matrix": "initial_conditions",
            "sequence_count": treatment_count,
            "treatment_count": treatment_count,
            "condition_count": condition_count,
            "event_count": event_count,
            "seed": seed,
            "dt": dt,
            "end": end,
            "samples": samples,
        },
    )


def subject_ablation_subjects() -> tuple[Subject, ...]:
    params = subject_ablation_base_params()
    flat_state = SubjectState.initial(
        MemoryTrace(
            vector=np.zeros(len(DEFAULT_VOCABULARY), dtype=float),
            retention=params.memory.retention,
            decay_tau=params.memory.decay_tau,
            max_age=params.memory.max_age,
        ),
        params.topology,
    )
    prior_state = SubjectState.initial(
        MemoryTrace(
            vector=np.linspace(0.1, 0.9, len(DEFAULT_VOCABULARY)),
            retention=params.memory.retention,
            decay_tau=params.memory.decay_tau,
            max_age=params.memory.max_age,
        ),
        params.topology,
    )
    return (
        Subject(
            id="flat",
            params=params,
            initial_state=flat_state,
            vocabulary=list(DEFAULT_VOCABULARY),
            sensorium=default_sensorium(),
        ),
        Subject(
            id="prior",
            params=params,
            initial_state=prior_state,
            vocabulary=list(DEFAULT_VOCABULARY),
            sensorium=default_sensorium(),
        ),
    )


def initial_condition_subjects(
    *,
    condition_count: int = 16,
) -> tuple[Subject, ...]:
    if condition_count <= 0:
        raise ValueError("condition_count must be positive")
    params = initial_conditions_base_params()
    side = int(np.ceil(np.sqrt(condition_count)))
    values = np.linspace(0.08, 0.92, side)
    subjects = []
    for index, (x_value, y_value) in enumerate(
        (pair for pair in ((x, y) for x in values for y in values))
    ):
        if index >= condition_count:
            break
        vector = _initial_memory_vector(float(x_value), float(y_value))
        trace = MemoryTrace(
            vector=vector,
            retention=params.memory.retention,
            decay_tau=params.memory.decay_tau,
            max_age=params.memory.max_age,
        )
        subject_id = f"init-{index:02d}"
        subjects.append(
            Subject(
                id=subject_id,
                params=params,
                initial_state=SubjectState.initial(trace, params.topology),
                vocabulary=list(DEFAULT_VOCABULARY),
                sensorium=default_sensorium(),
            )
        )
    return tuple(subjects)


def subject_ablation_variants() -> tuple[MatrixVariant, ...]:
    base = subject_ablation_base_params()
    return (
        MatrixVariant("baseline", "Baseline", base, comparison_role="baseline"),
        MatrixVariant(
            "zero-attention",
            "Zero attention",
            replace(
                base,
                attention=zero_attention_profile(),
            ),
            comparison_role="negative_control",
            control_for="baseline",
        ),
        MatrixVariant(
            "split-attention",
            "External split attention",
            replace(
                base,
                attention=external_only_attention_profile(
                    channel_weights={"visual": 0.5, "audio": 0.5},
                ),
            ),
            comparison_role="intervention_control",
            control_for="baseline",
        ),
        MatrixVariant(
            "surprise-learning",
            "Surprise learning",
            replace(
                base,
                learning_rule=ImportanceWeightedLearningRule(surprise_gain=0.5),
            ),
            comparison_role="mechanism_variant",
            control_for="baseline",
        ),
        MatrixVariant(
            "zero-predictor",
            "Zero predictor",
            base,
            predictor_factory=ZeroPredictor,
            comparison_role="negative_control",
            control_for="baseline",
        ),
    )


def subject_ablation_base_params() -> ModelParams:
    return replace(
        default_model_params(),
        attention=balanced_attention_profile(),
        topology=SubjectiveTopologyParams(prior=SubjectiveTopologyPrior()),
    )


def initial_conditions_base_params() -> ModelParams:
    return replace(
        default_model_params(),
        attention=balanced_attention_profile(),
        topology=SubjectiveTopologyParams(prior=SubjectiveTopologyPrior()),
    )


def _initial_memory_vector(projected_x: float, projected_y: float) -> np.ndarray:
    vector = np.full(len(DEFAULT_VOCABULARY), 0.5, dtype=float)
    by_feature = {name: index for index, name in enumerate(DEFAULT_VOCABULARY)}
    for feature in ("angularity", "symmetry", "sides"):
        vector[by_feature[feature]] = projected_x
    for feature in ("roundness", "hue", "saturation", "novelty"):
        vector[by_feature[feature]] = projected_y
    return vector


def run_matrix_cell(
    *,
    sequence_id: str,
    sequence,
    subject: Subject,
    variant: MatrixVariant,
    dt: float,
    end: float | None,
) -> MatrixRunRecord:
    variant_subject = Subject(
        id=subject.id,
        params=variant.params,
        initial_state=subject.initial_state,
        vocabulary=list(subject.vocabulary),
        sensorium=subject.sensorium,
    )
    model = ExperienceModel(
        sequence=sequence,
        subject_state=variant_subject.fresh_state(),
        params=variant.params,
        vocabulary=list(variant_subject.vocabulary),
        sensorium=variant_subject.sensorium,
        predictor=(
            MemoryVectorPredictor()
            if variant.predictor_factory is None
            else variant.predictor_factory()
        ),
    )
    run_end = sequence.duration if end is None else end
    states = model.run(start=0.0, end=run_end, dt=dt)
    episode = episode_from_cave_states(
        f"{subject.id}:{variant.id}",
        sequence,
        list(variant_subject.vocabulary),
        states,
        metadata={
            "source": "cave.demonstrations.simulation",
            "subject_id": subject.id,
            "variant_id": variant.id,
            "memory_decay_tau": variant.params.memory.decay_tau,
            "memory_max_age": variant.params.memory.max_age,
            "memory_retention": variant.params.memory.retention,
            "topology_params": variant.params.topology,
        },
    )
    run = SubjectRun(
        id=f"{sequence_id}:{subject.id}:{variant.id}",
        subject=variant_subject,
        sequence=sequence,
        episode=episode,
    )
    return MatrixRunRecord(
        id=run.id,
        label=f"{sequence_id}-{subject.id}-{variant.id}",
        sequence_id=sequence_id,
        subject_id=subject.id,
        variant_id=variant.id,
        run=run,
        metadata={
            "variant_label": variant.label,
            "control_for": variant.control_for,
        },
        factors={
            "treatment": factor_level(
                "treatment",
                sequence_id,
                label=f"Sequence {sequence_id}",
                role="shared_input",
            ),
            "start_condition": factor_level(
                "start_condition",
                subject.id,
                label=subject.id,
                role="initial_state",
            ),
            "condition": factor_level(
                "condition",
                variant.id,
                label=variant.label,
                role=variant.comparison_role,
                metadata={"control_for": variant.control_for},
            ),
        },
        comparison_role=variant.comparison_role,
        matched_set_id=sequence_id,
        replicate_id=subject.id,
        group_id=variant.id,
    )


def check_subject_ablation_matrix(records) -> dict[str, object]:
    errors = []
    by_key = {
        (record.subject_id, record.variant_id): record.run
        for record in records
    }
    required = [
        ("flat", "baseline"),
        ("flat", "zero-attention"),
        ("flat", "zero-predictor"),
        ("prior", "baseline"),
        ("prior", "zero-attention"),
    ]
    missing = [f"{subject}:{variant}" for subject, variant in required if (subject, variant) not in by_key]
    if missing:
        errors.append(f"missing matrix cells: {', '.join(missing)}")

    metrics = {}
    if not missing:
        flat_baseline = by_key[("flat", "baseline")]
        flat_zero_attention = by_key[("flat", "zero-attention")]
        flat_zero_predictor = by_key[("flat", "zero-predictor")]
        prior_baseline = by_key[("prior", "baseline")]
        prior_zero_attention = by_key[("prior", "zero-attention")]
        zero_attention_effect_distance = embedding_distance(
            state_effect_embedding(flat_zero_attention),
            state_effect_embedding(prior_zero_attention),
        )
        baseline_zero_attention_distance = embedding_distance(
            state_effect_embedding(flat_baseline),
            state_effect_embedding(flat_zero_attention),
        )
        baseline_prior_effect_distance = embedding_distance(
            state_effect_embedding(flat_baseline),
            state_effect_embedding(prior_baseline),
        )
        baseline_prior_observed_distance = embedding_distance(
            memory_trajectory_embedding(flat_baseline),
            memory_trajectory_embedding(prior_baseline),
        )
        baseline_prior_internal_distance = embedding_distance(
            subjective_trajectory_embedding(flat_baseline),
            subjective_trajectory_embedding(prior_baseline),
        )
        baseline_zero_predictor_internal_distance = embedding_distance(
            subjective_trajectory_embedding(flat_baseline),
            subjective_trajectory_embedding(flat_zero_predictor),
        )
        metrics = {
            "zero_attention_effect_distance": zero_attention_effect_distance,
            "baseline_zero_attention_effect_distance": baseline_zero_attention_distance,
            "baseline_prior_effect_distance": baseline_prior_effect_distance,
            "baseline_prior_observed_distance": baseline_prior_observed_distance,
            "baseline_prior_internal_distance": baseline_prior_internal_distance,
            "baseline_zero_predictor_internal_distance": baseline_zero_predictor_internal_distance,
        }
        if zero_attention_effect_distance > 1e-12:
            errors.append("zero attention did not collapse state effect across subjects")
        if baseline_zero_attention_distance <= 1e-12:
            errors.append("baseline did not diverge from zero attention")
        if baseline_prior_effect_distance > 1e-12:
            errors.append("prior subject changed isolated baseline state effect")
        if baseline_prior_observed_distance <= 1e-12:
            errors.append("prior subject did not change observed baseline memory")
        if baseline_prior_internal_distance <= 1e-12:
            errors.append("prior subject did not change internal baseline geometry")
        if baseline_zero_predictor_internal_distance <= 1e-12:
            errors.append("zero predictor did not change subjective trajectory geometry")

    return {
        "id": "subject_ablation_matrix",
        "ok": not errors,
        "errors": errors,
        "metrics": metrics,
    }


def check_population_clusters_matrix(records) -> dict[str, object]:
    errors = []
    labels = [record.label for record in records]
    if len(labels) != len(set(labels)):
        errors.append("population matrix labels are not unique")
    zero_attention = [
        record
        for record in records
        if record.variant_id == "zero-attention"
    ]
    if not zero_attention:
        errors.append("population matrix has no zero-attention records")

    zero_distances = []
    for index, first in enumerate(zero_attention):
        for second in zero_attention[index + 1:]:
            zero_distances.append(
                embedding_distance(
                    state_effect_embedding(first.run),
                    state_effect_embedding(second.run),
                )
            )
    max_zero_attention_effect_distance = max(zero_distances) if zero_distances else 0.0
    observed_subject_distance = _mean_pair_distance_for_variant(
        records,
        "baseline",
        lambda run: memory_trajectory_embedding(run),
    )
    internal_zero_predictor_distance = _mean_variant_pair_distance(
        records,
        "baseline",
        "zero-predictor",
        lambda run: subjective_trajectory_embedding(run),
    )
    baseline_sequence_effect_distance = _mean_sequence_pair_distance(
        records,
        "flat",
        "baseline",
        lambda run: state_effect_embedding(run),
    )

    if max_zero_attention_effect_distance > 1e-12:
        errors.append("zero-attention state effects did not collapse across population")
    if observed_subject_distance <= 0.0:
        errors.append("observed memory has no subject separation")
    if internal_zero_predictor_distance <= 0.0:
        errors.append("zero predictor has no internal-prediction separation")
    if baseline_sequence_effect_distance <= 0.0:
        errors.append("baseline state effect does not vary by sequence")

    return {
        "id": "population_clusters_matrix",
        "ok": not errors,
        "errors": errors,
        "metrics": {
            "max_zero_attention_effect_distance": max_zero_attention_effect_distance,
            "mean_observed_subject_distance_baseline": observed_subject_distance,
            "mean_internal_baseline_zero_predictor_distance": internal_zero_predictor_distance,
            "mean_baseline_sequence_effect_distance": baseline_sequence_effect_distance,
        },
    }


def check_initial_conditions_matrix(records) -> dict[str, object]:
    errors = []
    sequence_ids = {record.sequence_id for record in records}
    variant_ids = {record.variant_id for record in records}
    subject_ids = {record.subject_id for record in records}
    if variant_ids != {"baseline"}:
        errors.append("initial-condition matrix must use only baseline condition")
    expected_subject_ids = set(subject_ids)
    for sequence_id in sequence_ids:
        sequence_subject_ids = {
            record.subject_id
            for record in records
            if record.sequence_id == sequence_id
        }
        if sequence_subject_ids != expected_subject_ids:
            errors.append(
                f"initial-condition treatment {sequence_id} does not contain the full start set"
            )

    memory_distances = []
    by_sequence = {}
    for record in records:
        by_sequence.setdefault(record.sequence_id, []).append(record)
    for sequence_records in by_sequence.values():
        for index, first in enumerate(sequence_records):
            for second in sequence_records[index + 1:]:
                memory_distances.append(
                    embedding_distance(
                        memory_trajectory_embedding(first.run),
                        memory_trajectory_embedding(second.run),
                    )
                )
    treatment_distances = []
    by_subject = {}
    for record in records:
        by_subject.setdefault(record.subject_id, []).append(record)
    for subject_records in by_subject.values():
        for index, first in enumerate(subject_records):
            for second in subject_records[index + 1:]:
                treatment_distances.append(
                    embedding_distance(
                        memory_trajectory_embedding(first.run),
                        memory_trajectory_embedding(second.run),
                    )
                )
    mean_memory_distance = float(np.mean(memory_distances)) if memory_distances else 0.0
    mean_treatment_distance = (
        float(np.mean(treatment_distances)) if treatment_distances else 0.0
    )
    if mean_memory_distance <= 0.0:
        errors.append("different initial conditions did not separate memory trajectories")
    if len(sequence_ids) > 1 and mean_treatment_distance <= 0.0:
        errors.append("different treatments did not separate matched-start trajectories")

    return {
        "id": "initial_conditions_matrix",
        "ok": not errors,
        "errors": errors,
        "metrics": {
            "run_count": len(records),
            "sequence_count": len(sequence_ids),
            "variant_count": len(variant_ids),
            "subject_count": len(subject_ids),
            "mean_memory_trajectory_distance": mean_memory_distance,
            "mean_same_start_treatment_distance": mean_treatment_distance,
        },
    }


def _mean_pair_distance_for_variant(records, variant_id: str, embedding) -> float:
    distances = []
    by_sequence = {}
    for record in records:
        if record.variant_id == variant_id:
            by_sequence.setdefault(record.sequence_id, []).append(record)
    for sequence_records in by_sequence.values():
        if len(sequence_records) < 2:
            continue
        for index, first in enumerate(sequence_records):
            for second in sequence_records[index + 1:]:
                distances.append(embedding_distance(embedding(first.run), embedding(second.run)))
    return float(np.mean(distances)) if distances else 0.0


def _mean_variant_pair_distance(records, left_variant: str, right_variant: str, embedding) -> float:
    distances = []
    by_key = {
        (record.sequence_id, record.subject_id, record.variant_id): record
        for record in records
    }
    for record in records:
        if record.variant_id != left_variant:
            continue
        other = by_key.get((record.sequence_id, record.subject_id, right_variant))
        if other is not None:
            distances.append(embedding_distance(embedding(record.run), embedding(other.run)))
    return float(np.mean(distances)) if distances else 0.0


def _mean_sequence_pair_distance(records, subject_id: str, variant_id: str, embedding) -> float:
    selected = [
        record
        for record in records
        if record.subject_id == subject_id and record.variant_id == variant_id
    ]
    distances = []
    for index, first in enumerate(selected):
        for second in selected[index + 1:]:
            distances.append(embedding_distance(embedding(first.run), embedding(second.run)))
    return float(np.mean(distances)) if distances else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Cave matrix reports.")
    parser.add_argument(
        "matrix",
        choices=["initial-conditions", "population-clusters", "subject-ablation"],
        help="Matrix report to generate.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--sequences", type=int, default=6)
    parser.add_argument(
        "--starts",
        type=int,
        default=None,
        help="Initial start conditions for the initial-conditions matrix.",
    )
    parser.add_argument(
        "--treatments",
        type=int,
        default=1,
        help="Generated treatment sequences for the initial-conditions matrix.",
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--end", type=float, default=3.0)
    parser.add_argument("--samples", type=int, default=48)
    args = parser.parse_args()

    if args.matrix == "population-clusters":
        spec = population_clusters_matrix_report_spec(
            sequence_count=args.sequences,
            event_count=args.count,
            seed=args.seed,
            dt=args.dt,
            end=args.end,
            samples=args.samples,
        )
    elif args.matrix == "initial-conditions":
        start_count = args.sequences if args.starts is None else args.starts
        spec = initial_conditions_matrix_report_spec(
            condition_count=start_count,
            treatment_count=args.treatments,
            event_count=args.count,
            seed=args.seed,
            dt=args.dt,
            end=args.end,
            samples=args.samples,
        )
    elif args.matrix == "subject-ablation":
        spec = subject_ablation_matrix_report_spec(
            event_count=args.count,
            seed=args.seed,
            dt=args.dt,
            end=args.end,
            samples=args.samples,
        )
    else:  # pragma: no cover - argparse prevents this
        raise ValueError(args.matrix)

    output = args.output or Path("out/reports/cave/matrices") / spec.id
    outputs = write_matrix_report(spec, output)
    print(f"wrote {outputs.report_md}")


if __name__ == "__main__":
    main()
