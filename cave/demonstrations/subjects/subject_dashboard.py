from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from cave.demonstrations.examples import DEFAULT_VOCABULARY, default_model_params, random_experience_sequence
from cave.commitments.attention import AttentionProfile, INTERNAL_EXPECTATION_CHANNEL
from cave.commitments.memory import (
    MemoryTrace,
)
from cave.demonstrations.state import SubjectState
from cave.commitments.topology import SubjectiveTopologyParams, SubjectiveTopologyPrior
from cave.demonstrations.subjects import (
    state_effect_embedding,
    subjective_trajectory_embedding,
    make_subject,
    memory_trajectory_embedding,
    run_subject,
    save_subject_comparison_dashboard,
    threshold_clusters,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("subjects_dashboard.png"))
    parser.add_argument("--sequences", type=int, default=3)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--end", type=float, default=3.0)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--cluster-threshold", type=float, default=1e-12)
    args = parser.parse_args()

    runs, labels = controlled_subject_runs(
        sequence_count=args.sequences,
        event_count=args.count,
        seed=args.seed,
        dt=args.dt,
        end=args.end,
    )
    effect = lambda run: state_effect_embedding(run, samples=args.samples)
    observed = lambda run: memory_trajectory_embedding(run, samples=args.samples)
    internal = lambda run: subjective_trajectory_embedding(run, samples=args.samples)

    save_subject_comparison_dashboard(
        runs,
        labels,
        args.output,
        effect_embedding=effect,
        observed_embedding=observed,
        internal_embedding=internal,
        title=f"Subject Comparison: {args.sequences} sequences x 5 subjects",
    )
    print(f"wrote {args.output}")
    print("experience-effect clusters:")
    for cluster in threshold_clusters(
        runs,
        effect,
        threshold=args.cluster_threshold,
    ):
        print([labels[index] for index in cluster])


def controlled_subject_runs(
    *,
    sequence_count: int,
    event_count: int,
    seed: int,
    dt: float,
    end: float,
):
    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")
    if event_count <= 0:
        raise ValueError("event_count must be positive")

    base = default_model_params()
    flat_landscape = SubjectiveTopologyParams(prior=SubjectiveTopologyPrior())
    zero_params = replace(
        base,
        attention=AttentionProfile(mode="constant", level=0.0),
        topology=flat_landscape,
    )
    full_params = replace(
        base,
        attention=AttentionProfile(mode="constant", level=1.0),
        topology=flat_landscape,
    )
    sine_params = replace(
        base,
        attention=AttentionProfile(mode="sine", level=0.5, amplitude=0.5),
        topology=flat_landscape,
    )
    split_params = replace(
        base,
        attention=AttentionProfile(
            mode="constant",
            level=1.0,
            channel_weights={
                "visual": 0.25,
                "audio": 0.25,
                INTERNAL_EXPECTATION_CHANNEL: 0.5,
            },
        ),
        topology=flat_landscape,
    )
    prior_memory = SubjectState.initial(
        MemoryTrace(vector=np.linspace(0.1, 0.9, len(DEFAULT_VOCABULARY))),
        flat_landscape,
    )
    subjects = [
        make_subject("zero-flat", params=zero_params),
        make_subject("full-flat", params=full_params),
        make_subject("sine-flat", params=sine_params),
        make_subject("zero-prior", params=zero_params, initial_state=prior_memory),
        make_subject("half-chan", params=split_params),
    ]
    sequences = [
        random_experience_sequence(count=event_count, seed=seed + index)
        for index in range(sequence_count)
    ]

    runs = []
    labels = []
    for sequence_index, sequence in enumerate(sequences):
        for subject in subjects:
            runs.append(
                run_subject(
                    sequence,
                    subject,
                    dt=dt,
                    end=end,
                    run_id=f"q{sequence_index}:{subject.id}",
                )
            )
            labels.append(f"Q{sequence_index}-{subject.id}")
    return runs, labels


if __name__ == "__main__":
    main()
