from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from cave.commitments.agency import PreferenceActionPolicy, PreferenceProfile
from cave.commitments.attention import AttentionProfile
from cave.observation.experience import ExperienceObject, FeatureVector, InputSequence, TemporalExtent
from cave.demonstrations.examples import default_model_params
from cave.commitments.memory import MemoryParams
from cave.presentation.reports.specs import ReportSection, SubjectComparisonReportSpec
from cave.presentation.reports.subject_comparison import write_subject_comparison_report
from cave.demonstrations.subjects import make_subject, run_subject
from cave.demonstrations.subjects.subject_dashboard import controlled_subject_runs
from cave.commitments.topology import SubjectiveTopologyParams, SubjectiveTopologyPrior


def same_world_different_subjects_report_spec(
    *,
    event_count: int = 5,
    seed: int = 101,
    dt: float = 0.2,
    end: float = 3.0,
    samples: int = 48,
    cluster_threshold: float = 1e-12,
) -> SubjectComparisonReportSpec:
    def build_runs():
        return controlled_subject_runs(
            sequence_count=1,
            event_count=event_count,
            seed=seed,
            dt=dt,
            end=end,
        )

    return SubjectComparisonReportSpec(
        id="same-world-different-subjects",
        title="Cave Subject Scenario: Same World, Different Subjects",
        run_factory=build_runs,
        description=(
            "Runs one external sequence through controlled subjects with "
            "different attention and prior-state settings. The report checks "
            "that the world is held fixed while internal trajectories diverge."
        ),
        sections=(
            ReportSection(
                title="Shared World",
                body=(
                    "Every run uses the same external sequence. Differences in "
                    "the dashboard therefore come from subject parameters and "
                    "prior state rather than different input objects."
                ),
            ),
            ReportSection(
                title="Observed Memory Versus Experience Effect",
                body=(
                    "`zero-flat` and `zero-prior` both receive no attended input, "
                    "so their state effect should match. Their observed "
                    "memory can still differ because one subject starts from a "
                    "different prior memory state."
                ),
            ),
            ReportSection(
                title="Attention As Subject Difference",
                body=(
                    "The zero, full, sine, and split-channel subjects diverge in "
                    "experience-effect and internal-prediction embeddings even "
                    "though the world is unchanged."
                ),
            ),
        ),
        samples=samples,
        cluster_threshold=cluster_threshold,
        config={
            "scenario": "same_world_different_subjects",
            "sequence_count": 1,
            "event_count": event_count,
            "seed": seed,
            "dt": dt,
            "end": end,
            "samples": samples,
        },
    )


def preference_shaped_topology_report_spec(
    *,
    dt: float = 0.2,
    end: float = 1.0,
    samples: int = 24,
    cluster_threshold: float = 1e-12,
) -> SubjectComparisonReportSpec:
    def build_runs():
        sequence = preference_sequence()
        warm_subject = make_subject(
            "pref-warm",
            params=preference_params(
                PreferenceProfile(
                    feature_rewards={"warmth": 1.0},
                    feature_aversions={"threat": 0.1},
                    approach_gain=0.6,
                    avoid_gain=0.8,
                )
            ),
            vocabulary=["warmth", "threat"],
        )
        threat_sensitive_subject = make_subject(
            "pref-threat-avoid",
            params=preference_params(
                PreferenceProfile(
                    feature_rewards={"warmth": 0.1},
                    feature_aversions={"threat": 1.0},
                    approach_gain=0.6,
                    avoid_gain=0.8,
                )
            ),
            vocabulary=["warmth", "threat"],
        )
        runs = [
            run_subject(sequence, warm_subject, dt=dt, end=end, run_id="pref-warm"),
            run_subject(
                sequence,
                threat_sensitive_subject,
                dt=dt,
                end=end,
                run_id="pref-threat-avoid",
            ),
        ]
        return runs, ["pref-warm", "pref-threat-avoid"]

    return SubjectComparisonReportSpec(
        id="preference-shaped-topology",
        title="Cave Subject Scenario: Preference-Shaped Topology",
        run_factory=build_runs,
        description=(
            "Runs one external sequence through subjects with different "
            "preference profiles. The report checks that the shared world stays "
            "fixed while actions, exposure, memory, and topology diverge."
        ),
        sections=(
            ReportSection(
                title="Shared World",
                body=(
                    "Both subjects receive the same simultaneous warm and "
                    "threat events. The authored sequence is not changed by "
                    "agency."
                ),
            ),
            ReportSection(
                title="Preference Actions",
                body=(
                    "`pref-warm` approaches the warm event. "
                    "`pref-threat-avoid` avoids the threat event. These actions "
                    "change exposure before sensing and state update."
                ),
            ),
            ReportSection(
                title="Topology Consequence",
                body=(
                    "Different exposure produces different memory and topology "
                    "trajectories from the same world."
                ),
            ),
        ),
        samples=samples,
        cluster_threshold=cluster_threshold,
        config={
            "scenario": "preference_shaped_topology",
            "dt": dt,
            "end": end,
            "samples": samples,
        },
    )


def preference_sequence() -> InputSequence:
    return InputSequence(
        objects=[
            ExperienceObject(
                id="warm_event",
                temporal_extent=TemporalExtent(0.0, 1.0, 0),
                features=FeatureVector({"warmth": 1.0, "threat": 0.0}),
                kind="warm",
                salience=1.0,
            ),
            ExperienceObject(
                id="threat_event",
                temporal_extent=TemporalExtent(0.0, 1.0, 1),
                features=FeatureVector({"warmth": 0.0, "threat": 1.0}),
                kind="threat",
                salience=1.0,
            ),
        ]
    )


def preference_params(preferences: PreferenceProfile):
    return replace(
        default_model_params(),
        memory=MemoryParams(retention=0.75, decay_tau=2.0, max_age=4.0),
        attention=AttentionProfile(mode="constant", level=1.0),
        topology=SubjectiveTopologyParams(
            feature_x="warmth",
            feature_y="threat",
            prior=SubjectiveTopologyPrior(),
        ),
        action_policy=PreferenceActionPolicy(preferences),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Cave subject comparison reports.")
    parser.add_argument(
        "scenario",
        choices=["preference-shaped-topology", "same-world-different-subjects"],
        help="Subject comparison report to generate.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--end", type=float, default=3.0)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--cluster-threshold", type=float, default=1e-12)
    args = parser.parse_args()

    if args.scenario == "same-world-different-subjects":
        spec = same_world_different_subjects_report_spec(
            event_count=args.count,
            seed=args.seed,
            dt=args.dt,
            end=args.end,
            samples=args.samples,
            cluster_threshold=args.cluster_threshold,
        )
    elif args.scenario == "preference-shaped-topology":
        spec = preference_shaped_topology_report_spec(
            dt=args.dt,
            end=args.end,
            samples=args.samples,
            cluster_threshold=args.cluster_threshold,
        )
    else:  # pragma: no cover - argparse prevents this
        raise ValueError(args.scenario)

    output = args.output or Path("out/reports/cave/subjects") / spec.id
    outputs = write_subject_comparison_report(spec, output)
    print(f"wrote {outputs.report_md}")


if __name__ == "__main__":
    main()
