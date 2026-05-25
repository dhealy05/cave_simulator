from __future__ import annotations

import numpy as np

from cave.commitments.attention import AttentionProfile
from cave.observation.episodes import CaveProducer, Episode
from cave.demonstrations.examples import model_for_sequence
from cave.observation.experience import load_experience_document
from cave.commitments.learning import ImportanceWeightedLearningRule
from cave.commitments.memory import MemoryParams
from cave.demonstrations.scenarios._common import scenario_fixture_path
from cave.presentation.reports.specs import ProducerReportSpec, ReportSection, ReportViewAsset
from cave.demonstrations.simulation import ModelParams
from cave.commitments.topology import SubjectiveTopologyParams, SubjectiveTopologyPrior
from cave.observation.views import (
    CorrectionView,
    ExpectationActualView,
    SubjectiveTopologyView,
    TimelineView,
    default_views,
)


def expectation_violation_report_spec(
    *,
    dt: float = 0.1,
    fps: int = 12,
    include_assets: bool = True,
) -> ProducerReportSpec:
    fixture = scenario_fixture_path("expectation_violation")

    def build_episode() -> Episode:
        document = load_experience_document(fixture)
        model = model_for_sequence(
            document.sequence,
            params=expectation_violation_params(),
            vocabulary=document.vocabulary,
        )
        return CaveProducer(model).run(dt=dt)

    view_assets = ()
    if include_assets:
        view_assets = (
            ReportViewAsset(
                id="timeline",
                title="Timeline / Tape",
                views=[TimelineView()],
                filename="timeline.gif",
            ),
            ReportViewAsset(
                id="expectation_actual",
                title="Expectation / Actual",
                views=[ExpectationActualView()],
                filename="expectation_actual.gif",
            ),
            ReportViewAsset(
                id="correction",
                title="Prediction Correction Over Time",
                views=[CorrectionView()],
                filename="correction.gif",
            ),
            ReportViewAsset(
                id="subjective_topology",
                title="Subjective State Topology",
                views=[SubjectiveTopologyView()],
                filename="subjective_topology.gif",
            ),
        )

    return ProducerReportSpec(
        id="expectation-violation",
        title="Cave Scenario: Expectation Violation",
        episode_factory=build_episode,
        input_summary=f"{fixture.as_posix()} via CaveProducer(dt={dt})",
        description=(
            "A repeated input establishes an expectation, then an anomalous "
            "input arrives on the other feature axis. The scenario checks that "
            "prediction error falls during repetition and rises at the violation."
        ),
        views=default_views(),
        view_assets=view_assets,
        frame_time=1.5,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "cave",
            "scenario": "expectation_violation",
            "fixture": fixture.as_posix(),
            "dt": dt,
            "fps": fps,
            "surprise_gain": 0.25,
        },
        checks=(check_expectation_violation,),
        sections=(
            ReportSection(
                title="Claim",
                body=(
                    "The first three events repeat the same input vector, so "
                    "prediction error should shrink as memory learns the pattern. "
                    "The final event switches feature axes, making the expected "
                    "input wrong."
                ),
                asset_ids=("timeline",),
            ),
            ReportSection(
                title="Internal Correction",
                body=(
                    "The expectation and correction assets show the expected "
                    "vector converging during repetition and then separating "
                    "from the anomalous actual vector at the violation."
                ),
                asset_ids=("expectation_actual", "correction"),
            ),
            ReportSection(
                title="Topology",
                body=(
                    "The topology view places the learned pattern and violation "
                    "on the scenario's `energy` and `warmth` axes."
                ),
                asset_ids=("subjective_topology",),
            ),
        ),
    )


def expectation_violation_params() -> ModelParams:
    return ModelParams(
        memory=MemoryParams(retention=0.55, decay_tau=2.0, max_age=4.0),
        attention=AttentionProfile(mode="constant", level=1.0),
        topology=SubjectiveTopologyParams(
            feature_x="energy",
            feature_y="warmth",
            prior=SubjectiveTopologyPrior(),
        ),
        learning_rule=ImportanceWeightedLearningRule(surprise_gain=0.25),
    )


def check_expectation_violation(episode: Episode) -> dict[str, object]:
    ids = ["repeat_1", "repeat_2", "repeat_3", "violation"]
    first_observations = {}
    for observation in episode.observations:
        for input_id in observation.active_inputs:
            if input_id in ids and input_id not in first_observations:
                first_observations[input_id] = observation

    errors = []
    missing = [input_id for input_id in ids if input_id not in first_observations]
    if missing:
        errors.append(f"missing observations for {', '.join(missing)}")

    surprise = {
        input_id: float(first_observations[input_id].surprise)
        for input_id in ids
        if input_id in first_observations
    }
    learning_rate = {
        input_id: float(first_observations[input_id].learning_rate)
        for input_id in ids
        if input_id in first_observations
    }
    actual = {
        input_id: first_observations[input_id].actual
        for input_id in ids
        if input_id in first_observations
    }

    if not missing:
        if not surprise["repeat_1"] > surprise["repeat_2"] > surprise["repeat_3"]:
            errors.append("repeated inputs did not reduce surprise monotonically")
        if not surprise["violation"] > surprise["repeat_3"] * 2.0:
            errors.append("violation did not raise surprise enough above the learned pattern")
        if not learning_rate["violation"] > learning_rate["repeat_3"]:
            errors.append("surprise-weighted learning did not raise the violation learning rate")
        if not np.allclose(actual["violation"], np.array([0.0, 0.5])):
            errors.append("violation actual input is not the anomalous feature vector")

    return {
        "id": "expectation_violation",
        "ok": not errors,
        "errors": errors,
        "metrics": {
            "surprise": surprise,
            "learning_rate": learning_rate,
        },
    }
