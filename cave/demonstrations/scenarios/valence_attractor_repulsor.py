from __future__ import annotations

from cave.commitments.affect import MetadataValenceEvaluator
from cave.commitments.attention import AttentionProfile
from cave.observation.episodes import CaveProducer, Episode
from cave.demonstrations.examples import model_for_sequence
from cave.observation.experience import load_experience_document
from cave.commitments.memory import MemoryParams
from cave.commitments.objective import LinearObjectiveEvaluator
from cave.demonstrations.scenarios._common import (
    affect_scenario_assets,
    first_observation_with_active_input,
    scenario_fixture_path,
)
from cave.presentation.reports.specs import ProducerReportSpec, ReportSection
from cave.demonstrations.simulation import ModelParams
from cave.commitments.topology import SubjectiveTopologyParams, SubjectiveTopologyPrior
from cave.observation.views import AffectView, default_views


def valence_attractor_repulsor_report_spec(
    *,
    dt: float = 0.1,
    fps: int = 12,
    include_assets: bool = True,
) -> ProducerReportSpec:
    fixture = scenario_fixture_path("valence_attractor_repulsor")

    def build_episode() -> Episode:
        document = load_experience_document(fixture)
        model = model_for_sequence(
            document.sequence,
            params=valence_attractor_repulsor_params(),
            vocabulary=document.vocabulary,
        )
        return CaveProducer(model).run(dt=dt)

    return ProducerReportSpec(
        id="valence-attractor-repulsor",
        title="Cave Scenario: Valence Attractor / Repulsor",
        episode_factory=build_episode,
        input_summary=f"{fixture.as_posix()} via CaveProducer(dt={dt})",
        description=(
            "Neutral, pleasurable, and painful events carry authored affect "
            "metadata. The scenario checks that pain and pleasure are evaluated "
            "separately from prediction surprise."
        ),
        views=[*default_views(), AffectView()],
        view_assets=affect_scenario_assets() if include_assets else (),
        frame_time=1.7,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "cave",
            "scenario": "valence_attractor_repulsor",
            "fixture": fixture.as_posix(),
            "dt": dt,
            "fps": fps,
        },
        checks=(check_valence_attractor_repulsor,),
        sections=(
            ReportSection(
                title="Claim",
                body=(
                    "Affect is authored as metadata and evaluated as model "
                    "state. The neutral event should remain affectively flat, "
                    "the pleasant event should raise pleasure, and the painful "
                    "event should raise pain."
                ),
                asset_ids=("timeline", "affect"),
            ),
            ReportSection(
                title="Prediction Separation",
                body=(
                    "Surprise remains visible in the affect asset, but it is "
                    "not automatically pain under this configuration."
                ),
                asset_ids=("expectation_actual", "affect"),
            ),
        ),
    )


def valence_attractor_repulsor_params() -> ModelParams:
    return ModelParams(
        memory=MemoryParams(retention=0.75, decay_tau=2.0, max_age=4.0),
        attention=AttentionProfile(mode="constant", level=1.0),
        topology=SubjectiveTopologyParams(
            feature_x="comfort",
            feature_y="threat",
            prior=SubjectiveTopologyPrior(),
        ),
        valence_evaluator=MetadataValenceEvaluator(),
        objective_evaluator=LinearObjectiveEvaluator(prediction_weight=0.25),
    )


def check_valence_attractor_repulsor(episode: Episode) -> dict[str, object]:
    errors = []
    neutral = first_observation_with_active_input(episode, "neutral_event")
    pleasant = first_observation_with_active_input(episode, "pleasant_event")
    painful = first_observation_with_active_input(episode, "painful_event")
    if neutral is None:
        errors.append("missing active observation for neutral_event")
    if pleasant is None:
        errors.append("missing active observation for pleasant_event")
    if painful is None:
        errors.append("missing active observation for painful_event")

    metrics = {}
    observations = {
        "neutral": neutral,
        "pleasant": pleasant,
        "painful": painful,
    }
    for key, observation in observations.items():
        if observation is None:
            continue
        valence = observation.metadata.get("valence", {})
        objective = observation.metadata.get("objective", {})
        metrics[key] = {
            "pain": float(valence.get("pain", 0.0)),
            "pleasure": float(valence.get("pleasure", 0.0)),
            "net": float(valence.get("net", 0.0)),
            "utility": float(objective.get("utility", 0.0)),
            "surprise": float(observation.surprise),
        }

    if not errors:
        if not metrics["neutral"]["pain"] == metrics["neutral"]["pleasure"] == 0.0:
            errors.append("neutral event was not affectively flat")
        if not metrics["pleasant"]["pleasure"] > metrics["pleasant"]["pain"]:
            errors.append("pleasant event did not produce positive valence")
        if not metrics["painful"]["pain"] > metrics["painful"]["pleasure"]:
            errors.append("painful event did not produce negative valence")
        if metrics["painful"]["surprise"] > 0.0 and metrics["neutral"]["pain"] > 0.0:
            errors.append("surprise created pain despite zero surprise-pain gain")

    return {
        "id": "valence_attractor_repulsor",
        "ok": not errors,
        "errors": errors,
        "metrics": metrics,
    }
