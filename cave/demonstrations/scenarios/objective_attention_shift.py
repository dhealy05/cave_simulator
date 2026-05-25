from __future__ import annotations

import numpy as np

from cave.commitments.affect import MetadataValenceEvaluator
from cave.commitments.attention import AttentionProfile, ObjectiveAdaptiveAttentionPolicy
from cave.observation.episodes import CaveProducer, Episode
from cave.demonstrations.examples import model_for_sequence
from cave.observation.experience import load_experience_document
from cave.commitments.memory import MemoryParams
from cave.commitments.objective import LinearObjectiveEvaluator
from cave.demonstrations.scenarios._common import (
    affect_scenario_assets,
    scenario_fixture_path,
)
from cave.presentation.reports.specs import ProducerReportSpec, ReportSection
from cave.observation.sensing import FeatureSensor, Sensorium
from cave.demonstrations.simulation import ModelParams
from cave.commitments.topology import SubjectiveTopologyParams, SubjectiveTopologyPrior
from cave.observation.views import AffectView, default_views


def objective_attention_shift_report_spec(
    *,
    dt: float = 0.1,
    fps: int = 12,
    include_assets: bool = True,
) -> ProducerReportSpec:
    fixture = scenario_fixture_path("objective_attention_shift")

    def build_episode() -> Episode:
        document = load_experience_document(fixture)
        model = model_for_sequence(
            document.sequence,
            params=objective_attention_shift_params(),
            vocabulary=document.vocabulary,
        )
        model.sensorium = Sensorium(
            sensors=(
                FeatureSensor(modality="visual", channel="visual"),
                FeatureSensor(modality="audio", channel="audio"),
            )
        )
        return CaveProducer(model).run(dt=dt)

    return ProducerReportSpec(
        id="objective-attention-shift",
        title="Cave Scenario: Objective Attention Shift",
        episode_factory=build_episode,
        input_summary=f"{fixture.as_posix()} via CaveProducer(dt={dt})",
        description=(
            "A neutral visual signal and painful audio signal are sensed at the "
            "same time. The scenario checks that objective-driven attention "
            "moves the next timestep toward the valued channel."
        ),
        views=[*default_views(), AffectView()],
        view_assets=affect_scenario_assets() if include_assets else (),
        frame_time=0.1,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "cave",
            "scenario": "objective_attention_shift",
            "fixture": fixture.as_posix(),
            "dt": dt,
            "fps": fps,
            "initial_channel_weights": {"visual": 0.8, "audio": 0.2},
        },
        checks=(check_objective_attention_shift,),
        sections=(
            ReportSection(
                title="Claim",
                body=(
                    "The audio channel initially receives only partial "
                    "attention. Once its painful value enters state, the "
                    "attention policy shifts future channel weights toward it."
                ),
                asset_ids=("timeline", "affect"),
            ),
            ReportSection(
                title="Management Signal",
                body=(
                    "The affect/objective asset exposes the pressure that "
                    "drives attention. Observation metadata records both the "
                    "current and next channel distributions."
                ),
                asset_ids=("expectation_actual",),
            ),
        ),
    )


def objective_attention_shift_params() -> ModelParams:
    return ModelParams(
        memory=MemoryParams(retention=0.75, decay_tau=2.0, max_age=4.0),
        attention=AttentionProfile(
            mode="constant",
            level=1.0,
            channel_weights={"visual": 0.8, "audio": 0.2},
        ),
        attention_policy=ObjectiveAdaptiveAttentionPolicy(
            learning_rate=0.75,
            signal_gain=0.1,
            pain_gain=4.0,
            pleasure_gain=1.0,
        ),
        topology=SubjectiveTopologyParams(
            feature_x="visual_signal",
            feature_y="audio_signal",
            prior=SubjectiveTopologyPrior(),
        ),
        valence_evaluator=MetadataValenceEvaluator(),
        objective_evaluator=LinearObjectiveEvaluator(prediction_weight=0.25),
    )


def check_objective_attention_shift(episode: Episode) -> dict[str, object]:
    errors = []
    if len(episode.observations) < 2:
        errors.append("episode needs at least two observations")
        return {
            "id": "objective_attention_shift",
            "ok": False,
            "errors": errors,
            "metrics": {},
        }

    first = episode.observations[0]
    second = episode.observations[1]
    first_channels = first.metadata.get("attention_channels", {})
    next_channels = first.metadata.get("next_attention_channels", {})
    second_channels = second.metadata.get("attention_channels", {})
    valence = first.metadata.get("valence", {})
    metrics = {
        "first_attention_channels": dict(first_channels),
        "first_next_attention_channels": dict(next_channels),
        "second_attention_channels": dict(second_channels),
        "first_pain": float(valence.get("pain", 0.0)),
        "first_channel_pain": dict(valence.get("channel_pain", {})),
    }

    first_audio = float(first_channels.get("audio", 0.0))
    next_audio = float(next_channels.get("audio", 0.0))
    second_audio = float(second_channels.get("audio", 0.0))
    if not metrics["first_pain"] > 0.0:
        errors.append("painful audio did not enter valence state")
    if not next_audio > first_audio:
        errors.append("next attention did not shift toward the painful audio channel")
    if not np.isclose(second_audio, next_audio):
        errors.append("second timestep did not use the adaptive next channel weights")

    return {
        "id": "objective_attention_shift",
        "ok": not errors,
        "errors": errors,
        "metrics": metrics,
    }
