from __future__ import annotations

import numpy as np

from cave.commitments.attention import AttentionProfile
from cave.observation.episodes import CaveProducer, Episode
from cave.demonstrations.examples import model_for_sequence
from cave.observation.experience import load_experience_document
from cave.commitments.memory import MemoryParams
from cave.demonstrations.scenarios._common import (
    core_scenario_assets,
    first_observation_with_active_input,
    scenario_fixture_path,
)
from cave.presentation.reports.specs import ProducerReportSpec, ReportSection
from cave.demonstrations.simulation import ModelParams
from cave.commitments.topology import SubjectiveTopologyParams, SubjectiveTopologyPrior
from cave.observation.views import default_views


def importance_weighted_event_report_spec(
    *,
    dt: float = 0.1,
    fps: int = 12,
    include_assets: bool = True,
) -> ProducerReportSpec:
    fixture = scenario_fixture_path("importance_weighted_event")

    def build_episode() -> Episode:
        document = load_experience_document(fixture)
        model = model_for_sequence(
            document.sequence,
            params=importance_weighted_event_params(),
            vocabulary=document.vocabulary,
        )
        return CaveProducer(model).run(dt=dt)

    return ProducerReportSpec(
        id="importance-weighted-event",
        title="Cave Scenario: Importance-Weighted Event",
        episode_factory=build_episode,
        input_summary=f"{fixture.as_posix()} via CaveProducer(dt={dt})",
        description=(
            "Two matched visual events carry the same feature vector and "
            "salience, but the second has a higher learning weight. The "
            "scenario checks that this changes learning rate, memory movement, "
            "and object-level attention strength without using a special event "
            "category."
        ),
        views=default_views(),
        view_assets=core_scenario_assets() if include_assets else (),
        frame_time=0.8,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "cave",
            "scenario": "importance_weighted_event",
            "fixture": fixture.as_posix(),
            "dt": dt,
            "fps": fps,
        },
        checks=(check_importance_weighted_event,),
        sections=(
            ReportSection(
                title="Claim",
                body=(
                    "The two events have matching feature vectors and salience. "
                    "The second event has higher `learning_weight`, so the "
                    "difference should come from the shared update rule rather "
                    "than a special object category."
                ),
                asset_ids=("timeline",),
            ),
            ReportSection(
                title="Learning Effect",
                body=(
                    "The expectation/actual asset shows the higher-weight event "
                    "moving memory farther from the prior expected state."
                ),
                asset_ids=("expectation_actual",),
            ),
            ReportSection(
                title="Object Memory",
                body=(
                    "The topology view uses object-level memory strength, which "
                    "also reflects the higher learning weight."
                ),
                asset_ids=("subjective_topology",),
            ),
        ),
    )


def importance_weighted_event_params() -> ModelParams:
    return ModelParams(
        memory=MemoryParams(retention=0.8, decay_tau=2.0, max_age=4.0),
        attention=AttentionProfile(mode="constant", level=1.0),
        topology=SubjectiveTopologyParams(
            feature_x="impact",
            feature_y="context",
            prior=SubjectiveTopologyPrior(),
        ),
    )


def check_importance_weighted_event(episode: Episode) -> dict[str, object]:
    errors = []
    ordinary = first_observation_with_active_input(episode, "ordinary_event")
    important = first_observation_with_active_input(episode, "important_event")
    if ordinary is None:
        errors.append("missing active observation for ordinary_event")
    if important is None:
        errors.append("missing active observation for important_event")

    metrics = {}
    if ordinary is not None and important is not None:
        ordinary_delta = float(np.linalg.norm(ordinary.memory_state - ordinary.expected))
        important_delta = float(np.linalg.norm(important.memory_state - important.expected))
        ordinary_attention = float(ordinary.attention_weights.get("ordinary_event", 0.0))
        important_attention = float(important.attention_weights.get("important_event", 0.0))
        metrics = {
            "ordinary_learning_rate": float(ordinary.learning_rate),
            "important_learning_rate": float(important.learning_rate),
            "ordinary_memory_delta": ordinary_delta,
            "important_memory_delta": important_delta,
            "ordinary_attention_weight": ordinary_attention,
            "important_attention_weight": important_attention,
        }
        if not important.learning_rate > ordinary.learning_rate:
            errors.append("higher learning_weight did not raise learning rate")
        if not important_delta > ordinary_delta:
            errors.append("higher learning_weight did not move memory farther")
        if not important_attention > ordinary_attention:
            errors.append("higher learning_weight did not raise object attention strength")

    return {
        "id": "importance_weighted_event",
        "ok": not errors,
        "errors": errors,
        "metrics": metrics,
    }
