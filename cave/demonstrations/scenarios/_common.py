from __future__ import annotations

from pathlib import Path

from cave.commitments.attention import AttentionProfile
from cave.observation.episodes import Episode
from cave.commitments.memory import MemoryParams
from cave.presentation.reports.specs import ReportViewAsset
from cave.demonstrations.simulation import ModelParams
from cave.commitments.topology import SubjectiveTopologyParams, SubjectiveTopologyPrior
from cave.observation.views import (
    AffectView,
    ExpectationActualView,
    SubjectiveTopologyView,
    TimelineView,
)


def core_scenario_assets() -> tuple[ReportViewAsset, ...]:
    return (
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
            id="subjective_topology",
            title="Subjective State Topology",
            views=[SubjectiveTopologyView()],
            filename="subjective_topology.gif",
        ),
    )


def affect_scenario_assets() -> tuple[ReportViewAsset, ...]:
    return (
        *core_scenario_assets(),
        ReportViewAsset(
            id="affect",
            title="Affect / Objective",
            views=[AffectView()],
            filename="affect.gif",
        ),
    )


def first_observation_with_active_input(episode: Episode, input_id: str):
    for observation in episode.observations:
        if input_id in observation.active_inputs:
            return observation
    return None


def channel_scenario_params(channel_weights: dict[str, float]) -> ModelParams:
    return ModelParams(
        memory=MemoryParams(retention=0.75, decay_tau=2.0, max_age=4.0),
        attention=AttentionProfile(
            mode="constant",
            level=1.0,
            channel_weights=channel_weights,
        ),
        topology=SubjectiveTopologyParams(
            feature_x="visual_signal",
            feature_y="audio_signal",
            prior=SubjectiveTopologyPrior(),
        ),
    )


def scenario_fixture_path(name: str) -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "fixtures"
        / "cave"
        / "scenarios"
        / f"{name}.json"
    )
