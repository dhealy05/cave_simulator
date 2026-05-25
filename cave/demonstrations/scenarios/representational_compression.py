from __future__ import annotations

import numpy as np

from cave.commitments.attention import AttentionProfile
from cave.observation.episodes import CaveProducer, Episode
from cave.demonstrations.examples import model_for_sequence
from cave.observation.experience import load_experience_document
from cave.commitments.memory import MemoryParams
from cave.demonstrations.scenarios._common import (
    affect_scenario_assets,
    first_observation_with_active_input,
    scenario_fixture_path,
)
from cave.presentation.reports.specs import ProducerReportSpec, ReportSection
from cave.demonstrations.simulation import ModelParams
from cave.commitments.topology import SubjectiveTopologyParams, SubjectiveTopologyPrior
from cave.observation.views import AffectView, default_views
from cave.commitments.workspace import TopKWorkspaceCompressor


def representational_compression_report_spec(
    *,
    dt: float = 0.1,
    fps: int = 12,
    include_assets: bool = True,
) -> ProducerReportSpec:
    fixture = scenario_fixture_path("representational_compression")

    def build_episode() -> Episode:
        document = load_experience_document(fixture)
        model = model_for_sequence(
            document.sequence,
            params=representational_compression_params(),
            vocabulary=document.vocabulary,
        )
        return CaveProducer(model).run(dt=dt)

    return ProducerReportSpec(
        id="representational-compression",
        title="Cave Scenario: Representational Compression",
        episode_factory=build_episode,
        input_summary=f"{fixture.as_posix()} via CaveProducer(dt={dt})",
        description=(
            "A rich attended input is forced through a top-k workspace before "
            "prediction and memory update. The scenario checks that details can "
            "be dropped after attention admits the signal."
        ),
        views=[*default_views(), AffectView()],
        view_assets=affect_scenario_assets() if include_assets else (),
        frame_time=0.0,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "cave",
            "scenario": "representational_compression",
            "fixture": fixture.as_posix(),
            "dt": dt,
            "fps": fps,
            "workspace_capacity": 1,
            "workspace_input_mode": "workspace",
        },
        checks=(check_representational_compression,),
        sections=(
            ReportSection(
                title="Claim",
                body=(
                    "Attention admits the full signal, but the workspace keeps "
                    "only the strongest feature before the signal is used for "
                    "prediction and memory."
                ),
                asset_ids=("expectation_actual", "affect"),
            ),
            ReportSection(
                title="Compression Cost",
                body=(
                    "Observation metadata records the raw attended vector, the "
                    "workspace reconstruction, active features, reconstruction "
                    "error, and compression cost."
                ),
                asset_ids=("timeline",),
            ),
        ),
    )


def representational_compression_params() -> ModelParams:
    return ModelParams(
        memory=MemoryParams(retention=0.75, decay_tau=2.0, max_age=4.0),
        attention=AttentionProfile(mode="constant", level=1.0),
        topology=SubjectiveTopologyParams(
            feature_x="dominant",
            feature_y="secondary",
            prior=SubjectiveTopologyPrior(),
        ),
        workspace_compressor=TopKWorkspaceCompressor(capacity=1),
        workspace_input_mode="workspace",
    )


def check_representational_compression(episode: Episode) -> dict[str, object]:
    errors = []
    observation = first_observation_with_active_input(episode, "rich_signal")
    if observation is None:
        errors.append("missing active observation for rich_signal")
        return {
            "id": "representational_compression",
            "ok": False,
            "errors": errors,
            "metrics": {},
        }

    workspace = observation.metadata.get("workspace", {})
    attended_input = np.array(observation.metadata.get("attended_input", []), dtype=float)
    reconstructed = np.array(workspace.get("reconstructed", []), dtype=float)
    active_features = list(workspace.get("active_features", []))
    metrics = {
        "attended_input": attended_input.tolist(),
        "actual": observation.actual.tolist(),
        "workspace_reconstructed": reconstructed.tolist(),
        "active_features": active_features,
        "compression_cost": float(workspace.get("compression_cost", 0.0)),
        "reconstruction_error": float(workspace.get("reconstruction_error", 0.0)),
    }

    if not np.allclose(attended_input, np.array([0.5, 0.25, 0.125])):
        errors.append("raw attended input did not preserve the rich signal")
    if not np.allclose(observation.actual, np.array([0.5, 0.0, 0.0])):
        errors.append("actual state input was not workspace-compressed")
    if active_features != ["dominant"]:
        errors.append("workspace did not retain only the dominant feature")
    if metrics["compression_cost"] <= 0.0:
        errors.append("compression cost did not record dropped detail")

    return {
        "id": "representational_compression",
        "ok": not errors,
        "errors": errors,
        "metrics": metrics,
    }
