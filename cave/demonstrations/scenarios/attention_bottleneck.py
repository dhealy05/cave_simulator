from __future__ import annotations

import numpy as np

from cave.observation.episodes import CaveProducer, Episode
from cave.demonstrations.examples import model_for_sequence
from cave.observation.experience import load_experience_document
from cave.demonstrations.scenarios._common import (
    channel_scenario_params,
    core_scenario_assets,
    scenario_fixture_path,
)
from cave.presentation.reports.specs import ProducerReportSpec, ReportSection
from cave.observation.sensing import FeatureSensor, Sensorium
from cave.observation.views import default_views


def attention_bottleneck_report_spec(
    *,
    dt: float = 0.1,
    fps: int = 12,
    include_assets: bool = True,
) -> ProducerReportSpec:
    fixture = scenario_fixture_path("attention_bottleneck")

    def build_episode() -> Episode:
        document = load_experience_document(fixture)
        model = model_for_sequence(
            document.sequence,
            params=channel_scenario_params(
                channel_weights={"visual": 0.25, "audio": 0.75},
            ),
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
        id="attention-bottleneck",
        title="Cave Scenario: Attention Bottleneck",
        episode_factory=build_episode,
        input_summary=f"{fixture.as_posix()} via two-channel CaveProducer(dt={dt})",
        description=(
            "Visual and audio objects are presented simultaneously and both can "
            "be sensed. The scenario checks that the actual attended input "
            "follows the attention-channel distribution."
        ),
        views=default_views(),
        view_assets=core_scenario_assets() if include_assets else (),
        frame_time=0.0,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "cave",
            "scenario": "attention_bottleneck",
            "fixture": fixture.as_posix(),
            "dt": dt,
            "fps": fps,
            "channel_weights": {"visual": 0.25, "audio": 0.75},
        },
        checks=(check_attention_bottleneck,),
        sections=(
            ReportSection(
                title="Claim",
                body=(
                    "A visual marker and audio marker are active at the same "
                    "time and both are sensed. The attention distribution, not "
                    "mere external presence, determines how much each channel "
                    "enters actual input."
                ),
                asset_ids=("timeline",),
            ),
            ReportSection(
                title="Channel Gating",
                body=(
                    "With channel weights `visual=0.25` and `audio=0.75`, the "
                    "actual input should be `[0.25, 0.75]` for the first frame."
                ),
                asset_ids=("expectation_actual",),
            ),
            ReportSection(
                title="Topology Deposit",
                body=(
                    "The topology view shows the resulting state deposit after "
                    "attention has weighted the sensed channel responses."
                ),
                asset_ids=("subjective_topology",),
            ),
        ),
    )


def check_attention_bottleneck(episode: Episode) -> dict[str, object]:
    errors = []
    observation = episode.observations[0] if episode.observations else None
    metrics = {}
    if observation is None:
        errors.append("episode has no observations")
    else:
        expected_actual = np.array([0.25, 0.75])
        metrics = {
            "active_inputs": list(observation.active_inputs),
            "actual": observation.actual.tolist(),
            "attention_weights": dict(observation.attention_weights),
            "sensor_channels": list(
                observation.metadata.get("sensor_channels", {}).keys()
            ),
        }
        if observation.active_inputs != ["visual_marker", "audio_marker"]:
            errors.append("simultaneous active inputs were not both present")
        if not np.allclose(observation.actual, expected_actual):
            errors.append("actual input does not match attention-channel weighting")
        if set(metrics["sensor_channels"]) != {"visual", "audio"}:
            errors.append("two-channel sensorium did not report both channels")
        if not np.isclose(observation.attention_weights.get("visual_marker", 0.0), 0.25):
            errors.append("visual marker attention weight is not 0.25")
        if not np.isclose(observation.attention_weights.get("audio_marker", 0.0), 0.75):
            errors.append("audio marker attention weight is not 0.75")

    return {
        "id": "attention_bottleneck",
        "ok": not errors,
        "errors": errors,
        "metrics": metrics,
    }
