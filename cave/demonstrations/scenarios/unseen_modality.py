from __future__ import annotations

import numpy as np

from cave.observation.episodes import CaveProducer, Episode
from cave.demonstrations.examples import model_for_sequence
from cave.observation.experience import load_experience_document
from cave.demonstrations.scenarios._common import (
    channel_scenario_params,
    core_scenario_assets,
    first_observation_with_active_input,
    scenario_fixture_path,
)
from cave.presentation.reports.specs import ProducerReportSpec, ReportSection
from cave.observation.views import default_views


def unseen_modality_report_spec(
    *,
    dt: float = 0.1,
    fps: int = 12,
    include_assets: bool = True,
) -> ProducerReportSpec:
    fixture = scenario_fixture_path("unseen_modality")

    def build_episode() -> Episode:
        document = load_experience_document(fixture)
        model = model_for_sequence(
            document.sequence,
            params=channel_scenario_params(
                channel_weights={"visual": 1.0},
            ),
            vocabulary=document.vocabulary,
        )
        return CaveProducer(model).run(dt=dt)

    return ProducerReportSpec(
        id="unseen-modality",
        title="Cave Scenario: Unseen Modality",
        episode_factory=build_episode,
        input_summary=f"{fixture.as_posix()} via visual-only CaveProducer(dt={dt})",
        description=(
            "An audio object exists in the external sequence, but the default "
            "visual-only subject has no audio sensor. The scenario checks that "
            "the audio object appears as an input while contributing no actual "
            "attended input when it is active."
        ),
        views=default_views(),
        view_assets=core_scenario_assets() if include_assets else (),
        frame_time=0.6,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "cave",
            "scenario": "unseen_modality",
            "fixture": fixture.as_posix(),
            "dt": dt,
            "fps": fps,
            "sensorium": "default_visual_only",
        },
        checks=(check_unseen_modality,),
        sections=(
            ReportSection(
                title="Claim",
                body=(
                    "The audio event is part of the external sequence, but the "
                    "subject uses the default visual-only sensorium. It should "
                    "therefore appear in time without entering actual input."
                ),
                asset_ids=("timeline",),
            ),
            ReportSection(
                title="Internal Input",
                body=(
                    "The expectation/actual asset shows a visual response for "
                    "`visible_flash` and a zero actual vector while "
                    "`unheard_tone` is active."
                ),
                asset_ids=("expectation_actual",),
            ),
            ReportSection(
                title="State Consequence",
                body=(
                    "Because the audio object is unsensed, it does not deposit "
                    "new state through actual input for this subject."
                ),
                asset_ids=("subjective_topology",),
            ),
        ),
    )


def check_unseen_modality(episode: Episode) -> dict[str, object]:
    errors = []
    input_ids = [item.id for item in episode.inputs]
    if "visible_flash" not in input_ids or "unheard_tone" not in input_ids:
        errors.append("expected visual and audio inputs are not both present")

    visible = first_observation_with_active_input(episode, "visible_flash")
    unheard = first_observation_with_active_input(episode, "unheard_tone")
    if visible is None:
        errors.append("missing active observation for visible_flash")
    if unheard is None:
        errors.append("missing active observation for unheard_tone")

    metrics = {}
    if visible is not None:
        metrics["visible_actual"] = visible.actual.tolist()
        if not np.allclose(visible.actual, np.array([1.0, 0.0])):
            errors.append("visible object did not enter visual-only subject state")
    if unheard is not None:
        metrics["unheard_actual"] = unheard.actual.tolist()
        metrics["unheard_sensor_channels"] = list(
            unheard.metadata.get("sensor_channels", {}).keys()
        )
        if not np.allclose(unheard.actual, np.array([0.0, 0.0])):
            errors.append("unsensed audio object affected actual input")

    return {
        "id": "unseen_modality",
        "ok": not errors,
        "errors": errors,
        "metrics": metrics,
    }
