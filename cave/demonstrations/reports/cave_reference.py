from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

from cave.commitments.attention import (
    INTERNAL_EXPECTATION_CHANNEL,
    AttentionChannelCurve,
    AttentionProfile,
)
from cave.observation.episodes import CaveProducer
from cave.observation.experience import load_experience_document
from cave.demonstrations.examples import default_model_params, model_for_sequence
from cave.presentation.renderers.matplotlib_renderer import available_styles
from cave.presentation.renderers.topology_surface_renderer import save_topology_state_surface
from cave.presentation.reports.generate import write_producer_report
from cave.presentation.reports.specs import (
    ProducerReportSpec,
    ReportExtraAsset,
    ReportSection,
    ReportViewAsset,
)
from cave.observation.views import (
    CorrectionView,
    ExpectationActualView,
    MemoryLookbackView,
    PresentationView,
    SubjectiveTopologyView,
    TimelineView,
    default_views,
)


def reference_cave_report_spec(
    *,
    dt: float = 0.1,
    fps: int = 12,
    include_readme_assets: bool = True,
    style: str = "default",
) -> ProducerReportSpec:
    fixture = _reference_fixture_path()

    def build_episode():
        document = load_experience_document(fixture)
        model = model_for_sequence(
            document.sequence,
            params=_reference_model_params(),
            vocabulary=document.vocabulary,
        )
        return CaveProducer(model).run(dt=dt)

    view_assets = ()
    extra_assets = ()
    if include_readme_assets:
        view_assets = (
            ReportViewAsset(
                id="presentation_wall",
                title="Presentation / Wall POV",
                views=[PresentationView()],
                filename="01_presentation_wall.gif",
            ),
            ReportViewAsset(
                id="memory_lookback",
                title="Memory / Lookback",
                views=[MemoryLookbackView()],
                filename="02_memory_lookback.gif",
            ),
            ReportViewAsset(
                id="timeline_tape",
                title="Timeline / Tape",
                views=[TimelineView()],
                filename="03_timeline_tape.gif",
            ),
            ReportViewAsset(
                id="expectation_actual",
                title="Expectation / Actual",
                views=[ExpectationActualView()],
                filename="04_expectation_actual.gif",
            ),
            ReportViewAsset(
                id="prediction_correction",
                title="Prediction Correction Over Time",
                views=[CorrectionView()],
                filename="05_prediction_correction_over_time.gif",
            ),
            ReportViewAsset(
                id="subjective_topology",
                title="Subjective State Topology",
                views=[SubjectiveTopologyView()],
                filename="06_subjective_topology.gif",
            ),
            ReportViewAsset(
                id="multi_view_state",
                title="Multi-view State",
                views=default_views(),
                filename="07_multi_view_state.gif",
                columns=2,
            ),
        )
        extra_assets = (
            ReportExtraAsset(
                id="topology_state_surface",
                title="Topology State Surface",
                filename="08_topology_state_surface.png",
                writer=save_topology_state_surface,
            ),
        )

    return ProducerReportSpec(
        id="reference",
        title="Cave Reference Report",
        episode_factory=build_episode,
        input_summary=f"{fixture.as_posix()} via CaveProducer(dt={dt})",
        description=(
            "Reference Cave producer run generated from the authored example "
            "fixture. This report is the executable counterpart to the README "
            "figures for the native Cave model."
        ),
        views=default_views(),
        view_assets=view_assets,
        extra_assets=extra_assets,
        sections=reference_report_sections(),
        dt=dt,
        fps=fps,
        columns=2,
        style=style,
        config={
            "producer": "cave",
            "fixture": fixture.as_posix(),
            "dt": dt,
            "fps": fps,
            "style": style,
            "attention_schedule": "three-channel sine allocation",
        },
    )


def _reference_model_params():
    return replace(
        default_model_params(),
        attention=AttentionProfile(
            mode="sine",
            level=0.72,
            amplitude=0.22,
            phase=0.15,
            channel_weights={
                "visual": 0.45,
                "audio": 0.12,
                INTERNAL_EXPECTATION_CHANNEL: 0.43,
            },
            channel_curves={
                "visual": AttentionChannelCurve(
                    mode="sine",
                    level=0.52,
                    amplitude=0.34,
                    phase=0.0,
                    cycles=1.0,
                ),
                INTERNAL_EXPECTATION_CHANNEL: AttentionChannelCurve(
                    mode="sine",
                    level=0.48,
                    amplitude=0.34,
                    phase=math.pi,
                    cycles=1.0,
                ),
                "audio": AttentionChannelCurve(
                    mode="sine",
                    level=0.18,
                    amplitude=0.14,
                    phase=math.pi / 2.0,
                    cycles=2.0,
                ),
            },
        ),
    )


def reference_report_sections() -> tuple[ReportSection, ...]:
    return (
        ReportSection(
            title="Presentation",
            body=(
                "The presentation view shows the currently active experience "
                "object. Wall position is derived from the object's temporal "
                "phase; it is not stored as object identity."
            ),
            asset_ids=("presentation_wall",),
        ),
        ReportSection(
            title="Memory",
            body=(
                "The memory view shows object memories after active intervals "
                "end. Strength follows accumulated attention and decay, while "
                "the compressed memory vector remains available in `episode.json`."
            ),
            asset_ids=("memory_lookback",),
        ),
        ReportSection(
            title="Timeline And Attention",
            body=(
                "The timeline exposes the temporal substrate and sampled "
                "attention curve. These values drive input gating, memory "
                "encoding, and topology deposits."
            ),
            asset_ids=("timeline_tape",),
        ),
        ReportSection(
            title="Expectation And Correction",
            body=(
                "The expectation views show prediction before update, actual "
                "attended input, prediction error, and after-update memory. "
                "This is the internal correction path for the native Cave run."
            ),
            asset_ids=("expectation_actual", "prediction_correction"),
        ),
        ReportSection(
            title="Topology",
            body=(
                "The topology assets show accumulated subjective density over "
                "feature space. The reference feature plane uses a form "
                "projection over angularity, symmetry, and sides, crossed with "
                "a sensory-tone projection over roundness, hue, saturation, "
                "and novelty. The surface view flattens each topology frame "
                "and stacks the result over time."
            ),
            asset_ids=("subjective_topology", "topology_state_surface"),
        ),
        ReportSection(
            title="Multi-view Frame",
            body=(
                "The multi-view animation places the external and internal "
                "projections side by side. All panels consume the same episode "
                "frames; none of them advances model state."
            ),
            asset_ids=("multi_view_state",),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Cave reference report.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/reports/cave/reference"),
    )
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--style", default="default", choices=available_styles())
    parser.add_argument(
        "--skip-readme-assets",
        action="store_true",
        help="Only write the standard report frame and animation.",
    )
    args = parser.parse_args()
    spec = reference_cave_report_spec(
        dt=args.dt,
        fps=args.fps,
        include_readme_assets=not args.skip_readme_assets,
        style=args.style,
    )
    outputs = write_producer_report(spec, args.output)
    print(f"wrote {outputs.report_md}")


def _reference_fixture_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "artifacts"
        / "inputs"
        / "cave"
        / "reference.json"
    )


if __name__ == "__main__":
    main()
