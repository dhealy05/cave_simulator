from __future__ import annotations

import argparse
from pathlib import Path

from cave.observation.episodes import Episode
from cave.presentation.renderers.matplotlib_renderer import available_styles
from cave.presentation.reports.generate import write_producer_report
from cave.presentation.reports.specs import ProducerReportSpec, ReportSection, ReportViewAsset
from cave.observation.producers.sources.gpt2 import GPT2Producer
from cave.observation.views import (
    CorrectionView,
    ExpectationActualView,
    MemoryLookbackView,
    PresentationView,
    SubjectiveTopologyView,
    TimelineView,
    default_views,
)


def gpt2_reference_report_spec(
    *,
    text: str = "Hello, my name is Paul and I like to ",
    model_path: str | Path = "lib/models/gpt2",
    feature_count: int = 8,
    active_top_k: int = 8,
    fps: int = 8,
    include_assets: bool = True,
    episode_factory=None,
    style: str = "default",
) -> ProducerReportSpec:
    model_path = Path(model_path)

    def build_episode() -> Episode:
        if episode_factory is not None:
            return episode_factory()
        return GPT2Producer(
            model_path,
            feature_count=feature_count,
            active_top_k=active_top_k,
        ).run(text)

    view_assets = ()
    if include_assets:
        view_assets = (
            ReportViewAsset(
                id="presentation",
                title="Current Token",
                views=[PresentationView()],
                filename="presentation.gif",
            ),
            ReportViewAsset(
                id="context",
                title="Attention Context",
                views=[MemoryLookbackView(min_strength=0.0)],
                filename="context.gif",
            ),
            ReportViewAsset(
                id="timeline",
                title="Token Timeline",
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
                title="Prediction Correction",
                views=[CorrectionView()],
                filename="correction.gif",
            ),
            ReportViewAsset(
                id="subjective_topology",
                title="Derived Topology",
                views=[SubjectiveTopologyView()],
                filename="subjective_topology.gif",
            ),
        )

    return ProducerReportSpec(
        id="reference",
        title="GPT-2 Reference Report",
        episode_factory=build_episode,
        input_summary=f"{text!r} via GPT2Producer({model_path.as_posix()})",
        description=(
            "Reference GPT-2 producer report generated from one forward pass. "
            "The report uses the generic episode/report path while keeping the "
            "GPT-2-specific interpretation explicit."
        ),
        views=default_views(),
        view_assets=view_assets,
        sections=gpt2_report_sections(),
        fps=fps,
        columns=2,
        style=style,
        config={
            "producer": "gpt2",
            "text": text,
            "model_path": model_path.as_posix(),
            "feature_count": feature_count,
            "active_top_k": active_top_k,
            "fps": fps,
            "style": style,
        },
    )


def gpt2_report_sections() -> tuple[ReportSection, ...]:
    return (
        ReportSection(
            title="Token Stream",
            body=(
                "Every token becomes an episode input. Observations begin at "
                "the second token because GPT-2 predicts token `n` from the "
                "preceding context."
            ),
            asset_ids=("presentation", "timeline"),
        ),
        ReportSection(
            title="Attention Context",
            body=(
                "The memory/lookback panel is repurposed as a temporal attention "
                "context view. It shows the selected prior tokens and their "
                "retained attention weights."
            ),
            asset_ids=("context",),
        ),
        ReportSection(
            title="Prediction Geometry",
            body=(
                "Expected vectors come from the previous-position token "
                "distribution; actual vectors are current token embeddings; "
                "memory-like state is the final-layer hidden state."
            ),
            asset_ids=("expectation_actual", "correction"),
        ),
        ReportSection(
            title="Derived Topology",
            body=(
                "GPT-2 does not maintain Cave's native topology. The topology "
                "view is derived from the emitted episode using per-episode PCA "
                "features."
            ),
            asset_ids=("subjective_topology",),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a GPT-2 producer report.")
    parser.add_argument(
        "--text",
        default="Hello, my name is Paul and I like to ",
    )
    parser.add_argument("--model-path", type=Path, default=Path("lib/models/gpt2"))
    parser.add_argument("--output", type=Path, default=Path("out/reports/gpt2/reference"))
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument("--active-top-k", type=int, default=8)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--style", default="default", choices=available_styles())
    parser.add_argument(
        "--skip-assets",
        action="store_true",
        help="Only write the standard report frame and animation.",
    )
    args = parser.parse_args()
    spec = gpt2_reference_report_spec(
        text=args.text,
        model_path=args.model_path,
        feature_count=args.feature_count,
        active_top_k=args.active_top_k,
        fps=args.fps,
        include_assets=not args.skip_assets,
        style=args.style,
    )
    try:
        outputs = write_producer_report(spec, args.output)
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")
    print(f"wrote {outputs.report_md}")


if __name__ == "__main__":
    main()
