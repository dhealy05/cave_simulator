from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from cave.observation.episodes import Episode
from cave.presentation.renderers.matplotlib_renderer import available_styles
from cave.presentation.reports.generate import write_producer_report
from cave.presentation.reports.specs import ProducerReportSpec, ReportSection, ReportViewAsset
from cave.observation.producers.sources.conversation import ConversationProducer, ConversationTurn
from cave.observation.views import (
    CorrectionView,
    ExpectationActualView,
    MemoryLookbackView,
    PresentationView,
    SubjectiveTopologyView,
    TimelineView,
    default_views,
)


def conversation_reference_report_spec(
    *,
    turns: Sequence[ConversationTurn | tuple[str, str] | dict[str, object]] | None = None,
    model_path: str | Path = "lib/models/gpt2",
    feature_count: int = 8,
    context_selection: str = "attended_top_k",
    context_top_k: int = 8,
    fps: int = 8,
    include_assets: bool = True,
    episode_factory=None,
    style: str = "default",
) -> ProducerReportSpec:
    model_path = Path(model_path)
    turns = tuple(turns or default_conversation_turns())

    def build_episode() -> Episode:
        if episode_factory is not None:
            return episode_factory()
        return ConversationProducer(
            model_path,
            feature_count=feature_count,
            context_selection=context_selection,
            context_top_k=context_top_k,
        ).run(turns)

    view_assets = ()
    if include_assets:
        view_assets = (
            ReportViewAsset(
                id="presentation",
                title="Current Conversation Segment",
                views=[PresentationView()],
                filename="presentation.gif",
            ),
            ReportViewAsset(
                id="mock_memory",
                title="Prior Turns As Mock Memory",
                views=[MemoryLookbackView(min_strength=0.0)],
                filename="mock_memory.gif",
            ),
            ReportViewAsset(
                id="timeline",
                title="Conversation Timeline",
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
        title="Conversation Reference Report",
        episode_factory=build_episode,
        input_summary=f"{len(turns)} turns via ConversationProducer({model_path.as_posix()})",
        description=(
            "Reference conversation producer report. Prior turns are rendered as "
            "mock memory/context items supplied by the conversation protocol, not "
            "as transformer-stored memories."
        ),
        views=default_views(),
        view_assets=view_assets,
        checks=(check_conversation_reference,),
        sections=conversation_report_sections(),
        fps=fps,
        columns=2,
        style=style,
        config={
            "producer": "conversation",
            "model_path": model_path.as_posix(),
            "feature_count": feature_count,
            "context_selection": context_selection,
            "context_top_k": context_top_k,
            "fps": fps,
            "turn_count": len(turns),
            "style": style,
        },
    )


def default_conversation_turns() -> tuple[ConversationTurn, ...]:
    return (
        ConversationTurn("user", "I think prior context acts like memory."),
        ConversationTurn("assistant", "Yes, as a protocol-level memory."),
        ConversationTurn("user", "What should expected versus actual mean?"),
        ConversationTurn(
            "assistant",
            "The view can show whole turns while the metrics stay token-derived.",
        ),
    )


def conversation_report_sections() -> tuple[ReportSection, ...]:
    return (
        ReportSection(
            title="Conversation Segments",
            body=(
                "Each episode input is a whole turn. The presentation and "
                "timeline views stay at conversation scale instead of exposing "
                "individual tokens by default."
            ),
            asset_ids=("presentation", "timeline"),
        ),
        ReportSection(
            title="Mock Memory Context",
            body=(
                "Prior turns are selected and weighted as memory-like context "
                "items. This is an application-level interpretation of the "
                "conversation prefix, not a claim about durable transformer memory."
            ),
            asset_ids=("mock_memory",),
        ),
        ReportSection(
            title="Expected Versus Actual",
            body=(
                "Expected vectors are aggregated from token predictions over the "
                "current turn. Actual vectors summarize the observed turn, and "
                "surprise is mean negative log probability across its tokens."
            ),
            asset_ids=("expectation_actual", "correction"),
        ),
        ReportSection(
            title="Derived Topology",
            body=(
                "The topology is derived from the emitted episode vectors. It is "
                "a projection of conversation-state summaries, not a native "
                "language-model topology."
            ),
            asset_ids=("subjective_topology",),
        ),
    )


def check_conversation_reference(episode: Episode) -> dict[str, object]:
    errors = []
    if episode.metadata.get("adapter") != "ConversationProducer":
        errors.append("episode adapter is not ConversationProducer")
    if episode.metadata.get("memory_interpretation") != "mock_prior_context":
        errors.append("missing mock memory interpretation metadata")
    if episode.metadata.get("presentation_mode") != "current_conversation_segment":
        errors.append("presentation mode is not current_conversation_segment")
    if episode.metadata.get("lookback_mode") != "conversation_mock_memory":
        errors.append("lookback mode is not conversation_mock_memory")
    if len(episode.inputs) < 2:
        errors.append("conversation report requires at least two inputs")
    if len(episode.observations) != max(0, len(episode.inputs) - 1):
        errors.append("conversation observations should start at the second segment")
    if any(not observation.active_inputs for observation in episode.observations):
        errors.append("each observed segment should retain prior mock memory context")
    return {
        "id": "conversation_reference",
        "ok": not errors,
        "errors": errors,
        "metrics": {
            "segment_count": len(episode.inputs),
            "observation_count": len(episode.observations),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a conversation producer report."
    )
    parser.add_argument(
        "--turn",
        action="append",
        default=None,
        help="Conversation turn as ROLE=TEXT. Can be supplied multiple times.",
    )
    parser.add_argument("--model-path", type=Path, default=Path("lib/models/gpt2"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/reports/conversation/reference"),
    )
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument(
        "--context-selection",
        choices=["attended_top_k", "full_context", "recent_k"],
        default="attended_top_k",
    )
    parser.add_argument("--context-top-k", type=int, default=8)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--style", default="default", choices=available_styles())
    parser.add_argument(
        "--skip-assets",
        action="store_true",
        help="Only write the standard report frame and animation.",
    )
    args = parser.parse_args()
    turns = None
    if args.turn:
        try:
            turns = tuple(
                _parse_cli_turn(value, index)
                for index, value in enumerate(args.turn)
            )
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
    spec = conversation_reference_report_spec(
        turns=turns,
        model_path=args.model_path,
        feature_count=args.feature_count,
        context_selection=args.context_selection,
        context_top_k=args.context_top_k,
        fps=args.fps,
        include_assets=not args.skip_assets,
        style=args.style,
    )
    try:
        outputs = write_producer_report(spec, args.output)
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")
    print(f"wrote {outputs.report_md}")


def _parse_cli_turn(value: str, index: int) -> ConversationTurn:
    if "=" not in value:
        return ConversationTurn("user" if index % 2 == 0 else "assistant", value)
    role, text = value.split("=", 1)
    role = role.strip()
    if not role:
        raise argparse.ArgumentTypeError("--turn role must not be empty")
    return ConversationTurn(role, text)


if __name__ == "__main__":
    main()
