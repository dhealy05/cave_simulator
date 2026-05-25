from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from cave.observation.episodes import Episode
from cave.observation.population import factor_level
from cave.observation.source_reports.gpt2_matrices import _gpt2_subject_for_episode, _sequence_from_episode
from cave.presentation.reports.matrix import write_matrix_report
from cave.presentation.reports.specs import MatrixReportSpec, MatrixRunRecord, ReportSection
from cave.observation.producers.sources.conversation import ConversationProducer, ConversationTurn
from cave.demonstrations.subjects import (
    SubjectRun,
    active_context_embedding,
    embedding_distance,
    subjective_trajectory_embedding,
)


ConversationEpisodeFactory = Callable[
    [Sequence[ConversationTurn], "ConversationMatrixConfig"],
    Episode,
]


@dataclass(frozen=True)
class ConversationMatrixFixture:
    id: str
    turns: tuple[ConversationTurn, ...]


@dataclass(frozen=True)
class ConversationMatrixConfig:
    id: str
    label: str
    context_selection: str = "attended_top_k"
    context_top_k: int = 3
    attention_layer: int = -1
    feature_count: int = 4


def conversation_text_config_matrix_report_spec(
    *,
    fixtures: Sequence[ConversationMatrixFixture] | None = None,
    configs: Sequence[ConversationMatrixConfig] | None = None,
    model_path: str | Path = "lib/models/gpt2",
    feature_count: int = 4,
    samples: int = 32,
    episode_factory: ConversationEpisodeFactory | None = None,
) -> MatrixReportSpec:
    fixtures = tuple(fixtures or default_conversation_matrix_fixtures())
    configs = tuple(
        configs or default_conversation_matrix_configs(feature_count=feature_count)
    )
    model_path = Path(model_path)

    def build_records():
        records = []
        runtime_factory = (
            episode_factory
            if episode_factory is not None
            else _conversation_runtime_episode_factory(model_path)
        )
        for fixture in fixtures:
            for config in configs:
                episode = runtime_factory(fixture.turns, config)
                records.append(
                    conversation_episode_matrix_record(
                        fixture,
                        config,
                        episode,
                    )
                )
        return records

    return MatrixReportSpec(
        id="text-config",
        title="Conversation Matrix: Texts x Context Configs",
        run_factory=build_records,
        description=(
            "Compares fixed conversation fixtures across mock-memory context "
            "selection configs. Prior turns are treated as supplied context "
            "items, not durable transformer memories."
        ),
        sections=(
            ReportSection(
                title="Matrix Shape",
                body=(
                    "The input axis is a fixed conversation. The producer axis "
                    "is the conversation adapter. The variant axis changes how "
                    "prior turns are selected as mock memory context."
                ),
            ),
            ReportSection(
                title="Expected Controls",
                body=(
                    "All configs should preserve the same turn ids for a "
                    "conversation. Full-context keeps every prior turn, recent-k "
                    "keeps only the latest prior turns, and attended-top-k "
                    "normalizes attention mass over selected prior turns."
                ),
            ),
        ),
        checks=(check_conversation_text_config_matrix,),
        samples=samples,
        cluster_thresholds={
            "state_effect": 1e-12,
            "observed_memory": 0.02,
            "subjective_trajectory": 0.02,
            "active_context": 0.08,
        },
        config={
            "matrix": "conversation_text_config",
            "model_path": model_path.as_posix(),
            "feature_count": feature_count,
            "fixtures": [fixture.id for fixture in fixtures],
            "configs": [config.id for config in configs],
            "samples": samples,
        },
    )


def default_conversation_matrix_fixtures() -> tuple[ConversationMatrixFixture, ...]:
    return (
        ConversationMatrixFixture(
            "memory",
            (
                ConversationTurn("user", "Prior context is like memory."),
                ConversationTurn("assistant", "It functions that way in the protocol."),
                ConversationTurn("user", "Then expected versus actual is turn-level?"),
                ConversationTurn("assistant", "The view can be turn-level while metrics stay token-level."),
            ),
        ),
        ConversationMatrixFixture(
            "pattern",
            (
                ConversationTurn("user", "red blue red blue"),
                ConversationTurn("assistant", "The alternating pattern is clear."),
                ConversationTurn("user", "red blue red"),
                ConversationTurn("assistant", "blue is the likely continuation."),
            ),
        ),
    )


def default_conversation_matrix_configs(
    *,
    feature_count: int = 4,
) -> tuple[ConversationMatrixConfig, ...]:
    return (
        ConversationMatrixConfig(
            id="attended-top-1",
            label="Attended top 1",
            context_selection="attended_top_k",
            context_top_k=1,
            feature_count=feature_count,
        ),
        ConversationMatrixConfig(
            id="recent-2",
            label="Recent 2",
            context_selection="recent_k",
            context_top_k=2,
            feature_count=feature_count,
        ),
        ConversationMatrixConfig(
            id="full-context",
            label="Full context",
            context_selection="full_context",
            context_top_k=99,
            feature_count=feature_count,
        ),
    )


def _conversation_runtime_episode_factory(
    model_path: Path,
) -> ConversationEpisodeFactory:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Conversation support needs the optional GPT-2 runtime. Install it "
            'with: python -m pip install -e ".[gpt2]"'
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
    )

    def run(
        turns: Sequence[ConversationTurn],
        config: ConversationMatrixConfig,
    ) -> Episode:
        return ConversationProducer(
            model_path,
            feature_count=config.feature_count,
            context_selection=config.context_selection,
            context_top_k=config.context_top_k,
            attention_layer=config.attention_layer,
            model=model,
            tokenizer=tokenizer,
        ).run(turns)

    return run


def conversation_episode_matrix_record(
    fixture: ConversationMatrixFixture,
    config: ConversationMatrixConfig,
    episode: Episode,
) -> MatrixRunRecord:
    subject = _gpt2_subject_for_episode(episode)
    run = SubjectRun(
        id=f"{fixture.id}:conversation:{config.id}",
        subject=subject,
        sequence=_sequence_from_episode(episode),
        episode=episode,
    )
    return MatrixRunRecord(
        id=run.id,
        label=f"{fixture.id}-conversation-{config.id}",
        sequence_id=fixture.id,
        subject_id="conversation",
        variant_id=config.id,
        run=run,
        metadata={
            "turn_count": len(fixture.turns),
            "config_label": config.label,
            "context_selection": config.context_selection,
            "context_top_k": config.context_top_k,
            "attention_layer": config.attention_layer,
            "feature_count": config.feature_count,
        },
        factors={
            "treatment": factor_level(
                "treatment",
                fixture.id,
                label=fixture.id,
                role="fixed_conversation",
            ),
            "start_condition": factor_level(
                "start_condition",
                "conversation",
                label="Conversation producer",
                role="producer",
            ),
            "condition": factor_level(
                "condition",
                config.id,
                label=config.label,
                role="context_config",
            ),
        },
        comparison_role="context_config",
        matched_set_id=fixture.id,
        replicate_id="conversation",
        group_id=config.id,
    )


def check_conversation_text_config_matrix(records) -> dict[str, object]:
    errors = []
    by_fixture: dict[str, list[MatrixRunRecord]] = {}
    by_key = {}
    for record in records:
        by_fixture.setdefault(record.sequence_id, []).append(record)
        by_key[(record.sequence_id, record.variant_id)] = record

    variant_ids = {record.variant_id for record in records}
    for fixture_id in by_fixture:
        missing = variant_ids - {
            record.variant_id for record in by_fixture.get(fixture_id, ())
        }
        if missing:
            errors.append(f"{fixture_id}: missing configs {sorted(missing)}")

    for fixture_id, fixture_records in by_fixture.items():
        segment_sequences = {
            tuple(
                (
                    item.id,
                    item.metadata.get("role"),
                    item.metadata.get("text"),
                )
                for item in record.run.episode.inputs
            )
            for record in fixture_records
        }
        if len(segment_sequences) != 1:
            errors.append(f"{fixture_id}: configs did not preserve turn sequence")

    full_context_max_active = 0
    for record in records:
        if record.variant_id != "full-context":
            continue
        for observation in record.run.episode.observations:
            expected_count = int(observation.t)
            full_context_max_active = max(
                full_context_max_active,
                len(observation.active_inputs),
            )
            if len(observation.active_inputs) != expected_count:
                errors.append("full-context config did not keep all prior turns")
                break

    recent_max_active = max(
        (
            len(observation.active_inputs)
            for record in records
            if record.variant_id == "recent-2"
            for observation in record.run.episode.observations
        ),
        default=0,
    )
    if recent_max_active > 2:
        errors.append("recent-2 config kept more than two prior turns")

    top1_max_active = max(
        (
            len(observation.active_inputs)
            for record in records
            if record.variant_id == "attended-top-1"
            for observation in record.run.episode.observations
        ),
        default=0,
    )
    if top1_max_active > 1:
        errors.append("attended-top-1 config kept more than one prior turn")

    fixture_ids = sorted(by_fixture)
    different_fixture_distance = 0.0
    if len(fixture_ids) >= 2:
        first = by_key.get((fixture_ids[0], "full-context"))
        second = by_key.get((fixture_ids[1], "full-context"))
        if first is not None and second is not None:
            different_fixture_distance = embedding_distance(
                subjective_trajectory_embedding(first.run),
                subjective_trajectory_embedding(second.run),
            )
            if different_fixture_distance <= 0.0:
                errors.append("different conversations did not diverge internally")

    context_distances = []
    for fixture_id in fixture_ids:
        top1 = by_key.get((fixture_id, "attended-top-1"))
        full = by_key.get((fixture_id, "full-context"))
        if top1 is None or full is None:
            continue
        context_distances.append(
            embedding_distance(
                active_context_embedding(top1.run),
                active_context_embedding(full.run),
            )
        )
    same_fixture_context_config_distance = (
        float(np.mean(context_distances)) if context_distances else 0.0
    )
    if context_distances and same_fixture_context_config_distance <= 0.0:
        errors.append("active context embedding did not separate context configs")

    return {
        "id": "conversation_text_config_matrix",
        "ok": not errors,
        "errors": errors,
        "metrics": {
            "fixture_count": len(by_fixture),
            "config_count": len(variant_ids),
            "different_fixture_internal_distance": different_fixture_distance,
            "same_fixture_context_config_distance": same_fixture_context_config_distance,
            "attended_top_1_max_active": top1_max_active,
            "recent_2_max_active": recent_max_active,
            "full_context_max_active": full_context_max_active,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate conversation matrix reports."
    )
    parser.add_argument(
        "matrix",
        choices=["text-config"],
        help="Conversation matrix report to generate.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=Path("lib/models/gpt2"))
    parser.add_argument("--feature-count", type=int, default=4)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()

    if args.matrix == "text-config":
        spec = conversation_text_config_matrix_report_spec(
            model_path=args.model_path,
            feature_count=args.feature_count,
            samples=args.samples,
        )
    else:  # pragma: no cover - argparse prevents this
        raise ValueError(args.matrix)

    output = args.output or Path("out/reports/conversation/matrices") / spec.id
    try:
        outputs = write_matrix_report(spec, output)
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")
    print(f"wrote {outputs.report_md}")


if __name__ == "__main__":
    main()
