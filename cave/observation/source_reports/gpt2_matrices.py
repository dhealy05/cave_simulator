from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from cave.observation.episodes import Episode
from cave.observation.experience import ExperienceObject, FeatureVector, InputSequence, TemporalExtent
from cave.commitments.memory import MemoryTrace
from cave.presentation.reports.matrix import write_matrix_report
from cave.presentation.reports.specs import MatrixReportSpec, MatrixRunRecord, ReportSection
from cave.observation.population import factor_level
from cave.observation.producers.sources.gpt2 import GPT2Producer
from cave.demonstrations.state import SubjectState
from cave.demonstrations.subjects import (
    Subject,
    SubjectRun,
    active_context_embedding,
    embedding_distance,
    subjective_trajectory_embedding,
)
from cave.commitments.topology import SubjectiveTopologyParams, SubjectiveTopologyPrior
from cave.demonstrations.simulation import ModelParams
from cave.commitments.memory import MemoryParams


EpisodeFactory = Callable[[str, "GPT2MatrixConfig"], Episode]


@dataclass(frozen=True)
class GPT2MatrixText:
    id: str
    text: str


@dataclass(frozen=True)
class GPT2MatrixConfig:
    id: str
    label: str
    active_input_mode: str = "attended_top_k"
    active_top_k: int = 3
    attention_layer: int = -1
    feature_count: int = 4


def gpt2_text_config_matrix_report_spec(
    *,
    texts: Sequence[GPT2MatrixText | tuple[str, str]] | None = None,
    configs: Sequence[GPT2MatrixConfig] | None = None,
    model_path: str | Path = "lib/models/gpt2",
    feature_count: int = 4,
    samples: int = 32,
    episode_factory: EpisodeFactory | None = None,
) -> MatrixReportSpec:
    texts = _coerce_texts(texts or default_gpt2_matrix_texts())
    configs = tuple(configs or default_gpt2_matrix_configs(feature_count=feature_count))
    model_path = Path(model_path)

    def build_records():
        records = []
        runtime_factory = (
            episode_factory
            if episode_factory is not None
            else _gpt2_runtime_episode_factory(model_path)
        )
        for text_item in texts:
            for config in configs:
                episode = runtime_factory(text_item.text, config)
                records.append(
                    gpt2_episode_matrix_record(
                        text_item,
                        config,
                        episode,
                    )
                )
        return records

    return MatrixReportSpec(
        id="text-config",
        title="GPT-2 Matrix: Texts x Producer Configs",
        run_factory=build_records,
        description=(
            "Compares fixed text inputs across GPT-2 producer interpretation "
            "configs. This uses teacher-forced forward passes, not generation."
        ),
        sections=(
            ReportSection(
                title="Matrix Shape",
                body=(
                    "The input axis is fixed text. The subject profile axis "
                    "is the GPT-2 producer identity. The variant axis changes "
                    "how the producer selects active token context."
                ),
            ),
            ReportSection(
                title="What Can Vary",
                body=(
                    "This matrix can compare attention context modes, active "
                    "top-k values, attention layers, and feature projection "
                    "widths. Temperature is not included because this report "
                    "does not generate continuations."
                ),
            ),
            ReportSection(
                title="Expected Controls",
                body=(
                    "All configs should preserve the same token inputs for the "
                    "same text. Current-token context should keep one active "
                    "input per observation, while full-context should grow with "
                    "token position."
                ),
            ),
        ),
        checks=(check_gpt2_text_config_matrix,),
        samples=samples,
        cluster_thresholds={
            "state_effect": 1e-12,
            "observed_memory": 0.02,
            "subjective_trajectory": 0.02,
            "active_context": 0.08,
        },
        config={
            "matrix": "gpt2_text_config",
            "model_path": model_path.as_posix(),
            "feature_count": feature_count,
            "texts": [item.id for item in texts],
            "configs": [item.id for item in configs],
            "samples": samples,
        },
    )


def default_gpt2_matrix_texts() -> tuple[GPT2MatrixText, ...]:
    return (
        GPT2MatrixText("hello", "Hello, my name is Paul and I like to "),
        GPT2MatrixText("pattern", "red blue red blue red blue "),
        GPT2MatrixText("surprise", "The capital of France is banana "),
    )


def default_gpt2_matrix_configs(
    *,
    feature_count: int = 4,
) -> tuple[GPT2MatrixConfig, ...]:
    return (
        GPT2MatrixConfig(
            id="current-token",
            label="Current token",
            active_input_mode="current_token",
            active_top_k=1,
            feature_count=feature_count,
        ),
        GPT2MatrixConfig(
            id="attended-top-1",
            label="Attended top 1",
            active_input_mode="attended_top_k",
            active_top_k=1,
            feature_count=feature_count,
        ),
        GPT2MatrixConfig(
            id="attended-top-3",
            label="Attended top 3",
            active_input_mode="attended_top_k",
            active_top_k=3,
            feature_count=feature_count,
        ),
        GPT2MatrixConfig(
            id="full-context",
            label="Full context",
            active_input_mode="full_context",
            active_top_k=99,
            feature_count=feature_count,
        ),
    )


def _coerce_texts(
    texts: Sequence[GPT2MatrixText | tuple[str, str]],
) -> tuple[GPT2MatrixText, ...]:
    coerced = []
    for item in texts:
        if isinstance(item, GPT2MatrixText):
            coerced.append(item)
            continue
        text_id, text = item
        coerced.append(GPT2MatrixText(text_id, text))
    return tuple(coerced)


def _gpt2_runtime_episode_factory(model_path: Path) -> EpisodeFactory:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "GPT-2 support is optional. Install it with: "
            'python -m pip install -e ".[gpt2]"'
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="eager",
    )

    def run(text: str, config: GPT2MatrixConfig) -> Episode:
        return GPT2Producer(
            model_path,
            feature_count=config.feature_count,
            active_input_mode=config.active_input_mode,
            active_top_k=config.active_top_k,
            attention_layer=config.attention_layer,
            model=model,
            tokenizer=tokenizer,
        ).run(text)

    return run


def gpt2_episode_matrix_record(
    text_item: GPT2MatrixText,
    config: GPT2MatrixConfig,
    episode: Episode,
) -> MatrixRunRecord:
    subject = _gpt2_subject_for_episode(episode)
    run = SubjectRun(
        id=f"{text_item.id}:gpt2:{config.id}",
        subject=subject,
        sequence=_sequence_from_episode(episode),
        episode=episode,
    )
    return MatrixRunRecord(
        id=run.id,
        label=f"{text_item.id}-gpt2-{config.id}",
        sequence_id=text_item.id,
        subject_id="gpt2",
        variant_id=config.id,
        run=run,
        metadata={
            "text": text_item.text,
            "config_label": config.label,
            "active_input_mode": config.active_input_mode,
            "active_top_k": config.active_top_k,
            "attention_layer": config.attention_layer,
            "feature_count": config.feature_count,
        },
        factors={
            "treatment": factor_level(
                "treatment",
                text_item.id,
                label=text_item.id,
                role="fixed_text",
            ),
            "start_condition": factor_level(
                "start_condition",
                "gpt2",
                label="GPT-2 producer",
                role="producer",
            ),
            "condition": factor_level(
                "condition",
                config.id,
                label=config.label,
                role="producer_config",
            ),
        },
        comparison_role="producer_config",
        matched_set_id=text_item.id,
        replicate_id="gpt2",
        group_id=config.id,
    )


def check_gpt2_text_config_matrix(records) -> dict[str, object]:
    errors = []
    by_text: dict[str, list[MatrixRunRecord]] = {}
    by_key = {}
    for record in records:
        by_text.setdefault(record.sequence_id, []).append(record)
        by_key[(record.sequence_id, record.variant_id)] = record

    variant_ids = {record.variant_id for record in records}
    for text_id in by_text:
        missing = variant_ids - {
            record.variant_id for record in by_text.get(text_id, ())
        }
        if missing:
            errors.append(f"{text_id}: missing configs {sorted(missing)}")

    for text_id, text_records in by_text.items():
        token_sequences = {
            tuple(
                (
                    item.metadata.get("token_id"),
                    item.metadata.get("token_text"),
                )
                for item in record.run.episode.inputs
            )
            for record in text_records
        }
        if len(token_sequences) != 1:
            errors.append(f"{text_id}: configs did not preserve token sequence")

    current_records = [
        record for record in records if record.variant_id == "current-token"
    ]
    if any(
        len(observation.active_inputs) != 1
        for record in current_records
        for observation in record.run.episode.observations
    ):
        errors.append("current-token config did not keep exactly one active input")
    current_token_max_active = max(
        (
            len(observation.active_inputs)
            for record in current_records
            for observation in record.run.episode.observations
        ),
        default=0,
    )

    top1_records = [
        record for record in records if record.variant_id == "attended-top-1"
    ]
    if any(
        len(observation.active_inputs) > 1
        for record in top1_records
        for observation in record.run.episode.observations
    ):
        errors.append("attended-top-1 config kept more than one active input")

    full_context_ok = True
    full_context_max_active = 0
    for record in records:
        if record.variant_id != "full-context":
            continue
        for observation in record.run.episode.observations:
            expected_count = int(observation.t) + 1
            full_context_max_active = max(
                full_context_max_active,
                len(observation.active_inputs),
            )
            if len(observation.active_inputs) != expected_count:
                full_context_ok = False
                break
    if not full_context_ok:
        errors.append("full-context config did not grow active context by position")
    if full_context_max_active <= current_token_max_active:
        errors.append("full-context did not retain more inputs than current-token")

    text_ids = sorted(by_text)
    different_text_distance = 0.0
    same_text_context_config_distance = 0.0
    if len(text_ids) >= 2:
        first = by_key.get((text_ids[0], "attended-top-3"))
        second = by_key.get((text_ids[1], "attended-top-3"))
        if first is not None and second is not None:
            different_text_distance = embedding_distance(
                subjective_trajectory_embedding(first.run),
                subjective_trajectory_embedding(second.run),
            )
            if different_text_distance <= 0.0:
                errors.append("different texts did not diverge internally")
    context_distances = []
    for text_id in text_ids:
        current = by_key.get((text_id, "current-token"))
        full = by_key.get((text_id, "full-context"))
        if current is None or full is None:
            continue
        context_distances.append(
            embedding_distance(
                active_context_embedding(current.run),
                active_context_embedding(full.run),
            )
        )
    if context_distances:
        same_text_context_config_distance = float(np.mean(context_distances))
        if same_text_context_config_distance <= 0.0:
            errors.append("active context embedding did not separate context configs")

    return {
        "id": "gpt2_text_config_matrix",
        "ok": not errors,
        "errors": errors,
        "metrics": {
            "text_count": len(by_text),
            "config_count": len(variant_ids),
            "different_text_internal_distance": different_text_distance,
            "same_text_context_config_distance": same_text_context_config_distance,
            "current_token_max_active": current_token_max_active,
            "full_context_max_active": full_context_max_active,
        },
    }


def _gpt2_subject_for_episode(episode: Episode) -> Subject:
    params = ModelParams(
        memory=MemoryParams(retention=1.0, decay_tau=2.0, max_age=4.0),
        topology=episode.metadata.get(
            "topology_params",
            SubjectiveTopologyParams(prior=SubjectiveTopologyPrior()),
        ),
    )
    trace = MemoryTrace(
        vector=np.zeros(len(episode.vocabulary), dtype=float),
        retention=1.0,
        decay_tau=2.0,
        max_age=4.0,
    )
    return Subject(
        id="gpt2",
        params=params,
        initial_state=SubjectState.initial(trace, params.topology),
        vocabulary=list(episode.vocabulary),
        sensorium=None,
    )


def _sequence_from_episode(episode: Episode) -> InputSequence:
    objects = []
    for item in episode.inputs:
        objects.append(
            ExperienceObject(
                id=item.id,
                temporal_extent=TemporalExtent(
                    item.start,
                    item.end,
                    item.order_index,
                ),
                features=FeatureVector(
                    {
                        feature: float(item.features[index])
                        for index, feature in enumerate(episode.vocabulary)
                        if index < item.features.size
                    }
                ),
                kind=item.kind,
                presentation=item.presentation,
                salience=item.salience,
                learning_weight=item.learning_weight,
                modality=item.modality,
                metadata=dict(item.metadata),
            )
        )
    return InputSequence(objects)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GPT-2 matrix reports.")
    parser.add_argument(
        "matrix",
        choices=["text-config"],
        help="GPT-2 matrix report to generate.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=Path("lib/models/gpt2"))
    parser.add_argument(
        "--text",
        action="append",
        default=None,
        help="Text input as ID=TEXT. Can be supplied multiple times.",
    )
    parser.add_argument("--feature-count", type=int, default=4)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()

    texts = None
    if args.text:
        try:
            texts = tuple(
                _parse_cli_text(value, index)
                for index, value in enumerate(args.text)
            )
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))

    if args.matrix == "text-config":
        spec = gpt2_text_config_matrix_report_spec(
            texts=texts,
            model_path=args.model_path,
            feature_count=args.feature_count,
            samples=args.samples,
        )
    else:  # pragma: no cover - argparse prevents this
        raise ValueError(args.matrix)

    output = args.output or Path("out/reports/gpt2/matrices") / spec.id
    try:
        outputs = write_matrix_report(spec, output)
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")
    print(f"wrote {outputs.report_md}")


def _parse_cli_text(value: str, index: int) -> GPT2MatrixText:
    if "=" not in value:
        return GPT2MatrixText(f"text-{index}", value)
    text_id, text = value.split("=", 1)
    text_id = text_id.strip()
    if not text_id:
        raise argparse.ArgumentTypeError("--text id must not be empty")
    return GPT2MatrixText(text_id, text)


if __name__ == "__main__":
    main()
