from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cave.demonstrations.examples import DEFAULT_VOCABULARY, demo_model, model_for_sequence, random_experience_model
from cave.observation.episodes import CaveProducer, Episode
from cave.observation.experience import load_experience_document
from cave.observation.projections import encode_value, project_all, view_to_dict
from cave.demonstrations.simulation import ExperienceModel
from cave.observation.structural import episode_frames, structural_state_for_episode
from cave.observation.views import (
    AffectView,
    ActionView,
    CorrectionView,
    ExperienceView,
    ExpectationActualView,
    MemoryLookbackView,
    PresentationView,
    SubjectiveTopologyView,
    TimelineView,
    default_views,
)


VIEW_FACTORIES = {
    "presentation": PresentationView,
    "memory": MemoryLookbackView,
    "timeline": TimelineView,
    "expectation": ExpectationActualView,
    "actual": ExpectationActualView,
    "affect": AffectView,
    "action": ActionView,
    "agency": ActionView,
    "objective": AffectView,
    "expectation_actual": ExpectationActualView,
    "correction": CorrectionView,
    "correction_triangle": CorrectionView,
    "subjective_topology": SubjectiveTopologyView,
    "topology": SubjectiveTopologyView,
    "landscape": SubjectiveTopologyView,
}


def add_experience_source_args(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--experience",
        type=Path,
        help="Path to an authored experience JSON document.",
    )
    source.add_argument(
        "--demo",
        action="store_true",
        help="Use the built-in demo experience sequence.",
    )
    source.add_argument(
        "--random",
        action="store_true",
        help="Generate a seeded random experience sequence.",
    )
    parser.add_argument("--count", type=int, default=8, help="Random sequence length.")
    parser.add_argument("--seed", type=int, default=7, help="Demo/random sequence seed.")


def model_from_source_args(args: argparse.Namespace) -> ExperienceModel:
    if args.experience is not None:
        document = load_experience_document(args.experience)
        vocabulary = document.vocabulary or list(DEFAULT_VOCABULARY)
        return model_for_sequence(document.sequence, vocabulary=vocabulary)
    if args.random:
        return random_experience_model(count=args.count, seed=args.seed)
    return demo_model(seed=args.seed)


def producer_from_source_args(args: argparse.Namespace) -> CaveProducer:
    return CaveProducer(model_from_source_args(args))


episode_source_from_source_args = producer_from_source_args


def views_from_names(names: str | None) -> list[ExperienceView]:
    if names is None or names == "all":
        return default_views()
    views: list[ExperienceView] = []
    for raw_name in names.split(","):
        name = raw_name.strip().lower()
        if not name:
            continue
        try:
            views.append(VIEW_FACTORIES[name]())
        except KeyError as exc:
            choices = ", ".join(sorted(VIEW_FACTORIES))
            raise ValueError(f"unsupported view {name!r}; choose from {choices}, all") from exc
    if not views:
        raise ValueError("at least one view must be selected")
    return views


def run_episode(
    producer: CaveProducer,
    *,
    start: float = 0.0,
    end: float | None = None,
    dt: float = 0.1,
) -> Episode:
    return producer.run(start=start, end=end, dt=dt)


def run_payload(
    episode: Episode,
) -> dict[str, Any]:
    return episode_payload(episode)


def episode_payload(episode: Episode) -> dict[str, Any]:
    structural = structural_state_for_episode(episode)
    frames = episode_frames(episode, structural)
    return {
        "source_name": episode.source_name,
        "vocabulary": list(episode.vocabulary),
        "duration": episode.duration,
        "metadata": encode_value(episode.metadata),
        "inputs": [
            {
                "id": item.id,
                "kind": item.kind,
                "start": item.start,
                "end": item.end,
                "order_index": item.order_index,
                "salience": item.salience,
                "learning_weight": item.learning_weight,
                "modality": item.modality,
                "features": encode_value(item.features),
                "metadata": encode_value(item.metadata),
            }
            for item in episode.inputs
        ],
        "frames": [
            {
                "t": frame.observation.t,
                "t_normalized": frame.observation.t_normalized,
                "attention": frame.observation.attention,
                "attention_weights": encode_value(frame.observation.attention_weights),
                "active_input_ids": list(frame.observation.active_inputs),
                "actual": encode_value(frame.observation.actual),
                "expected": encode_value(frame.observation.expected),
                "prediction_error": encode_value(frame.observation.error),
                "memory_state": encode_value(frame.observation.memory_state),
                "surprise": frame.observation.surprise,
                "learning_rate": frame.observation.learning_rate,
                "metadata": encode_value(frame.observation.metadata),
                "memory_items": [
                    {
                        "source_id": item.source.id,
                        "age": item.age(frame.observation.t),
                        "strength": item.strength,
                    }
                    for item in frame.topology_frame.memory_items
                ],
                "topology": {
                    "feature_x": encode_value(frame.topology_frame.topology.feature_x),
                    "feature_y": encode_value(frame.topology_frame.topology.feature_y),
                    "bounds": encode_value(frame.topology_frame.topology.bounds),
                    "density": encode_value(frame.topology_frame.topology.density),
                    "expected_density": encode_value(
                        frame.topology_frame.topology.expected_density
                    ),
                    "actual_density": encode_value(
                        frame.topology_frame.topology.actual_density
                    ),
                },
                "correction": (
                    None
                    if frame.topology_frame.correction is None
                    else encode_value(frame.topology_frame.correction)
                ),
                "views": {
                    name: view_to_dict(view)
                    for name, view in project_all(frame).items()
                },
            }
            for frame in frames
        ],
    }


def write_json_payload(payload: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, indent=2)
    if output is None:
        print(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
