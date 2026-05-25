from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cave.observation.experience.authoring import (
    ExperienceQualities,
    resolve_experience_object,
)
from cave.observation.experience.objects import ExperienceObject, InputSequence, TemporalExtent
from cave.observation.experience.features import FeatureVector


@dataclass(frozen=True)
class ExperienceDocument:
    sequence: InputSequence
    vocabulary: list[str] | None = None
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_experience_document(path: str | Path) -> ExperienceDocument:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("experience document must be a JSON object")
    return experience_document_from_dict(data)


def experience_document_from_dict(data: dict[str, Any]) -> ExperienceDocument:
    raw_objects = data.get("objects", data.get("sequence"))
    if not isinstance(raw_objects, list):
        raise ValueError("experience document must contain an objects list")

    objects = [
        experience_object_from_dict(raw_obj, order_index=index)
        for index, raw_obj in enumerate(raw_objects)
    ]
    vocabulary = data.get("vocabulary")
    if vocabulary is not None:
        if not isinstance(vocabulary, list) or not all(isinstance(item, str) for item in vocabulary):
            raise ValueError("vocabulary must be a list of strings")
        vocabulary = list(vocabulary)

    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")

    name = data.get("name")
    if name is not None and not isinstance(name, str):
        raise ValueError("name must be a string")

    return ExperienceDocument(
        sequence=InputSequence(objects),
        vocabulary=vocabulary,
        name=name,
        metadata=dict(metadata),
    )


def experience_object_from_dict(
    data: dict[str, Any],
    *,
    order_index: int,
) -> ExperienceObject:
    if not isinstance(data, dict):
        raise ValueError("each experience object must be a JSON object")

    object_id = _required_str(data, "id")
    start = _number(data.get("start", data.get("temporal_start")), "start")
    end = _number(data.get("end", data.get("temporal_end")), "end")
    raw_features = data.get("features")
    if not isinstance(raw_features, dict):
        raise ValueError(f"{object_id}: features must be an object")

    features = {
        str(key): _number(value, f"{object_id}.features.{key}")
        for key, value in raw_features.items()
    }
    raw_metadata = data.get("metadata", {})
    if not isinstance(raw_metadata, dict):
        raise ValueError(f"{object_id}: metadata must be an object")
    metadata = dict(raw_metadata)

    raw_qualities = data.get("qualities", metadata.get("qualities"))
    qualities: ExperienceQualities | None = None
    if raw_qualities is not None:
        if not isinstance(raw_qualities, dict):
            raise ValueError(f"{object_id}: qualities must be an object")
        qualities = ExperienceQualities.from_mapping(raw_qualities)

    salience = _number(data.get("salience", 1.0), f"{object_id}.salience")
    learning_weight = _number(
        data.get("learning_weight", data.get("importance", 1.0)),
        f"{object_id}.learning_weight",
    )
    obj = ExperienceObject(
        id=object_id,
        temporal_extent=TemporalExtent(
            start=start,
            end=end,
            order_index=int(data.get("order_index", order_index)),
        ),
        features=FeatureVector(features),
        kind=str(data.get("kind", "experience")),
        salience=salience,
        learning_weight=learning_weight,
        modality=str(data.get("modality", "visual")),
        metadata=metadata,
    )
    if qualities is None:
        return obj
    return resolve_experience_object(
        obj,
        qualities,
        salience=salience if "salience" in data else None,
        learning_weight=(
            learning_weight
            if "learning_weight" in data or "importance" in data
            else None
        ),
    )


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    return float(value)
