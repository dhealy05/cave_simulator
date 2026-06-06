from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from cave.commitments.topology import SubjectiveTopologyParams, SubjectiveTopologyPrior
from cave.observation.episodes import Episode, EpisodeInput, EpisodeObservation
from cave.observation.experience import (
    ExperienceObject,
    FeatureVector,
    InputSequence,
    TemporalExtent,
    presentation_for_object,
)


Array = np.ndarray

DEFAULT_PRIMITIVE_VOCABULARY = ["primitive_x", "primitive_y"]
PRIMITIVE_ETA = 0.45


@dataclass(frozen=True)
class PrimitiveStep:
    t: int
    actual: float
    expected: float
    error: float
    surprise: float
    memory_previous: float
    memory: float


@dataclass(frozen=True)
class PrimitiveStep2D:
    t: int
    actual: tuple[float, float]
    expected: tuple[float, float]
    error: tuple[float, float]
    surprise: float
    memory_previous: tuple[float, float]
    memory: tuple[float, float]


@dataclass(frozen=True)
class PrimitiveVectorStep:
    t: int
    actual: tuple[float, ...]
    expected: tuple[float, ...]
    error: tuple[float, ...]
    surprise: float
    memory_previous: tuple[float, ...]
    memory: tuple[float, ...]


@dataclass(frozen=True)
class PrimitiveWorldObject:
    id: str
    label: str
    sprite: str
    features: tuple[float, float]
    value: float = 0.0


@dataclass(frozen=True)
class PrimitiveTreeVariant:
    scale: float
    features: tuple[float, float]


PRIMITIVE_OBJECTS: dict[str, PrimitiveWorldObject] = {
    "tree": PrimitiveWorldObject(
        id="tree",
        label="Tree",
        sprite="object_tree.png",
        features=(0.20, 0.80),
        value=0.20,
    ),
    "rock": PrimitiveWorldObject(
        id="rock",
        label="Rock",
        sprite="object_rock.png",
        features=(0.50, 0.30),
        value=0.00,
    ),
    "snake": PrimitiveWorldObject(
        id="snake",
        label="Snake",
        sprite="object_snake_1.png",
        features=(0.90, 0.90),
        value=-1.00,
    ),
}

PRIMITIVE_EPISODE = ("tree", "tree", "tree", "snake", "rock", "tree")
PRIMITIVE_MEMORY_INITIAL = PRIMITIVE_OBJECTS["tree"].features

PRIMITIVE_TREE_VARIANTS: tuple[PrimitiveTreeVariant, ...] = (
    PrimitiveTreeVariant(scale=1.28, features=(0.18, 0.74)),
    PrimitiveTreeVariant(scale=1.48, features=(0.22, 0.86)),
    PrimitiveTreeVariant(scale=1.16, features=(0.17, 0.79)),
    PrimitiveTreeVariant(scale=1.38, features=(0.24, 0.82)),
)

DEFAULT_PRIMITIVE_WEIGHTS = {"tree": 0.60, "rock": 0.22, "snake": 0.18}


class PrimitiveProducer:
    name = "primitive"

    def __init__(
        self,
        sequence: InputSequence | None = None,
        *,
        name: str = "primitive",
        eta: float = PRIMITIVE_ETA,
        memory_initial: Sequence[float] | None = PRIMITIVE_MEMORY_INITIAL,
        vocabulary: Sequence[str] | None = None,
    ) -> None:
        if eta < 0.0:
            raise ValueError("eta must be non-negative")
        self.sequence = sequence or primitive_input_sequence()
        self.name = name
        self.eta = float(eta)
        self.memory_initial = (
            None
            if memory_initial is None
            else tuple(float(value) for value in memory_initial)
        )
        self.vocabulary = list(vocabulary or DEFAULT_PRIMITIVE_VOCABULARY)

    def run(self) -> Episode:
        return primitive_episode_from_sequence(
            source_name=self.name,
            sequence=self.sequence,
            vocabulary=self.vocabulary,
            eta=self.eta,
            memory_initial=self.memory_initial,
        )


PrimitiveEpisodeSource = PrimitiveProducer


def rollout_vectors(
    inputs: Sequence[Sequence[float]],
    *,
    eta: float = 0.5,
    memory_initial: Sequence[float] | None = None,
) -> list[PrimitiveVectorStep]:
    """Run the primitive recurrence on arbitrary fixed-length vectors."""

    if eta < 0.0:
        raise ValueError("eta must be non-negative")
    if not inputs:
        raise ValueError("rollout requires at least one input vector")
    actuals = [np.array(values, dtype=float) for values in inputs]
    dimensions = actuals[0].shape
    if len(dimensions) != 1:
        raise ValueError("input vectors must be one-dimensional")
    for actual in actuals:
        if actual.shape != dimensions:
            raise ValueError("all input vectors must have the same length")
    if memory_initial is None:
        memory_previous = np.zeros(dimensions, dtype=float)
    else:
        memory_previous = np.array(memory_initial, dtype=float)
        if memory_previous.shape != dimensions:
            raise ValueError("memory_initial must match input vector length")

    rows: list[PrimitiveVectorStep] = []
    for index, actual in enumerate(actuals, start=1):
        expected = memory_previous
        error = actual - expected
        memory = memory_previous + eta * error
        rows.append(
            PrimitiveVectorStep(
                t=index,
                actual=_tuple(actual),
                expected=_tuple(expected),
                error=_tuple(error),
                surprise=float(np.linalg.norm(error)),
                memory_previous=_tuple(memory_previous),
                memory=_tuple(memory),
            )
        )
        memory_previous = memory
    return rows


def rollout_1d(
    inputs: Sequence[float] = (0.10, 0.20, 0.25, 0.80, 0.75, 0.30),
    *,
    eta: float = 0.5,
    memory_initial: float = 0.0,
) -> list[PrimitiveStep]:
    """Run the primitive scalar recurrence."""

    vector_rows = rollout_vectors(
        [(value,) for value in inputs],
        eta=eta,
        memory_initial=(memory_initial,),
    )
    return [
        PrimitiveStep(
            t=row.t,
            actual=row.actual[0],
            expected=row.expected[0],
            error=row.error[0],
            surprise=abs(row.error[0]),
            memory_previous=row.memory_previous[0],
            memory=row.memory[0],
        )
        for row in vector_rows
    ]


def rollout_2d(
    inputs: Sequence[tuple[float, float]] = (
        (0.10, 0.18),
        (0.22, 0.25),
        (0.28, 0.30),
        (0.82, 0.74),
        (0.76, 0.69),
        (0.34, 0.35),
    ),
    *,
    eta: float = 0.5,
    memory_initial: tuple[float, float] = (0.0, 0.0),
) -> list[PrimitiveStep2D]:
    """Run the primitive recurrence in a two-feature plane."""

    vector_rows = rollout_vectors(inputs, eta=eta, memory_initial=memory_initial)
    return [
        PrimitiveStep2D(
            t=row.t,
            actual=(row.actual[0], row.actual[1]),
            expected=(row.expected[0], row.expected[1]),
            error=(row.error[0], row.error[1]),
            surprise=row.surprise,
            memory_previous=(row.memory_previous[0], row.memory_previous[1]),
            memory=(row.memory[0], row.memory[1]),
        )
        for row in vector_rows
    ]


def primitive_episode_from_sequence(
    *,
    source_name: str,
    sequence: InputSequence,
    vocabulary: Sequence[str] | None = None,
    eta: float = PRIMITIVE_ETA,
    memory_initial: Sequence[float] | None = PRIMITIVE_MEMORY_INITIAL,
) -> Episode:
    if eta < 0.0:
        raise ValueError("eta must be non-negative")
    vocabulary = list(vocabulary or DEFAULT_PRIMITIVE_VOCABULARY)
    _validate_vocabulary(vocabulary)
    actuals = [
        obj.features.to_array(vocabulary) * obj.salience
        for obj in sequence.objects
    ]
    rows = rollout_vectors(actuals, eta=eta, memory_initial=memory_initial)
    inputs = [
        EpisodeInput(
            id=obj.id,
            kind=obj.kind,
            start=obj.temporal_extent.start,
            end=obj.temporal_extent.end,
            order_index=obj.temporal_extent.order_index,
            features=obj.features.to_array(vocabulary),
            modality=obj.modality,
            salience=obj.salience,
            learning_weight=obj.learning_weight,
            presentation=presentation_for_object(obj),
            metadata=dict(obj.metadata),
        )
        for obj in sequence.objects
    ]
    observations = [
        EpisodeObservation(
            t=obj.temporal_extent.center,
            t_normalized=_normalized_time(obj.temporal_extent.center, sequence.duration),
            expected=np.array(row.expected, dtype=float),
            actual=np.array(row.actual, dtype=float),
            memory_state=np.array(row.memory, dtype=float),
            surprise=row.surprise,
            learning_rate=eta * obj.learning_weight,
            attention=obj.salience,
            attention_weights={obj.id: obj.salience * obj.learning_weight},
            active_inputs=[obj.id],
            input_features={obj.id: obj.features.to_array(vocabulary)},
            metadata={
                "source_object_id": obj.id,
                "object_label": obj.metadata.get("label", obj.kind),
                "primitive_eta": eta,
                "memory_previous": list(row.memory_previous),
                "error": list(row.error),
            },
        )
        for obj, row in zip(sequence.objects, rows, strict=True)
    ]
    return Episode(
        source_name=source_name,
        vocabulary=vocabulary,
        inputs=inputs,
        observations=observations,
        duration=sequence.duration,
        metadata={
            "source": "primitive.recurrence",
            "adapter": "PrimitiveProducer",
            "eta": eta,
            "memory_initial": None if memory_initial is None else list(memory_initial),
            "topology_params": SubjectiveTopologyParams(
                feature_x=vocabulary[0],
                feature_y=vocabulary[min(1, len(vocabulary) - 1)],
                prior=SubjectiveTopologyPrior(),
            ),
        },
    )


def primitive_input_sequence(
    episode: Sequence[str] = PRIMITIVE_EPISODE,
    *,
    feature_names: Sequence[str] | None = None,
    start: float = 0.0,
    step_duration: float = 1.0,
    gap: float = 0.0,
) -> InputSequence:
    if step_duration <= 0.0:
        raise ValueError("step_duration must be positive")
    if gap < 0.0:
        raise ValueError("gap must be non-negative")
    feature_names = list(feature_names or DEFAULT_PRIMITIVE_VOCABULARY)
    _validate_world_feature_names(feature_names)

    objects: list[ExperienceObject] = []
    tree_count = 0
    for index, object_id in enumerate(episode):
        if object_id not in PRIMITIVE_OBJECTS:
            raise ValueError(f"unknown primitive object id: {object_id}")
        world_object = PRIMITIVE_OBJECTS[object_id]
        if object_id == "tree":
            features = primitive_tree_variant(tree_count).features
            tree_count += 1
        else:
            features = world_object.features
        object_start = start + index * (step_duration + gap)
        objects.append(
            ExperienceObject(
                id=f"primitive_{index:03d}_{object_id}",
                temporal_extent=TemporalExtent(
                    start=object_start,
                    end=object_start + step_duration,
                    order_index=index,
                ),
                features=FeatureVector(
                    {
                        feature_names[0]: features[0],
                        feature_names[1]: features[1],
                    }
                ),
                kind=object_id,
                modality="primitive",
                metadata={
                    "object_id": object_id,
                    "label": world_object.label,
                    "sprite": world_object.sprite,
                    "value": world_object.value,
                },
            )
        )
    return InputSequence(objects)


def primitive_episode_features(
    episode: Sequence[str] = PRIMITIVE_EPISODE,
) -> list[tuple[float, float]]:
    features: list[tuple[float, float]] = []
    tree_count = 0
    for object_id in episode:
        if object_id not in PRIMITIVE_OBJECTS:
            raise ValueError(f"unknown primitive object id: {object_id}")
        if object_id == "tree":
            features.append(primitive_tree_variant(tree_count).features)
            tree_count += 1
        else:
            features.append(PRIMITIVE_OBJECTS[object_id].features)
    return features


def primitive_random_episode(
    rng: np.random.Generator,
    length: int,
    *,
    weights: Mapping[str, float] | None = None,
) -> tuple[str, ...]:
    if length < 1:
        raise ValueError("length must be at least 1")
    weights = dict(weights or DEFAULT_PRIMITIVE_WEIGHTS)
    object_ids = tuple(PRIMITIVE_OBJECTS.keys())
    probability = np.array(
        [float(weights.get(object_id, 0.0)) for object_id in object_ids],
        dtype=float,
    )
    if np.any(probability < 0.0) or float(probability.sum()) <= 0.0:
        raise ValueError("weights must contain at least one positive non-negative value")
    probability /= probability.sum()
    return tuple(str(rng.choice(object_ids, p=probability)) for _ in range(length))


def primitive_jittered_features(
    episode: Sequence[str],
    rng: np.random.Generator,
    jitter: float,
) -> list[tuple[float, float]]:
    if jitter < 0.0:
        raise ValueError("jitter must be non-negative")
    features: list[tuple[float, float]] = []
    tree_count = 0
    for object_id in episode:
        if object_id not in PRIMITIVE_OBJECTS:
            raise ValueError(f"unknown primitive object id: {object_id}")
        if object_id == "tree":
            base = primitive_tree_variant(tree_count).features
            tree_count += 1
        else:
            base = PRIMITIVE_OBJECTS[object_id].features
        x = float(np.clip(base[0] + rng.normal(0.0, jitter), 0.0, 1.0))
        y = float(np.clip(base[1] + rng.normal(0.0, jitter), 0.0, 1.0))
        features.append((x, y))
    return features


def primitive_random_input_sequence(
    *,
    seed: int,
    length: int,
    jitter: float = 0.03,
    feature_names: Sequence[str] | None = None,
    weights: Mapping[str, float] | None = None,
) -> InputSequence:
    rng = np.random.default_rng(seed)
    episode = primitive_random_episode(rng, length, weights=weights)
    features = primitive_jittered_features(episode, rng, jitter)
    feature_names = list(feature_names or DEFAULT_PRIMITIVE_VOCABULARY)
    _validate_world_feature_names(feature_names)
    objects = []
    for index, (object_id, values) in enumerate(zip(episode, features, strict=True)):
        world_object = PRIMITIVE_OBJECTS[object_id]
        objects.append(
            ExperienceObject(
                id=f"primitive_{index:03d}_{object_id}",
                temporal_extent=TemporalExtent(float(index), float(index + 1), index),
                features=FeatureVector({feature_names[0]: values[0], feature_names[1]: values[1]}),
                kind=object_id,
                modality="primitive",
                metadata={
                    "object_id": object_id,
                    "label": world_object.label,
                    "sprite": world_object.sprite,
                    "value": world_object.value,
                    "seed": seed,
                    "jitter": jitter,
                },
            )
        )
    return InputSequence(objects)


def primitive_prototype_features() -> dict[str, tuple[float, float]]:
    return {object_id: obj.features for object_id, obj in PRIMITIVE_OBJECTS.items()}


def primitive_tree_variant(occurrence_index: int) -> PrimitiveTreeVariant:
    return PRIMITIVE_TREE_VARIANTS[occurrence_index % len(PRIMITIVE_TREE_VARIANTS)]


def nearest_prototype(
    vector: Sequence[float],
    prototypes: Mapping[str, Sequence[float]],
) -> tuple[str, float]:
    """Return the nearest prototype label and Euclidean distance."""

    point = np.array(vector, dtype=float)
    best_label = ""
    best_distance = float("inf")
    for label, values in prototypes.items():
        distance = float(np.linalg.norm(point - np.array(values, dtype=float)))
        if distance < best_distance:
            best_label = label
            best_distance = distance
    if not best_label:
        raise ValueError("at least one prototype is required")
    return best_label, best_distance


def _normalized_time(t: float, duration: float) -> float:
    if duration <= 0.0:
        return 0.0
    return min(1.0, max(0.0, t / duration))


def _validate_vocabulary(vocabulary: Sequence[str]) -> None:
    if not vocabulary:
        raise ValueError("vocabulary must include at least one feature")


def _validate_world_feature_names(feature_names: Sequence[str]) -> None:
    if len(feature_names) != 2:
        raise ValueError("primitive world objects require exactly two feature names")


def _tuple(values: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in values)
