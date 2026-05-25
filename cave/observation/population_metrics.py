from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, TypeVar

import numpy as np


Record = TypeVar("Record")
EmbeddingFn = Callable[[Record], np.ndarray]
LabelFn = Callable[[Record, str], str | None]


@dataclass(frozen=True)
class FactorSeparation:
    within_mean: float
    between_mean: float
    margin: float
    ratio: float
    within_count: int
    between_count: int

    def payload(self) -> dict[str, float | int]:
        return {
            "within_mean": self.within_mean,
            "between_mean": self.between_mean,
            "margin": self.margin,
            "ratio": self.ratio,
            "within_count": self.within_count,
            "between_count": self.between_count,
        }


def population_geometry_metrics(
    records: Sequence[Record],
    embeddings: Mapping[str, EmbeddingFn[Record]],
    *,
    label_fn: LabelFn[Record],
    treatment_factor: str = "treatment",
    start_factor: str = "start_condition",
    permutations: int = 199,
    seed: int = 0,
) -> dict[str, object]:
    if not records:
        raise ValueError("population geometry metrics require at least one record")
    treatment_labels = _factor_labels(records, treatment_factor, label_fn)
    start_labels = _factor_labels(records, start_factor, label_fn)
    treatment_levels = sorted(set(treatment_labels))
    start_levels = sorted(set(start_labels))
    chance_treatment = 1.0 / len(treatment_levels) if treatment_levels else 0.0
    chance_start = 1.0 / len(start_levels) if start_levels else 0.0

    by_embedding = {}
    for name, embedding_fn in embeddings.items():
        matrix = _embedding_matrix(records, embedding_fn)
        distances = _distance_matrix(matrix)
        treatment = _factor_separation(distances, treatment_labels)
        start = _factor_separation(distances, start_labels)
        treatment_decoding = _nearest_centroid_accuracy(
            matrix,
            target_labels=treatment_labels,
            holdout_labels=start_labels,
        )
        start_decoding = _nearest_centroid_accuracy(
            matrix,
            target_labels=start_labels,
            holdout_labels=treatment_labels,
        )
        matched_treatment = _matched_factor_distance(
            distances,
            group_labels=start_labels,
            compared_labels=treatment_labels,
        )
        matched_start = _matched_factor_distance(
            distances,
            group_labels=treatment_labels,
            compared_labels=start_labels,
        )
        null = _permutation_null(
            distances,
            treatment_labels=treatment_labels,
            start_labels=start_labels,
            observed_margin=treatment.margin,
            permutations=permutations,
            seed=seed,
        )
        by_embedding[name] = {
            "treatment": treatment.payload(),
            "start": start.payload(),
            "matched_start_treatment_distance": matched_treatment,
            "same_treatment_start_distance": matched_start,
            "treatment_decoding": {
                "accuracy": treatment_decoding,
                "chance": chance_treatment,
                "lift": treatment_decoding - chance_treatment,
                "held_out_factor": start_factor,
            },
            "start_decoding": {
                "accuracy": start_decoding,
                "chance": chance_start,
                "lift": start_decoding - chance_start,
                "held_out_factor": treatment_factor,
            },
            "permutation_null": null,
        }

    return {
        "run_count": len(records),
        "treatment_count": len(treatment_levels),
        "start_count": len(start_levels),
        "treatments": treatment_levels,
        "starts": start_levels,
        "permutations": permutations,
        "embedding_metrics": by_embedding,
    }


def _factor_labels(
    records: Sequence[Record],
    factor: str,
    label_fn: LabelFn[Record],
) -> list[str]:
    labels = []
    for record in records:
        label = label_fn(record, factor)
        if label is None:
            raise ValueError(f"record is missing factor {factor!r}")
        labels.append(label)
    return labels


def _embedding_matrix(
    records: Sequence[Record],
    embedding_fn: EmbeddingFn[Record],
) -> np.ndarray:
    flattened = [np.asarray(embedding_fn(record), dtype=float).ravel() for record in records]
    width = max((item.size for item in flattened), default=0)
    matrix = np.zeros((len(flattened), width), dtype=float)
    for index, item in enumerate(flattened):
        matrix[index, : item.size] = item
    return matrix


def _distance_matrix(matrix: np.ndarray) -> np.ndarray:
    count = matrix.shape[0]
    distances = np.zeros((count, count), dtype=float)
    if count <= 1:
        return distances
    normalizer = np.sqrt(max(1, matrix.shape[1]))
    for i in range(count):
        delta = matrix[i + 1 :] - matrix[i]
        values = np.linalg.norm(delta, axis=1) / normalizer
        distances[i, i + 1 :] = values
        distances[i + 1 :, i] = values
    return distances


def _factor_separation(
    distances: np.ndarray,
    labels: Sequence[str],
) -> FactorSeparation:
    within = []
    between = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                within.append(float(distances[i, j]))
            else:
                between.append(float(distances[i, j]))
    within_mean = float(np.mean(within)) if within else 0.0
    between_mean = float(np.mean(between)) if between else 0.0
    return FactorSeparation(
        within_mean=within_mean,
        between_mean=between_mean,
        margin=between_mean - within_mean,
        ratio=_safe_ratio(between_mean, within_mean),
        within_count=len(within),
        between_count=len(between),
    )


def _matched_factor_distance(
    distances: np.ndarray,
    *,
    group_labels: Sequence[str],
    compared_labels: Sequence[str],
) -> dict[str, float | int]:
    values = []
    for i in range(len(group_labels)):
        for j in range(i + 1, len(group_labels)):
            if group_labels[i] == group_labels[j] and compared_labels[i] != compared_labels[j]:
                values.append(float(distances[i, j]))
    return {
        "mean": float(np.mean(values)) if values else 0.0,
        "count": len(values),
    }


def _nearest_centroid_accuracy(
    matrix: np.ndarray,
    *,
    target_labels: Sequence[str],
    holdout_labels: Sequence[str],
) -> float:
    if len(set(target_labels)) <= 1 or len(set(holdout_labels)) <= 1:
        return 0.0
    correct = 0
    scored = 0
    target_levels = sorted(set(target_labels))
    for holdout in sorted(set(holdout_labels)):
        train_indices = [i for i, label in enumerate(holdout_labels) if label != holdout]
        test_indices = [i for i, label in enumerate(holdout_labels) if label == holdout]
        centroids = {}
        for target in target_levels:
            selected = [i for i in train_indices if target_labels[i] == target]
            if selected:
                centroids[target] = np.mean(matrix[selected], axis=0)
        if len(centroids) <= 1:
            continue
        for index in test_indices:
            prediction = min(
                centroids,
                key=lambda target: float(np.linalg.norm(matrix[index] - centroids[target])),
            )
            correct += int(prediction == target_labels[index])
            scored += 1
    return float(correct / scored) if scored else 0.0


def _permutation_null(
    distances: np.ndarray,
    *,
    treatment_labels: Sequence[str],
    start_labels: Sequence[str],
    observed_margin: float,
    permutations: int,
    seed: int,
) -> dict[str, float | int]:
    if permutations <= 0 or len(set(treatment_labels)) <= 1 or len(set(start_labels)) <= 1:
        return {
            "observed_margin": observed_margin,
            "null_mean": 0.0,
            "null_std": 0.0,
            "p_value": 1.0,
            "permutations": 0,
        }
    rng = np.random.default_rng(seed)
    by_start: dict[str, list[int]] = {}
    for index, start in enumerate(start_labels):
        by_start.setdefault(start, []).append(index)
    null_margins = []
    for _ in range(permutations):
        shuffled = list(treatment_labels)
        for indices in by_start.values():
            values = [shuffled[index] for index in indices]
            rng.shuffle(values)
            for index, value in zip(indices, values):
                shuffled[index] = value
        null_margins.append(_factor_separation(distances, shuffled).margin)
    null_array = np.asarray(null_margins, dtype=float)
    exceed = int(np.sum(null_array >= observed_margin))
    p_value = float((exceed + 1) / (len(null_array) + 1))
    return {
        "observed_margin": observed_margin,
        "null_mean": float(np.mean(null_array)),
        "null_std": float(np.std(null_array)),
        "p_value": p_value,
        "permutations": int(len(null_array)),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return float("inf") if numerator > 0.0 else 0.0
    return float(numerator / denominator)
