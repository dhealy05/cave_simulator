from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from cave.demonstrations.subjects.distances import EmbeddingFn, pairwise_distance_matrix
from cave.demonstrations.subjects.runs import SubjectRun


RunT = TypeVar("RunT")


def save_subject_comparison_dashboard(
    runs: Sequence[SubjectRun],
    labels: Sequence[str],
    output: str | Path,
    *,
    effect_embedding: EmbeddingFn,
    observed_embedding: EmbeddingFn,
    internal_embedding: EmbeddingFn | None = None,
    title: str = "Subject Comparison",
    dpi: int = 150,
) -> None:
    if len(runs) == 0:
        raise ValueError("at least one run is required")
    if len(runs) != len(labels):
        raise ValueError("runs and labels must have the same length")
    save_episode_comparison_dashboard(
        [run.as_labeled_episode(label) for run, label in zip(runs, labels)],
        output,
        effect_embedding=lambda item: effect_embedding(_run_from_labeled(item, runs)),
        observed_embedding=lambda item: observed_embedding(_run_from_labeled(item, runs)),
        internal_embedding=(
            None
            if internal_embedding is None
            else lambda item: internal_embedding(_run_from_labeled(item, runs))
        ),
        title=title,
        dpi=dpi,
    )


def save_episode_comparison_dashboard(
    runs: Sequence[RunT],
    output: str | Path,
    *,
    effect_embedding: EmbeddingFn[RunT],
    observed_embedding: EmbeddingFn[RunT],
    internal_embedding: EmbeddingFn[RunT] | None = None,
    title: str = "Episode Comparison",
    dpi: int = 150,
) -> None:
    if len(runs) == 0:
        raise ValueError("at least one run is required")

    effect_distances = pairwise_distance_matrix(runs, effect_embedding)
    observed_distances = pairwise_distance_matrix(runs, observed_embedding)
    effect_coords = classical_mds(effect_distances)
    observed_coords = classical_mds(observed_distances)
    internal_distances = (
        None
        if internal_embedding is None
        else pairwise_distance_matrix(runs, internal_embedding)
    )
    internal_coords = (
        None
        if internal_distances is None
        else classical_mds(internal_distances)
    )
    labels = [_display_label(run) for run in runs]

    columns = 2 if internal_embedding is None else 3
    figure = plt.figure(figsize=(9.0 * columns, 12.0))
    grid = figure.add_gridspec(2, columns, height_ratios=[1.2, 1.0])
    axes = [
        figure.add_subplot(grid[row, column])
        for row in range(2)
        for column in range(columns)
    ]

    draw_distance_matrix(
        axes[0],
        effect_distances,
        labels,
        "State-effect distance",
        figure,
    )
    draw_distance_matrix(
        axes[1],
        observed_distances,
        labels,
        "Observed memory distance",
        figure,
    )
    if internal_distances is not None and internal_coords is not None:
        draw_distance_matrix(
            axes[2],
            internal_distances,
            labels,
            "Subjective trajectory distance",
            figure,
        )
        scatter_axes = axes[3:6]
    else:
        scatter_axes = axes[2:4]

    draw_mds_scatter(scatter_axes[0], runs, labels, effect_coords, "MDS: state effect")
    draw_mds_scatter(scatter_axes[1], runs, labels, observed_coords, "MDS: observed memory")
    if internal_coords is not None and len(scatter_axes) > 2:
        draw_mds_scatter(
            scatter_axes[2],
            runs,
            labels,
            internal_coords,
            "MDS: subjective trajectory",
        )

    figure.suptitle(title, x=0.01, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def draw_distance_matrix(
    axis,
    distances: np.ndarray,
    labels: Sequence[str],
    title: str,
    figure,
) -> None:
    image = axis.imshow(distances, cmap="magma")
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=90, fontsize=7)
    axis.set_yticklabels(labels, fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def draw_mds_scatter(
    axis,
    runs: Sequence[object],
    labels: Sequence[str],
    coords: np.ndarray,
    title: str,
) -> None:
    subject_ids = sorted({_group_key(run) for run in runs})
    color_map = {
        subject_id: plt.get_cmap("tab10")(index % 10)
        for index, subject_id in enumerate(subject_ids)
    }
    sequence_keys = _sequence_keys(labels)
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    marker_map = {
        key: marker_cycle[index % len(marker_cycle)]
        for index, key in enumerate(sequence_keys)
    }

    for index, (run, label) in enumerate(zip(runs, labels)):
        axis.scatter(
            coords[index, 0],
            coords[index, 1],
            color=color_map[_group_key(run)],
            marker=marker_map[_label_sequence_key(label)],
            s=85,
            edgecolor="black",
            linewidth=0.4,
        )
        axis.text(
            coords[index, 0],
            coords[index, 1],
            label,
            fontsize=7,
            ha="left",
            va="bottom",
        )

    axis.axhline(0, color="#cccccc", linewidth=0.8)
    axis.axvline(0, color="#cccccc", linewidth=0.8)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlabel("component 1")
    axis.set_ylabel("component 2")


def classical_mds(distances: np.ndarray, dimensions: int = 2) -> np.ndarray:
    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distances must be a square matrix")
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")
    count = distances.shape[0]
    centering = np.eye(count) - np.ones((count, count)) / count
    gram = -0.5 * centering @ (distances**2) @ centering
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    positive = np.maximum(eigvals[:dimensions], 0.0)
    coords = eigvecs[:, :dimensions] * np.sqrt(positive)
    if coords.shape[1] < dimensions:
        padding = np.zeros((count, dimensions - coords.shape[1]), dtype=float)
        coords = np.concatenate([coords, padding], axis=1)
    return coords


def _sequence_keys(labels: Sequence[str]) -> list[str]:
    keys = []
    for label in labels:
        key = _label_sequence_key(label)
        if key not in keys:
            keys.append(key)
    return keys


def _label_sequence_key(label: str) -> str:
    if "-" in label:
        return label.split("-", 1)[0]
    return label


def _display_label(run: object) -> str:
    value = getattr(run, "display_label", None)
    if isinstance(value, str):
        return value
    label = getattr(run, "label", None)
    if isinstance(label, str):
        return label
    run_id = getattr(run, "id", None)
    if isinstance(run_id, str):
        return run_id
    return str(run)


def _group_key(run: object) -> str:
    group = getattr(run, "group", None)
    if isinstance(group, str):
        return group
    subject = getattr(run, "subject", None)
    subject_id = getattr(subject, "id", None)
    if isinstance(subject_id, str):
        return subject_id
    source_name = getattr(getattr(run, "episode", None), "source_name", None)
    if isinstance(source_name, str):
        return source_name
    return "episode"


def _run_from_labeled(item: object, runs: Sequence[SubjectRun]) -> SubjectRun:
    item_id = getattr(item, "id", None)
    for run in runs:
        if run.id == item_id:
            return run
    raise ValueError(f"no SubjectRun found for labeled episode {item_id!r}")
