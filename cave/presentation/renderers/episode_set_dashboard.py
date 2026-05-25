from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from cave.demonstrations.subjects.dashboard import (
    classical_mds,
    draw_distance_matrix,
    draw_mds_scatter,
)
from cave.observation.episode_runs import EpisodeSet, LabeledEpisode
from cave.observation.projections import encode_value
from cave.observation.structural import (
    actual_trajectory_embedding,
    memory_trajectory_embedding,
    state_effect_embedding,
    subjective_trajectory_embedding,
)


EmbeddingFn = Callable[[LabeledEpisode], np.ndarray]


def default_episode_set_embeddings(samples: int = 48) -> dict[str, EmbeddingFn]:
    return {
        "state_effect": lambda item: state_effect_embedding(item.episode, samples=samples),
        "observed_memory": lambda item: memory_trajectory_embedding(
            item.episode,
            samples=samples,
        ),
        "subjective_trajectory": lambda item: subjective_trajectory_embedding(
            item.episode,
            samples=samples,
        ),
        "actual_input": lambda item: actual_trajectory_embedding(
            item.episode,
            samples=samples,
        ),
    }


def episode_set_distance_payload(
    episode_set: EpisodeSet,
    *,
    samples: int = 48,
    embeddings: dict[str, EmbeddingFn] | None = None,
) -> dict[str, object]:
    embeddings = embeddings or default_episode_set_embeddings(samples=samples)
    labels = [episode.display_label for episode in episode_set.episodes]
    distances = {
        name: _distance_matrix(episode_set.episodes, embedding_fn).tolist()
        for name, embedding_fn in embeddings.items()
    }
    return {
        "id": "episode_set_distances",
        "episode_set": {
            "id": episode_set.id,
            "title": episode_set.title,
            "comparison_axis": episode_set.comparison_axis,
            "metadata": episode_set.metadata,
        },
        "samples": samples,
        "distance_normalization": "zero-padded flattened embeddings",
        "labels": labels,
        "distances": distances,
    }


def save_episode_set_distances_json(
    episode_set: EpisodeSet,
    output: str | Path,
    *,
    samples: int = 48,
    embeddings: dict[str, EmbeddingFn] | None = None,
) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = episode_set_distance_payload(
        episode_set,
        samples=samples,
        embeddings=embeddings,
    )
    output.write_text(
        json.dumps(encode_value(payload), indent=2) + "\n",
        encoding="utf-8",
    )


def save_episode_set_dashboard(
    episode_set: EpisodeSet,
    output: str | Path,
    *,
    samples: int = 48,
    embeddings: dict[str, EmbeddingFn] | None = None,
    title: str | None = None,
    dpi: int = 150,
) -> None:
    embeddings = embeddings or default_episode_set_embeddings(samples=samples)
    selected = {
        key: embeddings[key]
        for key in ("state_effect", "observed_memory", "subjective_trajectory")
        if key in embeddings
    }
    if not selected:
        raise ValueError("at least one embedding is required")

    labels = [episode.display_label for episode in episode_set.episodes]
    distance_by_name = {
        name: _distance_matrix(episode_set.episodes, embedding_fn)
        for name, embedding_fn in selected.items()
    }
    coord_by_name = {
        name: classical_mds(distances)
        for name, distances in distance_by_name.items()
    }

    count = len(selected)
    figure = plt.figure(
        figsize=(5.8 * count, 8.2),
        facecolor="#f7f4ef",
    )
    grid = figure.add_gridspec(
        2,
        count,
        height_ratios=(1.1, 1.0),
        left=0.055,
        right=0.985,
        top=0.88,
        bottom=0.065,
        wspace=0.28,
        hspace=0.32,
    )
    for col, (name, distances) in enumerate(distance_by_name.items()):
        matrix_axis = figure.add_subplot(grid[0, col])
        draw_distance_matrix(
            matrix_axis,
            distances,
            labels,
            _embedding_title(name),
            figure,
        )
        scatter_axis = figure.add_subplot(grid[1, col])
        draw_mds_scatter(
            scatter_axis,
            episode_set.episodes,
            labels,
            coord_by_name[name],
            f"MDS: {_embedding_title(name)}",
        )

    figure.text(
        0.055,
        0.965,
        title or episode_set.title or "Episode Set Dashboard",
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="#111827",
    )
    axis_label = episode_set.comparison_axis or "episode"
    figure.text(
        0.055,
        0.925,
        f"Comparison axis: {axis_label}; samples: {samples}",
        ha="left",
        va="top",
        fontsize=9.5,
        color="#344054",
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.14)
    plt.close(figure)


def _distance_matrix(
    episodes: Sequence[LabeledEpisode],
    embedding_fn: EmbeddingFn,
) -> np.ndarray:
    embeddings = [embedding_fn(episode) for episode in episodes]
    count = len(embeddings)
    distances = np.zeros((count, count), dtype=float)
    for i in range(count):
        for j in range(i + 1, count):
            distance = _aligned_embedding_distance(embeddings[i], embeddings[j])
            distances[i, j] = distance
            distances[j, i] = distance
    return distances


def _aligned_embedding_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = np.asarray(a, dtype=float).ravel()
    b_flat = np.asarray(b, dtype=float).ravel()
    size = max(a_flat.size, b_flat.size)
    if size == 0:
        return 0.0
    a_aligned = np.zeros(size, dtype=float)
    b_aligned = np.zeros(size, dtype=float)
    a_aligned[: a_flat.size] = a_flat
    b_aligned[: b_flat.size] = b_flat
    return float(np.linalg.norm(a_aligned - b_aligned) / np.sqrt(size))


def _embedding_title(name: str) -> str:
    return name.replace("_", " ").title()
