from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from cave.demonstrations.reports.cave_matrices import (
    initial_conditions_matrix_report_spec,
    population_clusters_matrix_report_spec,
    subject_ablation_matrix_report_spec,
)
from cave.demonstrations.examples import default_model_params
from cave.demonstrations.scenarios.topology_atlas import topology_atlas_params
from cave.observation.episode_runs import EpisodeSet, LabeledEpisode
from cave.presentation.renderers.episode_set_dashboard import (
    save_episode_set_dashboard,
    save_episode_set_distances_json,
)
from cave.presentation.renderers.topology_atlas_renderer import (
    save_topology_atlas,
    save_topology_atlas_metrics,
    topology_atlas_results,
)
from cave.presentation.renderers.topology_population_renderer import (
    save_topology_population_animation,
    save_topology_population_dashboard,
    save_topology_scatter_migration,
)
from cave.demonstrations.subjects.dashboard import classical_mds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render matrix runs as top-down topology atlas plus population geometry."
    )
    parser.add_argument(
        "matrix",
        nargs="?",
        choices=("initial-conditions", "subject-ablation", "population-clusters"),
        default="subject-ablation",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results/cave/topology-comparison"))
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument(
        "--starts",
        type=int,
        default=None,
        help="Initial start conditions for the initial-conditions matrix.",
    )
    parser.add_argument(
        "--treatments",
        type=int,
        default=1,
        help="Generated treatment sequences for the initial-conditions matrix.",
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--end", type=float, default=3.0)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument(
        "--projection",
        choices=("native", "atlas"),
        default="native",
        help="Topology projection to use for the top-down atlas.",
    )
    args = parser.parse_args()

    if args.matrix == "subject-ablation":
        spec = subject_ablation_matrix_report_spec(
            event_count=args.count,
            seed=args.seed,
            dt=args.dt,
            end=args.end,
            samples=args.samples,
        )
    elif args.matrix == "population-clusters":
        spec = population_clusters_matrix_report_spec(
            sequence_count=args.sequences,
            event_count=args.count,
            seed=args.seed,
            dt=args.dt,
            end=args.end,
            samples=args.samples,
        )
    else:
        start_count = args.sequences if args.starts is None else args.starts
        spec = initial_conditions_matrix_report_spec(
            condition_count=start_count,
            treatment_count=args.treatments,
            event_count=args.count,
            seed=args.seed,
            dt=args.dt,
            end=args.end,
            samples=args.samples,
        )

    records = list(spec.run_factory())
    population_records = tuple(record.to_population_record() for record in records)
    episode_set = _episode_set_from_records(records, title=f"{spec.title}: Topology Comparison")
    params = default_model_params().topology if args.projection == "native" else topology_atlas_params()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    atlas_png = output_dir / "topology_atlas.png"
    atlas_metrics_json = output_dir / "topology_atlas_metrics.json"
    population_png = output_dir / "population_dashboard.png"
    topology_population_png = output_dir / "topology_population.png"
    topology_delta_population_png = output_dir / "topology_delta_population.png"
    topology_migration_gif = output_dir / "topology_migration.gif"
    topology_scatter_migration_gif = output_dir / "topology_scatter_migration.gif"
    topology_subjective_trajectory_gif = output_dir / "topology_subjective_trajectory.gif"
    topology_distances_json = output_dir / "topology_distances.json"
    distances_json = output_dir / "episode_set_distances.json"

    save_topology_atlas(episode_set, atlas_png, params)
    save_topology_atlas_metrics(episode_set, atlas_metrics_json, params)
    save_episode_set_dashboard(
        episode_set,
        population_png,
        samples=args.samples,
        title=f"{spec.title}: Population Geometry",
    )
    save_episode_set_distances_json(
        episode_set,
        distances_json,
        samples=args.samples,
    )
    _save_topology_population(
        episode_set,
        params,
        topology_delta_population_png,
        topology_distances_json,
    )
    save_topology_population_dashboard(
        population_records,
        topology_population_png,
        params,
        title=(
            f"{spec.title}: Treatments x Starts"
            if args.matrix == "initial-conditions" and args.treatments > 1
            else f"{spec.title}: Same Treatment, Different Starts"
        ),
    )
    save_topology_population_animation(
        population_records,
        topology_migration_gif,
        params,
        title=f"{spec.title}: Topology Migration",
        fps=args.fps,
    )
    save_topology_scatter_migration(
        population_records,
        topology_scatter_migration_gif,
        params,
        title=f"{spec.title}: Topology Scatter Migration",
        color_factor="start_condition",
        marker_factor="condition",
        point_kind="centroid",
        fps=args.fps,
    )
    save_topology_scatter_migration(
        population_records,
        topology_subjective_trajectory_gif,
        params,
        title=f"{spec.title}: Subjective State Trajectories",
        color_factor=(
            "treatment"
            if args.matrix == "initial-conditions" and args.treatments > 1
            else "start_condition"
        ),
        marker_factor="condition",
        point_kind="subjective",
        fps=args.fps,
    )

    print(f"wrote {atlas_png}")
    print(f"wrote {population_png}")
    print(f"wrote {topology_population_png}")
    print(f"wrote {topology_delta_population_png}")
    print(f"wrote {topology_migration_gif}")
    print(f"wrote {topology_scatter_migration_gif}")
    print(f"wrote {topology_subjective_trajectory_gif}")
    print(f"wrote {atlas_metrics_json}")
    print(f"wrote {topology_distances_json}")
    print(f"wrote {distances_json}")


def _episode_set_from_records(records, *, title: str) -> EpisodeSet:
    entries = []
    for record in records:
        labeled = record.to_population_record().to_labeled_episode(
            group_factor="condition",
            series_factor="start_condition",
        )
        entries.append(
            LabeledEpisode(
                id=labeled.id,
                label=_compact_label(record),
                episode=labeled.episode,
                group=labeled.group,
                series=labeled.series,
                metadata=labeled.metadata,
            )
        )
    entries = tuple(entries)
    return EpisodeSet(
        id="matrix_topology_comparison",
        title=title,
        comparison_axis="matrix cell",
        episodes=entries,
        metadata={"source": "cave.demonstrations.reports.topology_comparison"},
    )


def _compact_label(record) -> str:
    return f"{record.sequence_id} / {record.subject_id} / {record.variant_id}"


def _save_topology_population(
    episode_set: EpisodeSet,
    params,
    output: Path,
    distances_output: Path,
) -> None:
    results = topology_atlas_results(episode_set, params)
    labels = [result.label for result in results]
    embeddings = [result.experienced_delta.ravel() for result in results]
    distances = _distance_matrix(embeddings)
    coords = classical_mds(distances)

    distances_output.write_text(
        json.dumps(
            {
                "id": "topology_distances",
                "embedding": "experienced_topology_delta",
                "labels": labels,
                "distances": distances.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    figure = plt.figure(figsize=(14.5, 6.4), facecolor="#f7f4ef")
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 1.08),
        left=0.065,
        right=0.985,
        top=0.84,
        bottom=0.24,
        wspace=0.28,
    )
    matrix_axis = figure.add_subplot(grid[0, 0])
    image = matrix_axis.imshow(distances, cmap="magma", interpolation="nearest")
    matrix_axis.set_title("Topology Delta Distance", loc="left", fontsize=12, fontweight="bold")
    matrix_axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    matrix_axis.set_yticks(range(len(labels)), labels, fontsize=7)
    figure.colorbar(image, ax=matrix_axis, fraction=0.046, pad=0.04)

    scatter_axis = figure.add_subplot(grid[0, 1])
    groups = [episode.group or "episode" for episode in episode_set.episodes]
    series = [episode.series or "episode" for episode in episode_set.episodes]
    colors = {
        group: plt.get_cmap("tab10")(index % 10)
        for index, group in enumerate(sorted(set(groups)))
    }
    marker_cycle = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
    markers = {
        name: marker_cycle[index % len(marker_cycle)]
        for index, name in enumerate(sorted(set(series)))
    }
    for index, label in enumerate(labels):
        scatter_axis.scatter(
            coords[index, 0],
            coords[index, 1],
            color=colors[groups[index]],
            marker=markers[series[index]],
            s=62,
            edgecolor="#111827",
            linewidth=0.4,
            alpha=0.88,
        )
    scatter_axis.axhline(0.0, color="#d1d5db", linewidth=0.8)
    scatter_axis.axvline(0.0, color="#d1d5db", linewidth=0.8)
    scatter_axis.set_title("MDS: Topology Delta", loc="left", fontsize=12, fontweight="bold")
    scatter_axis.set_xlabel("component 1")
    scatter_axis.set_ylabel("component 2")
    group_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=group,
            markerfacecolor=color,
            markeredgecolor="#111827",
            markersize=7,
        )
        for group, color in colors.items()
    ]
    series_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="#111827",
            label=series_name,
            linestyle="None",
            markersize=7,
        )
        for series_name, marker in markers.items()
    ]
    scatter_axis.legend(
        handles=group_handles + series_handles,
        loc="upper right",
        fontsize=7,
        frameon=False,
        ncol=2,
    )

    figure.text(
        0.065,
        0.95,
        f"{episode_set.title}: Topology Population",
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="#111827",
    )
    figure.text(
        0.065,
        0.905,
        f"Embedding: final experienced topology delta; projection: {params.feature_x.name} x {params.feature_y.name}",
        ha="left",
        va="top",
        fontsize=9.5,
        color="#344054",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight", pad_inches=0.14)
    plt.close(figure)


def _distance_matrix(embeddings: list[np.ndarray]) -> np.ndarray:
    count = len(embeddings)
    distances = np.zeros((count, count), dtype=float)
    for i in range(count):
        for j in range(i + 1, count):
            distance = float(np.linalg.norm(embeddings[i] - embeddings[j]) / np.sqrt(max(1, embeddings[i].size)))
            distances[i, j] = distance
            distances[j, i] = distance
    return distances


if __name__ == "__main__":
    main()
