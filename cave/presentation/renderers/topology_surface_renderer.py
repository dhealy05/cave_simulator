from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from cave.demonstrations.examples import demo_model, random_experience_model
from cave.observation.episodes import CaveProducer, Episode
from cave.observation.experience import FeatureAxis, FeatureProjection, feature_axis_label
from cave.observation.structural import EpisodeFrame, episode_frames, structural_state_for_episode
from cave.commitments.topology import SubjectiveTopologyState


EXPECTED_COLOR = "#0072B2"
ACTUAL_COLOR = "#009E73"
NEUTRAL_COLOR = "#E2E6EA"
EVENT_COLOR = "#D55E00"


@dataclass(frozen=True)
class FlattenedTopologyState:
    feature_x: FeatureAxis
    feature_y: FeatureAxis
    coordinates: np.ndarray
    density: np.ndarray
    expected_density: np.ndarray
    actual_density: np.ndarray


@dataclass(frozen=True)
class TopologySurface:
    times: np.ndarray
    feature_x: FeatureAxis
    feature_y: FeatureAxis
    coordinates: np.ndarray
    density: np.ndarray
    expected_density: np.ndarray
    actual_density: np.ndarray


def flatten_topology_state(
    topology: SubjectiveTopologyState,
    resolution: int | None = None,
) -> FlattenedTopologyState:
    grid_x, grid_y, density = topology.grid(resolution)
    _, _, expected_density = topology.expected_grid(resolution)
    _, _, actual_density = topology.actual_grid(resolution)
    coordinates = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
    return FlattenedTopologyState(
        feature_x=topology.feature_x,
        feature_y=topology.feature_y,
        coordinates=coordinates,
        density=density.reshape(-1),
        expected_density=expected_density.reshape(-1),
        actual_density=actual_density.reshape(-1),
    )


def topology_state_surface(
    episode: Episode,
    *,
    resolution: int = 36,
) -> TopologySurface:
    structural = structural_state_for_episode(episode)
    frames = episode_frames(episode, structural)
    return topology_state_surface_from_episode_frames(frames, resolution=resolution)


def topology_state_surface_from_episode_frames(
    frames: Sequence[EpisodeFrame],
    *,
    resolution: int = 36,
) -> TopologySurface:
    if not frames:
        raise ValueError("at least one frame is required")
    flattened = [
        flatten_topology_state(frame.topology_frame.topology, resolution)
        for frame in frames
    ]
    first = flattened[0]
    coordinates = first.coordinates
    return TopologySurface(
        times=np.array([frame.observation.t for frame in frames], dtype=float),
        feature_x=first.feature_x,
        feature_y=first.feature_y,
        coordinates=coordinates,
        density=np.stack([item.density for item in flattened], axis=0),
        expected_density=np.stack(
            [item.expected_density for item in flattened],
            axis=0,
        ),
        actual_density=np.stack(
            [item.actual_density for item in flattened],
            axis=0,
        ),
    )


def save_topology_state_surface(
    episode: Episode,
    output: str | Path,
    *,
    resolution: int = 36,
    dpi: int = 140,
) -> None:
    surface = topology_state_surface(episode, resolution=resolution)

    figure = plt.figure(figsize=(12.8, 7.2))
    axis = figure.add_axes((0.0, -0.02, 0.98, 1.0), projection="3d")
    draw_topology_state_surface(axis, surface)
    draw_3d_event_guides(axis, episode, surface.density.shape[1])

    figure.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def draw_topology_state_surface(axis: Axes, surface: TopologySurface) -> None:
    cell_indices = np.arange(surface.coordinates.shape[0], dtype=float)
    grid_x, grid_y = np.meshgrid(surface.times, cell_indices)
    grid_z = np.clip(surface.density.T, 0.0, 1.0)
    facecolors = _source_facecolors(
        grid_z,
        surface.expected_density.T,
        surface.actual_density.T,
    )

    axis.plot_surface(
        grid_x,
        grid_y,
        grid_z,
        facecolors=facecolors,
        vmin=0.0,
        vmax=1.0,
        alpha=0.9,
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
    )
    axis.plot_wireframe(
        grid_x,
        grid_y,
        grid_z,
        color="#1f2933",
        linewidth=0.12,
        alpha=0.08,
        rstride=max(1, cell_indices.size // 72),
        cstride=max(1, surface.times.size // 48),
    )

    _style_surface_axis(axis)
    axis.set_title(
        "Subjective Trajectory Surface",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=-2,
    )
    axis.set_xlabel("time")
    axis.set_ylabel("feature plane cell")
    axis.text2D(
        0.66,
        0.035,
        f"feature plane cells: {_feature_axis_summary(surface.feature_x)} x "
        f"{_feature_axis_summary(surface.feature_y)}",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )
    axis.set_zlabel("")
    axis.text2D(
        0.93,
        0.56,
        "state density",
        transform=axis.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=9,
    )
    axis.set_zlim(0.0, 1.16)
    axis.set_yticks(_coordinate_tick_indices(surface.coordinates))
    axis.set_yticklabels(_coordinate_tick_labels(surface.coordinates), fontsize=8)
    axis.view_init(elev=25, azim=-58)
    axis.set_box_aspect((3.25, 1.18, 0.72))
    _draw_source_legend(axis)


def draw_3d_event_guides(
    axis: Axes,
    episode: Episode,
    cell_count: int,
) -> None:
    y_min = -0.5
    y_max = cell_count - 0.5
    for obj in episode.inputs:
        start = obj.start
        end = obj.end
        axis.plot(
            [start, start],
            [y_min, y_max],
            [0.0, 0.0],
            color=EVENT_COLOR,
            linewidth=0.7,
            alpha=0.26,
        )
        axis.plot(
            [start, end],
            [y_min, y_min],
            [1.03, 1.03],
            color=EVENT_COLOR,
            linewidth=1.6,
            alpha=0.42,
        )


def _coordinate_tick_indices(coordinates: np.ndarray) -> np.ndarray:
    if coordinates.shape[0] <= 6:
        return np.arange(coordinates.shape[0], dtype=float)
    return np.linspace(0, coordinates.shape[0] - 1, 6, dtype=int).astype(float)


def _coordinate_tick_labels(coordinates: np.ndarray) -> list[str]:
    return [
        f"({coordinates[index, 0]:.1f}, {coordinates[index, 1]:.1f})"
        for index in _coordinate_tick_indices(coordinates).astype(int)
    ]


def _feature_axis_summary(axis: FeatureAxis) -> str:
    label = feature_axis_label(axis)
    if not isinstance(axis, FeatureProjection):
        return label
    keys = "/".join(axis.weights)
    return f"{label} ({keys})"


def _style_surface_axis(axis: Axes) -> None:
    axis.tick_params(axis="both", which="major", labelsize=8, pad=0)
    axis.tick_params(axis="z", which="major", labelsize=8, pad=1)
    for pane_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        pane_axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        pane_axis.pane.set_edgecolor((0.78, 0.80, 0.83, 0.55))
    for axis_name in ("xaxis", "yaxis", "zaxis"):
        info = getattr(axis, axis_name)._axinfo
        info["grid"]["color"] = (0.56, 0.59, 0.63, 0.25)
        info["grid"]["linewidth"] = 0.6


def _draw_source_legend(axis: Axes) -> None:
    handles = [
        Patch(facecolor=EXPECTED_COLOR, edgecolor="none", alpha=0.88, label="expected memory"),
        Patch(facecolor=ACTUAL_COLOR, edgecolor="none", alpha=0.88, label="lived input"),
        Patch(
            facecolor=EVENT_COLOR,
            edgecolor="none",
            alpha=0.45,
            label="event window",
        ),
    ]
    axis.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.9),
        frameon=False,
        fontsize=8,
    )


def _source_facecolors(
    density: np.ndarray,
    expected_density: np.ndarray,
    actual_density: np.ndarray,
) -> np.ndarray:
    density = np.clip(np.asarray(density, dtype=float), 0.0, 1.0)
    expected = np.clip(np.asarray(expected_density, dtype=float), 0.0, 1.0)
    actual = np.clip(np.asarray(actual_density, dtype=float), 0.0, 1.0)
    expected_color = np.array(to_rgba(EXPECTED_COLOR), dtype=float)
    actual_color = np.array(to_rgba(ACTUAL_COLOR), dtype=float)
    neutral_color = np.array(to_rgba(NEUTRAL_COLOR), dtype=float)
    highlight = np.array(to_rgba("#F8FAFC"), dtype=float)
    shadow = np.array(to_rgba("#263238"), dtype=float)
    source_mass = expected + actual
    actual_share = np.divide(
        actual,
        source_mass,
        out=np.zeros_like(actual, dtype=float),
        where=source_mass > 1e-12,
    )
    colors = (
        expected_color.reshape(1, 1, 4) * (1.0 - actual_share[..., None])
        + actual_color.reshape(1, 1, 4) * actual_share[..., None]
    )
    light = 0.18 + 0.82 * np.sqrt(density)
    shaded = shadow.reshape(1, 1, 4) * (1.0 - light[..., None]) + colors * light[..., None]
    shaded = 0.86 * shaded + 0.14 * highlight.reshape(1, 1, 4) * density[..., None]
    colors = np.where(
        (source_mass > 1e-12)[..., None],
        shaded,
        neutral_color.reshape(1, 1, 4),
    )
    colors[..., 3] = 0.22 + 0.70 * density
    return colors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render subjective topology density over time as a static 3D surface."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("topology_state_surface.png"),
    )
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resolution", type=int, default=36)
    parser.add_argument(
        "--random",
        action="store_true",
        help="Generate a seeded random experience sequence instead of the fixed demo.",
    )
    parser.add_argument("--count", type=int, default=8, help="Random sequence length.")
    args = parser.parse_args()

    model = (
        random_experience_model(count=args.count, seed=args.seed)
        if args.random
        else demo_model(seed=args.seed)
    )
    episode = CaveProducer(model).run(dt=args.dt)
    save_topology_state_surface(
        episode,
        args.output,
        resolution=args.resolution,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
