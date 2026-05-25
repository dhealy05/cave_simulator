from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from cave.commitments.topology import SubjectiveTopologyParams, SubjectiveTopologyState
from cave.observation.experience import feature_axis_label
from cave.observation.population import PopulationRunRecord
from cave.observation.structural import EpisodeTopologyFrame, structural_state_for_episode


@dataclass(frozen=True)
class TopologyTrajectory:
    run_id: str
    label: str
    treatment_id: str
    start_condition_id: str
    condition_id: str
    comparison_role: str
    matched_set_id: str | None
    times: np.ndarray
    density_deltas: np.ndarray
    centroids: np.ndarray
    mass: np.ndarray
    spread: np.ndarray
    peak: np.ndarray
    expected_points: np.ndarray
    actual_points: np.ndarray
    after_points: np.ndarray
    correction_distance: np.ndarray


def topology_trajectories(
    records: Sequence[PopulationRunRecord],
    params: SubjectiveTopologyParams,
) -> tuple[TopologyTrajectory, ...]:
    if not records:
        raise ValueError("at least one population run is required")
    initial = SubjectiveTopologyState.initial(
        feature_x=params.feature_x,
        feature_y=params.feature_y,
        bounds=params.bounds,
        resolution=params.resolution,
        prior=params.prior,
    )
    return tuple(
        _trajectory_for_record(record, params=params, initial=initial)
        for record in records
    )


def save_topology_population_dashboard(
    records: Sequence[PopulationRunRecord],
    output: str | Path,
    params: SubjectiveTopologyParams,
    *,
    title: str = "Same Treatment, Different Starts",
    selected_time: float | None = None,
    dpi: int = 150,
) -> None:
    trajectories = topology_trajectories(records, params)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    condition_colors = _condition_colors(trajectories)
    start_markers = _start_markers(trajectories)
    lower, upper = params.bounds
    extent = (lower, upper, lower, upper)
    density_scale = _density_scale(trajectories)

    figure = plt.figure(figsize=(13.4, 7.2), facecolor="#f7f4ef")
    grid = figure.add_gridspec(
        3,
        3,
        width_ratios=(1.35, 1.0, 1.0),
        height_ratios=(1.0, 1.0, 0.72),
        left=0.06,
        right=0.985,
        top=0.86,
        bottom=0.12,
        wspace=0.34,
        hspace=0.38,
    )
    map_axis = figure.add_subplot(grid[:, 0])
    mass_axis = figure.add_subplot(grid[0, 1])
    spread_axis = figure.add_subplot(grid[1, 1])
    peak_axis = figure.add_subplot(grid[2, 1])
    correction_axis = figure.add_subplot(grid[0:2, 2])
    legend_axis = figure.add_subplot(grid[2, 2])

    frame_indices = [
        _frame_index_for_time(trajectory, selected_time)
        for trajectory in trajectories
    ]
    stain = _stain_image(
        [
            trajectory.density_deltas[index]
            for trajectory, index in zip(trajectories, frame_indices)
        ],
        [condition_colors[trajectory.condition_id] for trajectory in trajectories],
        scale=density_scale,
    )
    map_axis.imshow(
        stain,
        origin="lower",
        extent=extent,
        interpolation="bilinear",
        aspect="equal",
    )
    for trajectory, frame_index in zip(trajectories, frame_indices):
        color = condition_colors[trajectory.condition_id]
        marker = start_markers[trajectory.start_condition_id]
        path = trajectory.centroids[: frame_index + 1]
        valid = np.isfinite(path[:, 0]) & np.isfinite(path[:, 1])
        if np.any(valid):
            valid_path = path[valid]
            map_axis.plot(
                valid_path[:, 0],
                valid_path[:, 1],
                color=color,
                linewidth=1.45,
                alpha=0.64,
            )
            map_axis.scatter(
                [valid_path[-1, 0]],
                [valid_path[-1, 1]],
                color=color,
                marker=marker,
                s=58,
                edgecolor="#111827",
                linewidth=0.55,
                zorder=5,
            )
    map_axis.set_xlim(lower, upper)
    map_axis.set_ylim(lower, upper)
    map_axis.set_xlabel(feature_axis_label(params.feature_x))
    map_axis.set_ylabel(feature_axis_label(params.feature_y))
    map_axis.set_title("Topology density delta and centroid paths", loc="left", fontweight="bold")
    map_axis.grid(True, color="#d8d2c8", linewidth=0.6, alpha=0.75)

    _draw_metric_axis(
        mass_axis,
        trajectories,
        condition_colors,
        start_markers,
        "mass",
        "Topology mass",
    )
    _draw_metric_axis(
        spread_axis,
        trajectories,
        condition_colors,
        start_markers,
        "spread",
        "Topology spread",
    )
    _draw_metric_axis(
        peak_axis,
        trajectories,
        condition_colors,
        start_markers,
        "peak",
        "Topology peak",
    )
    _draw_correction_axis(
        correction_axis,
        trajectories,
        condition_colors,
        start_markers,
    )
    _draw_dashboard_legend(legend_axis, trajectories, condition_colors, start_markers)

    figure.text(
        0.06,
        0.955,
        title,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="#111827",
    )
    figure.text(
        0.06,
        0.912,
        _dashboard_subtitle(trajectories, params, selected_time),
        ha="left",
        va="top",
        fontsize=9.5,
        color="#344054",
    )
    figure.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.14)
    plt.close(figure)


def save_topology_population_animation(
    records: Sequence[PopulationRunRecord],
    output: str | Path,
    params: SubjectiveTopologyParams,
    *,
    title: str = "Topology Population: Time Migration",
    fps: int = 6,
) -> None:
    trajectories = topology_trajectories(records, params)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    frame_count = min(trajectory.times.size for trajectory in trajectories)
    if frame_count <= 0:
        raise ValueError("topology animation requires at least one frame")

    condition_colors = _condition_colors(trajectories)
    start_markers = _start_markers(trajectories)
    lower, upper = params.bounds
    extent = (lower, upper, lower, upper)
    density_scale = _density_scale(trajectories)

    figure = plt.figure(figsize=(10.8, 6.2), facecolor="#f7f4ef")
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.1, 0.74),
        left=0.07,
        right=0.98,
        top=0.84,
        bottom=0.12,
        wspace=0.22,
    )
    map_axis = figure.add_subplot(grid[0, 0])
    legend_axis = figure.add_subplot(grid[0, 1])

    def update(frame_index: int):
        map_axis.clear()
        legend_axis.clear()
        stain = _stain_image(
            [trajectory.density_deltas[frame_index] for trajectory in trajectories],
            [condition_colors[trajectory.condition_id] for trajectory in trajectories],
            scale=density_scale,
        )
        map_axis.imshow(
            stain,
            origin="lower",
            extent=extent,
            interpolation="bilinear",
            aspect="equal",
        )
        for trajectory in trajectories:
            color = condition_colors[trajectory.condition_id]
            marker = start_markers[trajectory.start_condition_id]
            path = trajectory.centroids[: frame_index + 1]
            valid = np.isfinite(path[:, 0]) & np.isfinite(path[:, 1])
            if not np.any(valid):
                continue
            valid_path = path[valid]
            map_axis.plot(
                valid_path[:, 0],
                valid_path[:, 1],
                color=color,
                linewidth=1.35,
                alpha=0.58,
            )
            map_axis.scatter(
                valid_path[-1, 0],
                valid_path[-1, 1],
                color=color,
                marker=marker,
                s=44,
                edgecolor="#111827",
                linewidth=0.45,
                zorder=5,
            )
        t = trajectories[0].times[frame_index]
        map_axis.set_xlim(lower, upper)
        map_axis.set_ylim(lower, upper)
        map_axis.set_xlabel(feature_axis_label(params.feature_x))
        map_axis.set_ylabel(feature_axis_label(params.feature_y))
        map_axis.set_title(f"Topology stain migration, t={t:0.2f}", loc="left", fontweight="bold")
        map_axis.grid(True, color="#d8d2c8", linewidth=0.6, alpha=0.75)
        _draw_animation_legend(legend_axis, trajectories, condition_colors, start_markers)
        figure.suptitle(
            title,
            x=0.07,
            y=0.965,
            ha="left",
            fontsize=16,
            fontweight="bold",
            color="#111827",
        )
        return map_axis, legend_axis

    anim = animation.FuncAnimation(
        figure,
        update,
        frames=frame_count,
        interval=1000 / max(1, fps),
        blit=False,
    )
    anim.save(output, writer=animation.PillowWriter(fps=fps), dpi=120)
    plt.close(figure)


def save_topology_scatter_migration(
    records: Sequence[PopulationRunRecord],
    output: str | Path,
    params: SubjectiveTopologyParams,
    *,
    title: str = "Topology Scatter Migration",
    color_factor: str = "start_condition",
    marker_factor: str = "condition",
    point_kind: str = "centroid",
    fps: int = 6,
) -> None:
    trajectories = topology_trajectories(records, params)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    frame_count = min(trajectory.times.size for trajectory in trajectories)
    if frame_count <= 0:
        raise ValueError("topology scatter migration requires at least one frame")

    color_values = sorted(
        {_trajectory_factor(trajectory, color_factor) for trajectory in trajectories}
    )
    marker_values = sorted(
        {_trajectory_factor(trajectory, marker_factor) for trajectory in trajectories}
    )
    colors = {
        value: np.array(plt.get_cmap("tab10")(index % 10)[:3], dtype=float)
        for index, value in enumerate(color_values)
    }
    markers = {
        value: marker
        for value, marker in zip(
            marker_values,
            ("o", "s", "^", "D", "P", "X", "v", "<", ">"),
        )
    }
    offsets = _scatter_offsets(
        trajectories,
        color_factor=color_factor,
        marker_factor=marker_factor,
    )
    lower, upper = params.bounds

    figure = plt.figure(figsize=(10.2, 6.0), facecolor="#f7f4ef")
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.15, 0.62),
        left=0.075,
        right=0.98,
        top=0.84,
        bottom=0.12,
        wspace=0.18,
    )
    scatter_axis = figure.add_subplot(grid[0, 0])
    legend_axis = figure.add_subplot(grid[0, 1])

    def update(frame_index: int):
        scatter_axis.clear()
        legend_axis.clear()
        for trajectory in trajectories:
            points = _trajectory_points(trajectory, point_kind)
            path = points[: frame_index + 1]
            valid = np.isfinite(path[:, 0]) & np.isfinite(path[:, 1])
            if not np.any(valid):
                continue
            valid_path = path[valid]
            color_value = _trajectory_factor(trajectory, color_factor)
            marker_value = _trajectory_factor(trajectory, marker_factor)
            color = colors[color_value]
            offset_path = valid_path + offsets[trajectory.run_id]
            scatter_axis.plot(
                offset_path[:, 0],
                offset_path[:, 1],
                color=color,
                linewidth=1.15,
                alpha=0.36,
                zorder=2,
            )
            scatter_axis.scatter(
                offset_path[-1, 0],
                offset_path[-1, 1],
                color=color,
                marker=markers[marker_value],
                s=58,
                edgecolor="#111827",
                linewidth=0.55,
                alpha=0.9,
                zorder=5,
            )
        t = trajectories[0].times[frame_index]
        scatter_axis.set_xlim(lower, upper)
        scatter_axis.set_ylim(lower, upper)
        scatter_axis.set_aspect("equal")
        scatter_axis.set_xlabel(feature_axis_label(params.feature_x))
        scatter_axis.set_ylabel(feature_axis_label(params.feature_y))
        scatter_axis.set_title(
            f"{_point_kind_label(point_kind).title()} scatter, t={t:0.2f}",
            loc="left",
            fontweight="bold",
        )
        scatter_axis.grid(True, color="#d8d2c8", linewidth=0.6, alpha=0.75)
        _draw_scatter_legend(
            legend_axis,
            colors,
            markers,
            color_factor=color_factor,
            marker_factor=marker_factor,
            point_kind=point_kind,
        )
        figure.suptitle(
            title,
            x=0.075,
            y=0.965,
            ha="left",
            fontsize=16,
            fontweight="bold",
            color="#111827",
        )
        return scatter_axis, legend_axis

    anim = animation.FuncAnimation(
        figure,
        update,
        frames=frame_count,
        interval=1000 / max(1, fps),
        blit=False,
    )
    anim.save(output, writer=animation.PillowWriter(fps=fps), dpi=120)
    plt.close(figure)


def _trajectory_for_record(
    record: PopulationRunRecord,
    *,
    params: SubjectiveTopologyParams,
    initial: SubjectiveTopologyState,
) -> TopologyTrajectory:
    structural = structural_state_for_episode(record.episode, topology_params=params)
    frames = tuple(structural.topology_frames)
    if not frames:
        raise ValueError(f"population run {record.id!r} has no topology frames")
    density_deltas = np.stack(
        [
            np.clip(frame.topology.density - initial.density, 0.0, None)
            for frame in frames
        ],
        axis=0,
    )
    centroids = np.array(
        [_density_centroid(density, params.bounds) for density in density_deltas],
        dtype=float,
    )
    mass = np.sum(density_deltas, axis=(1, 2))
    peak = np.max(density_deltas, axis=(1, 2))
    spread = np.array(
        [
            _density_spread(density, centroid, params.bounds)
            for density, centroid in zip(density_deltas, centroids)
        ],
        dtype=float,
    )
    expected, actual, after = _correction_points(frames)
    correction_distance = np.linalg.norm(actual - expected, axis=1)
    invalid = ~np.isfinite(actual[:, 0]) | ~np.isfinite(expected[:, 0])
    correction_distance[invalid] = np.nan
    return TopologyTrajectory(
        run_id=record.id,
        label=record.display_label,
        treatment_id=record.factor_id("treatment", "treatment") or "treatment",
        start_condition_id=(
            record.factor_id("start_condition", "start") or "start"
        ),
        condition_id=record.factor_id("condition", record.group_id) or "condition",
        comparison_role=record.comparison_role,
        matched_set_id=record.matched_set_id,
        times=np.array([frame.t for frame in frames], dtype=float),
        density_deltas=density_deltas,
        centroids=centroids,
        mass=mass,
        spread=spread,
        peak=peak,
        expected_points=expected,
        actual_points=actual,
        after_points=after,
        correction_distance=correction_distance,
    )


def _trajectory_points(
    trajectory: TopologyTrajectory,
    point_kind: str,
) -> np.ndarray:
    if point_kind == "centroid":
        return trajectory.centroids
    if point_kind == "expected":
        return trajectory.expected_points
    if point_kind == "actual":
        return trajectory.actual_points
    if point_kind == "after":
        return trajectory.after_points
    if point_kind in {"subjective", "memory", "memory_state"}:
        return trajectory.after_points
    raise ValueError(
        "point_kind must be one of: centroid, expected, actual, after, "
        "subjective, memory, memory_state"
    )


def _point_kind_label(point_kind: str) -> str:
    if point_kind in {"after", "subjective", "memory", "memory_state"}:
        return "subjective state"
    return point_kind


def _trajectory_factor(
    trajectory: TopologyTrajectory,
    factor: str,
) -> str:
    if factor == "treatment":
        return trajectory.treatment_id
    if factor == "start_condition":
        return trajectory.start_condition_id
    if factor == "condition":
        return trajectory.condition_id
    if factor == "comparison_role":
        return trajectory.comparison_role
    if factor == "matched_set":
        return trajectory.matched_set_id or "unmatched"
    raise ValueError(
        "factor must be one of: treatment, start_condition, condition, "
        "comparison_role, matched_set"
    )


def _draw_scatter_legend(
    axis,
    colors: dict[str, np.ndarray],
    markers: dict[str, str],
    *,
    color_factor: str,
    marker_factor: str,
    point_kind: str,
) -> None:
    axis.set_axis_off()
    axis.text(
        0.0,
        0.98,
        f"Color: {color_factor}",
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#111827",
        transform=axis.transAxes,
    )
    color_items = list(colors.items())
    color_columns = 2 if len(color_items) > 10 else 1
    color_rows = int(np.ceil(len(color_items) / color_columns))
    row_gap = 0.065 if color_columns > 1 else 0.085
    for index, (value, color) in enumerate(color_items):
        column = index // color_rows
        row = index % color_rows
        x = 0.04 + 0.46 * column
        y = 0.86 - row_gap * row
        axis.scatter(
            x,
            y,
            color=color,
            s=58,
            marker="o",
            edgecolor="#111827",
            linewidth=0.45,
            transform=axis.transAxes,
        )
        axis.text(
            x + 0.08,
            y,
            value,
            ha="left",
            va="center",
            fontsize=8.1 if color_columns > 1 else 9,
            color="#24303f",
            transform=axis.transAxes,
        )
    y = 0.86 - row_gap * max(1, color_rows) - 0.08
    axis.text(
        0.0,
        y,
        f"Marker: {marker_factor}",
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#111827",
        transform=axis.transAxes,
    )
    y -= 0.12
    for value, marker in markers.items():
        if y < 0.08:
            break
        axis.scatter(
            0.04,
            y,
            color="#ffffff",
            s=72,
            marker=marker,
            edgecolor="#111827",
            linewidth=0.85,
            transform=axis.transAxes,
        )
        axis.text(
            0.12,
            y,
            value,
            ha="left",
            va="center",
            fontsize=8.5,
            color="#24303f",
            transform=axis.transAxes,
        )
        y -= 0.075
    axis.text(
        0.0,
        0.06,
        f"Point: {_point_kind_label(point_kind)}\n"
        "Trails show previous positions.\n"
        "Small dodge separates overlaps.",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#475467",
        linespacing=1.35,
        transform=axis.transAxes,
    )


def _scatter_offsets(
    trajectories: Sequence[TopologyTrajectory],
    *,
    color_factor: str,
    marker_factor: str,
) -> dict[str, np.ndarray]:
    keys = sorted(
        {
            (
                _trajectory_factor(trajectory, color_factor),
                _trajectory_factor(trajectory, marker_factor),
            )
            for trajectory in trajectories
        }
    )
    if len(keys) <= 1:
        return {
            trajectory.run_id: np.zeros(2, dtype=float)
            for trajectory in trajectories
        }
    radius = 0.018
    by_key = {}
    for index, key in enumerate(keys):
        angle = 2.0 * np.pi * index / len(keys)
        by_key[key] = radius * np.array([np.cos(angle), np.sin(angle)], dtype=float)
    return {
        trajectory.run_id: by_key[
            (
                _trajectory_factor(trajectory, color_factor),
                _trajectory_factor(trajectory, marker_factor),
            )
        ]
        for trajectory in trajectories
    }


def _correction_points(
    frames: Sequence[EpisodeTopologyFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = []
    actual = []
    after = []
    missing = np.array([np.nan, np.nan], dtype=float)
    for frame in frames:
        if frame.correction is None:
            expected.append(missing)
            actual.append(missing)
            after.append(missing)
            continue
        expected.append(frame.correction.expected_point)
        actual.append(frame.correction.actual_point)
        after.append(frame.correction.after_point)
    return (
        np.array(expected, dtype=float),
        np.array(actual, dtype=float),
        np.array(after, dtype=float),
    )


def _draw_metric_axis(
    axis,
    trajectories: Sequence[TopologyTrajectory],
    colors: dict[str, np.ndarray],
    markers: dict[str, str],
    attribute: str,
    title: str,
) -> None:
    for trajectory in trajectories:
        values = np.asarray(getattr(trajectory, attribute), dtype=float)
        axis.plot(
            trajectory.times,
            values,
            color=colors[trajectory.condition_id],
            linewidth=1.45,
            alpha=0.78,
        )
        axis.scatter(
            [trajectory.times[-1]],
            [values[-1]],
            color=colors[trajectory.condition_id],
            marker=markers[trajectory.start_condition_id],
            s=34,
            edgecolor="#111827",
            linewidth=0.4,
            zorder=5,
        )
    axis.set_title(title, loc="left", fontweight="bold", fontsize=10.5)
    axis.set_xlabel("time")
    axis.grid(True, color="#d8d2c8", linewidth=0.6, alpha=0.65)


def _draw_correction_axis(
    axis,
    trajectories: Sequence[TopologyTrajectory],
    colors: dict[str, np.ndarray],
    markers: dict[str, str],
) -> None:
    for trajectory in trajectories:
        axis.plot(
            trajectory.times,
            trajectory.correction_distance,
            color=colors[trajectory.condition_id],
            linewidth=1.55,
            alpha=0.82,
        )
        valid = np.isfinite(trajectory.correction_distance)
        if np.any(valid):
            index = np.where(valid)[0][-1]
            axis.scatter(
                [trajectory.times[index]],
                [trajectory.correction_distance[index]],
                color=colors[trajectory.condition_id],
                marker=markers[trajectory.start_condition_id],
                s=38,
                edgecolor="#111827",
                linewidth=0.4,
                zorder=5,
            )
    axis.set_title("Expected vs actual correction", loc="left", fontweight="bold", fontsize=10.5)
    axis.set_xlabel("time")
    axis.set_ylabel("distance")
    axis.grid(True, color="#d8d2c8", linewidth=0.6, alpha=0.65)


def _draw_dashboard_legend(
    axis,
    trajectories: Sequence[TopologyTrajectory],
    colors: dict[str, np.ndarray],
    markers: dict[str, str],
) -> None:
    axis.set_axis_off()
    condition_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"{condition} ({_condition_role(trajectories, condition)})",
            markerfacecolor=color,
            markeredgecolor="#111827",
            markersize=7,
        )
        for condition, color in colors.items()
    ]
    start_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="#111827",
            label=start,
            linestyle="None",
            markersize=7,
        )
        for start, marker in markers.items()
    ]
    axis.legend(
        handles=condition_handles + start_handles,
        loc="center left",
        frameon=False,
        fontsize=7.4,
        ncol=1,
    )


def _draw_animation_legend(
    axis,
    trajectories: Sequence[TopologyTrajectory],
    colors: dict[str, np.ndarray],
    markers: dict[str, str],
) -> None:
    axis.set_axis_off()
    axis.text(
        0.0,
        0.98,
        "Conditions",
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#111827",
        transform=axis.transAxes,
    )
    y = 0.88
    for condition, color in colors.items():
        axis.scatter(
            0.03,
            y,
            color=color,
            s=70,
            marker="o",
            edgecolor="#111827",
            linewidth=0.4,
            transform=axis.transAxes,
        )
        axis.text(
            0.1,
            y,
            f"{condition} ({_condition_role(trajectories, condition)})",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#24303f",
            transform=axis.transAxes,
        )
        y -= 0.075
    y -= 0.035
    axis.text(
        0.0,
        y,
        "Starting Conditions",
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#111827",
        transform=axis.transAxes,
    )
    y -= 0.1
    for start, marker in markers.items():
        axis.scatter(
            0.03,
            y,
            color="#ffffff",
            s=70,
            marker=marker,
            edgecolor="#111827",
            linewidth=0.8,
            transform=axis.transAxes,
        )
        axis.text(
            0.1,
            y,
            start,
            ha="left",
            va="center",
            fontsize=9,
            color="#24303f",
            transform=axis.transAxes,
        )
        y -= 0.075
    axis.text(
        0.0,
        0.18,
        "Stain opacity follows current topology density.\n"
        "Paths show density centroids over time.\n"
        "All runs share the same projection and prior.",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#475467",
        linespacing=1.3,
        transform=axis.transAxes,
    )


def _condition_colors(
    trajectories: Sequence[TopologyTrajectory],
) -> dict[str, np.ndarray]:
    condition_ids = sorted({trajectory.condition_id for trajectory in trajectories})
    return {
        condition: np.array(plt.get_cmap("tab10")(index % 10)[:3], dtype=float)
        for index, condition in enumerate(condition_ids)
    }


def _start_markers(trajectories: Sequence[TopologyTrajectory]) -> dict[str, str]:
    marker_cycle = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
    start_ids = sorted({trajectory.start_condition_id for trajectory in trajectories})
    return {
        start: marker_cycle[index % len(marker_cycle)]
        for index, start in enumerate(start_ids)
    }


def _condition_role(
    trajectories: Sequence[TopologyTrajectory],
    condition: str,
) -> str:
    for trajectory in trajectories:
        if trajectory.condition_id == condition:
            return trajectory.comparison_role
    return "condition"


def _dashboard_subtitle(
    trajectories: Sequence[TopologyTrajectory],
    params: SubjectiveTopologyParams,
    selected_time: float | None,
) -> str:
    treatments = sorted({trajectory.treatment_id for trajectory in trajectories})
    matched_sets = sorted(
        {
            trajectory.matched_set_id
            for trajectory in trajectories
            if trajectory.matched_set_id is not None
        }
    )
    time_label = "final frame" if selected_time is None else f"t={selected_time:0.2f}"
    return (
        f"treatments: {', '.join(treatments)} | matched sets: {', '.join(matched_sets)} | "
        f"projection: {feature_axis_label(params.feature_x)} x "
        f"{feature_axis_label(params.feature_y)} | map: {time_label}"
    )


def _frame_index_for_time(
    trajectory: TopologyTrajectory,
    selected_time: float | None,
) -> int:
    if selected_time is None:
        return int(trajectory.times.size - 1)
    return int(np.argmin(np.abs(trajectory.times - selected_time)))


def _density_centroid(
    density: np.ndarray,
    bounds: tuple[float, float],
) -> tuple[float, float]:
    total = float(np.sum(density))
    if total <= 1e-12:
        return (float("nan"), float("nan"))
    lower, upper = bounds
    axis = np.linspace(lower, upper, density.shape[0])
    grid_x, grid_y = np.meshgrid(axis, axis)
    return (
        float(np.sum(grid_x * density) / total),
        float(np.sum(grid_y * density) / total),
    )


def _density_spread(
    density: np.ndarray,
    centroid: np.ndarray,
    bounds: tuple[float, float],
) -> float:
    total = float(np.sum(density))
    if total <= 1e-12 or not np.all(np.isfinite(centroid)):
        return 0.0
    lower, upper = bounds
    axis = np.linspace(lower, upper, density.shape[0])
    grid_x, grid_y = np.meshgrid(axis, axis)
    distance_sq = (grid_x - centroid[0]) ** 2 + (grid_y - centroid[1]) ** 2
    return float(np.sqrt(np.sum(distance_sq * density) / total))


def _density_scale(trajectories: Sequence[TopologyTrajectory]) -> float:
    scale = max(
        float(np.max(trajectory.density_deltas))
        for trajectory in trajectories
    )
    return max(scale, 1e-9)


def _stain_image(
    densities: list[np.ndarray],
    colors: list[np.ndarray],
    *,
    scale: float,
) -> np.ndarray:
    image = np.ones((*densities[0].shape, 3), dtype=float)
    for density, color in zip(densities, colors):
        normalized = np.clip(density / scale, 0.0, 1.0)
        alpha = 0.42 * normalized
        image = image * (1.0 - alpha[..., None]) + color.reshape(1, 1, 3) * alpha[..., None]
    return np.clip(image, 0.0, 1.0)
