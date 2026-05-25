from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from cave.commitments.topology import (
    SubjectiveTopologyParams,
    SubjectiveTopologyPrior,
    SubjectiveTopologyState,
)
from cave.observation.episode_runs import EpisodeSet, LabeledEpisode
from cave.observation.projections import encode_value
from cave.observation.structural import structural_state_for_episode


TopologyAtlasEntry = LabeledEpisode


@dataclass(frozen=True)
class TopologyAtlasResult:
    id: str
    label: str
    experienced_delta: np.ndarray
    expected_density: np.ndarray
    actual_density: np.ndarray
    actual_expected_delta: np.ndarray
    metrics: dict[str, float]


def topology_atlas_results(
    entries: EpisodeSet | Sequence[LabeledEpisode],
    params: SubjectiveTopologyParams,
) -> tuple[TopologyAtlasResult, ...]:
    labeled_entries = _coerce_entries(entries)
    initial = SubjectiveTopologyState.initial(
        feature_x=params.feature_x,
        feature_y=params.feature_y,
        bounds=params.bounds,
        resolution=params.resolution,
        prior=params.prior,
    )
    results = []
    for entry in labeled_entries:
        structural = structural_state_for_episode(entry.episode, topology_params=params)
        final_density = structural.topology_frames[-1].topology.density
        experienced_delta = np.clip(final_density - initial.density, 0.0, None)
        expected_points = [
            frame.correction.expected_point
            for frame in structural.topology_frames
            if frame.correction is not None
        ]
        actual_points = [
            frame.correction.actual_point
            for frame in structural.topology_frames
            if frame.correction is not None
        ]
        expected_density = _counterfactual_density(expected_points, params)
        actual_density = _counterfactual_density(actual_points, params)
        actual_expected_delta = actual_density - expected_density
        metrics = _atlas_metrics(
            experienced_delta=experienced_delta,
            expected_density=expected_density,
            actual_density=actual_density,
            params=params,
        )
        results.append(
            TopologyAtlasResult(
                id=entry.id,
                label=entry.display_label,
                experienced_delta=experienced_delta,
                expected_density=expected_density,
                actual_density=actual_density,
                actual_expected_delta=actual_expected_delta,
                metrics=metrics,
            )
        )
    return tuple(results)


def topology_atlas_metrics_payload(
    entries: EpisodeSet | Sequence[LabeledEpisode],
    params: SubjectiveTopologyParams,
) -> dict[str, object]:
    results = topology_atlas_results(entries, params)
    return {
        "id": "topology_atlas",
        "projection": {
            "feature_x": _axis_name(params.feature_x),
            "feature_y": _axis_name(params.feature_y),
            "bounds": list(params.bounds),
            "resolution": params.resolution,
            "prior": params.prior.mode,
        },
        "episode_set": _episode_set_payload(entries),
        "entries": {
            result.id: {
                "label": result.label,
                "metrics": result.metrics,
            }
            for result in results
        },
    }


def save_topology_atlas(
    entries: EpisodeSet | Sequence[LabeledEpisode],
    output: str | Path,
    params: SubjectiveTopologyParams,
    *,
    dpi: int = 150,
) -> None:
    results = topology_atlas_results(entries, params)
    if not results:
        raise ValueError("at least one topology atlas entry is required")

    density_max = max(
        max(
            float(np.max(result.experienced_delta)),
            float(np.max(result.expected_density)),
            float(np.max(result.actual_density)),
        )
        for result in results
    )
    density_max = max(density_max, 1e-9)
    delta_abs_max = max(
        float(np.max(np.abs(result.actual_expected_delta)))
        for result in results
    )
    delta_abs_max = max(delta_abs_max, 1e-9)

    columns = (
        ("experienced_delta", "Experienced delta", "magma", 0.0, density_max),
        ("expected_density", "Expected path", "magma", 0.0, density_max),
        ("actual_density", "Actual path", "magma", 0.0, density_max),
        (
            "actual_expected_delta",
            "Actual - expected",
            "RdBu_r",
            -delta_abs_max,
            delta_abs_max,
        ),
    )
    row_height = 1.28
    figure = plt.figure(
        figsize=(15.6, max(6.2, 1.05 + row_height * len(results))),
        facecolor="#f7f4ef",
    )
    grid = figure.add_gridspec(
        len(results) + 1,
        len(columns) + 2,
        width_ratios=(1.35, 1.0, 1.0, 1.0, 1.0, 1.15),
        height_ratios=(0.28, *([1.0] * len(results))),
        left=0.035,
        right=0.985,
        top=0.9,
        bottom=0.035,
        wspace=0.13,
        hspace=0.14,
    )
    lower, upper = params.bounds
    extent = (lower, upper, lower, upper)
    _draw_atlas_header(
        figure,
        params=params,
        density_max=density_max,
        delta_abs_max=delta_abs_max,
    )

    headers = ("Episode", *(column[1] for column in columns), "Metrics")
    for col, header in enumerate(headers):
        axis = figure.add_subplot(grid[0, col])
        axis.set_axis_off()
        axis.text(
            0.5 if col else 0.0,
            0.35,
            header,
            ha="center" if col else "left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#1f2933",
        )

    for row, result in enumerate(results):
        label_axis = figure.add_subplot(grid[row + 1, 0])
        label_axis.set_axis_off()
        label_axis.add_patch(
            plt.Rectangle(
                (0.0, 0.08),
                0.96,
                0.84,
                facecolor="#ebe4da",
                edgecolor="#d2c7b8",
                linewidth=0.8,
                transform=label_axis.transAxes,
            )
        )
        label_axis.text(
            0.06,
            0.55,
            result.label,
            ha="left",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="#24303f",
            transform=label_axis.transAxes,
        )
        label_axis.text(
            0.06,
            0.33,
            result.id,
            ha="left",
            va="center",
            fontsize=6.8,
            color="#667085",
            transform=label_axis.transAxes,
        )

        for col, (attribute, _title, cmap, vmin, vmax) in enumerate(columns):
            axis = figure.add_subplot(grid[row + 1, col + 1])
            axis.imshow(
                getattr(result, attribute),
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="bilinear",
                aspect="equal",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_facecolor("#111827")
            for spine in axis.spines.values():
                spine.set_color("#2f3a4a")
                spine.set_linewidth(0.8)
            if attribute == "experienced_delta":
                _draw_centroid_marker(axis, result.metrics)

        metric_axis = figure.add_subplot(grid[row + 1, len(columns) + 1])
        metric_axis.set_axis_off()
        metric_axis.add_patch(
            plt.Rectangle(
                (0.0, 0.08),
                0.96,
                0.84,
                facecolor="#ffffff",
                edgecolor="#ddd6cc",
                linewidth=0.8,
                transform=metric_axis.transAxes,
            )
        )
        metric_axis.text(
            0.08,
            0.5,
            _metric_caption(result.metrics),
            ha="left",
            va="center",
            fontsize=7.4,
            color="#24303f",
            linespacing=1.25,
            transform=metric_axis.transAxes,
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.14)
    plt.close(figure)


def save_topology_atlas_metrics(
    entries: EpisodeSet | Sequence[LabeledEpisode],
    output: str | Path,
    params: SubjectiveTopologyParams,
) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = topology_atlas_metrics_payload(entries, params)
    output.write_text(
        json.dumps(encode_value(payload), indent=2) + "\n",
        encoding="utf-8",
    )


def shared_topology_params_for_episode_set(
    entries: EpisodeSet | Sequence[LabeledEpisode],
) -> SubjectiveTopologyParams:
    labeled_entries = _coerce_entries(entries)
    first = labeled_entries[0].episode
    params = first.metadata.get("topology_params") or SubjectiveTopologyParams()
    return replace(params, prior=SubjectiveTopologyPrior())


def _counterfactual_density(
    points: Sequence[np.ndarray],
    params: SubjectiveTopologyParams,
) -> np.ndarray:
    state = SubjectiveTopologyState.initial(
        feature_x=params.feature_x,
        feature_y=params.feature_y,
        bounds=params.bounds,
        resolution=params.resolution,
        prior=params.prior,
    )
    density = np.array(state.density, dtype=float)
    for point in points:
        density = state._diffuse(density * params.decay, params.diffusion)
        density += params.deposit_strength * state._gaussian_grid(
            np.asarray(point, dtype=float),
            params.deposit_width,
        )
        density = np.clip(density, 0.0, params.max_density)
    return density


def _coerce_entries(
    entries: EpisodeSet | Sequence[LabeledEpisode],
) -> tuple[LabeledEpisode, ...]:
    if isinstance(entries, EpisodeSet):
        return entries.episodes
    labeled_entries = tuple(entries)
    if not labeled_entries:
        raise ValueError("at least one topology atlas entry is required")
    return labeled_entries


def _episode_set_payload(entries: EpisodeSet | Sequence[LabeledEpisode]) -> dict[str, object]:
    if isinstance(entries, EpisodeSet):
        return {
            "id": entries.id,
            "title": entries.title,
            "comparison_axis": entries.comparison_axis,
            "metadata": entries.metadata,
        }
    return {
        "id": None,
        "title": None,
        "comparison_axis": None,
        "metadata": {},
    }


def _atlas_metrics(
    *,
    experienced_delta: np.ndarray,
    expected_density: np.ndarray,
    actual_density: np.ndarray,
    params: SubjectiveTopologyParams,
) -> dict[str, float]:
    actual_expected_delta = actual_density - expected_density
    centroid_x, centroid_y = _centroid(experienced_delta, params.bounds)
    return {
        "experienced_mass": float(np.sum(experienced_delta)),
        "experienced_peak": float(np.max(experienced_delta)),
        "experienced_centroid_x": centroid_x,
        "experienced_centroid_y": centroid_y,
        "experienced_spread": _spread(experienced_delta, params.bounds),
        "expected_mass": float(np.sum(expected_density)),
        "actual_mass": float(np.sum(actual_density)),
        "actual_expected_l2": float(np.linalg.norm(actual_expected_delta)),
        "actual_expected_l1": float(np.sum(np.abs(actual_expected_delta))),
    }


def _centroid(density: np.ndarray, bounds: tuple[float, float]) -> tuple[float, float]:
    total = float(np.sum(density))
    if total <= 1e-12:
        return 0.0, 0.0
    lower, upper = bounds
    axis = np.linspace(lower, upper, density.shape[0])
    grid_x, grid_y = np.meshgrid(axis, axis)
    return (
        float(np.sum(grid_x * density) / total),
        float(np.sum(grid_y * density) / total),
    )


def _spread(density: np.ndarray, bounds: tuple[float, float]) -> float:
    total = float(np.sum(density))
    if total <= 1e-12:
        return 0.0
    centroid_x, centroid_y = _centroid(density, bounds)
    lower, upper = bounds
    axis = np.linspace(lower, upper, density.shape[0])
    grid_x, grid_y = np.meshgrid(axis, axis)
    squared = (grid_x - centroid_x) ** 2 + (grid_y - centroid_y) ** 2
    return float(np.sqrt(np.sum(squared * density) / total))


def _axis_name(axis: object) -> str:
    return getattr(axis, "name", str(axis))


def _draw_atlas_header(
    figure: plt.Figure,
    *,
    params: SubjectiveTopologyParams,
    density_max: float,
    delta_abs_max: float,
) -> None:
    figure.text(
        0.035,
        0.965,
        "Topology Atlas",
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="#111827",
    )
    figure.text(
        0.035,
        0.912,
        "Same initial prior, shared projection, derived overhead readouts",
        ha="left",
        va="top",
        fontsize=9.5,
        color="#344054",
    )
    figure.text(
        0.985,
        0.95,
        (
            f"{_axis_name(params.feature_x)} x {_axis_name(params.feature_y)}  |  "
            f"density 0-{density_max:.2f}  |  diff +/-{delta_abs_max:.2f}"
        ),
        ha="right",
        va="top",
        fontsize=8.5,
        color="#667085",
    )


def _draw_centroid_marker(axis: plt.Axes, metrics: dict[str, float]) -> None:
    axis.scatter(
        [metrics["experienced_centroid_x"]],
        [metrics["experienced_centroid_y"]],
        marker="+",
        s=34,
        color="#ffffff",
        linewidths=0.9,
        alpha=0.85,
        zorder=5,
    )


def _metric_caption(metrics: dict[str, float]) -> str:
    return (
        f"mass      {metrics['experienced_mass']:.2f}\n"
        f"peak      {metrics['experienced_peak']:.2f}\n"
        f"center    {metrics['experienced_centroid_x']:.2f}, "
        f"{metrics['experienced_centroid_y']:.2f}\n"
        f"spread    {metrics['experienced_spread']:.2f}\n"
        f"AE delta  {metrics['actual_expected_l2']:.2f}"
    )
