from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch

from cave.demonstrations.examples import demo_model, random_experience_model
from cave.observation.episodes import CaveProducer, Episode, EpisodeInput
from cave.observation.structural import EpisodeFrame, episode_frames, structural_state_for_episode
from cave.observation.views import presentation_for_episode_input
from cave.presentation.renderers.matplotlib_renderer.glyphs import draw_glyph


EXPECTED_COLOR = "#D55E00"
ACTUAL_COLOR = "#0072B2"
AFTER_COLOR = "#009E73"
ERROR_COLOR = "#CC79A7"
ATTENTION_COLOR = "#2563EB"
GRID_COLOR = "#CBD5E1"
TEXT_COLOR = "#111827"
PANEL_COLOR = "#F8FAFC"


def save_experience_trajectory_strip(
    episode: Episode,
    output: str | Path,
    *,
    dpi: int = 140,
    max_frames: int = 72,
) -> None:
    """Render a full episode as an unrolled experience-to-state strip.

    This is intentionally separate from the standard view stack. It is an
    explanatory static artifact: time runs left-to-right, each frame keeps the
    current correction feature-plane geometry inside a local time slice, and the
    observed presentations provide identity marks without changing the geometry.
    """

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    structural = structural_state_for_episode(episode)
    frames = _sample_frames(episode_frames(episode, structural), max_frames=max_frames)
    if not frames:
        raise ValueError("episode has no frames")

    figure = plt.figure(figsize=(16.0, 8.2), dpi=dpi, facecolor="white")
    grid = figure.add_gridspec(2, 1, height_ratios=(0.72, 1.65), hspace=0.24)
    input_axis = figure.add_subplot(grid[0, 0])
    strip_axis = figure.add_subplot(grid[1, 0])
    figure.suptitle(
        "Experience Trajectory Strip: observed episode unrolled into subjective correction",
        fontsize=16,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    _draw_input_attention_strip(input_axis, episode)
    _draw_correction_strip(strip_axis, episode, frames)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)


def _sample_frames(frames: Sequence[EpisodeFrame], *, max_frames: int) -> list[EpisodeFrame]:
    if len(frames) <= max_frames:
        return list(frames)
    indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
    return [frames[int(index)] for index in indices]


def _draw_input_attention_strip(axis: Axes, episode: Episode) -> None:
    _style_panel(axis, "Observed inputs and attention/admission")
    duration = max(float(episode.duration), 1e-9)
    axis.set_xlim(0.0, duration)
    axis.set_ylim(0.0, 1.0)
    axis.axhline(0.34, color="#94A3B8", linewidth=2.0, alpha=0.46)

    observations = episode.observations
    if observations:
        times = np.array([obs.t for obs in observations], dtype=float)
        attention = np.array([obs.attention for obs in observations], dtype=float)
        axis.fill_between(times, 0.02, 0.02 + 0.22 * attention, color=ATTENTION_COLOR, alpha=0.14)
        axis.plot(times, 0.02 + 0.22 * attention, color=ATTENTION_COLOR, linewidth=1.7, alpha=0.78)

    for item in episode.inputs:
        midpoint = 0.5 * (item.start + item.end)
        attention = _mean_attention_for_input(episode, item)
        axis.axvspan(
            item.start,
            item.end,
            ymin=0.08,
            ymax=0.92,
            color=ACTUAL_COLOR,
            alpha=0.035 + 0.12 * attention,
            linewidth=0,
        )
        axis.plot(
            [item.start, item.end],
            [0.34, 0.34],
            color=ACTUAL_COLOR,
            linewidth=1.2 + 4.0 * attention,
            alpha=0.18 + 0.52 * attention,
        )
        presentation = presentation_for_episode_input(item, episode.vocabulary)
        draw_glyph(
            axis,
            presentation,
            midpoint,
            0.60,
            0.075,
            alpha=max(0.20, min(1.0, attention)),
        )
        axis.text(
            midpoint,
            0.86,
            _short_label(item),
            ha="center",
            va="center",
            fontsize=8,
            color=TEXT_COLOR,
        )

    axis.text(
        0.01,
        0.05,
        "blue curve/bands = attention-weighted admission; glyphs = observed inputs",
        transform=axis.transAxes,
        fontsize=9,
        color=TEXT_COLOR,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 4},
    )
    axis.set_xticks([])
    axis.set_yticks([])


def _draw_correction_strip(
    axis: Axes,
    episode: Episode,
    frames: Sequence[EpisodeFrame],
) -> None:
    _style_panel(axis, "Unrolled subjective correction: expectation -> actual -> memory")
    bounds = frames[0].topology_frame.topology.bounds
    duration = max(float(episode.duration), 1e-9)
    slice_width = duration / max(5.0, len(episode.inputs) * 3.0)
    strip_end = duration + slice_width
    axis.set_xlim(-0.05 * slice_width, strip_end)
    axis.set_ylim(-0.04, 1.04)

    xx, yy, density = _correction_strip_density(frames, bounds, slice_width, strip_end)
    axis.contourf(xx, yy, density, levels=18, cmap="viridis", alpha=0.70)
    axis.contour(xx, yy, density, levels=8, colors="white", linewidths=0.32, alpha=0.34)

    after_points: list[tuple[float, float]] = []
    for frame in frames:
        correction = frame.topology_frame.correction
        if correction is None:
            continue
        x_offset = float(frame.observation.t)
        expected = _strip_point(correction.expected_point, bounds, x_offset, slice_width)
        actual = _strip_point(correction.actual_point, bounds, x_offset, slice_width)
        after = _strip_point(correction.after_point, bounds, x_offset, slice_width)
        after_points.append(after)
        actual_attention = _external_input_attention(frame)
        expected_attention = _internal_expectation_attention(frame)
        comparison_alpha = float(np.clip(0.10 + 0.42 * max(actual_attention, expected_attention), 0.0, 0.68))
        error_width = 0.55 + 3.2 * float(np.clip(frame.observation.surprise, 0.0, 1.0))
        axis.add_patch(
            FancyArrowPatch(
                expected,
                actual,
                arrowstyle="->",
                color=ERROR_COLOR,
                linewidth=error_width,
                mutation_scale=8 + 11 * float(np.clip(frame.observation.surprise, 0.0, 1.0)),
                alpha=comparison_alpha,
                zorder=4,
            )
        )
        axis.plot(
            [actual[0], after[0]],
            [actual[1], after[1]],
            color=AFTER_COLOR,
            linewidth=0.7 + 3.2 * actual_attention,
            alpha=0.18 + 0.58 * actual_attention,
            zorder=5,
        )
        axis.scatter(
            [expected[0]],
            [expected[1]],
            s=14 + 48 * expected_attention,
            color=EXPECTED_COLOR,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.24 + 0.66 * expected_attention,
            zorder=6,
        )
        axis.scatter(
            [actual[0]],
            [actual[1]],
            s=16 + 52 * actual_attention + 85 * float(np.clip(frame.observation.surprise, 0.0, 1.0)),
            color=ACTUAL_COLOR,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.22 + 0.68 * actual_attention,
            zorder=7,
        )

    if after_points:
        memory_path = np.array(after_points, dtype=float)
        axis.plot(memory_path[:, 0], memory_path[:, 1], color="white", linewidth=4.4, alpha=0.38, zorder=8)
        axis.plot(memory_path[:, 0], memory_path[:, 1], color=AFTER_COLOR, linewidth=2.0, alpha=0.90, zorder=9)
        axis.scatter(
            memory_path[:, 0],
            memory_path[:, 1],
            s=22,
            color=AFTER_COLOR,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.82,
            zorder=10,
        )

    for item in episode.inputs:
        _draw_event_correction_anchor(axis, episode, frames, item, bounds, slice_width)
    for item in episode.inputs:
        _draw_input_marker_on_strip(axis, episode, frames, item, bounds, slice_width)

    for item in episode.inputs:
        axis.axvline(item.start, color="white", linewidth=0.75, alpha=0.28, zorder=2)
        axis.text(
            item.start + 0.015 * duration,
            0.035,
            _short_label(item),
            fontsize=8,
            color="white",
            alpha=0.92,
            zorder=11,
        )

    axis.text(
        0.012,
        0.94,
        "x = time + projected feature-x; y = projected feature-y; arrows/dots are model correction state",
        transform=axis.transAxes,
        fontsize=9,
        color="white",
        bbox={"facecolor": "#111827", "alpha": 0.72, "edgecolor": "none", "pad": 4},
        zorder=12,
    )
    axis.set_xlabel("time offset + projected feature x")
    axis.set_ylabel("projected feature y")
    axis.set_yticks([0.0, 0.5, 1.0])


def _draw_event_correction_anchor(
    axis: Axes,
    episode: Episode,
    frames: Sequence[EpisodeFrame],
    item: EpisodeInput,
    bounds: tuple[float, float],
    slice_width: float,
) -> None:
    frame = min(frames, key=lambda candidate: abs(candidate.observation.t - item.center))
    correction = frame.topology_frame.correction
    if correction is None:
        return
    x_offset = float(frame.observation.t)
    expected = _strip_point(correction.expected_point, bounds, x_offset, slice_width)
    actual = _strip_point(correction.actual_point, bounds, x_offset, slice_width)
    after = _strip_point(correction.after_point, bounds, x_offset, slice_width)
    attention = max(_external_input_attention(frame), _mean_attention_for_input(episode, item))
    surprise = float(np.clip(frame.observation.surprise, 0.0, 1.0))
    axis.add_patch(
        FancyArrowPatch(
            expected,
            actual,
            arrowstyle="->",
            color=ERROR_COLOR,
            linewidth=1.8 + 4.0 * surprise,
            mutation_scale=13 + 16 * surprise,
            alpha=0.55 + 0.35 * attention,
            zorder=12,
        )
    )
    axis.plot(
        [actual[0], after[0]],
        [actual[1], after[1]],
        color=AFTER_COLOR,
        linewidth=1.6 + 3.6 * attention,
        alpha=0.62 + 0.25 * attention,
        zorder=13,
    )
    axis.scatter(
        [expected[0]],
        [expected[1]],
        s=80,
        color=EXPECTED_COLOR,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.94,
        zorder=14,
    )
    axis.scatter(
        [actual[0]],
        [actual[1]],
        s=95 + 160 * surprise,
        color=ACTUAL_COLOR,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.94,
        zorder=14,
    )
    axis.scatter(
        [after[0]],
        [after[1]],
        s=86,
        color=AFTER_COLOR,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.94,
        zorder=14,
    )


def _draw_input_marker_on_strip(
    axis: Axes,
    episode: Episode,
    frames: Sequence[EpisodeFrame],
    item: EpisodeInput,
    bounds: tuple[float, float],
    slice_width: float,
) -> None:
    frame = min(frames, key=lambda candidate: abs(candidate.observation.t - item.center))
    correction = frame.topology_frame.correction
    if correction is None:
        return
    point = _strip_point(correction.actual_point, bounds, float(frame.observation.t), slice_width)
    attention = _mean_attention_for_input(episode, item)
    presentation = presentation_for_episode_input(item, episode.vocabulary)
    draw_glyph(
        axis,
        presentation,
        point[0],
        point[1],
        0.070 * max(0.65, min(1.45, slice_width)),
        alpha=max(0.22, min(1.0, attention)),
    )


def _correction_strip_density(
    frames: Sequence[EpisodeFrame],
    bounds: tuple[float, float],
    slice_width: float,
    strip_end: float,
    *,
    grid_size_x: int = 520,
    grid_size_y: int = 120,
    sigma_x: float | None = None,
    sigma_y: float = 0.030,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma_x = sigma_x if sigma_x is not None else max(0.018, 0.050 * slice_width)
    x_axis = np.linspace(-0.05 * slice_width, strip_end, grid_size_x)
    y_axis = np.linspace(0.0, 1.0, grid_size_y)
    xx, yy = np.meshgrid(x_axis, y_axis)
    density = np.zeros_like(xx)
    for frame in frames:
        correction = frame.topology_frame.correction
        if correction is None:
            continue
        t = float(frame.observation.t)
        actual_attention = _external_input_attention(frame)
        expected_attention = _internal_expectation_attention(frame)
        expected = _strip_point(correction.expected_point, bounds, t, slice_width)
        actual = _strip_point(correction.actual_point, bounds, t, slice_width)
        after = _strip_point(correction.after_point, bounds, t, slice_width)
        density += 0.50 * expected_attention * _gaussian_strip(xx, yy, expected, sigma_x, sigma_y)
        density += 1.00 * actual_attention * _gaussian_strip(xx, yy, actual, sigma_x, sigma_y)
        density += 0.78 * actual_attention * _gaussian_strip(xx, yy, after, sigma_x, sigma_y)
    max_density = float(np.max(density))
    if max_density > 0.0:
        density = density / max_density
    return xx, yy, density


def _gaussian_strip(
    xx: np.ndarray,
    yy: np.ndarray,
    center: tuple[float, float],
    sigma_x: float,
    sigma_y: float,
) -> np.ndarray:
    cx, cy = center
    return np.exp(-(((xx - cx) ** 2) / (2.0 * sigma_x**2) + ((yy - cy) ** 2) / (2.0 * sigma_y**2)))


def _strip_point(
    point: np.ndarray,
    bounds: tuple[float, float],
    time_offset: float,
    slice_width: float,
) -> tuple[float, float]:
    lower, upper = bounds
    span = max(float(upper - lower), 1e-12)
    normalized = np.clip((np.asarray(point, dtype=float) - lower) / span, 0.0, 1.0)
    return float(time_offset + slice_width * normalized[0]), float(normalized[1])


def _mean_attention_for_input(episode: Episode, item: EpisodeInput) -> float:
    values = [
        float(obs.attention_weights.get(item.id, obs.attention))
        for obs in episode.observations
        if item.id in obs.active_inputs
    ]
    if not values:
        return 0.0
    return float(np.clip(np.mean(values), 0.0, 1.0))


def _external_input_attention(frame: EpisodeFrame) -> float:
    effective = frame.observation.metadata.get("effective_attention", {})
    if isinstance(effective, dict) and "external_input" in effective:
        return float(np.clip(effective["external_input"], 0.0, 1.0))
    return float(np.clip(frame.observation.attention, 0.0, 1.0))


def _internal_expectation_attention(frame: EpisodeFrame) -> float:
    effective = frame.observation.metadata.get("effective_attention", {})
    if isinstance(effective, dict) and "internal_expectation" in effective:
        return float(np.clip(effective["internal_expectation"], 0.0, 1.0))
    return float(np.clip(frame.observation.attention, 0.0, 1.0))


def _short_label(item: EpisodeInput) -> str:
    label = str(item.metadata.get("label", item.id))
    for prefix in ("evt_", "event_", "input_"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
    return label.replace("_", " ")


def _style_panel(axis: Axes, title: str) -> None:
    axis.set_title(title, loc="left", fontsize=12, fontweight="bold", color=TEXT_COLOR)
    axis.set_facecolor(PANEL_COLOR)
    axis.grid(True, color=GRID_COLOR, linewidth=0.75, alpha=0.56)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color("#CBD5E1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a full episode as an unrolled trajectory strip.")
    parser.add_argument("--output", type=Path, default=Path("out/experience_trajectory_strip.png"))
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--random", action="store_true", help="Use a seeded random experience sequence.")
    parser.add_argument("--count", type=int, default=8, help="Random sequence length.")
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--max-frames", type=int, default=72)
    args = parser.parse_args()

    model = (
        random_experience_model(count=args.count, seed=args.seed)
        if args.random
        else demo_model(seed=args.seed)
    )
    episode = CaveProducer(model).run(dt=args.dt)
    save_experience_trajectory_strip(
        episode,
        args.output,
        dpi=args.dpi,
        max_frames=args.max_frames,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
