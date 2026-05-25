from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from primitive_engine import PrimitiveStep, PrimitiveStep2D, rollout_1d, rollout_2d


PHASES = ("expectation", "actual", "error", "update", "carry")
COLORS = {
    "actual": "#0072B2",
    "expected": "#D55E00",
    "error": "#CC79A7",
    "memory": "#009E73",
    "prior": "#6B7280",
    "grid": "#D1D5DB",
    "text": "#111827",
}


def save_rollout_csv(rows: Sequence[PrimitiveStep], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def save_json(data: object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_primitive_1d_frame(rows: Sequence[PrimitiveStep], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), dpi=130)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.075, top=0.88, wspace=0.14, hspace=0.30)
    _draw_1d_dashboard(fig, axes.ravel(), rows, len(rows) - 1, "carry", 1.0)
    fig.savefig(output)
    plt.close(fig)


def save_primitive_1d_animation(
    rows: Sequence[PrimitiveStep],
    output: Path,
    *,
    fps: int = 4,
    subframes: int = 3,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = _animation_frames(len(rows), subframes=subframes)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), dpi=130)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.075, top=0.88, wspace=0.14, hspace=0.30)

    def update(frame: tuple[int, str, float]):
        index, phase, alpha = frame
        _draw_1d_dashboard(fig, axes.ravel(), rows, index, phase, alpha)
        return axes.ravel().tolist()

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 / max(1, fps),
        blit=False,
    )
    anim.save(output, writer=animation.PillowWriter(fps=fps), dpi=130)
    plt.close(fig)


def save_primitive_2d_frame(rows: Sequence[PrimitiveStep2D], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.4), dpi=130)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.065, top=0.91, wspace=0.18, hspace=0.34)
    _draw_2d_dashboard(fig, axes.ravel(), rows, len(rows) - 1, "carry", 1.0)
    fig.savefig(output)
    plt.close(fig)


def save_primitive_2d_animation(
    rows: Sequence[PrimitiveStep2D],
    output: Path,
    *,
    fps: int = 4,
    subframes: int = 3,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = _animation_frames(len(rows), subframes=subframes)
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.4), dpi=130)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.065, top=0.91, wspace=0.18, hspace=0.34)

    def update(frame: tuple[int, str, float]):
        index, phase, alpha = frame
        _draw_2d_dashboard(fig, axes.ravel(), rows, index, phase, alpha)
        return axes.ravel().tolist()

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 / max(1, fps),
        blit=False,
    )
    anim.save(output, writer=animation.PillowWriter(fps=fps), dpi=130)
    plt.close(fig)


def save_primitive_trajectory_surface(
    rows: Sequence[PrimitiveStep2D],
    output: Path,
    *,
    grid_size: int = 90,
    sigma: float = 0.055,
    decay: float = 0.94,
) -> None:
    """Render the primitive subjective trajectory and accumulated topology."""

    output.parent.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.0, 1.0, grid_size)
    xx, yy = np.meshgrid(grid, grid)
    density = np.zeros_like(xx)
    density_by_step: list[float] = []
    weights = {
        "expected": 0.55,
        "actual": 1.00,
        "memory": 0.82,
    }
    for row in rows:
        density *= decay
        density += weights["expected"] * _gaussian_2d(xx, yy, row.expected, sigma)
        density += weights["actual"] * _gaussian_2d(xx, yy, row.actual, sigma)
        density += weights["memory"] * _gaussian_2d(xx, yy, row.memory, sigma)
        density_by_step.append(float(density.sum()))

    fig = plt.figure(figsize=(12.6, 5.8), dpi=140)
    fig.suptitle(
        "Primitive subjective trajectory and topology-like state",
        fontsize=15,
        fontweight="bold",
        color=COLORS["text"],
    )
    trajectory_axis = fig.add_subplot(1, 2, 1)
    surface_axis = fig.add_subplot(1, 2, 2)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.10, top=0.86, wspace=0.18)
    _draw_static_trajectory(trajectory_axis, rows)
    _draw_static_surface(surface_axis, xx, yy, density, rows, density_by_step)
    fig.savefig(output)
    plt.close(fig)


def _animation_frames(step_count: int, *, subframes: int) -> list[tuple[int, str, float]]:
    frames: list[tuple[int, str, float]] = []
    for index in range(step_count):
        for phase in PHASES:
            for subframe in range(subframes):
                alpha = 1.0 if subframes == 1 else subframe / (subframes - 1)
                frames.append((index, phase, alpha))
    return frames


def _draw_1d_dashboard(
    fig,
    axes: Sequence[Axes],
    rows: Sequence[PrimitiveStep],
    index: int,
    phase: str,
    alpha: float,
) -> None:
    fig.suptitle(
        "Primitive Cave: expectation, error, and memory update",
        fontsize=15,
        fontweight="bold",
        color=COLORS["text"],
    )
    for axis in axes:
        axis.clear()
    row = rows[index]
    stage_rank = PHASES.index(phase)
    _draw_current_input(axes[0], row, stage_rank)
    _draw_expectation_actual_1d(axes[1], row, stage_rank)
    _draw_memory_update_1d(axes[2], row, stage_rank, alpha)
    _draw_timeline_1d(axes[3], rows, index, stage_rank, alpha)


def _draw_current_input(axis: Axes, row: PrimitiveStep, stage_rank: int) -> None:
    _prepare_value_axis(axis, "A. Current admitted input")
    axis.text(0.5, 1.05, f"t = {row.t}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    if stage_rank >= PHASES.index("actual"):
        axis.bar([0], [row.actual], width=0.42, color=COLORS["actual"])
        axis.text(0, row.actual + 0.04, f"U_t = {row.actual:.3f}", ha="center", fontsize=10)
    else:
        axis.text(0, 0.50, "actual not admitted yet", ha="center", color=COLORS["prior"], fontsize=10)
    axis.set_xlim(-0.7, 0.7)
    axis.set_xticks([])


def _draw_expectation_actual_1d(axis: Axes, row: PrimitiveStep, stage_rank: int) -> None:
    _prepare_value_axis(axis, "B. Expectation before actual")
    axis.axvline(0, color=COLORS["grid"], linewidth=2)
    axis.scatter([0], [row.expected], s=95, color=COLORS["expected"], zorder=4)
    axis.text(0.05, row.expected, f"E_t = {row.expected:.3f}", va="center", fontsize=10)
    if stage_rank >= PHASES.index("actual"):
        axis.scatter([0], [row.actual], s=95, color=COLORS["actual"], zorder=5)
        axis.text(0.05, row.actual, f"U_t = {row.actual:.3f}", va="center", fontsize=10)
    if stage_rank >= PHASES.index("error"):
        axis.annotate(
            "",
            xy=(0, row.actual),
            xytext=(0, row.expected),
            arrowprops={"arrowstyle": "<->", "color": COLORS["error"], "linewidth": 2.5},
        )
        midpoint = (row.actual + row.expected) / 2
        axis.text(-0.45, midpoint, f"P_t = {row.error:+.3f}", color=COLORS["error"], fontsize=10)
    axis.set_xlim(-0.55, 0.75)
    axis.set_xticks([])


def _draw_memory_update_1d(axis: Axes, row: PrimitiveStep, stage_rank: int, alpha: float) -> None:
    _prepare_value_axis(axis, "C. Memory update")
    visible_memory = row.memory_previous
    if stage_rank >= PHASES.index("update"):
        visible_memory = row.memory_previous + alpha * (row.memory - row.memory_previous)
    if stage_rank >= PHASES.index("carry"):
        visible_memory = row.memory
    axis.scatter([0], [row.memory_previous], s=80, color=COLORS["prior"], label="prior memory")
    axis.scatter([0.45], [visible_memory], s=110, color=COLORS["memory"], label="memory now")
    axis.plot([0, 0.45], [row.memory_previous, visible_memory], color=COLORS["memory"], linewidth=2)
    if stage_rank >= PHASES.index("actual"):
        axis.scatter([0.9], [row.actual], s=85, color=COLORS["actual"], label="actual")
    axis.text(0, row.memory_previous + 0.04, f"M_{{t-1}} = {row.memory_previous:.3f}", ha="center", fontsize=9)
    axis.text(0.45, visible_memory + 0.04, f"M_t = {visible_memory:.3f}", ha="center", fontsize=9)
    axis.set_xlim(-0.25, 1.15)
    axis.set_xticks([0, 0.45, 0.9])
    axis.set_xticklabels(["prior", "updated", "actual"], fontsize=9)
    axis.legend(loc="upper left", fontsize=8, frameon=False)


def _draw_timeline_1d(
    axis: Axes,
    rows: Sequence[PrimitiveStep],
    index: int,
    stage_rank: int,
    alpha: float,
) -> None:
    _style_axis(axis, "D. Running tape")
    times = np.array([row.t for row in rows])
    shown = slice(0, index + 1)
    actual = np.array([row.actual for row in rows], dtype=float)
    expected = np.array([row.expected for row in rows], dtype=float)
    memory = np.array([row.memory for row in rows], dtype=float)
    error = np.array([row.error for row in rows], dtype=float)
    if stage_rank < PHASES.index("actual"):
        actual[index] = np.nan
    if stage_rank < PHASES.index("error"):
        error[index] = np.nan
    if stage_rank < PHASES.index("update"):
        memory[index] = rows[index].memory_previous
    else:
        memory[index] = rows[index].memory_previous + alpha * (rows[index].memory - rows[index].memory_previous)
    axis.plot(times[shown], expected[shown], color=COLORS["expected"], marker="o", label="expected")
    axis.plot(times[shown], actual[shown], color=COLORS["actual"], marker="o", label="actual")
    axis.plot(times[shown], memory[shown], color=COLORS["memory"], marker="o", label="memory")
    axis.bar(times[shown], error[shown], color=COLORS["error"], alpha=0.25, label="signed error")
    axis.axvline(rows[index].t, color=COLORS["text"], linewidth=1.2, alpha=0.55)
    axis.set_xlim(0.6, len(rows) + 0.4)
    axis.set_ylim(-0.42, 1.02)
    axis.set_xlabel("timestep")
    axis.legend(loc="upper left", ncol=2, fontsize=8, frameon=False)


def _draw_2d_dashboard(
    fig,
    axes: Sequence[Axes],
    rows: Sequence[PrimitiveStep2D],
    index: int,
    phase: str,
    alpha: float,
) -> None:
    fig.suptitle(
        "Primitive Cave in a two-feature plane",
        fontsize=15,
        fontweight="bold",
        color=COLORS["text"],
    )
    for axis in axes:
        axis.clear()
    stage_rank = PHASES.index(phase)
    row = rows[index]
    _draw_feature_plane(axes[0], rows, index, row, stage_rank, alpha)
    _draw_2d_tape(axes[1], rows, index, stage_rank, alpha)
    _draw_static_trajectory(axes[2], rows[: index + 1])
    _draw_static_surface_panel(axes[3], rows[: index + 1])


def _draw_feature_plane(
    axis: Axes,
    rows: Sequence[PrimitiveStep2D],
    index: int,
    row: PrimitiveStep2D,
    stage_rank: int,
    alpha: float,
) -> None:
    _style_axis(axis, "Expectation and actual as feature vectors")
    axis.set_xlim(-0.05, 0.95)
    axis.set_ylim(-0.05, 0.95)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("feature x: angularity")
    axis.set_ylabel("feature y: brightness")
    actual = np.array(row.actual)
    expected = np.array(row.expected)
    memory_previous = np.array(row.memory_previous)
    memory = np.array(row.memory)
    visible_memory = memory_previous
    if stage_rank >= PHASES.index("update"):
        visible_memory = memory_previous + alpha * (memory - memory_previous)
    if stage_rank >= PHASES.index("carry"):
        visible_memory = memory
    previous_memories = np.array([r.memory for r in rows[:index]], dtype=float)
    if len(previous_memories):
        axis.plot(previous_memories[:, 0], previous_memories[:, 1], color=COLORS["memory"], alpha=0.45, linewidth=1.8)
    axis.scatter([expected[0]], [expected[1]], s=100, color=COLORS["expected"], label="E_t expected", zorder=4)
    if stage_rank >= PHASES.index("actual"):
        axis.scatter([actual[0]], [actual[1]], s=100, color=COLORS["actual"], label="U_t actual", zorder=5)
    if stage_rank >= PHASES.index("error"):
        arrow = FancyArrowPatch(
            expected,
            actual,
            arrowstyle="->",
            color=COLORS["error"],
            linewidth=2.5,
            mutation_scale=14,
        )
        axis.add_patch(arrow)
    axis.scatter([visible_memory[0]], [visible_memory[1]], s=90, color=COLORS["memory"], label="M_t memory", zorder=6)
    axis.text(0.02, 0.90, f"t = {row.t}", fontsize=12, fontweight="bold")
    axis.text(0.02, 0.84, f"|P_t| = {row.surprise:.3f}", fontsize=10, color=COLORS["error"])
    axis.legend(loc="lower right", fontsize=8, frameon=False)


def _draw_2d_tape(axis: Axes, rows: Sequence[PrimitiveStep2D], index: int, stage_rank: int, alpha: float) -> None:
    _style_axis(axis, "Feature values over time")
    times = np.array([row.t for row in rows])
    actual_x = np.array([row.actual[0] for row in rows], dtype=float)
    expected_x = np.array([row.expected[0] for row in rows], dtype=float)
    memory_x = np.array([row.memory[0] for row in rows], dtype=float)
    surprise = np.array([row.surprise for row in rows], dtype=float)
    if stage_rank < PHASES.index("actual"):
        actual_x[index] = np.nan
    if stage_rank < PHASES.index("update"):
        memory_x[index] = rows[index].memory_previous[0]
    else:
        memory_x[index] = rows[index].memory_previous[0] + alpha * (
            rows[index].memory[0] - rows[index].memory_previous[0]
        )
    shown = slice(0, index + 1)
    axis.plot(times[shown], expected_x[shown], color=COLORS["expected"], marker="o", label="expected x")
    axis.plot(times[shown], actual_x[shown], color=COLORS["actual"], marker="o", label="actual x")
    axis.plot(times[shown], memory_x[shown], color=COLORS["memory"], marker="o", label="memory x")
    axis.bar(times[shown], surprise[shown], color=COLORS["error"], alpha=0.20, label="surprise")
    axis.axvline(rows[index].t, color=COLORS["text"], linewidth=1.2, alpha=0.55)
    axis.set_xlim(0.6, len(rows) + 0.4)
    axis.set_ylim(0.0, 0.95)
    axis.set_xlabel("timestep")
    axis.legend(loc="upper left", ncol=2, fontsize=8, frameon=False)


def _draw_static_trajectory(axis: Axes, rows: Sequence[PrimitiveStep2D]) -> None:
    _style_axis(axis, "Subjective trajectory: E_t -> U_t -> M_t")
    axis.set_xlim(-0.05, 0.95)
    axis.set_ylim(-0.05, 0.95)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("feature x: angularity")
    axis.set_ylabel("feature y: brightness")
    expected = np.array([row.expected for row in rows], dtype=float)
    actual = np.array([row.actual for row in rows], dtype=float)
    memory = np.array([row.memory for row in rows], dtype=float)
    axis.plot(memory[:, 0], memory[:, 1], color=COLORS["memory"], linewidth=2.2, alpha=0.8, label="memory path")
    axis.scatter(expected[:, 0], expected[:, 1], s=72, color=COLORS["expected"], label="E_t expected", zorder=4)
    axis.scatter(actual[:, 0], actual[:, 1], s=72, color=COLORS["actual"], label="U_t actual", zorder=5)
    axis.scatter(memory[:, 0], memory[:, 1], s=72, color=COLORS["memory"], label="M_t memory", zorder=6)
    for row in rows:
        start = np.array(row.expected)
        actual_point = np.array(row.actual)
        memory_point = np.array(row.memory)
        axis.add_patch(
            FancyArrowPatch(
                start,
                actual_point,
                arrowstyle="->",
                color=COLORS["error"],
                linewidth=1.6,
                alpha=0.8,
                mutation_scale=10,
            )
        )
        axis.plot(
            [actual_point[0], memory_point[0]],
            [actual_point[1], memory_point[1]],
            color=COLORS["memory"],
            linewidth=1.2,
            alpha=0.55,
        )
        axis.text(memory_point[0] + 0.012, memory_point[1] + 0.012, str(row.t), fontsize=8)
    axis.text(
        0.02,
        0.91,
        "Each numbered point is the updated memory state after comparison.",
        fontsize=9,
        color=COLORS["text"],
    )
    axis.legend(loc="lower right", fontsize=8, frameon=False)


def _draw_static_surface(
    axis: Axes,
    xx: np.ndarray,
    yy: np.ndarray,
    density: np.ndarray,
    rows: Sequence[PrimitiveStep2D],
    density_by_step: Sequence[float],
) -> None:
    _style_axis(axis, "Accumulated topology-like field L_t(x, y)")
    image = axis.contourf(xx, yy, density, levels=16, cmap="viridis")
    axis.contour(xx, yy, density, levels=8, colors="white", linewidths=0.45, alpha=0.55)
    memory = np.array([row.memory for row in rows], dtype=float)
    actual = np.array([row.actual for row in rows], dtype=float)
    expected = np.array([row.expected for row in rows], dtype=float)
    axis.plot(memory[:, 0], memory[:, 1], color="white", linewidth=2.2, alpha=0.95)
    axis.scatter(expected[:, 0], expected[:, 1], s=34, color=COLORS["expected"], edgecolor="white", linewidth=0.5)
    axis.scatter(actual[:, 0], actual[:, 1], s=34, color=COLORS["actual"], edgecolor="white", linewidth=0.5)
    axis.scatter(memory[:, 0], memory[:, 1], s=42, color=COLORS["memory"], edgecolor="white", linewidth=0.6)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("feature x: angularity")
    axis.set_ylabel("feature y: brightness")
    axis.text(
        0.02,
        0.94,
        f"final density mass = {density_by_step[-1]:.1f}",
        fontsize=9,
        color="white",
        bbox={"facecolor": "#111827", "alpha": 0.72, "edgecolor": "none", "pad": 4},
    )
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("accumulated density")


def _draw_static_surface_panel(axis: Axes, rows: Sequence[PrimitiveStep2D]) -> None:
    xx, yy, density, density_by_step = _topology_density(rows)
    _style_axis(axis, "Accumulated topology-like field L_t(x, y)")
    axis.contourf(xx, yy, density, levels=14, cmap="viridis")
    axis.contour(xx, yy, density, levels=7, colors="white", linewidths=0.4, alpha=0.55)
    memory = np.array([row.memory for row in rows], dtype=float)
    actual = np.array([row.actual for row in rows], dtype=float)
    expected = np.array([row.expected for row in rows], dtype=float)
    axis.plot(memory[:, 0], memory[:, 1], color="white", linewidth=2.0, alpha=0.95)
    axis.scatter(expected[:, 0], expected[:, 1], s=30, color=COLORS["expected"], edgecolor="white", linewidth=0.4)
    axis.scatter(actual[:, 0], actual[:, 1], s=30, color=COLORS["actual"], edgecolor="white", linewidth=0.4)
    axis.scatter(memory[:, 0], memory[:, 1], s=36, color=COLORS["memory"], edgecolor="white", linewidth=0.5)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("feature x: angularity")
    axis.set_ylabel("feature y: brightness")
    axis.text(
        0.03,
        0.94,
        f"density mass = {density_by_step[-1]:.1f}",
        transform=axis.transAxes,
        fontsize=8,
        color="white",
        bbox={"facecolor": "#111827", "alpha": 0.72, "edgecolor": "none", "pad": 3},
    )


def _topology_density(
    rows: Sequence[PrimitiveStep2D],
    *,
    grid_size: int = 90,
    sigma: float = 0.055,
    decay: float = 0.94,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    grid = np.linspace(0.0, 1.0, grid_size)
    xx, yy = np.meshgrid(grid, grid)
    density = np.zeros_like(xx)
    density_by_step: list[float] = []
    weights = {"expected": 0.55, "actual": 1.00, "memory": 0.82}
    for row in rows:
        density *= decay
        density += weights["expected"] * _gaussian_2d(xx, yy, row.expected, sigma)
        density += weights["actual"] * _gaussian_2d(xx, yy, row.actual, sigma)
        density += weights["memory"] * _gaussian_2d(xx, yy, row.memory, sigma)
        density_by_step.append(float(density.sum()))
    return xx, yy, density, density_by_step


def _gaussian_2d(
    xx: np.ndarray,
    yy: np.ndarray,
    center: tuple[float, float],
    sigma: float,
) -> np.ndarray:
    cx, cy = center
    distance_squared = (xx - cx) ** 2 + (yy - cy) ** 2
    return np.exp(-distance_squared / (2 * sigma**2))


def _prepare_value_axis(axis: Axes, title: str) -> None:
    _style_axis(axis, title)
    axis.set_ylim(-0.05, 1.05)
    axis.set_ylabel("scalar feature value")


def _style_axis(axis: Axes, title: str) -> None:
    axis.set_title(title, loc="left", fontsize=11, fontweight="bold", color=COLORS["text"])
    axis.grid(True, color=COLORS["grid"], linewidth=0.8, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def run_demo(output_dir: Path, *, fps: int = 4, subframes: int = 3) -> dict[str, str]:
    rows_1d = rollout_1d()
    rows_2d = rollout_2d()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "rollout_1d_csv": output_dir / "primitive_rollout_1d.csv",
        "rollout_1d_json": output_dir / "primitive_rollout_1d.json",
        "frame_1d": output_dir / "primitive_loop_1d.png",
        "animation_1d": output_dir / "primitive_loop_1d.gif",
        "rollout_2d_json": output_dir / "primitive_rollout_2d.json",
        "frame_2d": output_dir / "primitive_loop_2d.png",
        "animation_2d": output_dir / "primitive_loop_2d.gif",
        "trajectory_surface": output_dir / "primitive_trajectory_surface.png",
    }

    save_rollout_csv(rows_1d, paths["rollout_1d_csv"])
    save_json([asdict(row) for row in rows_1d], paths["rollout_1d_json"])
    save_primitive_1d_frame(rows_1d, paths["frame_1d"])
    save_primitive_1d_animation(rows_1d, paths["animation_1d"], fps=fps, subframes=subframes)

    save_json([asdict(row) for row in rows_2d], paths["rollout_2d_json"])
    save_primitive_2d_frame(rows_2d, paths["frame_2d"])
    save_primitive_2d_animation(rows_2d, paths["animation_2d"], fps=fps, subframes=subframes)
    save_primitive_trajectory_surface(rows_2d, paths["trajectory_surface"])

    return {name: str(path) for name, path in paths.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the primitive Cave recurrence demo.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "out",
        help="Directory for generated rollout data, frames, and GIFs.",
    )
    parser.add_argument("--fps", type=int, default=4, help="GIF frames per second.")
    parser.add_argument("--subframes", type=int, default=3, help="Interpolation frames per phase.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_demo(args.output_dir, fps=args.fps, subframes=args.subframes)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
