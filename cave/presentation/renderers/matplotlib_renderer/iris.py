"""Iris-diaphragm "expression" renderer for evolved-subject episodes.

The evolved subject's entire interface to the world is a single scalar gate
(exposure) plus a hidden memory vector. This renderer embodies exactly that and
nothing more: a mechanical iris diaphragm whose aperture *is* the exposure, a
charge ring that shows the hidden state carrying a cue across the delay, and a
paired exposure-over-time trace that certifies the anticipation quantitatively.

This is the "expression" register only: a fixed (already-evolved) controller
enacting its disposition over one lifetime. It deliberately shows no
within-lifetime learning, because the weights are frozen during a run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Polygon, Wedge

from cave.observation.episodes import Episode
from cave.presentation.renderers.matplotlib_renderer.styles import (
    RendererStyle,
    apply_axis_style,
    apply_figure_style,
    resolve_style,
)


_IRIS_CENTER = (0.5, 0.54)
_IRIS_RADIUS = 0.34
_APERTURE_MIN = 0.035
_APERTURE_MAX = 0.30
_BLADE_COUNT = 8


@dataclass(frozen=True)
class IrisFrame:
    """Per-timestep state for the iris expression view."""

    t: float
    aperture: float        # exposure in [0, 1] -> aperture diameter
    phase: str             # "cue" | "gap" | "outcome" | "neutral"
    valence: float         # -1 / 0 / +1, sign of the externally present field
    charge: float          # signed latent carry in [-1, 1]
    utility: float
    outcome_value: float
    label: str


def build_iris_frames(episode: Episode) -> list[IrisFrame]:
    """Project an evolved-subject episode into iris frames.

    Reads exposure from ``observation.attention`` and the carried latent charge
    from ``observation.memory_state`` (the hidden state). The charge axis is the
    good-vs-bad direction in hidden space, so a positive charge literally means
    "this hidden state is decodable as heading toward a good outcome".
    """

    observations = list(episode.observations)
    if not observations:
        raise ValueError("episode has no observations")

    charges = _latent_charge_series(observations)
    frames: list[IrisFrame] = []
    for observation, charge in zip(observations, charges):
        evolved = observation.metadata.get("evolved_subject", {})
        if not isinstance(evolved, dict):
            evolved = {}
        input_id = observation.active_inputs[0] if observation.active_inputs else ""
        outcome_value = float(evolved.get("outcome_value", 0.0))
        future_outcome = str(evolved.get("future_outcome", "neutral"))
        phase, valence, label = _phase_from(input_id, outcome_value, future_outcome)
        frames.append(
            IrisFrame(
                t=float(observation.t),
                aperture=float(np.clip(observation.attention, 0.0, 1.0)),
                phase=phase,
                valence=valence,
                charge=float(charge),
                utility=float(evolved.get("utility", 0.0)),
                outcome_value=outcome_value,
                label=label,
            )
        )
    return frames


def save_iris_expression_animation(
    episode: Episode,
    output: str | Path,
    *,
    fps: int = 4,
    style: str | RendererStyle | None = None,
    max_frames: int = 96,
    title: str = "Iris diaphragm — exposure expression",
) -> None:
    """Render the iris + paired exposure trace as a gif."""

    frames = build_iris_frames(episode)
    resolved_style = resolve_style(style)
    if max_frames > 0 and len(frames) > max_frames:
        indices = np.unique(np.linspace(0, len(frames) - 1, max_frames, dtype=int))
    else:
        indices = np.arange(len(frames), dtype=int)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(resolved_style.rc_params()):
        figure = plt.figure(figsize=(9.4, 4.2), dpi=120)
        apply_figure_style(figure, resolved_style)
        grid = GridSpec(1, 2, width_ratios=[1.0, 1.45], figure=figure, wspace=0.18)
        iris_axis = figure.add_subplot(grid[0, 0])
        trace_axis = figure.add_subplot(grid[0, 1])

        def update(step: int):
            pointer = int(indices[step])
            iris_axis.clear()
            trace_axis.clear()
            draw_iris(iris_axis, frames[pointer], resolved_style, title=title)
            draw_exposure_trace(trace_axis, frames, pointer, resolved_style)
            return iris_axis, trace_axis

        anim = animation.FuncAnimation(
            figure,
            update,
            frames=len(indices),
            interval=1000 / max(1, fps),
            blit=False,
        )
        anim.save(output, writer=animation.PillowWriter(fps=fps), dpi=120)
        plt.close(figure)


def draw_iris(
    axis: Axes,
    frame: IrisFrame,
    style: RendererStyle | str | None = None,
    *,
    title: str = "Iris diaphragm",
) -> None:
    """Draw one mechanical iris diaphragm at the frame's aperture and charge."""

    resolved = resolve_style(style)
    apply_axis_style(axis, resolved, title)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])

    center = np.array(_IRIS_CENTER)
    aperture = _APERTURE_MIN + (_APERTURE_MAX - _APERTURE_MIN) * float(
        np.clip(frame.aperture, 0.0, 1.0)
    )

    _draw_utility_halo(axis, center, frame, resolved)
    _draw_field(axis, center, frame, resolved)
    _draw_blades(axis, center, aperture, resolved)
    _draw_housing(axis, center, resolved)
    _draw_charge_ring(axis, center, frame, resolved)
    _draw_iris_readout(axis, frame, resolved)


def draw_exposure_trace(
    axis: Axes,
    frames: list[IrisFrame],
    pointer: int,
    style: RendererStyle | str | None = None,
) -> None:
    """Draw exposure over time with cue/gap/outcome context and a pointer.

    This is the certifier: in good cycles the exposure line lifts during the
    cue and the neutral gap, *before* the outcome arrives; in bad cycles it
    stays low. Anticipation is read directly off the curve.
    """

    resolved = resolve_style(style)
    apply_axis_style(axis, resolved, "Exposure over time — opens for good, shut for bad", grid=True, grid_axis="y")
    if not frames:
        return

    times = np.array([frame.t for frame in frames], dtype=float)
    apertures = np.array([frame.aperture for frame in frames], dtype=float)
    step = float(times[1] - times[0]) if len(times) > 1 else 1.0

    good = resolved.color("pleasure", "#2f855a")
    bad = resolved.color("pain", "#b42318")

    cues = [(frame.t, frame.valence) for frame in frames if frame.phase == "cue"]
    boundaries = [t for t, _ in cues] + [times[-1] + step]
    for index, (start, valence) in enumerate(cues):
        axis.axvspan(
            start,
            boundaries[index + 1],
            color=good if valence > 0 else bad,
            alpha=0.06,
            zorder=0,
        )

    axis.plot(
        times,
        apertures,
        color=resolved.color("exposure", "#111827"),
        linewidth=resolved.linewidth(2.0),
        zorder=3,
    )

    for frame in frames:
        if frame.phase == "cue":
            axis.scatter(
                frame.t,
                0.04,
                marker="^",
                s=resolved.markersize(40),
                color=good if frame.valence > 0 else bad,
                edgecolor=resolved.color("memory_edge", "#1f2933"),
                linewidth=resolved.linewidth(0.5),
                alpha=0.85,
                zorder=4,
            )
        elif frame.phase == "outcome":
            axis.scatter(
                frame.t,
                0.04,
                marker="o",
                s=resolved.markersize(46),
                color=good if frame.outcome_value > 0 else bad,
                edgecolor=resolved.color("memory_edge", "#1f2933"),
                linewidth=resolved.linewidth(0.5),
                zorder=4,
            )

    current = frames[pointer]
    axis.axvline(
        current.t,
        color=resolved.color("current", "#c2410c"),
        linewidth=resolved.linewidth(1.2),
        alpha=0.7,
        zorder=5,
    )
    axis.scatter(
        [current.t],
        [current.aperture],
        s=resolved.markersize(70),
        color=resolved.color("current", "#c2410c"),
        edgecolor=resolved.color("current_edge", "#7c2d12"),
        linewidth=resolved.linewidth(0.8),
        zorder=6,
    )

    axis.set_xlim(float(times[0]) - step, float(times[-1]) + step)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("time")
    axis.set_ylabel("exposure (aperture)")


def _draw_utility_halo(axis: Axes, center: np.ndarray, frame: IrisFrame, style: RendererStyle) -> None:
    utility = float(np.clip(frame.utility, -1.0, 1.0))
    if abs(utility) < 1e-6:
        return
    color = style.color("pleasure", "#2f855a") if utility >= 0.0 else style.color("pain", "#b42318")
    axis.add_patch(
        Circle(
            tuple(center),
            radius=_IRIS_RADIUS + 0.1,
            facecolor=color,
            edgecolor="none",
            alpha=0.06 + 0.2 * abs(utility),
            zorder=0,
        )
    )


def _draw_field(axis: Axes, center: np.ndarray, frame: IrisFrame, style: RendererStyle) -> None:
    if frame.valence > 0:
        color = style.color("pleasure", "#2f855a")
    elif frame.valence < 0:
        color = style.color("pain", "#b42318")
    else:
        color = style.color("neutral", "#6b7280")
    axis.add_patch(
        Circle(
            tuple(center),
            radius=_IRIS_RADIUS,
            facecolor=color,
            edgecolor="none",
            alpha=0.85 if frame.phase == "outcome" else 0.6,
            zorder=1,
        )
    )


def _draw_blades(axis: Axes, center: np.ndarray, aperture: float, style: RendererStyle) -> None:
    beta = 2.0 * np.pi / _BLADE_COUNT
    twist = beta * 0.6
    blade_light = "#aeb6c2"
    blade_dark = "#878f9c"
    edge = style.color("memory_edge", "#1f2933")
    for k in range(_BLADE_COUNT):
        a1 = k * beta
        a2 = (k + 1) * beta
        outer_1 = center + _IRIS_RADIUS * np.array([np.cos(a1), np.sin(a1)])
        outer_2 = center + _IRIS_RADIUS * np.array([np.cos(a2), np.sin(a2)])
        inner_2 = center + aperture * np.array([np.cos(a2 + twist), np.sin(a2 + twist)])
        inner_1 = center + aperture * np.array([np.cos(a1 + twist), np.sin(a1 + twist)])
        axis.add_patch(
            Polygon(
                [outer_1, outer_2, inner_2, inner_1],
                closed=True,
                facecolor=blade_light if k % 2 == 0 else blade_dark,
                edgecolor=edge,
                linewidth=style.linewidth(0.7),
                joinstyle="miter",
                zorder=3,
            )
        )


def _draw_housing(axis: Axes, center: np.ndarray, style: RendererStyle) -> None:
    axis.add_patch(
        Circle(
            tuple(center),
            radius=_IRIS_RADIUS,
            facecolor="none",
            edgecolor=style.color("memory_edge", "#1f2933"),
            linewidth=style.linewidth(3.0),
            zorder=4,
        )
    )


def _draw_charge_ring(axis: Axes, center: np.ndarray, frame: IrisFrame, style: RendererStyle) -> None:
    inner = _IRIS_RADIUS + 0.025
    width = 0.03
    # faint full track so the ring is always legible
    axis.add_patch(
        Wedge(
            tuple(center),
            inner + width,
            0.0,
            360.0,
            width=width,
            facecolor=style.color("corridor", "#d6d9de"),
            edgecolor="none",
            alpha=0.5,
            zorder=2,
        )
    )
    charge = float(np.clip(frame.charge, -1.0, 1.0))
    if abs(charge) < 1e-6:
        return
    sweep = 350.0 * abs(charge)
    color = style.color("pleasure", "#2f855a") if charge > 0 else style.color("pain", "#b42318")
    axis.add_patch(
        Wedge(
            tuple(center),
            inner + width,
            90.0,
            90.0 + sweep,
            width=width,
            facecolor=color,
            edgecolor="none",
            alpha=0.35 + 0.55 * abs(charge),
            zorder=5,
        )
    )


def _draw_iris_readout(axis: Axes, frame: IrisFrame, style: RendererStyle) -> None:
    axis.text(
        0.5,
        0.965,
        frame.label,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color=style.text_color,
        bbox=style.text_box(alpha=0.72),
    )
    axis.text(
        0.5,
        0.03,
        f"aperture {frame.aperture:.2f}    charge {frame.charge:+.2f}    utility {frame.utility:+.2f}",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color=style.muted_text_color,
    )


def _phase_from(
    input_id: str,
    outcome_value: float,
    future_outcome: str,
) -> tuple[str, float, str]:
    if "cue" in input_id:
        valence = 1.0 if future_outcome == "good" else (-1.0 if future_outcome == "bad" else 0.0)
        return "cue", valence, f"cue: {future_outcome}"
    if "delay" in input_id:
        return "gap", 0.0, "delay"
    if "outcome" in input_id or abs(outcome_value) > 1e-9:
        valence = 1.0 if outcome_value > 0 else (-1.0 if outcome_value < 0 else 0.0)
        return "outcome", valence, f"outcome: {'good' if outcome_value > 0 else 'bad'}"
    return "neutral", 0.0, input_id or "neutral"


def _latent_charge_series(observations) -> np.ndarray:
    hidden = [np.asarray(obs.memory_state, dtype=float) for obs in observations]
    if not hidden or hidden[0].size == 0:
        return np.zeros(len(observations), dtype=float)
    labels = []
    for obs in observations:
        evolved = obs.metadata.get("evolved_subject", {})
        labels.append(str(evolved.get("future_outcome", "neutral")) if isinstance(evolved, dict) else "neutral")
    matrix = np.stack(hidden, axis=0)
    good = matrix[[label == "good" for label in labels]]
    bad = matrix[[label == "bad" for label in labels]]
    if good.size == 0 or bad.size == 0:
        return np.zeros(len(observations), dtype=float)
    axis_vector = np.mean(good, axis=0) - np.mean(bad, axis=0)
    norm = float(np.linalg.norm(axis_vector))
    if norm <= 1e-12:
        return np.zeros(len(observations), dtype=float)
    axis_vector = axis_vector / norm
    raw = matrix @ axis_vector
    scale = float(np.max(np.abs(raw)))
    if scale <= 1e-12:
        return np.zeros(len(observations), dtype=float)
    return raw / scale
