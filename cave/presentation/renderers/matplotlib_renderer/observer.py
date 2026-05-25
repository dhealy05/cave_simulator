from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Ellipse

from cave.observation.views import ObserverViewState
from cave.presentation.renderers.matplotlib_renderer.styles import RendererStyle, apply_axis_style, resolve_style


def draw_observer(
    axis: Axes,
    view_state: ObserverViewState,
    style: RendererStyle | str | None = None,
) -> None:
    resolved_style = resolve_style(style)
    apply_axis_style(axis, resolved_style, view_state.title)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])

    _draw_trail(axis, view_state, resolved_style)
    _draw_eye(axis, view_state, resolved_style)
    _draw_readout(axis, view_state, resolved_style)


def _draw_eye(axis: Axes, view_state: ObserverViewState, style: RendererStyle) -> None:
    openness = float(np.clip(view_state.openness, 0.0, 1.0))
    aperture = 0.08 + 0.34 * openness
    center = np.array([0.5 + 0.055 * view_state.gaze_x, 0.56 + 0.035 * view_state.gaze_y])
    utility = float(np.clip(view_state.utility, -1.0, 1.0))
    if utility >= 0.0:
        halo_color = style.color("pleasure", "#2f855a")
        halo_alpha = 0.08 + 0.22 * abs(utility)
    else:
        halo_color = style.color("pain", "#b42318")
        halo_alpha = 0.08 + 0.22 * abs(utility)
    axis.add_patch(
        Ellipse(
            (0.5, 0.56),
            width=0.78,
            height=0.38 + 0.16 * openness,
            facecolor=halo_color,
            edgecolor="none",
            alpha=halo_alpha,
            zorder=1,
        )
    )
    axis.add_patch(
        Ellipse(
            (0.5, 0.56),
            width=0.72,
            height=aperture,
            facecolor=style.color("well_fill", "#f7f1dd"),
            edgecolor=style.color("memory_edge", "#1f2933"),
            linewidth=style.linewidth(2.0),
            zorder=3,
        )
    )
    iris_radius = 0.095 + 0.035 * openness
    iris_color = _mix_color(
        style.color("negative_error", "#2563eb"),
        style.color("positive_error", "#c2410c"),
        float(np.clip(max(view_state.error, view_state.surprise), 0.0, 1.0)),
    )
    axis.add_patch(
        Circle(
            center,
            radius=iris_radius,
            facecolor=iris_color,
            edgecolor=style.color("memory_edge", "#1f2933"),
            linewidth=style.linewidth(1.4),
            alpha=0.92,
            zorder=4,
        )
    )
    pupil_radius = 0.026 + 0.052 * float(np.clip(view_state.pupil_scale, 0.0, 1.0))
    axis.add_patch(
        Circle(
            center,
            radius=pupil_radius,
            facecolor=style.color("after", "#111827"),
            edgecolor=style.color("after", "#111827"),
            linewidth=style.linewidth(0.8),
            zorder=5,
        )
    )
    axis.add_patch(
        Circle(
            (center[0] - 0.025, center[1] + 0.03),
            radius=0.012,
            facecolor="#ffffff",
            edgecolor="none",
            alpha=0.72,
            zorder=6,
        )
    )
    upper_y = 0.56 + aperture * 0.58
    lower_y = 0.56 - aperture * 0.58
    axis.plot(
        [0.17, 0.32, 0.5, 0.68, 0.83],
        [0.56, upper_y + 0.04, upper_y, upper_y + 0.04, 0.56],
        color=style.color("memory_edge", "#1f2933"),
        linewidth=style.linewidth(1.6),
        zorder=7,
    )
    axis.plot(
        [0.17, 0.32, 0.5, 0.68, 0.83],
        [0.56, lower_y - 0.03, lower_y, lower_y - 0.03, 0.56],
        color=style.color("memory_edge", "#1f2933"),
        linewidth=style.linewidth(1.2),
        alpha=0.75,
        zorder=7,
    )


def _draw_trail(axis: Axes, view_state: ObserverViewState, style: RendererStyle) -> None:
    points = view_state.trail_points
    if not points:
        return
    xs = np.array([0.5 + 0.28 * point.x for point in points], dtype=float)
    ys = np.array([0.18 + 0.12 * point.y for point in points], dtype=float)
    axis.plot(
        xs,
        ys,
        color=style.color("memory", "#2f855a"),
        linewidth=style.linewidth(1.5),
        alpha=0.58,
        zorder=2,
    )
    attentions = np.array([point.attention for point in points], dtype=float)
    sizes = 12 + 44 * np.clip(attentions, 0.0, 1.0)
    axis.scatter(
        xs,
        ys,
        s=style.markersize(float(np.mean(sizes))),
        color=style.color("memory", "#2f855a"),
        edgecolor=style.color("memory_edge", "#1f2933"),
        linewidth=style.linewidth(0.5),
        alpha=0.34,
        zorder=2,
    )
    axis.scatter(
        [xs[-1]],
        [ys[-1]],
        s=style.markersize(68),
        color=style.color("current", "#c2410c"),
        edgecolor=style.color("current_edge", "#7c2d12"),
        linewidth=style.linewidth(0.8),
        zorder=6,
    )


def _draw_readout(axis: Axes, view_state: ObserverViewState, style: RendererStyle) -> None:
    axis.text(
        0.5,
        0.93,
        f"gaze {view_state.gaze_label}",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color=style.text_color,
        bbox=style.text_box(alpha=0.72),
    )
    axis.text(
        0.5,
        0.035,
        f"open {view_state.openness:.2f}   error {view_state.error:.2f}   utility {view_state.utility:.2f}",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color=style.muted_text_color,
    )


def _mix_color(low: str, high: str, amount: float) -> tuple[float, float, float]:
    amount = float(np.clip(amount, 0.0, 1.0))
    low_rgb = np.array(to_rgb(low), dtype=float)
    high_rgb = np.array(to_rgb(high), dtype=float)
    mixed = (1.0 - amount) * low_rgb + amount * high_rgb
    return float(mixed[0]), float(mixed[1]), float(mixed[2])
