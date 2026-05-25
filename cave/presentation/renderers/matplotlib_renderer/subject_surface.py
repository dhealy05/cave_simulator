from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Wedge

from cave.observation.views import SubjectSurfaceViewState
from cave.presentation.renderers.matplotlib_renderer.styles import RendererStyle, apply_axis_style, resolve_style


_CENTER = np.array([0.5, 0.55])
_BODY_RADIUS = 0.31


def draw_subject_surface(
    axis: Axes,
    view_state: SubjectSurfaceViewState,
    style: RendererStyle | str | None = None,
) -> None:
    resolved_style = resolve_style(style)
    apply_axis_style(axis, resolved_style, view_state.title)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])

    _draw_pressure_field(axis, view_state, resolved_style)
    _draw_surprise_ripples(axis, view_state, resolved_style)
    _draw_body(axis, view_state, resolved_style)
    _draw_receptors(axis, view_state, resolved_style)
    _draw_gate(axis, view_state, resolved_style)
    _draw_carry(axis, view_state, resolved_style)
    _draw_input_pulse(axis, view_state, resolved_style)
    _draw_trail(axis, view_state, resolved_style)
    _draw_readout(axis, view_state, resolved_style)


def _draw_pressure_field(axis: Axes, view_state: SubjectSurfaceViewState, style: RendererStyle) -> None:
    if view_state.valence > 0:
        color = style.color("pleasure", "#2f855a")
    elif view_state.valence < 0:
        color = style.color("pain", "#b42318")
    else:
        color = style.color("neutral", "#6b7280")
    pressure = float(np.clip(view_state.pressure, 0.0, 1.0))
    axis.add_patch(
        Circle(
            tuple(_CENTER),
            radius=_BODY_RADIUS + 0.12 + 0.03 * pressure,
            facecolor=color,
            edgecolor="none",
            alpha=0.06 + 0.16 * pressure,
            zorder=0,
        )
    )


def _draw_surprise_ripples(axis: Axes, view_state: SubjectSurfaceViewState, style: RendererStyle) -> None:
    surprise = float(np.clip(view_state.surprise, 0.0, 1.0))
    if surprise <= 1e-6:
        return
    color = style.color("positive_error", "#c2410c")
    for index, radius in enumerate((_BODY_RADIUS + 0.04, _BODY_RADIUS + 0.095)):
        axis.add_patch(
            Circle(
                tuple(_CENTER),
                radius=radius + 0.025 * surprise * index,
                facecolor="none",
                edgecolor=color,
                linewidth=style.linewidth(0.8 + 0.4 * surprise),
                alpha=(0.24 - 0.07 * index) * surprise,
                zorder=1,
            )
        )


def _draw_body(axis: Axes, view_state: SubjectSurfaceViewState, style: RendererStyle) -> None:
    carry = abs(float(np.clip(view_state.carry, -1.0, 1.0)))
    axis.add_patch(
        Circle(
            tuple(_CENTER),
            radius=_BODY_RADIUS,
            facecolor=style.color("well_fill", "#f7f1dd"),
            edgecolor=style.color("memory_edge", "#1f2933"),
            linewidth=style.linewidth(2.0),
            alpha=0.93,
            zorder=3,
        )
    )
    axis.add_patch(
        Circle(
            tuple(_CENTER),
            radius=_BODY_RADIUS * (0.72 + 0.08 * carry),
            facecolor=style.color("corridor", "#d6d9de"),
            edgecolor="none",
            alpha=0.22,
            zorder=4,
        )
    )


def _draw_receptors(axis: Axes, view_state: SubjectSurfaceViewState, style: RendererStyle) -> None:
    channels = view_state.active_channels or ("input",)
    count = max(3, min(7, len(channels) + 2))
    angles = np.linspace(25.0, 155.0, count)
    active = min(len(channels), count)
    for index, angle in enumerate(angles):
        theta = np.deg2rad(angle)
        pos = _CENTER + (_BODY_RADIUS + 0.018) * np.array([np.cos(theta), np.sin(theta)])
        active_alpha = 0.9 if index < active else 0.34
        axis.add_patch(
            Circle(
                tuple(pos),
                radius=0.014 if index < active else 0.01,
                facecolor=style.color("attention", "#2f855a"),
                edgecolor=style.color("memory_edge", "#1f2933"),
                linewidth=style.linewidth(0.45),
                alpha=active_alpha,
                zorder=7,
            )
        )


def _draw_gate(axis: Axes, view_state: SubjectSurfaceViewState, style: RendererStyle) -> None:
    aperture = float(np.clip(view_state.aperture, 0.0, 1.0))
    gate_radius = 0.045 + 0.105 * aperture
    gate_center = _CENTER + np.array([0.0, 0.065])
    axis.add_patch(
        Ellipse(
            tuple(gate_center),
            width=0.12 + 0.23 * aperture,
            height=0.06 + 0.19 * aperture,
            facecolor=style.axes_facecolor,
            edgecolor=style.color("memory_edge", "#1f2933"),
            linewidth=style.linewidth(1.2),
            alpha=0.96,
            zorder=8,
        )
    )
    axis.add_patch(
        Circle(
            tuple(gate_center),
            radius=gate_radius,
            facecolor=style.color("exposure", "#111827"),
            edgecolor="none",
            alpha=0.12 + 0.46 * aperture,
            zorder=9,
        )
    )
    for sign in (-1.0, 1.0):
        axis.plot(
            [gate_center[0] - 0.15, gate_center[0] + 0.15],
            [gate_center[1] + sign * (0.035 + 0.095 * aperture)] * 2,
            color=style.color("memory_edge", "#1f2933"),
            linewidth=style.linewidth(1.0),
            alpha=0.5,
            zorder=10,
        )


def _draw_carry(axis: Axes, view_state: SubjectSurfaceViewState, style: RendererStyle) -> None:
    carry = float(np.clip(view_state.carry, -1.0, 1.0))
    axis.add_patch(
        Wedge(
            tuple(_CENTER),
            _BODY_RADIUS * 0.66,
            0.0,
            360.0,
            width=0.026,
            facecolor=style.color("corridor", "#d6d9de"),
            edgecolor="none",
            alpha=0.46,
            zorder=5,
        )
    )
    if abs(carry) <= 1e-6:
        return
    color = style.color("pleasure", "#2f855a") if carry > 0.0 else style.color("pain", "#b42318")
    sweep = 320.0 * abs(carry)
    axis.add_patch(
        Wedge(
            tuple(_CENTER),
            _BODY_RADIUS * 0.66,
            90.0,
            90.0 + sweep,
            width=0.026,
            facecolor=color,
            edgecolor="none",
            alpha=0.34 + 0.52 * abs(carry),
            zorder=6,
        )
    )
    angle = np.deg2rad(90.0 + sweep)
    bead = _CENTER + _BODY_RADIUS * 0.54 * np.array([np.cos(angle), np.sin(angle)])
    axis.add_patch(
        Circle(
            tuple(bead),
            radius=0.022 + 0.012 * abs(carry),
            facecolor=color,
            edgecolor=style.color("memory_edge", "#1f2933"),
            linewidth=style.linewidth(0.6),
            alpha=0.9,
            zorder=11,
        )
    )


def _draw_input_pulse(axis: Axes, view_state: SubjectSurfaceViewState, style: RendererStyle) -> None:
    if view_state.valence > 0:
        color = style.color("pleasure", "#2f855a")
    elif view_state.valence < 0:
        color = style.color("pain", "#b42318")
    else:
        color = style.color("attention", "#2f855a")
    aperture = float(np.clip(view_state.aperture, 0.0, 1.0))
    start = (0.5, 0.96)
    end = (0.5, 0.69 + 0.035 * aperture)
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10 + 12 * aperture,
            linewidth=style.linewidth(0.9 + aperture),
            color=color,
            alpha=0.18 + 0.52 * aperture,
            zorder=2,
        )
    )
    axis.add_patch(
        Circle(
            start,
            radius=0.018 + 0.025 * float(np.clip(view_state.pressure, 0.0, 1.0)),
            facecolor=color,
            edgecolor=style.color("memory_edge", "#1f2933"),
            linewidth=style.linewidth(0.5),
            alpha=0.62,
            zorder=12,
        )
    )


def _draw_trail(axis: Axes, view_state: SubjectSurfaceViewState, style: RendererStyle) -> None:
    points = view_state.trail_points[-40:]
    if not points:
        return
    xs = np.linspace(0.22, 0.78, len(points))
    apertures = np.array([point.aperture for point in points], dtype=float)
    carries = np.array([point.carry for point in points], dtype=float)
    axis.plot(
        xs,
        0.12 + 0.075 * np.clip(apertures, 0.0, 1.0),
        color=style.color("exposure", "#111827"),
        linewidth=style.linewidth(1.3),
        alpha=0.7,
        zorder=4,
    )
    axis.plot(
        xs,
        0.12 + 0.04 * np.clip(carries, -1.0, 1.0),
        color=style.color("memory", "#2f855a"),
        linewidth=style.linewidth(1.0),
        alpha=0.58,
        zorder=4,
    )
    axis.scatter(
        [xs[-1]],
        [0.12 + 0.075 * float(np.clip(apertures[-1], 0.0, 1.0))],
        s=style.markersize(36),
        color=style.color("current", "#c2410c"),
        edgecolor=style.color("current_edge", "#7c2d12"),
        linewidth=style.linewidth(0.5),
        zorder=12,
    )


def _draw_readout(axis: Axes, view_state: SubjectSurfaceViewState, style: RendererStyle) -> None:
    axis.text(
        0.5,
        0.03,
        (
            f"{view_state.input_label}   gate {view_state.aperture:.2f}   "
            f"carry {view_state.carry:+.2f}   utility {view_state.utility:+.2f}"
        ),
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color=style.muted_text_color,
    )
    axis.text(
        0.04,
        0.93,
        view_state.mode,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=style.text_color,
        bbox=style.text_box(alpha=0.72),
    )
