from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from cave.presentation.renderers.matplotlib_renderer.styles import RendererStyle, apply_axis_style, resolve_style
from cave.observation.views import AffectViewState


def draw_affect(
    axis: Axes,
    view_state: AffectViewState,
    style: RendererStyle | str | None = None,
) -> None:
    resolved_style = resolve_style(style)
    apply_axis_style(axis, resolved_style, view_state.title)

    duration = max(0.001, view_state.duration)
    axis.set_xlim(0.0, duration)
    values = []
    for point in view_state.points:
        values.extend(
            [
                point.pain,
                point.pleasure,
                abs(point.net),
                point.surprise,
                abs(point.utility),
            ]
        )
    y_max = max(1.0, max(values) if values else 1.0)
    axis.set_ylim(-y_max * 1.05, y_max * 1.12)
    axis.set_xlabel("time")
    axis.set_ylabel("value")
    axis.grid(True, color=resolved_style.grid_color, linewidth=resolved_style.linewidth(0.8), alpha=0.8)
    axis.set_axisbelow(True)
    axis.axhline(
        0.0,
        color=resolved_style.color("neutral", "#6b7280"),
        linewidth=resolved_style.linewidth(0.8),
        alpha=0.7,
    )

    times = np.array([point.t for point in view_state.points], dtype=float)
    if times.size:
        series = [
            ("pain", [point.pain for point in view_state.points], resolved_style.color("pain", "#b42318"), 1.9, "-"),
            (
                "pleasure",
                [point.pleasure for point in view_state.points],
                resolved_style.color("pleasure", "#2f855a"),
                1.9,
                "-",
            ),
            ("net", [point.net for point in view_state.points], resolved_style.color("after", "#111827"), 1.7, "--"),
            (
                "surprise",
                [point.surprise for point in view_state.points],
                resolved_style.color("positive_error", "#c2410c"),
                1.2,
                ":",
            ),
            (
                "attention",
                [point.attention for point in view_state.points],
                resolved_style.color("strength", "#2563eb"),
                1.2,
                "-.",
            ),
        ]
        for label, values, color, linewidth, linestyle in series:
            axis.plot(
                times,
                np.array(values, dtype=float),
                color=color,
                linewidth=resolved_style.linewidth(linewidth),
                linestyle=linestyle,
                alpha=0.88,
                label=label,
            )

    current = view_state.current
    axis.axvline(
        current.t,
        color=resolved_style.color("interval_label", "#334e68"),
        linewidth=resolved_style.linewidth(1.0),
        alpha=0.7,
    )
    axis.scatter(
        [current.t, current.t],
        [current.pain, current.pleasure],
        s=[resolved_style.markersize(48), resolved_style.markersize(48)],
        color=[
            resolved_style.color("pain", "#b42318"),
            resolved_style.color("pleasure", "#2f855a"),
        ],
        edgecolor=resolved_style.color("memory_edge", "#1f2933"),
        linewidth=resolved_style.linewidth(0.7),
        zorder=5,
    )
    axis.text(
        0.99,
        0.97,
        f"pain {current.pain:.3f}\npleasure {current.pleasure:.3f}\nutility {current.utility:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=resolved_style.text_color,
        bbox=resolved_style.text_box(),
    )
    axis.legend(loc="lower left", fontsize=8, ncol=3, frameon=False, labelcolor=resolved_style.text_color)
