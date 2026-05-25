from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from cave.presentation.renderers.matplotlib_renderer.styles import RendererStyle, apply_axis_style, resolve_style
from cave.observation.views import ActionViewState


def draw_action(
    axis: Axes,
    view_state: ActionViewState,
    style: RendererStyle | str | None = None,
) -> None:
    resolved_style = resolve_style(style)
    apply_axis_style(axis, resolved_style, view_state.title)

    duration = max(0.001, view_state.duration)
    axis.set_xlim(0.0, duration)
    axis.set_ylim(-0.05, 1.75)
    axis.set_xlabel("time")
    axis.set_ylabel("strength / exposure")
    axis.grid(True, color=resolved_style.grid_color, linewidth=resolved_style.linewidth(0.8), alpha=0.8)
    axis.set_axisbelow(True)
    axis.axhline(
        1.0,
        color=resolved_style.color("neutral", "#6b7280"),
        linewidth=resolved_style.linewidth(0.8),
        linestyle=":",
        alpha=0.8,
    )

    times = np.array([point.t for point in view_state.points], dtype=float)
    if times.size:
        strengths = np.array([point.strength for point in view_state.points], dtype=float)
        exposures = np.array([point.exposure for point in view_state.points], dtype=float)
        utility = np.array(
            [point.expected_utility_delta for point in view_state.points],
            dtype=float,
        )
        utility = np.clip(utility, -1.0, 1.0)
        axis.plot(
            times,
            strengths,
            color=resolved_style.color("strength", "#2563eb"),
            linewidth=resolved_style.linewidth(1.8),
            label="strength",
        )
        axis.plot(
            times,
            exposures,
            color=resolved_style.color("exposure", "#111827"),
            linewidth=resolved_style.linewidth(1.8),
            linestyle="--",
            label="exposure",
        )
        axis.plot(
            times,
            utility,
            color=resolved_style.color("utility", "#2f855a"),
            linewidth=resolved_style.linewidth(1.3),
            linestyle=":",
            label="utility delta",
        )
        for point in view_state.points:
            if point.kind == "maintain":
                continue
            color = (
                resolved_style.color("pleasure", "#2f855a")
                if point.kind == "approach"
                else resolved_style.color("pain", "#b42318")
            )
            axis.scatter(
                [point.t],
                [point.exposure],
                s=resolved_style.markersize(48),
                color=color,
                edgecolor=resolved_style.color("memory_edge", "#1f2933"),
                linewidth=resolved_style.linewidth(0.7),
                zorder=5,
            )

    current = view_state.current
    axis.axvline(
        current.t,
        color=resolved_style.color("interval_label", "#334e68"),
        linewidth=resolved_style.linewidth(1.0),
        alpha=0.7,
    )
    label_target = current.target_id or "none"
    axis.text(
        0.99,
        0.97,
        f"{current.kind}\n{label_target}\nexposure {current.exposure:.2f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=resolved_style.text_color,
        bbox=resolved_style.text_box(),
    )
    axis.legend(loc="lower left", fontsize=8, ncol=3, frameon=False, labelcolor=resolved_style.text_color)
