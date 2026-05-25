from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from cave.presentation.renderers.matplotlib_renderer.styles import RendererStyle, apply_axis_style, resolve_style
from cave.observation.views import ExpectationActualViewState


def draw_expectation_actual(
    axis: Axes,
    view_state: ExpectationActualViewState,
    style: RendererStyle | str | None = None,
) -> None:
    resolved_style = resolve_style(style)
    apply_axis_style(axis, resolved_style, view_state.title)

    vocabulary = view_state.vocabulary
    count = len(vocabulary)
    if count == 0:
        axis.axis("off")
        return

    x = np.arange(count, dtype=float)
    expected = np.asarray(view_state.expected_before, dtype=float)
    actual = np.asarray(view_state.actual, dtype=float)
    error = np.asarray(view_state.error, dtype=float)
    after = np.asarray(view_state.expected_after, dtype=float)
    expected_attention = _attention_alpha(view_state.expected_attention)
    actual_attention = _attention_alpha(view_state.actual_attention)
    max_abs = max(
        1.0,
        float(np.max(np.abs(expected))) if expected.size else 0.0,
        float(np.max(np.abs(actual))) if actual.size else 0.0,
        float(np.max(np.abs(error))) if error.size else 0.0,
        float(np.max(np.abs(after))) if after.size else 0.0,
    )

    axis.axhline(
        0.0,
        color=resolved_style.color("neutral", "#6b7280"),
        linewidth=resolved_style.linewidth(0.8),
        alpha=0.7,
    )
    axis.bar(
        x - 0.18,
        expected,
        width=0.34,
        color=resolved_style.color("expected_fill", "#91a7c7"),
        alpha=0.08 + 0.56 * expected_attention,
        edgecolor=resolved_style.color("expected_edge", "#3f5f87"),
        linewidth=resolved_style.linewidth(0.4 + 1.5 * expected_attention),
        label="attended expected",
    )
    axis.bar(
        x + 0.18,
        actual,
        width=0.34,
        color=resolved_style.color("actual", "#2f855a"),
        alpha=0.08 + 0.72 * actual_attention,
        linewidth=resolved_style.linewidth(0.4 + 1.2 * actual_attention),
        label="attended actual",
    )
    error_colors = [
        resolved_style.color("positive_error", "#c2410c")
        if value >= 0.0
        else resolved_style.color("negative_error", "#2563eb")
        for value in error
    ]
    axis.bar(
        x,
        error,
        width=0.12,
        color=error_colors,
        alpha=0.9,
        label="error",
    )
    for index, value in enumerate(after):
        axis.scatter(
            [x[index]],
            [value],
            s=resolved_style.markersize(42),
            facecolors="none",
            edgecolors=resolved_style.color("after", "#111827"),
            linewidth=resolved_style.linewidth(0.5 + 1.8 * actual_attention),
            alpha=0.2 + 0.75 * actual_attention,
            zorder=5,
        )

    axis.set_ylim(-max_abs * 1.15, max_abs * 1.15)
    axis.set_xlim(-0.6, count - 0.4)
    axis.set_xticks(x)
    axis.set_xticklabels(vocabulary, rotation=45, ha="right", fontsize=8)
    axis.set_ylabel("value")
    axis.grid(
        True,
        axis="y",
        color=resolved_style.grid_color,
        linewidth=resolved_style.linewidth(0.8),
        alpha=0.8,
    )
    axis.text(
        0.99,
        0.97,
        f"surprise {view_state.surprise:.3f}\n"
        f"learning {view_state.learning_rate:.3f}\n"
        f"ext {view_state.actual_attention:.2f} / int {view_state.expected_attention:.2f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=resolved_style.text_color,
        bbox=resolved_style.text_box(),
    )
    _draw_attention_gauge(
        axis,
        y=0.12,
        value=actual_attention,
        label="external",
        color=resolved_style.color("actual", "#2f855a"),
        style=resolved_style,
    )
    _draw_attention_gauge(
        axis,
        y=0.05,
        value=expected_attention,
        label="internal",
        color=resolved_style.color("expected_edge", "#3f5f87"),
        style=resolved_style,
    )
    axis.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=resolved_style.text_color)


def _attention_alpha(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _draw_attention_gauge(
    axis: Axes,
    *,
    y: float,
    value: float,
    label: str,
    color: str,
    style: RendererStyle,
) -> None:
    value = _attention_alpha(value)
    x0 = 0.06
    x1 = 0.28
    axis.plot(
        [x0, x1],
        [y, y],
        transform=axis.transAxes,
        color=style.color("neutral", "#6b7280"),
        linewidth=style.linewidth(5.0),
        alpha=0.18,
        solid_capstyle="butt",
        zorder=9,
    )
    axis.plot(
        [x0, x0 + (x1 - x0) * value],
        [y, y],
        transform=axis.transAxes,
        color=color,
        linewidth=style.linewidth(5.0),
        alpha=0.9,
        solid_capstyle="butt",
        zorder=10,
    )
    axis.text(
        x1 + 0.015,
        y,
        f"{label} {value:.2f}",
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=8,
        color=style.text_color,
        zorder=10,
    )
