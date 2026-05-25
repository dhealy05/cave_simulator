from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np
from matplotlib.axes import Axes

from cave.observation.experience import feature_axis_label
from cave.presentation.renderers.matplotlib_renderer.styles import RendererStyle, apply_axis_style, resolve_style
from cave.observation.views import BaseViewState, CorrectionViewState


def normalize_correction_series(
    view_states_by_frame: Sequence[Sequence[BaseViewState]],
) -> list[list[BaseViewState]]:
    correction_states: list[CorrectionViewState] = []
    for view_states in view_states_by_frame:
        for view_state in view_states:
            if not isinstance(view_state, CorrectionViewState):
                continue
            if (
                view_state.expected_point is None
                or view_state.actual_point is None
                or view_state.after_point is None
            ):
                continue
            correction_states.append(view_state)

    if not correction_states:
        return [list(view_states) for view_states in view_states_by_frame]

    point_matrix = np.vstack(
        [
            np.asarray(point, dtype=float)
            for view_state in correction_states
            for point in (
                view_state.expected_point,
                view_state.actual_point,
                view_state.after_point,
            )
            if point is not None
        ]
    )
    minimum = point_matrix.min(axis=0)
    maximum = point_matrix.max(axis=0)
    span = maximum - minimum
    constant_axis = span <= 1e-12
    safe_span = np.where(constant_axis, 1.0, span)
    times = np.array([view_state.t for view_state in correction_states], dtype=float)
    time_min = float(times.min())
    time_span = float(times.max() - time_min)
    normalized_times = (
        np.full(times.shape, 0.5, dtype=float)
        if time_span <= 1e-12
        else (times - time_min) / time_span
    )

    def normalize_point(point: np.ndarray | None) -> np.ndarray | None:
        if point is None:
            return None
        normalized = (np.asarray(point, dtype=float) - minimum) / safe_span
        if np.any(constant_axis):
            normalized = normalized.copy()
            normalized[constant_axis] = 0.5
        return np.clip(normalized, 0.0, 1.0)

    expected_series = np.vstack(
        [normalize_point(view_state.expected_point) for view_state in correction_states]
    )
    actual_series = np.vstack(
        [normalize_point(view_state.actual_point) for view_state in correction_states]
    )
    after_series = np.vstack(
        [normalize_point(view_state.after_point) for view_state in correction_states]
    )
    expected_attention_series = np.array(
        [view_state.expected_attention for view_state in correction_states],
        dtype=float,
    )
    actual_attention_series = np.array(
        [view_state.actual_attention for view_state in correction_states],
        dtype=float,
    )
    series_index_by_t = {
        id(view_state): index
        for index, view_state in enumerate(correction_states)
    }

    def normalize_time(value: float) -> float:
        if time_span <= 1e-12:
            return 0.5
        return min(1.0, max(0.0, (float(value) - time_min) / time_span))

    normalized_frames: list[list[BaseViewState]] = []
    for view_states in view_states_by_frame:
        normalized_view_states: list[BaseViewState] = []
        for view_state in view_states:
            if isinstance(view_state, CorrectionViewState):
                series_index = series_index_by_t.get(id(view_state))
                series_slice = slice(0, 0 if series_index is None else series_index + 1)
                normalized_view_states.append(
                    replace(
                        view_state,
                        title="Prediction Correction Over Time",
                        bounds=(0.0, 1.0),
                        expected_point=normalize_point(view_state.expected_point),
                        actual_point=normalize_point(view_state.actual_point),
                        after_point=normalize_point(view_state.after_point),
                        experience_times=[
                            normalize_time(time)
                            for time in view_state.experience_times
                        ],
                        normalized=True,
                        series_times=normalized_times[series_slice].copy(),
                        expected_series=expected_series[series_slice].copy(),
                        actual_series=actual_series[series_slice].copy(),
                        after_series=after_series[series_slice].copy(),
                        expected_attention_series=expected_attention_series[
                            series_slice
                        ].copy(),
                        actual_attention_series=actual_attention_series[
                            series_slice
                        ].copy(),
                    )
                )
            else:
                normalized_view_states.append(view_state)
        normalized_frames.append(normalized_view_states)
    return normalized_frames


def draw_correction(
    axis: Axes,
    view_state: CorrectionViewState,
    style: RendererStyle | str | None = None,
) -> None:
    resolved_style = resolve_style(style)
    if (
        view_state.normalized
        and view_state.series_times is not None
        and view_state.expected_series is not None
        and view_state.actual_series is not None
        and view_state.after_series is not None
    ):
        draw_correction_time_series(axis, view_state, resolved_style)
        return

    apply_axis_style(axis, resolved_style, view_state.title)

    lower, upper = view_state.bounds
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    x_label = feature_axis_label(view_state.feature_x)
    y_label = feature_axis_label(view_state.feature_y)
    if view_state.normalized:
        x_label = f"{x_label} (series-normalized)"
        y_label = f"{y_label} (series-normalized)"
        axis.set_xticks([0.0, 0.5, 1.0])
        axis.set_yticks([0.0, 0.5, 1.0])
        origin = 0.5
    else:
        origin = 0.0
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.axhline(origin, color=resolved_style.grid_color, linewidth=resolved_style.linewidth(0.8), zorder=0)
    axis.axvline(origin, color=resolved_style.grid_color, linewidth=resolved_style.linewidth(0.8), zorder=0)
    axis.grid(True, color=resolved_style.grid_color, linewidth=resolved_style.linewidth(0.8), alpha=0.7)
    axis.set_axisbelow(True)

    expected = view_state.expected_point
    actual = view_state.actual_point
    after = view_state.after_point

    if expected is not None and actual is not None and after is not None:
        surprise = view_state.surprise
        learning_rate = view_state.learning_rate
        expected_attention = _attention_alpha(view_state.expected_attention)
        actual_attention = _attention_alpha(view_state.actual_attention)
        comparison_attention = _attention_alpha(
            0.5 * (view_state.expected_attention + view_state.actual_attention)
        )

        axis.plot(
            [float(expected[0]), float(actual[0])],
            [float(expected[1]), float(actual[1])],
            color=resolved_style.color("positive_error", "#c2410c"),
            linewidth=resolved_style.linewidth(0.7 + 3.2 * comparison_attention),
            linestyle=":",
            alpha=min(1.0, 0.12 + 0.6 * comparison_attention + 0.4 * surprise),
            zorder=2,
        )
        axis.plot(
            [float(actual[0]), float(after[0])],
            [float(actual[1]), float(after[1])],
            color=resolved_style.color("after", "#111827"),
            linewidth=resolved_style.linewidth(
                0.7 + 2.2 * actual_attention + 4.0 * learning_rate
            ),
            alpha=min(1.0, 0.2 + 0.65 * actual_attention),
            zorder=3,
        )

        points = [
            (
                "attended expected",
                expected,
                resolved_style.color("expected_edge", "#3f5f87"),
                24.0 + 78.0 * expected_attention,
                expected_attention,
            ),
            (
                "attended actual",
                actual,
                resolved_style.color("positive_error", "#c2410c"),
                28.0 + 92.0 * actual_attention + 110.0 * surprise,
                actual_attention,
            ),
            (
                "after",
                after,
                resolved_style.color("after", "#111827"),
                34.0 + 78.0 * actual_attention + 110.0 * learning_rate,
                actual_attention,
            ),
        ]
        for label, point, color, size, alpha in points:
            axis.scatter(
                [float(point[0])],
                [float(point[1])],
                s=resolved_style.markersize(size),
                color=color,
                edgecolor=resolved_style.color("memory_edge", "#1f2933"),
                alpha=alpha,
                linewidth=resolved_style.linewidth(0.5 + 1.4 * alpha),
                zorder=4,
            )
            axis.annotate(
                label,
                xy=(float(point[0]), float(point[1])),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )
    else:
        axis.text(
            0.5,
            0.5,
            "no correction yet",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color=resolved_style.color("neutral", "#6b7280"),
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


def draw_correction_time_series(
    axis: Axes,
    view_state: CorrectionViewState,
    style: RendererStyle | str | None = None,
) -> None:
    resolved_style = resolve_style(style)
    apply_axis_style(axis, resolved_style, view_state.title)

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.03, 1.03)
    axis.set_xlabel("time (normalized)")
    axis.set_ylabel("series-normalized value")
    axis.set_xticks([0.0, 0.5, 1.0])
    axis.set_yticks([0.0, 0.5, 1.0])
    axis.grid(True, color=resolved_style.grid_color, linewidth=resolved_style.linewidth(0.8), alpha=0.8)
    axis.set_axisbelow(True)

    times = np.asarray(view_state.series_times, dtype=float)
    series_items = [
        ("expected", np.asarray(view_state.expected_series, dtype=float), resolved_style.color("expected_edge", "#3f5f87")),
        ("actual", np.asarray(view_state.actual_series, dtype=float), resolved_style.color("positive_error", "#c2410c")),
        ("after", np.asarray(view_state.after_series, dtype=float), resolved_style.color("after", "#111827")),
    ]
    feature_labels = [
        feature_axis_label(view_state.feature_x),
        feature_axis_label(view_state.feature_y),
    ]
    for marker_time, marker_label in zip(
        view_state.experience_times,
        view_state.experience_labels,
    ):
        axis.axvline(
            float(marker_time),
            color=resolved_style.color("interval", "#7a90a8"),
            linewidth=resolved_style.linewidth(0.9),
            linestyle="--",
            alpha=0.55,
            zorder=1,
        )
        axis.text(
            float(marker_time),
            1.01,
            f" {marker_label}",
            rotation=90,
            va="top",
            ha="left",
            fontsize=8,
            color=resolved_style.color("interval_label", "#334e68"),
            alpha=0.8,
        )

    for label, values, color in series_items:
        if times.size == 0 or values.size == 0:
            continue
        attention = _series_attention(view_state, label)
        _plot_weighted_series(
            axis,
            times,
            values[:, 0],
            attention,
            color=color,
            linestyle="-",
            label=f"{label} {feature_labels[0]}",
            style=resolved_style,
        )
        _plot_weighted_series(
            axis,
            times,
            values[:, 1],
            attention,
            color=color,
            linestyle="--",
            label=f"{label} {feature_labels[1]}",
            style=resolved_style,
        )
        current_attention = float(attention[-1]) if attention.size else 1.0
        axis.scatter(
            [times[-1], times[-1]],
            [values[-1, 0], values[-1, 1]],
            s=[
                resolved_style.markersize(16 + 44 * current_attention),
                resolved_style.markersize(14 + 36 * current_attention),
            ],
            color=color,
            edgecolor=resolved_style.color("memory_edge", "#1f2933"),
            alpha=0.2 + 0.75 * current_attention,
            linewidth=resolved_style.linewidth(0.5 + current_attention),
            zorder=4,
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
    axis.legend(loc="lower left", fontsize=7, ncol=2, frameon=False, labelcolor=resolved_style.text_color)


def _attention_alpha(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _series_attention(view_state: CorrectionViewState, label: str) -> np.ndarray:
    if label == "expected" and view_state.expected_attention_series is not None:
        return np.clip(
            np.asarray(view_state.expected_attention_series, dtype=float),
            0.0,
            1.0,
        )
    if label in {"actual", "after"} and view_state.actual_attention_series is not None:
        return np.clip(
            np.asarray(view_state.actual_attention_series, dtype=float),
            0.0,
            1.0,
        )
    return np.ones_like(np.asarray(view_state.series_times, dtype=float))


def _plot_weighted_series(
    axis: Axes,
    times: np.ndarray,
    values: np.ndarray,
    attention: np.ndarray,
    *,
    color: str,
    linestyle: str,
    label: str,
    style: RendererStyle,
) -> None:
    if times.size == 1:
        current_attention = float(attention[0]) if attention.size else 1.0
        axis.scatter(
            [times[0]],
            [values[0]],
            s=style.markersize(16 + 44 * current_attention),
            color=color,
            alpha=0.2 + 0.75 * current_attention,
            label=label,
            zorder=4,
        )
        return
    for index in range(times.size - 1):
        segment_attention = float(0.5 * (attention[index] + attention[index + 1]))
        axis.plot(
            times[index : index + 2],
            values[index : index + 2],
            color=color,
            linewidth=style.linewidth(0.25 + 5.5 * segment_attention),
            linestyle=linestyle,
            alpha=0.08 + 0.9 * segment_attention,
            label=label if index == 0 else None,
            solid_capstyle="round",
            zorder=3,
        )
