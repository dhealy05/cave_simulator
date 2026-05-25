from __future__ import annotations

from matplotlib.axes import Axes

from cave.presentation.renderers.matplotlib_renderer.styles import RendererStyle, apply_axis_style, resolve_style
from cave.observation.views import TimelineViewState


def draw_timeline(
    axis: Axes,
    view_state: TimelineViewState,
    style: RendererStyle | str | None = None,
) -> None:
    resolved_style = resolve_style(style)
    apply_axis_style(axis, resolved_style, view_state.title, grid=True)

    duration = max(0.001, view_state.duration)
    axis.set_xlim(0.0, duration)
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlabel("time (s)")
    axis.set_ylabel("attention allocation")

    times = [point.t for point in view_state.attention_points]
    values = [point.value for point in view_state.attention_points]
    if times and values:
        axis.plot(
            times,
            values,
            color=resolved_style.color("neutral", "#6b7280"),
            linewidth=resolved_style.linewidth(1.1),
            linestyle="--",
            alpha=0.5,
            label="capacity",
        )

    for channel, points in view_state.channel_attention_points.items():
        channel_times = [point.t for point in points]
        channel_values = [point.value for point in points]
        if not channel_times:
            continue
        axis.plot(
            channel_times,
            channel_values,
            color=_channel_color(channel, resolved_style),
            linewidth=resolved_style.linewidth(1.8),
            alpha=0.88,
            label=_channel_label(channel),
        )

    for interval in view_state.intervals:
        axis.axvline(
            interval.start,
            color=resolved_style.color("interval", "#7a90a8"),
            linewidth=resolved_style.linewidth(1.0),
            linestyle="--",
            alpha=0.7,
        )
        axis.text(
            interval.start,
            1.02,
            f" {interval.kind}",
            rotation=90,
            va="top",
            ha="left",
            fontsize=9,
            color=resolved_style.color("interval_label", "#334e68"),
        )

    t = view_state.pointer_t
    for channel, value in view_state.pointer_channel_attention.items():
        axis.scatter(
            [t],
            [value],
            s=resolved_style.markersize(54),
            color=_channel_color(channel, resolved_style),
            edgecolor=resolved_style.color("current_edge", "#7c2d12"),
            linewidth=resolved_style.linewidth(1.0),
            zorder=5,
        )
    axis.scatter(
        [t],
        [view_state.pointer_attention],
        s=resolved_style.markersize(46),
        facecolors="none",
        edgecolors=resolved_style.color("neutral", "#6b7280"),
        linewidth=resolved_style.linewidth(1.2),
        zorder=4,
    )
    axis.legend(
        loc="lower right",
        fontsize=7,
        frameon=False,
        labelcolor=resolved_style.text_color,
        ncol=2,
    )


def _channel_color(channel: str, style: RendererStyle) -> str:
    palette = {
        "visual": style.color("actual", "#2f855a"),
        "audio": "#7c3aed",
        "internal_expectation": style.color("expected_edge", "#3f5f87"),
    }
    if channel in palette:
        return palette[channel]
    colors = ["#0f766e", "#b45309", "#be123c", "#0369a1", "#4d7c0f"]
    return colors[sum(ord(char) for char in channel) % len(colors)]


def _channel_label(channel: str) -> str:
    return channel.replace("_", " ")
