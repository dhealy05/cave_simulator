from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from cave.presentation.renderers.matplotlib_renderer.glyphs import draw_glyph, prepare_axis
from cave.presentation.renderers.matplotlib_renderer.styles import RendererStyle, resolve_style
from cave.observation.views import MemoryLookbackViewState


def draw_memory(
    axis: Axes,
    view_state: MemoryLookbackViewState,
    style: RendererStyle | str | None = None,
) -> None:
    resolved_style = resolve_style(style)
    prepare_axis(axis, view_state.title, resolved_style)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal", adjustable="box")
    if "Temporal Lookback" in view_state.title or "Mock Memory" in view_state.title:
        axis.axhline(
            0.5,
            color=resolved_style.guide_color,
            linewidth=resolved_style.linewidth(1),
            linestyle="--",
            alpha=0.8,
        )
        axis.text(0.12, 0.12, "past", fontsize=8, color=resolved_style.muted_text_color, ha="left")
        axis.text(0.88, 0.12, "current", fontsize=8, color=resolved_style.muted_text_color, ha="right")
        label = (
            "prior turns as mock memories"
            if "Mock Memory" in view_state.title
            else "context influence over prior time"
        )
        axis.text(0.5, 0.86, label, fontsize=8, color=resolved_style.muted_text_color, ha="center")
        cell_count = max(1, len(view_state.items))
        cell_width = min(0.1, 0.72 / cell_count)
        for item in view_state.items:
            strength = min(1.0, max(0.0, item.strength))
            alpha = 0.12 + 0.78 * strength
            rect = Rectangle(
                (item.x - cell_width / 2, 0.36),
                cell_width,
                0.22,
                facecolor=resolved_style.color("memory", "#2f855a"),
                edgecolor=resolved_style.color("memory_edge", "#1f2933"),
                linewidth=resolved_style.linewidth(0.4),
                alpha=alpha,
            )
            axis.add_patch(rect)
            if item.age <= 1e-9:
                axis.axvline(
                    item.x,
                    color=resolved_style.color("current", "#c2410c"),
                    linewidth=resolved_style.linewidth(1.6),
                    alpha=0.9,
                )
        return
    horizon_y = 0.78
    axis.axhline(
        horizon_y,
        color=resolved_style.guide_color,
        linewidth=resolved_style.linewidth(1),
        linestyle="--",
        alpha=0.8,
    )
    axis.plot(
        [0.12, 0.5],
        [0.1, horizon_y],
        color=resolved_style.color("corridor", "#d6d9de"),
        linewidth=resolved_style.linewidth(1),
    )
    axis.plot(
        [0.88, 0.5],
        [0.1, horizon_y],
        color=resolved_style.color("corridor", "#d6d9de"),
        linewidth=resolved_style.linewidth(1),
    )
    for item in sorted(view_state.items, key=lambda memory_item: memory_item.depth, reverse=True):
        size = 0.26 * item.scale
        alpha = min(1.0, max(0.08, item.strength))
        draw_glyph(axis, item.presentation, item.x, item.y, size, alpha=alpha, style=resolved_style)
