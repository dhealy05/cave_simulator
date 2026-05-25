from __future__ import annotations

from matplotlib.axes import Axes

from cave.presentation.renderers.matplotlib_renderer.glyphs import draw_glyph, prepare_axis
from cave.presentation.renderers.matplotlib_renderer.styles import RendererStyle, resolve_style
from cave.observation.views import PresentationViewState


def draw_presentation(
    axis: Axes,
    view_state: PresentationViewState,
    style: RendererStyle | str | None = None,
) -> None:
    resolved_style = resolve_style(style)
    prepare_axis(axis, view_state.title, resolved_style)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axhline(
        0.5,
        color=resolved_style.color("wall", "#8795a1"),
        linewidth=resolved_style.linewidth(1),
    )
    axis.axvline(
        0.5,
        color=resolved_style.color("center", "#cbd5df"),
        linewidth=resolved_style.linewidth(1),
        linestyle="--",
    )
    for item in view_state.items:
        draw_glyph(
            axis,
            item.presentation,
            item.phase,
            0.5,
            0.09,
            alpha=item.opacity,
            style=resolved_style,
        )
