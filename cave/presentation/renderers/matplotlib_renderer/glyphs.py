from __future__ import annotations

import textwrap

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Polygon, Rectangle

from cave.observation.experience import Presentation, ShapePresentation, TextPresentation
from cave.presentation.renderers.matplotlib_renderer.styles import RendererStyle, apply_axis_style, resolve_style


def prepare_axis(axis: Axes, title: str, style: RendererStyle | str | None = None) -> None:
    resolved_style = resolve_style(style)
    apply_axis_style(axis, resolved_style, title)
    axis.set_xticks([])
    axis.set_yticks([])


def draw_glyph(
    axis: Axes,
    presentation: Presentation,
    x: float,
    y: float,
    size: float,
    *,
    alpha: float,
    style: RendererStyle | str | None = None,
) -> None:
    resolved_style = resolve_style(style)
    fill = "#f7f1dd"
    stroke = "#1f2933"
    shape_type = "object"
    if isinstance(presentation, TextPresentation):
        draw_text_glyph(axis, presentation, x, y, size, alpha=alpha, style=resolved_style)
        return
    if isinstance(presentation, ShapePresentation):
        fill = str(presentation.style.get("fill", fill))
        stroke = str(presentation.style.get("stroke", stroke))
        shape_type = presentation.shape_type
        size *= max(0.1, float(presentation.style.get("size", 1.0)))
    if not resolved_style.preserve_glyph_colors:
        fill = resolved_style.glyph_fill or fill
        stroke = resolved_style.glyph_stroke or stroke

    if shape_type == "circle":
        patch = Circle((x, y), radius=size, facecolor=fill, edgecolor=stroke, alpha=alpha)
    elif shape_type == "square":
        patch = Rectangle(
            (x - size, y - size),
            2 * size,
            2 * size,
            facecolor=fill,
            edgecolor=stroke,
            alpha=alpha,
        )
    elif shape_type == "line":
        axis.plot(
            [x - size, x + size],
            [y, y],
            color=stroke,
            linewidth=resolved_style.linewidth(4),
            alpha=alpha,
        )
        return
    elif isinstance(presentation, ShapePresentation) and presentation.points:
        points = np.array(presentation.points, dtype=float) * size
        points[:, 0] += x
        points[:, 1] += y
        patch = Polygon(points, closed=True, facecolor=fill, edgecolor=stroke, alpha=alpha)
    else:
        points = np.array(
            [
                [x, y + size],
                [x - 0.95 * size, y - 0.8 * size],
                [x + 0.95 * size, y - 0.8 * size],
            ]
        )
        patch = Polygon(points, closed=True, facecolor=fill, edgecolor=stroke, alpha=alpha)
    axis.add_patch(patch)


def draw_text_glyph(
    axis: Axes,
    presentation: TextPresentation,
    x: float,
    y: float,
    size: float,
    *,
    alpha: float,
    style: RendererStyle | str | None = None,
) -> None:
    resolved_style = resolve_style(style)
    fill = str(presentation.style.get("fill", "#ffffff"))
    stroke = str(presentation.style.get("stroke", "#1f2933"))
    text_color = str(presentation.style.get("text_color", "#111827"))
    if not resolved_style.preserve_glyph_colors:
        fill = resolved_style.glyph_fill or fill
        stroke = resolved_style.glyph_stroke or stroke
        text_color = resolved_style.glyph_text_color or text_color
    raw_text = presentation.text
    wrap_width = int(presentation.style.get("wrap_width", 28))
    text = "\n".join(textwrap.wrap(raw_text, width=wrap_width)) if len(raw_text) > wrap_width else raw_text
    line_count = max(1, text.count("\n") + 1)
    fontsize = min(24.0, max(8.0, 180.0 * size / (line_count ** 0.35)))
    axis.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        alpha=alpha,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": fill,
            "edgecolor": stroke,
            "linewidth": 1.0,
            "alpha": min(1.0, max(0.0, alpha)),
        },
        clip_on=True,
    )


def normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low = float(np.min(values))
    high = float(np.max(values))
    if high - low < 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)
