from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from matplotlib.axes import Axes
from matplotlib.figure import Figure


@dataclass(frozen=True)
class RendererStyle:
    name: str
    figure_facecolor: str
    axes_facecolor: str
    text_color: str
    muted_text_color: str
    spine_color: str
    grid_color: str
    guide_color: str
    box_facecolor: str
    box_edgecolor: str
    palette: Mapping[str, str] = field(default_factory=dict)
    glyph_fill: str | None = None
    glyph_stroke: str | None = None
    glyph_text_color: str | None = None
    preserve_glyph_colors: bool = True
    surface_cmap: str = "viridis"
    line_width_scale: float = 1.0
    marker_scale: float = 1.0
    font_family: str | None = None
    path_sketch: tuple[float, float, float] | None = None
    scanlines: bool = False

    def color(self, key: str, fallback: str) -> str:
        return str(self.palette.get(key, fallback))

    def linewidth(self, value: float) -> float:
        return value * self.line_width_scale

    def markersize(self, value: float) -> float:
        return value * self.marker_scale

    def text_box(self, *, alpha: float = 0.86) -> dict[str, object]:
        return {
            "boxstyle": "round,pad=0.25",
            "facecolor": self.box_facecolor,
            "edgecolor": self.box_edgecolor,
            "alpha": alpha,
        }

    def rc_params(self) -> dict[str, object]:
        params: dict[str, object] = {
            "text.color": self.text_color,
            "axes.labelcolor": self.text_color,
            "axes.edgecolor": self.spine_color,
            "xtick.color": self.muted_text_color,
            "ytick.color": self.muted_text_color,
            "grid.color": self.grid_color,
        }
        if self.font_family is not None:
            params["font.family"] = self.font_family
        if self.path_sketch is not None:
            params["path.sketch"] = self.path_sketch
        return params


DEFAULT_STYLE = RendererStyle(
    name="default",
    figure_facecolor="#ffffff",
    axes_facecolor="#f7f8fa",
    text_color="#111827",
    muted_text_color="#66788a",
    spine_color="#d6d9de",
    grid_color="#e3e6eb",
    guide_color="#cbd5df",
    box_facecolor="#ffffff",
    box_edgecolor="#d6d9de",
    palette={
        "attention": "#2f855a",
        "current": "#c2410c",
        "current_edge": "#7c2d12",
        "interval": "#7a90a8",
        "interval_label": "#334e68",
        "wall": "#8795a1",
        "center": "#cbd5df",
        "expected_fill": "#91a7c7",
        "expected_edge": "#3f5f87",
        "actual": "#2f855a",
        "positive_error": "#c2410c",
        "negative_error": "#2563eb",
        "after": "#111827",
        "neutral": "#6b7280",
        "memory": "#2f855a",
        "memory_edge": "#1f2933",
        "corridor": "#d6d9de",
        "well_fill": "#f7f1dd",
        "well_edge": "#1f2933",
        "pain": "#b42318",
        "pleasure": "#2f855a",
        "strength": "#2563eb",
        "exposure": "#111827",
        "utility": "#2f855a",
    },
)

CRT_STYLE = RendererStyle(
    name="crt",
    figure_facecolor="#020805",
    axes_facecolor="#03130b",
    text_color="#9dffbd",
    muted_text_color="#4ccf76",
    spine_color="#177a3f",
    grid_color="#0f4d2b",
    guide_color="#1fa058",
    box_facecolor="#041d10",
    box_edgecolor="#34d26f",
    palette={
        "attention": "#39ff88",
        "current": "#f6ff5f",
        "current_edge": "#adff2f",
        "interval": "#21a85b",
        "interval_label": "#8cffac",
        "wall": "#39ff88",
        "center": "#1fa058",
        "expected_fill": "#135d35",
        "expected_edge": "#75ff9d",
        "actual": "#39ff88",
        "positive_error": "#f6ff5f",
        "negative_error": "#4ee7ff",
        "after": "#d5ffd5",
        "neutral": "#67d787",
        "memory": "#39ff88",
        "memory_edge": "#bcffd0",
        "corridor": "#177a3f",
        "well_fill": "#0f4d2b",
        "well_edge": "#a5ffc0",
        "pain": "#ff6b6b",
        "pleasure": "#39ff88",
        "strength": "#4ee7ff",
        "exposure": "#d5ffd5",
        "utility": "#f6ff5f",
    },
    glyph_fill="#061f11",
    glyph_stroke="#9dffbd",
    glyph_text_color="#9dffbd",
    preserve_glyph_colors=False,
    surface_cmap="Greens",
    line_width_scale=1.15,
    marker_scale=1.08,
    font_family="monospace",
    scanlines=True,
)

PAPER_STYLE = RendererStyle(
    name="paper",
    figure_facecolor="#f5f0e6",
    axes_facecolor="#fbf6ea",
    text_color="#2f2a24",
    muted_text_color="#746758",
    spine_color="#c8b99f",
    grid_color="#ded1bb",
    guide_color="#b9a98e",
    box_facecolor="#fffaf0",
    box_edgecolor="#bba98e",
    palette={
        "attention": "#3d7a4f",
        "current": "#9a4a2a",
        "current_edge": "#6d351f",
        "interval": "#8c7a65",
        "interval_label": "#5d5144",
        "wall": "#7d7467",
        "center": "#b9a98e",
        "expected_fill": "#a6b6c5",
        "expected_edge": "#516b7d",
        "actual": "#3d7a4f",
        "positive_error": "#9a4a2a",
        "negative_error": "#3f6f9f",
        "after": "#2f2a24",
        "neutral": "#746758",
        "memory": "#3d7a4f",
        "memory_edge": "#3a332c",
        "corridor": "#c8b99f",
        "well_fill": "#fff1bd",
        "well_edge": "#3a332c",
        "pain": "#9d3f34",
        "pleasure": "#3d7a4f",
        "strength": "#3f6f9f",
        "exposure": "#2f2a24",
        "utility": "#6c7a3d",
    },
    surface_cmap="YlGn",
    line_width_scale=1.08,
    marker_scale=0.96,
    path_sketch=(1.0, 120.0, 2.0),
)


STYLES: dict[str, RendererStyle] = {
    "default": DEFAULT_STYLE,
    "none": DEFAULT_STYLE,
    "no-style": DEFAULT_STYLE,
    "crt": CRT_STYLE,
    "paper": PAPER_STYLE,
    "hand-drawn": PAPER_STYLE,
    "hand_drawn": PAPER_STYLE,
}


def available_styles() -> tuple[str, ...]:
    return tuple(sorted(STYLES))


def resolve_style(style: str | RendererStyle | None = None) -> RendererStyle:
    if style is None:
        return DEFAULT_STYLE
    if isinstance(style, RendererStyle):
        return style
    try:
        return STYLES[style.strip().lower()]
    except KeyError as exc:
        choices = ", ".join(available_styles())
        raise ValueError(f"unsupported render style {style!r}; choose from {choices}") from exc


def apply_figure_style(figure: Figure, style: RendererStyle) -> None:
    figure.set_facecolor(style.figure_facecolor)


def apply_axis_style(
    axis: Axes,
    style: RendererStyle,
    title: str,
    *,
    grid: bool = False,
    grid_axis: str = "both",
) -> None:
    axis.set_title(title, loc="left", fontsize=11, fontweight="bold", color=style.text_color)
    axis.set_facecolor(style.axes_facecolor)
    for spine in axis.spines.values():
        spine.set_color(style.spine_color)
    axis.tick_params(colors=style.muted_text_color)
    axis.xaxis.label.set_color(style.text_color)
    axis.yaxis.label.set_color(style.text_color)
    if hasattr(axis, "zaxis"):
        axis.zaxis.label.set_color(style.text_color)
        axis.tick_params(axis="z", colors=style.muted_text_color)
    if grid:
        axis.grid(True, axis=grid_axis, color=style.grid_color, linewidth=style.linewidth(0.8), alpha=0.8)
        axis.set_axisbelow(True)


def apply_scanline_effect(axis: Axes, style: RendererStyle) -> None:
    if not style.scanlines:
        return
    for y in [index / 24 for index in range(1, 24, 2)]:
        axis.plot(
            [0.0, 1.0],
            [y, y],
            transform=axis.transAxes,
            color="#b8ffc8",
            linewidth=0.35,
            alpha=0.08,
            zorder=50,
            clip_on=False,
        )
