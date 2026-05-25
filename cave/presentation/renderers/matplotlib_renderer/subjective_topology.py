from __future__ import annotations

from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
import numpy as np
from matplotlib.axes import Axes

from cave.observation.experience import FeatureAxis, FeatureProjection, feature_axis_label
from cave.presentation.renderers.matplotlib_renderer.styles import RendererStyle, apply_axis_style, resolve_style
from cave.observation.views import SubjectiveTopologyViewState


EXPECTED_COLOR = "#0072B2"
ACTUAL_COLOR = "#009E73"
NEUTRAL_COLOR = "#E2E6EA"


def draw_subjective_topology(
    axis: Axes,
    view_state: SubjectiveTopologyViewState,
    style: RendererStyle | str | None = None,
) -> None:
    resolved_style = resolve_style(style)
    apply_axis_style(axis, resolved_style, view_state.title)
    topology = view_state.topology
    lower, upper = topology.bounds
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel(_feature_axis_display(topology.feature_x))
    axis.set_ylabel(_feature_axis_display(topology.feature_y))
    axis.set_zlabel("state density")
    axis.set_zlim(0.0, 1.05)
    _style_topology_axis(axis)

    grid_x = view_state.grid_x
    grid_y = view_state.grid_y
    density = view_state.density

    if grid_x is None or grid_y is None or density is None:
        return

    rendered_density = np.clip(density, 0.0, 1.0)
    facecolors = _source_facecolors(
        rendered_density,
        view_state.expected_density,
        view_state.actual_density,
    )
    axis.plot_surface(
        grid_x,
        grid_y,
        rendered_density,
        facecolors=facecolors,
        vmin=0.0,
        vmax=1.0,
        alpha=0.82,
        linewidth=0,
        antialiased=True,
        rstride=2,
        cstride=2,
    )
    _draw_source_legend(axis)


def _style_topology_axis(axis: Axes) -> None:
    axis.tick_params(axis="both", which="major", labelsize=8, pad=0)
    axis.tick_params(axis="z", which="major", labelsize=8, pad=1)
    for pane_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        pane_axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        pane_axis.pane.set_edgecolor((0.78, 0.80, 0.83, 0.55))
    for axis_name in ("xaxis", "yaxis", "zaxis"):
        info = getattr(axis, axis_name)._axinfo
        info["grid"]["color"] = (0.56, 0.59, 0.63, 0.25)
        info["grid"]["linewidth"] = 0.6


def _draw_source_legend(axis: Axes) -> None:
    handles = [
        Patch(facecolor=EXPECTED_COLOR, edgecolor="none", alpha=0.88, label="expected memory"),
        Patch(facecolor=ACTUAL_COLOR, edgecolor="none", alpha=0.88, label="lived input"),
    ]
    axis.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.92),
        frameon=False,
        fontsize=8,
    )


def _feature_axis_display(axis: FeatureAxis) -> str:
    label = feature_axis_label(axis)
    if not isinstance(axis, FeatureProjection):
        return label
    return f"{label} projection"


def _source_facecolors(
    density: np.ndarray,
    expected_density: np.ndarray | None,
    actual_density: np.ndarray | None,
) -> np.ndarray:
    density = np.clip(np.asarray(density, dtype=float), 0.0, 1.0)
    expected = (
        np.zeros_like(density, dtype=float)
        if expected_density is None
        else np.clip(np.asarray(expected_density, dtype=float), 0.0, 1.0)
    )
    actual = (
        np.zeros_like(density, dtype=float)
        if actual_density is None
        else np.clip(np.asarray(actual_density, dtype=float), 0.0, 1.0)
    )
    expected_color = np.array(to_rgba(EXPECTED_COLOR), dtype=float)
    actual_color = np.array(to_rgba(ACTUAL_COLOR), dtype=float)
    neutral_color = np.array(to_rgba(NEUTRAL_COLOR), dtype=float)
    highlight = np.array(to_rgba("#F8FAFC"), dtype=float)
    shadow = np.array(to_rgba("#263238"), dtype=float)
    source_mass = expected + actual
    actual_share = np.divide(
        actual,
        source_mass,
        out=np.zeros_like(actual, dtype=float),
        where=source_mass > 1e-12,
    )
    colors = (
        expected_color.reshape(1, 1, 4) * (1.0 - actual_share[..., None])
        + actual_color.reshape(1, 1, 4) * actual_share[..., None]
    )
    light = 0.18 + 0.82 * np.sqrt(density)
    shaded = shadow.reshape(1, 1, 4) * (1.0 - light[..., None]) + colors * light[..., None]
    shaded = 0.86 * shaded + 0.14 * highlight.reshape(1, 1, 4) * density[..., None]
    colors = np.where(
        (source_mass > 1e-12)[..., None],
        shaded,
        neutral_color.reshape(1, 1, 4),
    )
    colors[..., 3] = 0.22 + 0.70 * density
    return colors
