from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from cave.observation.episodes import CaveProducer, Episode
from cave.observation.pipeline import views_from_names
from cave.demonstrations.examples import demo_model, random_experience_model
from cave.presentation.renderers.matplotlib_renderer.action import draw_action
from cave.presentation.renderers.matplotlib_renderer.affect import draw_affect
from cave.presentation.renderers.matplotlib_renderer.correction import (
    draw_correction,
    normalize_correction_series,
)
from cave.presentation.renderers.matplotlib_renderer.expectation_actual import draw_expectation_actual
from cave.presentation.renderers.matplotlib_renderer.memory import draw_memory
from cave.presentation.renderers.matplotlib_renderer.observer import draw_observer
from cave.presentation.renderers.matplotlib_renderer.presentation import draw_presentation
from cave.presentation.renderers.matplotlib_renderer.subject_surface import draw_subject_surface
from cave.presentation.renderers.matplotlib_renderer.subjective_topology import draw_subjective_topology
from cave.presentation.renderers.matplotlib_renderer.styles import (
    RendererStyle,
    apply_figure_style,
    apply_scanline_effect,
    available_styles,
    resolve_style,
)
from cave.presentation.renderers.matplotlib_renderer.timeline import draw_timeline
from cave.observation.structural import EpisodeFrame, episode_frames, structural_state_for_episode
from cave.observation.views import (
    ActionViewState,
    AffectViewState,
    BaseViewState,
    CorrectionViewState,
    ExperienceView,
    ExpectationActualViewState,
    MemoryLookbackViewState,
    ObserverViewState,
    PresentationViewState,
    SubjectSurfaceViewState,
    SubjectiveTopologyViewState,
    TimelineViewState,
    default_views,
)


@dataclass(frozen=True)
class LayoutSpec:
    columns: int = 1
    figsize_per_cell: tuple[float, float] = (6.0, 6.0)
    dpi: int = 120

    def shape_for(self, count: int) -> tuple[int, int]:
        if count <= 0:
            raise ValueError("at least one view is required")
        columns = max(1, min(self.columns, count))
        rows = math.ceil(count / columns)
        return rows, columns

    def figsize_for(self, count: int) -> tuple[float, float]:
        rows, columns = self.shape_for(count)
        cell_w, cell_h = self.figsize_per_cell
        return columns * cell_w, rows * cell_h


class MatplotlibRenderer:
    def __init__(
        self,
        layout: LayoutSpec | None = None,
        *,
        style: str | RendererStyle | None = None,
    ) -> None:
        self.layout = layout or LayoutSpec()
        self.style = resolve_style(style)

    def render_state(
        self,
        state: EpisodeFrame,
        views: Sequence[ExperienceView],
    ) -> tuple[Figure, list[Axes]]:
        with plt.rc_context(self.style.rc_params()):
            view_states = [view.project(state) for view in views]
            figure, axes = self._make_figure(view_states)
            self.draw_view_states(axes, view_states)
            figure.tight_layout()
            return figure, axes

    def save_frame(
        self,
        state: EpisodeFrame,
        views: Sequence[ExperienceView],
        output: str | Path,
    ) -> None:
        with plt.rc_context(self.style.rc_params()):
            figure, _ = self.render_state(state, views)
            figure.savefig(output, dpi=self.layout.dpi)
            plt.close(figure)

    def save_animation(
        self,
        episode: Episode,
        views: Sequence[ExperienceView],
        output: str | Path,
        *,
        start: float = 0.0,
        end: float | None = None,
        dt: float = 0.05,
        fps: int = 20,
    ) -> None:
        with plt.rc_context(self.style.rc_params()):
            if dt <= 0.0:
                raise ValueError("dt must be positive")
            structural = structural_state_for_episode(episode)
            states = [
                frame
                for frame in episode_frames(episode, structural)
                if start
                <= frame.observation.t
                <= (episode.duration if end is None else end)
            ]
            view_states_by_frame = [
                [view.project(state) for view in views]
                for state in states
            ]
            if not view_states_by_frame:
                raise ValueError("animation has no frames")
            view_states_by_frame = normalize_correction_series(view_states_by_frame)
            figure, axes = self._make_figure(view_states_by_frame[0])

            def update(frame_index: int) -> list[Axes]:
                self.draw_view_states(axes, view_states_by_frame[frame_index])
                figure.tight_layout()
                return axes

            anim = animation.FuncAnimation(
                figure,
                update,
                frames=len(view_states_by_frame),
                interval=1000 / fps,
                blit=False,
            )
            output = Path(output)
            if output.suffix.lower() == ".gif":
                writer = animation.PillowWriter(fps=fps)
            else:
                writer = animation.FFMpegWriter(fps=fps)
            anim.save(output, writer=writer, dpi=self.layout.dpi)
            plt.close(figure)

    def draw_view_states(
        self,
        axes: Sequence[Axes],
        view_states: Sequence[BaseViewState],
    ) -> None:
        for axis, view_state in zip(axes, view_states):
            axis.clear()
            self.draw_view_state(axis, view_state)
            apply_scanline_effect(axis, self.style)
        for axis in axes[len(view_states) :]:
            axis.clear()
            axis.axis("off")

    def draw_view_state(self, axis: Axes, view_state: BaseViewState) -> None:
        if isinstance(view_state, PresentationViewState):
            draw_presentation(axis, view_state, self.style)
        elif isinstance(view_state, SubjectSurfaceViewState):
            draw_subject_surface(axis, view_state, self.style)
        elif isinstance(view_state, ObserverViewState):
            draw_observer(axis, view_state, self.style)
        elif isinstance(view_state, ActionViewState):
            draw_action(axis, view_state, self.style)
        elif isinstance(view_state, MemoryLookbackViewState):
            draw_memory(axis, view_state, self.style)
        elif isinstance(view_state, TimelineViewState):
            draw_timeline(axis, view_state, self.style)
        elif isinstance(view_state, ExpectationActualViewState):
            draw_expectation_actual(axis, view_state, self.style)
        elif isinstance(view_state, CorrectionViewState):
            draw_correction(axis, view_state, self.style)
        elif isinstance(view_state, AffectViewState):
            draw_affect(axis, view_state, self.style)
        elif isinstance(view_state, SubjectiveTopologyViewState):
            draw_subjective_topology(axis, view_state, self.style)
        else:
            raise TypeError(f"unsupported view state: {type(view_state).__name__}")

    def _make_figure(
        self,
        view_states: Sequence[BaseViewState] | int,
    ) -> tuple[Figure, list[Axes]]:
        if isinstance(view_states, int):
            count = view_states
            projections: list[str | None] = [None] * count
        else:
            count = len(view_states)
            projections = [
                "3d" if isinstance(vs, SubjectiveTopologyViewState) else None
                for vs in view_states
            ]
        rows, columns = self.layout.shape_for(count)
        figure = plt.figure(figsize=self.layout.figsize_for(count))
        apply_figure_style(figure, self.style)
        axes: list[Axes] = []
        for index, projection in enumerate(projections):
            kwargs = {"projection": projection} if projection else {}
            axes.append(figure.add_subplot(rows, columns, index + 1, **kwargs))
        return figure, axes


def main() -> None:
    parser = argparse.ArgumentParser(description="Render cave views with matplotlib.")
    parser.add_argument("--output", type=Path, default=Path("animation.gif"))
    parser.add_argument("--frame", action="store_true", help="Render one still frame instead of an animation.")
    parser.add_argument("--time", type=float, default=2.4, help="Frame time for --frame.")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--columns", type=int, default=1)
    parser.add_argument(
        "--views",
        default="all",
        help="Comma-separated view names, or `all` for the default multi-view set.",
    )
    parser.add_argument(
        "--style",
        default="default",
        choices=available_styles(),
        help="Named renderer style.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--random",
        action="store_true",
        help="Generate a seeded random experience sequence instead of the fixed demo.",
    )
    parser.add_argument("--count", type=int, default=8, help="Random sequence length.")
    args = parser.parse_args()

    renderer = MatplotlibRenderer(
        layout=LayoutSpec(columns=args.columns),
        style=args.style,
    )
    views = views_from_names(args.views)
    model = (
        random_experience_model(count=args.count, seed=args.seed)
        if args.random
        else demo_model(seed=args.seed)
    )
    episode = CaveProducer(model).run(dt=args.dt)
    if args.frame:
        structural = structural_state_for_episode(episode)
        frames = [
            frame for frame in episode_frames(episode, structural)
            if frame.observation.t <= args.time
        ]
        if not frames:
            raise ValueError("frame has no episode observation")
        renderer.save_frame(frames[-1], views, args.output)
    else:
        renderer.save_animation(
            episode,
            views,
            args.output,
            fps=args.fps,
        )
