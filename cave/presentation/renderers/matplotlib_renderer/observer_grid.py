from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from cave.observation.episodes import Episode
from cave.observation.structural import episode_frames, structural_state_for_episode
from cave.observation.views import ObserverView
from cave.presentation.renderers.matplotlib_renderer import LayoutSpec, MatplotlibRenderer
from cave.presentation.renderers.matplotlib_renderer.styles import apply_figure_style, resolve_style


def save_observer_comparison_animation(
    episodes: Mapping[str, Episode],
    output: str | Path,
    *,
    fps: int = 4,
    style: str | None = None,
    max_frames: int = 72,
) -> None:
    if not episodes:
        raise ValueError("at least one episode is required")
    labels = list(episodes)
    frame_sets = [
        episode_frames(episode, structural_state_for_episode(episode))
        for episode in episodes.values()
    ]
    frame_count = min(len(frames) for frames in frame_sets)
    if frame_count <= 0:
        raise ValueError("observer comparison has no frames")
    if max_frames > 0 and frame_count > max_frames:
        indices = np.unique(np.linspace(0, frame_count - 1, max_frames, dtype=int))
    else:
        indices = np.arange(frame_count, dtype=int)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    layout = LayoutSpec(
        columns=min(4, len(labels)),
        figsize_per_cell=(3.2, 3.2),
        dpi=110,
    )
    renderer = MatplotlibRenderer(layout=layout, style=style)
    resolved_style = resolve_style(style)
    views = [ObserverView(title=label) for label in labels]

    with plt.rc_context(resolved_style.rc_params()):
        rows, columns = layout.shape_for(len(labels))
        figure = plt.figure(figsize=layout.figsize_for(len(labels)))
        apply_figure_style(figure, resolved_style)
        axes = [
            figure.add_subplot(rows, columns, index + 1)
            for index in range(len(labels))
        ]

        def update(frame_index: int):
            source_index = int(indices[frame_index])
            view_states = [
                view.project(frames[source_index])
                for view, frames in zip(views, frame_sets)
            ]
            renderer.draw_view_states(axes, view_states)
            figure.tight_layout()
            return axes

        anim = animation.FuncAnimation(
            figure,
            update,
            frames=len(indices),
            interval=1000 / fps,
            blit=False,
        )
        writer = animation.PillowWriter(fps=fps)
        anim.save(output, writer=writer, dpi=layout.dpi)
        plt.close(figure)
