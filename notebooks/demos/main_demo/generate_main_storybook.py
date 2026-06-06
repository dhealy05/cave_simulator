"""Generate the static panels for the full-system Cave storybook.

This is the sibling of ``notebooks/demos/primitive_demo`` (Jimmy and the primitive
recurrence), but driven by the *full* Cave model: the canonical
triangle / circle / square / gap demo sequence, nine-dimensional feature
vectors, and the six standard views.

Every panel is produced by the existing ``MatplotlibRenderer`` (the same one
behind the README multi-view GIF), just frozen one instant at a time. The pages
build the views up one at a time and end on the full six-panel dashboard.

Run from the repository root:

    python notebooks/demos/main_demo/generate_main_storybook.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from cave import CaveProducer, default_views, demo_model
from cave.observation.structural import episode_frames, structural_state_for_episode
from cave.presentation.renderers.matplotlib_renderer.renderer import (
    LayoutSpec,
    MatplotlibRenderer,
)

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "storybook_assets"

# Build-up arc: each page adds a view, ending on the full six-panel dashboard.
ALL_SIX = [
    "PresentationView",
    "MemoryLookbackView",
    "TimelineView",
    "ExpectationActualView",
    "CorrectionView",
    "SubjectiveTopologyView",
]
PAGES = [
    {"t": 0.0, "slug": "01_triangle_presentation", "views": ["PresentationView"]},
    {"t": 0.8, "slug": "02_triangle_vector", "views": ["PresentationView", "ExpectationActualView"]},
    {"t": 1.3, "slug": "03_gap_memory", "views": ["PresentationView", "ExpectationActualView", "MemoryLookbackView"]},
    {"t": 1.5, "slug": "04_circle_violation", "views": ["PresentationView", "ExpectationActualView", "CorrectionView"]},
    {"t": 2.0, "slug": "05_circle_time_topology", "views": ["PresentationView", "TimelineView", "SubjectiveTopologyView"]},
    {"t": 2.0, "slug": "06_full_dashboard", "views": ALL_SIX},
]


def _layout_for(count: int) -> LayoutSpec:
    if count <= 1:
        return LayoutSpec(columns=1, figsize_per_cell=(5.4, 4.4))
    if count <= 3:
        return LayoutSpec(columns=count, figsize_per_cell=(5.0, 4.2))
    return LayoutSpec(columns=3, figsize_per_cell=(4.8, 4.0))  # six -> 3x2


def main() -> None:
    episode = CaveProducer(demo_model()).run(dt=0.1)
    frames = list(episode_frames(episode, structural_state_for_episode(episode)))
    by_name = {type(v).__name__: v for v in default_views()}

    def frame_at(t: float):
        return min(frames, key=lambda f: abs(f.observation.t - t))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for page in PAGES:
        views = [by_name[name] for name in page["views"]]
        renderer = MatplotlibRenderer(layout=_layout_for(len(views)))
        frame = frame_at(page["t"])
        output = OUT_DIR / f"{page['slug']}.png"
        renderer.save_frame(frame, views, output)
        obs = frame.observation
        print(
            f"{page['slug']}: t={obs.t:.1f} "
            f"att={obs.attention:.2f} surprise={obs.surprise:.3f} "
            f"({len(views)} panel{'s' if len(views) != 1 else ''})"
        )


if __name__ == "__main__":
    main()
