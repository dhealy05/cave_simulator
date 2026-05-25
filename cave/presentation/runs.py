from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from cave.observation.episodes import Episode
from cave.observation.pipeline import episode_payload
from cave.presentation.renderers.matplotlib_renderer import LayoutSpec, MatplotlibRenderer
from cave.observation.structural import frame_for_time, structural_state_for_episode
from cave.observation.views import ExperienceView, default_views


@dataclass(frozen=True)
class ExperienceRunOutputs:
    directory: Path
    episode_json: Path
    metadata_json: Path
    frame_png: Path | None = None
    animation_gif: Path | None = None


@dataclass(frozen=True)
class ExperienceRun:
    id: str
    episode: Episode
    input_summary: str | None = None
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def source_name(self) -> str:
        return self.episode.source_name

    @property
    def slug(self) -> str:
        return slugify(self.id)

    @property
    def source_slug(self) -> str:
        return slugify(self.source_name)

    def output_dir(self, root: str | Path = "out/episodes") -> Path:
        return Path(root) / self.source_slug / self.slug

    def write_outputs(
        self,
        *,
        root: str | Path = "out/episodes",
        views: Sequence[ExperienceView] | None = None,
        frame_time: float | None = None,
        write_frame: bool = True,
        write_animation: bool = True,
        fps: int = 8,
        columns: int = 2,
        style: str = "default",
    ) -> ExperienceRunOutputs:
        directory = self.output_dir(root)
        directory.mkdir(parents=True, exist_ok=True)
        episode_json = directory / "episode.json"
        metadata_json = directory / "metadata.json"
        self.write_json(episode_json)
        self.write_metadata(metadata_json)

        selected_views = list(default_views() if views is None else views)
        renderer = MatplotlibRenderer(
            layout=LayoutSpec(columns=columns),
            style=style,
        )
        frame_png = None
        if write_frame:
            frame_png = directory / "frame.png"
            self.render_frame(
                frame_png,
                selected_views,
                frame_time=frame_time,
                renderer=renderer,
            )

        animation_gif = None
        if write_animation:
            animation_gif = directory / "animation.gif"
            self.render_animation(
                animation_gif,
                selected_views,
                fps=fps,
                renderer=renderer,
            )

        return ExperienceRunOutputs(
            directory=directory,
            episode_json=episode_json,
            metadata_json=metadata_json,
            frame_png=frame_png,
            animation_gif=animation_gif,
        )

    def write_json(self, output: str | Path) -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(episode_payload(self.episode), indent=2) + "\n",
            encoding="utf-8",
        )
        return output

    def write_metadata(self, output: str | Path) -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": self.id,
            "source_name": self.source_name,
            "input_summary": self.input_summary,
            "config": self.config,
            "episode_metadata": self.episode.metadata,
        }
        output.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        return output

    def render_frame(
        self,
        output: str | Path,
        views: Sequence[ExperienceView],
        *,
        frame_time: float | None = None,
        renderer: MatplotlibRenderer | None = None,
        style: str = "default",
    ) -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        structural = structural_state_for_episode(self.episode)
        if frame_time is None:
            frame_time = self._default_frame_time()
        frame = frame_for_time(self.episode, frame_time, structural)
        renderer = renderer or MatplotlibRenderer(style=style)
        renderer.save_frame(frame, views, output)
        return output

    def render_animation(
        self,
        output: str | Path,
        views: Sequence[ExperienceView],
        *,
        fps: int = 8,
        renderer: MatplotlibRenderer | None = None,
        style: str = "default",
    ) -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        renderer = renderer or MatplotlibRenderer(style=style)
        renderer.save_animation(self.episode, views, output, fps=fps)
        return output

    def _default_frame_time(self) -> float:
        if self.episode.observations:
            return self.episode.observations[len(self.episode.observations) // 2].t
        return 0.0


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "run"
