from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cave.observation.episodes import Episode


@dataclass(frozen=True)
class LabeledEpisode:
    id: str
    episode: Episode
    label: str | None = None
    group: str | None = None
    series: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def display_label(self) -> str:
        return self.label or self.id


@dataclass(frozen=True)
class EpisodeSet:
    id: str
    episodes: tuple[LabeledEpisode, ...]
    title: str | None = None
    comparison_axis: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episodes:
            raise ValueError("EpisodeSet requires at least one episode")


def labeled_episode(
    episode: Episode,
    *,
    id: str | None = None,
    label: str | None = None,
    group: str | None = None,
    series: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> LabeledEpisode:
    return LabeledEpisode(
        id=id or episode.source_name,
        episode=episode,
        label=label,
        group=group,
        series=series,
        metadata={} if metadata is None else dict(metadata),
    )


def episode_set(
    episodes: list[LabeledEpisode] | tuple[LabeledEpisode, ...],
    *,
    id: str,
    title: str | None = None,
    comparison_axis: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> EpisodeSet:
    return EpisodeSet(
        id=id,
        title=title,
        comparison_axis=comparison_axis,
        episodes=tuple(episodes),
        metadata={} if metadata is None else dict(metadata),
    )
