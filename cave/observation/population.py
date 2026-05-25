from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from cave.observation.episode_runs import LabeledEpisode
from cave.observation.episodes import Episode


@dataclass(frozen=True)
class FactorLevel:
    factor: str
    id: str
    label: str | None = None
    role: str = "axis"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def display_label(self) -> str:
        return self.label or self.id


@dataclass(frozen=True)
class PopulationRunRecord:
    id: str
    episode: Episode
    factors: Mapping[str, FactorLevel]
    label: str | None = None
    comparison_role: str = "treatment"
    matched_set_id: str | None = None
    replicate_id: str | None = None
    group_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "factors", dict(self.factors))

    @property
    def display_label(self) -> str:
        return self.label or self.id

    def factor(self, name: str) -> FactorLevel | None:
        return self.factors.get(name)

    def factor_id(self, name: str, default: str | None = None) -> str | None:
        level = self.factor(name)
        if level is None:
            return default
        return level.id

    def factor_label(self, name: str, default: str | None = None) -> str | None:
        level = self.factor(name)
        if level is None:
            return default
        return level.display_label

    def to_labeled_episode(
        self,
        *,
        group_factor: str = "condition",
        series_factor: str = "start_condition",
    ) -> LabeledEpisode:
        return LabeledEpisode(
            id=self.id,
            episode=self.episode,
            label=self.display_label,
            group=self.factor_id(group_factor, self.group_id),
            series=self.factor_id(series_factor, self.replicate_id),
            metadata={
                **self.metadata,
                "comparison_role": self.comparison_role,
                "matched_set_id": self.matched_set_id,
                "replicate_id": self.replicate_id,
                "group_id": self.group_id,
                "factors": factor_levels_payload(self.factors),
            },
        )


@dataclass(frozen=True)
class PopulationExperiment:
    id: str
    runs: tuple[PopulationRunRecord, ...]
    title: str | None = None
    comparison_axis: str | None = None
    factor_order: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "runs", tuple(self.runs))
        if not self.runs:
            raise ValueError("PopulationExperiment requires at least one run")

    def levels(self, factor: str) -> tuple[FactorLevel, ...]:
        by_id: dict[str, FactorLevel] = {}
        for run in self.runs:
            level = run.factor(factor)
            if level is not None and level.id not in by_id:
                by_id[level.id] = level
        return tuple(by_id.values())

    def to_labeled_episodes(
        self,
        *,
        group_factor: str = "condition",
        series_factor: str = "start_condition",
    ) -> tuple[LabeledEpisode, ...]:
        return tuple(
            run.to_labeled_episode(
                group_factor=group_factor,
                series_factor=series_factor,
            )
            for run in self.runs
        )


def factor_level(
    factor: str,
    id: str,
    *,
    label: str | None = None,
    role: str = "axis",
    metadata: dict[str, Any] | None = None,
) -> FactorLevel:
    return FactorLevel(
        factor=factor,
        id=id,
        label=label,
        role=role,
        metadata={} if metadata is None else dict(metadata),
    )


def factor_levels_payload(
    levels: Mapping[str, FactorLevel],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "factor": level.factor,
            "id": level.id,
            "label": level.label,
            "role": level.role,
            "metadata": level.metadata,
        }
        for name, level in sorted(levels.items())
    }


def factor_level_counts(
    runs: tuple[PopulationRunRecord, ...] | list[PopulationRunRecord],
    factor: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run in runs:
        level_id = run.factor_id(factor)
        if level_id is None:
            continue
        counts[level_id] = counts.get(level_id, 0) + 1
    return counts
