from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from cave.observation.episodes import Episode
from cave.demonstrations.subjects import SubjectRun
from cave.observation.population import (
    FactorLevel,
    PopulationRunRecord,
    factor_level,
)
from cave.observation.views import ExperienceView


EpisodeFactory = Callable[[], Episode]
ExtraAssetWriter = Callable[[Episode, Path], None]
ReportCheck = Callable[[Episode], dict[str, object]]
SubjectRunFactory = Callable[[], tuple[Sequence[SubjectRun], Sequence[str]]]
MatrixRunFactory = Callable[[], Sequence["MatrixRunRecord"]]
MatrixCheck = Callable[[Sequence["MatrixRunRecord"]], dict[str, object]]


@dataclass(frozen=True)
class ReportViewAsset:
    id: str
    title: str
    views: Sequence[ExperienceView]
    filename: str
    kind: str = "animation"
    columns: int = 1
    style: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"animation", "frame"}:
            raise ValueError("ReportViewAsset.kind must be animation or frame")


@dataclass(frozen=True)
class ReportExtraAsset:
    id: str
    title: str
    filename: str
    writer: ExtraAssetWriter


@dataclass(frozen=True)
class ReportSection:
    title: str
    body: str
    asset_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProducerReportSpec:
    id: str
    title: str
    episode_factory: EpisodeFactory
    input_summary: str
    description: str = ""
    views: Sequence[ExperienceView] = field(default_factory=tuple)
    view_assets: Sequence[ReportViewAsset] = field(default_factory=tuple)
    extra_assets: Sequence[ReportExtraAsset] = field(default_factory=tuple)
    checks: Sequence[ReportCheck] = field(default_factory=tuple)
    sections: Sequence[ReportSection] = field(default_factory=tuple)
    frame_time: float | None = None
    dt: float = 0.1
    fps: int = 12
    columns: int = 2
    style: str = "default"
    config: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SubjectComparisonReportSpec:
    id: str
    title: str
    run_factory: SubjectRunFactory
    description: str = ""
    sections: Sequence[ReportSection] = field(default_factory=tuple)
    samples: int = 48
    cluster_threshold: float = 1e-12
    config: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MatrixRunRecord:
    id: str
    label: str
    sequence_id: str
    subject_id: str
    variant_id: str
    run: SubjectRun
    metadata: dict[str, object] = field(default_factory=dict)
    factors: dict[str, FactorLevel] = field(default_factory=dict)
    comparison_role: str = "treatment"
    matched_set_id: str | None = None
    replicate_id: str | None = None
    group_id: str | None = None

    def __post_init__(self) -> None:
        factors = dict(self.factors)
        defaults = {
            "sequence": factor_level("sequence", self.sequence_id),
            "treatment": factor_level(
                "treatment",
                self.sequence_id,
                role="shared_input",
            ),
            "subject": factor_level("subject", self.subject_id),
            "start_condition": factor_level(
                "start_condition",
                self.subject_id,
                role="initial_state",
            ),
            "condition": factor_level(
                "condition",
                self.variant_id,
                role=self.comparison_role,
            ),
            "mechanism_variant": factor_level(
                "mechanism_variant",
                self.variant_id,
                role=self.comparison_role,
            ),
        }
        for name, level in defaults.items():
            factors.setdefault(name, level)
        object.__setattr__(self, "factors", factors)
        if self.matched_set_id is None:
            object.__setattr__(self, "matched_set_id", self.sequence_id)
        if self.replicate_id is None:
            object.__setattr__(self, "replicate_id", self.subject_id)
        if self.group_id is None:
            object.__setattr__(self, "group_id", self.variant_id)

    def factor(self, name: str) -> FactorLevel | None:
        return self.factors.get(name)

    def factor_id(self, name: str, default: str | None = None) -> str | None:
        level = self.factor(name)
        if level is None:
            return default
        return level.id

    def to_population_record(self) -> PopulationRunRecord:
        return PopulationRunRecord(
            id=self.id,
            label=self.label,
            episode=self.run.episode,
            factors=self.factors,
            comparison_role=self.comparison_role,
            matched_set_id=self.matched_set_id,
            replicate_id=self.replicate_id,
            group_id=self.group_id,
            metadata={
                **self.metadata,
                "sequence_id": self.sequence_id,
                "subject_id": self.subject_id,
                "variant_id": self.variant_id,
                "run_id": self.run.id,
            },
        )


@dataclass(frozen=True)
class MatrixReportSpec:
    id: str
    title: str
    run_factory: MatrixRunFactory
    description: str = ""
    sections: Sequence[ReportSection] = field(default_factory=tuple)
    checks: Sequence[MatrixCheck] = field(default_factory=tuple)
    samples: int = 48
    cluster_thresholds: dict[str, float] = field(default_factory=dict)
    config: dict[str, object] = field(default_factory=dict)
