from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cave.commitments.attention import AttentionProfile, FixedAttentionPolicy, ObjectiveAdaptiveAttentionPolicy
from cave.commitments.memory import MemoryParams
from cave.commitments.topology import SubjectiveTopologyParams, SubjectiveTopologyPrior
from cave.demonstrations.examples import model_for_sequence
from cave.demonstrations.simulation import ModelParams
from cave.observation.episodes import CaveProducer, Episode
from cave.observation.experience import ExperienceObject, FeatureVector, InputSequence, TemporalExtent
from cave.observation.projections import encode_value
from cave.observation.sensing import FeatureSensor, Sensorium
from cave.observation.views import default_views
from cave.presentation.reports.specs import ProducerReportSpec, ReportExtraAsset, ReportSection
from cave.substrates.cavenet import CaveNet, CaveNetProducer


REGULATION_VARIANTS = ("cave-fixed", "cave-adaptive", "cavenet-fixed", "cavenet-adaptive")
REGULATION_VOCABULARY = ["visual_signal", "audio_signal"]


@dataclass(frozen=True)
class RegulationRun:
    variant: str
    episode: Episode
    metrics: dict[str, float | str]


def regulation_recovery_report_spec(
    *,
    dt: float = 1.0,
    fps: int = 4,
    include_assets: bool = True,
) -> ProducerReportSpec:
    def build_episode() -> Episode:
        return build_regulation_episode("cave-adaptive", dt=dt)

    extra_assets = ()
    if include_assets:
        extra_assets = (
            ReportExtraAsset(
                id="regulation_metrics",
                title="Regulation Metrics JSON",
                filename="regulation_metrics.json",
                writer=lambda episode, output: write_regulation_metrics_json(output, dt=dt),
            ),
        )

    return ProducerReportSpec(
        id="regulation-recovery",
        title="Regulation Recovery",
        episode_factory=build_episode,
        input_summary="fixed and objective-adaptive attention variants",
        description=(
            "Tests whether objective/value pressure changes future coupling to an "
            "event stream through attention-channel regulation."
        ),
        views=default_views(),
        extra_assets=extra_assets,
        checks=(lambda episode: check_regulation_recovery(dt=dt),),
        frame_time=0.0,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "regulation_recovery",
            "scenario": "regulation_recovery",
            "role": "regulation",
            "variants": list(REGULATION_VARIANTS),
        },
        sections=(
            ReportSection(
                title="Question",
                body=(
                    "Regulation is present when an affective/objective signal changes "
                    "future coupling. Here the coupling variable is next-step channel "
                    "attention."
                ),
                asset_ids=("regulation_metrics",),
            ),
        ),
    )


def build_regulation_episode(variant: str, *, dt: float = 1.0) -> Episode:
    adaptive = variant.endswith("adaptive")
    if variant.startswith("cave"):
        model = model_for_sequence(
            _regulation_sequence(),
            params=_regulation_params(adaptive=adaptive),
            vocabulary=list(REGULATION_VOCABULARY),
        )
        model.sensorium = _regulation_sensorium()
        return CaveProducer(model, name=f"regulation:{variant}").run(dt=dt)
    if variant.startswith("cavenet"):
        model = model_for_sequence(
            _regulation_sequence(),
            params=_regulation_params(adaptive=adaptive),
            vocabulary=list(REGULATION_VOCABULARY),
        )
        model.sensorium = _regulation_sensorium()
        cavenet = CaveNet.from_subject_state(
            sequence=model.sequence,
            subject_state=model.subject_state,
            params=model.params,
            vocabulary=model.vocabulary,
            sensorium=model.sensorium,
        )
        return CaveNetProducer(cavenet, name=f"regulation:{variant}").run(dt=dt)
    raise ValueError(f"unsupported regulation variant: {variant}")


def regulation_runs(*, dt: float = 1.0) -> tuple[RegulationRun, ...]:
    return tuple(
        RegulationRun(variant, episode, _regulation_metrics(episode))
        for variant in REGULATION_VARIANTS
        for episode in (build_regulation_episode(variant, dt=dt),)
    )


def check_regulation_recovery(*, dt: float = 1.0) -> dict[str, object]:
    runs = regulation_runs(dt=dt)
    metrics = {run.variant: run.metrics for run in runs}
    roles = {
        "future_attention_regulation": {
            "cave_fixed_audio_delta": metrics["cave-fixed"]["audio_delta"],
            "cave_adaptive_audio_delta": metrics["cave-adaptive"]["audio_delta"],
            "cavenet_fixed_audio_delta": metrics["cavenet-fixed"]["audio_delta"],
            "cavenet_adaptive_audio_delta": metrics["cavenet-adaptive"]["audio_delta"],
            "cave_adaptive_next_audio": metrics["cave-adaptive"]["next_audio_attention"],
            "cavenet_adaptive_next_audio": metrics["cavenet-adaptive"]["next_audio_attention"],
        }
    }
    errors: list[str] = []
    if not abs(float(roles["future_attention_regulation"]["cave_fixed_audio_delta"])) <= 1e-12:
        errors.append("fixed Cave attention changed")
    if not float(roles["future_attention_regulation"]["cave_adaptive_audio_delta"]) > 0.2:
        errors.append("adaptive Cave did not increase audio attention")
    if not abs(float(roles["future_attention_regulation"]["cavenet_fixed_audio_delta"])) <= 1e-12:
        errors.append("fixed CaveNet attention changed")
    if not float(roles["future_attention_regulation"]["cavenet_adaptive_audio_delta"]) > 0.2:
        errors.append("adaptive CaveNet did not increase audio attention")
    return {
        "id": "regulation_recovery",
        "ok": not errors,
        "errors": errors,
        "metrics": metrics,
        "roles": roles,
    }


def write_regulation_metrics_json(output: Path, *, dt: float = 1.0) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(encode_value(check_regulation_recovery(dt=dt)), indent=2) + "\n",
        encoding="utf-8",
    )


def _regulation_metrics(episode: Episode) -> dict[str, float | str]:
    first = episode.observations[0]
    current = first.metadata.get("attention_channels", {})
    next_weights = first.metadata.get("next_attention_channels", {})
    current_audio = float(current.get("audio", 0.0))
    next_audio = float(next_weights.get("audio", current_audio))
    return {
        "current_audio_attention": current_audio,
        "next_audio_attention": next_audio,
        "audio_delta": next_audio - current_audio,
        "pain": float(first.metadata.get("valence", {}).get("pain", 0.0)),
        "adapter": str(episode.metadata.get("adapter", "")),
    }


def _regulation_sequence() -> InputSequence:
    return InputSequence(
        [
            ExperienceObject(
                id="visual_marker",
                temporal_extent=TemporalExtent(0.0, 1.0, 0),
                features=FeatureVector({"visual_signal": 1.0}),
                modality="visual",
            ),
            ExperienceObject(
                id="painful_audio",
                temporal_extent=TemporalExtent(0.0, 1.0, 1),
                features=FeatureVector({"audio_signal": 1.0}),
                modality="audio",
                metadata={"affect": {"pain": 1.0}},
            ),
            ExperienceObject(
                id="next_audio",
                temporal_extent=TemporalExtent(1.0, 2.0, 2),
                features=FeatureVector({"audio_signal": 1.0}),
                modality="audio",
            ),
        ]
    )


def _regulation_params(*, adaptive: bool) -> ModelParams:
    return ModelParams(
        memory=MemoryParams(retention=0.5, decay_tau=2.0, max_age=4.0),
        attention=AttentionProfile(
            mode="constant",
            level=1.0,
            channel_weights={"visual": 0.8, "audio": 0.2},
        ),
        attention_policy=(
            ObjectiveAdaptiveAttentionPolicy(learning_rate=0.7, pain_gain=2.0)
            if adaptive
            else FixedAttentionPolicy()
        ),
        topology=SubjectiveTopologyParams(
            feature_x="visual_signal",
            feature_y="audio_signal",
            prior=SubjectiveTopologyPrior(),
        ),
    )


def _regulation_sensorium() -> Sensorium:
    return Sensorium(
        sensors=(
            FeatureSensor(modality="visual", channel="visual"),
            FeatureSensor(modality="audio", channel="audio"),
        )
    )
