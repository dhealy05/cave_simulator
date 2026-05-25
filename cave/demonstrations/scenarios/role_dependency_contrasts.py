from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cave.observation.episodes import Episode, EpisodeInput, EpisodeObservation
from cave.observation.projections import encode_value
from cave.observation.views import default_views
from cave.presentation.reports.specs import (
    ProducerReportSpec,
    ReportExtraAsset,
    ReportSection,
)


CONTROL_VOCABULARY = ["visual", "audio", "value"]


@dataclass(frozen=True)
class ProbeVariant:
    id: str
    label: str
    expected_absences: tuple[str, ...]


def role_dependency_contrast_variants() -> tuple[ProbeVariant, ...]:
    return (
        ProbeVariant("positive-control", "Cave-like positive control", ()),
        ProbeVariant(
            "passive-recorder",
            "Passive external-input recorder",
            (
                "prediction_temporal_dependency",
                "unseen_modality_boundary",
                "attention_gate",
                "value_future_attention",
            ),
        ),
        ProbeVariant(
            "random-recurrent",
            "Random recurrent state",
            (
                "prediction_temporal_dependency",
                "attention_gate",
                "value_future_attention",
            ),
        ),
        ProbeVariant(
            "cosmetic-topology",
            "Cosmetic topology faker",
            (
                "prediction_temporal_dependency",
                "value_future_attention",
            ),
        ),
    )


def role_dependency_contrasts_report_spec(
    *,
    dt: float = 0.2,
    fps: int = 8,
    include_assets: bool = True,
) -> ProducerReportSpec:
    extra_assets = ()
    if include_assets:
        extra_assets = (
            ReportExtraAsset(
                id="contrast_metrics_json",
                title="Contrast Metrics JSON",
                filename="contrast_metrics.json",
                writer=write_role_dependency_contrast_metrics_json,
            ),
        )

    return ProducerReportSpec(
        id="role-dependency-contrasts",
        title="Causal Probe: Role Dependency Contrasts",
        episode_factory=lambda: build_role_dependency_contrast_episode("positive-control"),
        input_summary=(
            "synthetic control sequence with repetition, violation, modality, "
            "attention, and objective-pressure probes"
        ),
        description=(
            "Runs one authored probe sequence through a Cave-like positive control "
            "and three intervention controls. The result is a set of observed "
            "contrasts: which claimed role dependencies remain, disappear, or can "
            "be cosmetically imitated."
        ),
        views=default_views(),
        view_assets=(),
        extra_assets=extra_assets,
        frame_time=2.0,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "role_dependency_contrasts",
            "scenario": "role_dependency_contrasts",
            "dt": dt,
            "fps": fps,
        },
        checks=(check_role_dependency_contrasts,),
        sections=(
            ReportSection(
                title="Hypothesis",
                body=(
                    "The probe asks whether four relations are carried by the "
                    "episode dynamics: history-shaped prediction, modality "
                    "exclusion, attention-weighted composition, and value-shaped "
                    "future attention."
                ),
            ),
            ReportSection(
                title="Interventions And Contrasts",
                body=(
                    "The interventions are not bad runs. They are contrasts. The "
                    "passive recorder copies external inputs, the recurrent control "
                    "changes state without the tested dependencies, and the "
                    "cosmetic-topology control emits plausible-looking topology "
                    "metadata while leaving temporal dependencies absent."
                ),
                asset_ids=("contrast_metrics_json",),
            ),
        ),
    )


def build_role_dependency_contrast_episode(variant: str) -> Episode:
    inputs = _control_inputs()
    builders = {
        "positive-control": _positive_observations,
        "passive-recorder": _passive_recorder_observations,
        "random-recurrent": _random_recurrent_observations,
        "cosmetic-topology": _cosmetic_topology_observations,
    }
    try:
        observations = builders[variant](inputs)
    except KeyError as exc:
        choices = ", ".join(sorted(builders))
        raise ValueError(f"unknown role dependency contrast variant {variant!r}; choose from {choices}") from exc

    metadata = {
        "source": "cave.demonstrations.scenarios.role_dependency_contrasts",
        "adapter": "RoleDependencyContrast",
        "control_variant": variant,
        "control_kind": _control_label(variant),
    }
    if variant == "cosmetic-topology":
        metadata["cosmetic_topology_mass"] = 500.0

    return Episode(
        source_name=f"causal-probe:{variant}",
        vocabulary=list(CONTROL_VOCABULARY),
        inputs=inputs,
        observations=observations,
        duration=8.0,
        metadata=metadata,
    )


def check_role_dependency_contrasts(episode: Episode) -> dict[str, object]:
    variants = {
        variant.id: build_role_dependency_contrast_episode(variant.id)
        for variant in role_dependency_contrast_variants()
    }
    metrics = {
        variant_id: role_dependency_metrics(control_episode)
        for variant_id, control_episode in variants.items()
    }
    expected_absences = {
        variant.id: list(variant.expected_absences)
        for variant in role_dependency_contrast_variants()
    }
    role_ids = tuple(metrics["positive-control"]["roles"])
    relation_presence = {
        variant_id: dict(variant_metrics["roles"])
        for variant_id, variant_metrics in metrics.items()
    }
    observed_absences = {
        variant_id: [
            role_id
            for role_id, passed in variant_metrics["roles"].items()
            if not passed
        ]
        for variant_id, variant_metrics in metrics.items()
    }
    relation_presence_counts = {
        variant_id: sum(1 for passed in variant_metrics["roles"].values() if passed)
        for variant_id, variant_metrics in metrics.items()
    }
    missing_expected_absences = {
        variant.id: [
            role_id
            for role_id in variant.expected_absences
            if role_id not in observed_absences[variant.id]
        ]
        for variant in role_dependency_contrast_variants()
    }

    errors: list[str] = []
    if observed_absences["positive-control"]:
        errors.append(
            "positive control is missing expected relations: "
            + ", ".join(observed_absences["positive-control"])
        )
    for variant in role_dependency_contrast_variants():
        if variant.id == "positive-control":
            continue
        if missing_expected_absences[variant.id]:
            errors.append(
                f"{variant.id} retained relations expected to be absent: "
                + ", ".join(missing_expected_absences[variant.id])
            )
        if relation_presence_counts[variant.id] == len(role_ids):
            errors.append(f"{variant.id} preserved every relation; no contrast observed")

    return {
        "id": "role_dependency_contrasts",
        "ok": not errors,
        "errors": errors,
        "metrics": metrics,
        "roles": {
            "relation_presence": relation_presence,
            "observed_absences": observed_absences,
            "expected_absences": expected_absences,
            "missing_expected_absences": missing_expected_absences,
            "relation_presence_counts": relation_presence_counts,
            "cosmetic_topology_exceeds_control": (
                metrics["cosmetic-topology"]["raw"]["cosmetic_topology_mass"]
                > metrics["positive-control"]["raw"]["cosmetic_topology_mass"]
            ),
            "cosmetic_topology_does_not_supply_prediction_history": not metrics[
                "cosmetic-topology"
            ]["roles"]["prediction_temporal_dependency"],
        },
        "contrasts": role_dependency_contrast_results(metrics),
    }


def role_dependency_contrast_results(
    metrics: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    observed_by_role = {
        role_id: {
            variant_id: bool(variant_metrics["roles"][role_id])
            for variant_id, variant_metrics in metrics.items()
        }
        for role_id in metrics["positive-control"]["roles"]
    }
    return {
        "prediction_temporal_dependency": {
            "hypothesis": (
                "Expectation is history-dependent: repetition should lower "
                "surprise and a violation should raise it."
            ),
            "intervention_control": (
                "Compare the Cave-like control with passive copying, arbitrary "
                "recurrence, and cosmetic topology."
            ),
            "expected_contrast": (
                "Only an episode with prediction-history structure should show "
                "the repeat-down / violation-up pattern."
            ),
            "observed_contrast": observed_by_role["prediction_temporal_dependency"],
            "interpretation": (
                "The relation is absent in all three controls, including the "
                "cosmetic topology run, so topology-like output alone does not "
                "supply prediction history."
            ),
        },
        "unseen_modality_boundary": {
            "hypothesis": (
                "External presence and subject input are separable when a modality "
                "is unavailable."
            ),
            "intervention_control": (
                "Compare the Cave-like control with a passive recorder and a "
                "recurrent state control that do not enforce the modality boundary."
            ),
            "expected_contrast": (
                "The unheard input should have zero actual mass only when the "
                "episode enforces the sensor boundary."
            ),
            "observed_contrast": observed_by_role["unseen_modality_boundary"],
            "interpretation": (
                "The passive and recurrent controls leak the unheard event into "
                "actual state; the cosmetic control can imitate this simpler "
                "boundary without reproducing prediction or value dynamics."
            ),
        },
        "attention_gate": {
            "hypothesis": (
                "Simultaneous sensed inputs should be composed by attention weights, "
                "not by mere co-presence."
            ),
            "intervention_control": (
                "Compare the Cave-like control with input copying, arbitrary "
                "recurrence, and a cosmetic control that explicitly imitates the "
                "attention-weighted vector."
            ),
            "expected_contrast": (
                "The bottleneck vector should equal the attention-weighted "
                "composition only when attention gating is represented or imitated."
            ),
            "observed_contrast": observed_by_role["attention_gate"],
            "interpretation": (
                "The cosmetic run can match this local relation, which marks "
                "attention gating as an observable contrast but not sufficient "
                "evidence for the full subject trajectory."
            ),
        },
        "value_future_attention": {
            "hypothesis": (
                "A value-bearing event should alter a later attention distribution, "
                "not only label the current frame."
            ),
            "intervention_control": (
                "Compare the Cave-like control with copy, recurrence, and cosmetic "
                "metadata controls that do not propagate value into the next frame."
            ),
            "expected_contrast": (
                "The next frame should use the value-shifted attention distribution "
                "only in the Cave-like control."
            ),
            "observed_contrast": observed_by_role["value_future_attention"],
            "interpretation": (
                "The controls can carry input or metadata, but they do not preserve "
                "the value-to-future-attention dependency."
            ),
        },
    }


def role_dependency_metrics(episode: Episode) -> dict[str, object]:
    by_active = {
        tuple(observation.active_inputs): observation
        for observation in episode.observations
    }
    repeat_1 = by_active[("repeat_1",)]
    repeat_2 = by_active[("repeat_2",)]
    repeat_3 = by_active[("repeat_3",)]
    violation = by_active[("violation",)]
    unheard = by_active[("unheard_tone",)]
    bottleneck = by_active[("visual_marker", "audio_marker")]
    painful = by_active[("painful_audio",)]
    followup = by_active[("followup_audio",)]

    bottleneck_expected = np.array([0.25, 0.75, 0.0], dtype=float)
    unheard_zero = _norm(unheard.actual) <= 1e-9
    prediction_dependency = (
        repeat_1.surprise > repeat_2.surprise > repeat_3.surprise
        and violation.surprise > repeat_3.surprise * 2.0
        and _norm(repeat_3.expected) > _norm(repeat_1.expected)
    )
    attention_gate = np.allclose(bottleneck.actual, bottleneck_expected, atol=1e-9)
    value_shift = (
        float(painful.metadata.get("valence", {}).get("pain", 0.0)) > 0.0
        and painful.metadata.get("next_attention_channels", {}).get("audio", 0.0)
        > painful.metadata.get("attention_channels", {}).get("audio", 0.0)
        and followup.metadata.get("attention_channels", {}).get("audio", 0.0)
        == painful.metadata.get("next_attention_channels", {}).get("audio", None)
    )

    return {
        "roles": {
            "prediction_temporal_dependency": bool(prediction_dependency),
            "unseen_modality_boundary": bool(unheard_zero),
            "attention_gate": bool(attention_gate),
            "value_future_attention": bool(value_shift),
        },
        "raw": {
            "surprise": {
                "repeat_1": repeat_1.surprise,
                "repeat_2": repeat_2.surprise,
                "repeat_3": repeat_3.surprise,
                "violation": violation.surprise,
            },
            "repeat_3_expected_mass": _norm(repeat_3.expected),
            "unheard_actual_mass": _norm(unheard.actual),
            "bottleneck_actual": bottleneck.actual.tolist(),
            "painful_attention_channels": dict(
                painful.metadata.get("attention_channels", {})
            ),
            "painful_next_attention_channels": dict(
                painful.metadata.get("next_attention_channels", {})
            ),
            "followup_attention_channels": dict(
                followup.metadata.get("attention_channels", {})
            ),
            "cosmetic_topology_mass": float(
                episode.metadata.get("cosmetic_topology_mass", 0.0)
            ),
        },
    }


def write_role_dependency_contrast_metrics_json(episode: Episode, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = check_role_dependency_contrasts(episode)
    output.write_text(
        json.dumps(encode_value(payload), indent=2) + "\n",
        encoding="utf-8",
    )


def _control_inputs() -> list[EpisodeInput]:
    specs = (
        ("repeat_1", 0.0, [1.0, 0.0, 0.0], "visual"),
        ("repeat_2", 1.0, [1.0, 0.0, 0.0], "visual"),
        ("repeat_3", 2.0, [1.0, 0.0, 0.0], "visual"),
        ("violation", 3.0, [0.0, 1.0, 0.0], "visual"),
        ("unheard_tone", 4.0, [0.0, 1.0, 0.0], "audio"),
        ("visual_marker", 5.0, [1.0, 0.0, 0.0], "visual"),
        ("audio_marker", 5.0, [0.0, 1.0, 0.0], "audio"),
        ("painful_audio", 6.0, [0.0, 1.0, -1.0], "audio"),
        ("followup_audio", 7.0, [0.0, 1.0, 0.0], "audio"),
    )
    return [
        EpisodeInput(
            id=input_id,
            kind="control",
            start=start,
            end=start + 0.8,
            order_index=index,
            features=np.array(features, dtype=float),
            modality=modality,
        )
        for index, (input_id, start, features, modality) in enumerate(specs)
    ]


def _positive_observations(inputs: list[EpisodeInput]) -> list[EpisodeObservation]:
    return [
        _obs(0.0, ["repeat_1"], expected=[0.0, 0.0, 0.0], actual=[1.0, 0.0, 0.0], memory=[0.45, 0.0, 0.0], surprise=1.0),
        _obs(1.0, ["repeat_2"], expected=[0.75, 0.0, 0.0], actual=[1.0, 0.0, 0.0], memory=[0.8, 0.0, 0.0], surprise=0.25),
        _obs(2.0, ["repeat_3"], expected=[0.96, 0.0, 0.0], actual=[1.0, 0.0, 0.0], memory=[0.96, 0.0, 0.0], surprise=0.04),
        _obs(3.0, ["violation"], expected=[0.98, 0.0, 0.0], actual=[0.0, 1.0, 0.0], memory=[0.6, 0.35, 0.0], surprise=1.4),
        _obs(
            4.0,
            ["unheard_tone"],
            expected=[0.0, 0.0, 0.0],
            actual=[0.0, 0.0, 0.0],
            memory=[0.55, 0.32, 0.0],
            surprise=0.0,
            attention_weights={"unheard_tone": 0.0},
            metadata={"sensor_channels": {"visual": [0.0, 0.0, 0.0]}},
        ),
        _obs(
            5.0,
            ["visual_marker", "audio_marker"],
            expected=[0.0, 0.0, 0.0],
            actual=[0.25, 0.75, 0.0],
            memory=[0.48, 0.55, 0.0],
            surprise=0.79,
            attention_weights={"visual_marker": 0.25, "audio_marker": 0.75},
            metadata={"attention_channels": {"visual": 0.25, "audio": 0.75}},
        ),
        _obs(
            6.0,
            ["painful_audio"],
            expected=[0.0, 0.35, 0.0],
            actual=[0.0, 0.2, -0.2],
            memory=[0.4, 0.5, -0.1],
            surprise=0.25,
            attention_weights={"painful_audio": 0.2},
            metadata={
                "attention_channels": {"visual": 0.8, "audio": 0.2},
                "next_attention_channels": {"visual": 0.256, "audio": 0.744},
                "valence": {"pain": 0.8, "pleasure": 0.0, "utility": -1.0},
            },
        ),
        _obs(
            7.0,
            ["followup_audio"],
            expected=[0.0, 0.5, 0.0],
            actual=[0.0, 0.744, 0.0],
            memory=[0.32, 0.63, -0.08],
            surprise=0.244,
            attention_weights={"followup_audio": 0.744},
            metadata={"attention_channels": {"visual": 0.256, "audio": 0.744}},
        ),
    ]


def _passive_recorder_observations(inputs: list[EpisodeInput]) -> list[EpisodeObservation]:
    observations = []
    memory = np.zeros(3, dtype=float)
    for item in inputs:
        actual = item.features.copy()
        expected = actual.copy()
        memory = actual.copy()
        active = [item.id]
        if item.id == "visual_marker":
            audio = _input_by_id(inputs, "audio_marker")
            actual = item.features + audio.features
            expected = actual.copy()
            memory = actual.copy()
            active = ["visual_marker", "audio_marker"]
        if item.id == "audio_marker":
            continue
        observations.append(
            _obs(
                item.start,
                active,
                expected=expected,
                actual=actual,
                memory=memory,
                surprise=0.0,
                attention_weights={input_id: 1.0 for input_id in active},
            )
        )
    return observations


def _random_recurrent_observations(inputs: list[EpisodeInput]) -> list[EpisodeObservation]:
    rows = (
        ([0.2, 0.1, 0.0], [0.1, 0.0, 0.1], 0.4),
        ([0.0, 0.3, 0.1], [0.2, 0.1, 0.0], 0.9),
        ([0.4, 0.0, 0.2], [0.0, 0.2, 0.0], 0.2),
        ([0.1, 0.5, 0.0], [0.3, 0.0, 0.1], 0.7),
        ([0.0, 0.2, 0.3], [0.2, 0.2, 0.0], 0.6),
        ([0.7, 0.1, 0.0], [0.1, 0.4, 0.0], 0.3),
        ([0.2, 0.6, 0.2], [0.0, 0.1, 0.2], 0.8),
        ([0.5, 0.2, 0.1], [0.3, 0.2, 0.0], 0.5),
    )
    active_sets = (
        ["repeat_1"],
        ["repeat_2"],
        ["repeat_3"],
        ["violation"],
        ["unheard_tone"],
        ["visual_marker", "audio_marker"],
        ["painful_audio"],
        ["followup_audio"],
    )
    observations = []
    for index, (actual, expected, surprise) in enumerate(rows):
        active = active_sets[index]
        observations.append(
            _obs(
                float(index),
                active,
                expected=expected,
                actual=actual,
                memory=np.roll(np.array(actual, dtype=float), 1),
                surprise=surprise,
                attention_weights={input_id: 0.5 for input_id in active},
                metadata={"attention_channels": {"visual": 0.5, "audio": 0.5}},
            )
        )
    return observations


def _cosmetic_topology_observations(inputs: list[EpisodeInput]) -> list[EpisodeObservation]:
    observations = []
    active_sets = (
        ["repeat_1"],
        ["repeat_2"],
        ["repeat_3"],
        ["violation"],
        ["unheard_tone"],
        ["visual_marker", "audio_marker"],
        ["painful_audio"],
        ["followup_audio"],
    )
    actuals = (
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.25, 0.75, 0.0],
        [0.0, 0.2, -0.2],
        [0.0, 0.2, 0.0],
    )
    for index, active in enumerate(active_sets):
        metadata = {
            "cosmetic_topology_density": 50.0 + index * 10.0,
            "attention_channels": {"visual": 0.5, "audio": 0.5},
        }
        if active == ["unheard_tone"]:
            metadata["sensor_channels"] = {"visual": [0.0, 0.0, 0.0]}
        observations.append(
            _obs(
                float(index),
                active,
                expected=[0.0, 0.0, 0.0],
                actual=actuals[index],
                memory=[0.5, 0.5, 0.5],
                surprise=0.5,
                attention_weights={
                    input_id: (
                        0.25
                        if input_id == "visual_marker"
                        else 0.75
                        if input_id == "audio_marker"
                        else 0.0
                        if input_id == "unheard_tone"
                        else 0.5
                    )
                    for input_id in active
                },
                metadata=metadata,
            )
        )
    return observations


def _obs(
    t: float,
    active_inputs: list[str],
    *,
    expected,
    actual,
    memory,
    surprise: float,
    learning_rate: float = 0.2,
    attention: float = 1.0,
    attention_weights: dict[str, float] | None = None,
    metadata: dict[str, object] | None = None,
) -> EpisodeObservation:
    actual_array = np.asarray(actual, dtype=float)
    weights = attention_weights or {input_id: attention for input_id in active_inputs}
    input_features = {
        input_id: _feature_for_input_id(input_id)
        for input_id in active_inputs
    }
    return EpisodeObservation(
        t=t,
        t_normalized=t / 8.0,
        expected=np.asarray(expected, dtype=float),
        actual=actual_array,
        memory_state=np.asarray(memory, dtype=float),
        surprise=float(surprise),
        learning_rate=float(learning_rate),
        attention=float(attention),
        attention_weights=dict(weights),
        active_inputs=list(active_inputs),
        input_features=input_features,
        metadata=dict(metadata or {}),
    )


def _input_by_id(inputs: list[EpisodeInput], input_id: str) -> EpisodeInput:
    for item in inputs:
        if item.id == input_id:
            return item
    raise KeyError(input_id)


def _feature_for_input_id(input_id: str) -> np.ndarray:
    features = {
        "repeat_1": [1.0, 0.0, 0.0],
        "repeat_2": [1.0, 0.0, 0.0],
        "repeat_3": [1.0, 0.0, 0.0],
        "violation": [0.0, 1.0, 0.0],
        "unheard_tone": [0.0, 1.0, 0.0],
        "visual_marker": [1.0, 0.0, 0.0],
        "audio_marker": [0.0, 1.0, 0.0],
        "painful_audio": [0.0, 1.0, -1.0],
        "followup_audio": [0.0, 1.0, 0.0],
    }
    return np.asarray(features[input_id], dtype=float)


def _control_label(variant: str) -> str:
    for control_variant in role_dependency_contrast_variants():
        if control_variant.id == variant:
            return control_variant.label
    return variant


def _norm(value) -> float:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.linalg.norm(array.ravel()))
