from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path

import numpy as np

from cave.commitments.attention import AttentionProfile
from cave.commitments.memory import MemoryParams
from cave.commitments.topology import SubjectiveTopologyParams, SubjectiveTopologyPrior
from cave.commitments.workspace import TopKWorkspaceCompressor
from cave.demonstrations.examples import model_for_sequence
from cave.demonstrations.scenarios._common import core_scenario_assets
from cave.demonstrations.simulation import ModelParams
from cave.observation.episodes import CaveProducer, Episode
from cave.observation.experience import ExperienceObject, FeatureVector, InputSequence, TemporalExtent
from cave.observation.projections import encode_value
from cave.observation.views import default_views
from cave.presentation.reports.specs import ProducerReportSpec, ReportExtraAsset, ReportSection
from cave.substrates.cavenet import CaveNet, CaveNetProducer
from cave.substrates.minimal_subject import MinimalSubjectConfig, run_minimal_subject


COMMON_BEHAVIOR_VOCABULARY = [
    "cue",
    "expected",
    "violation",
    "distractor",
    "good",
    "bad",
]

SUBSTRATES = ("cave", "cavenet", "minimal_subject")
PROBES = ("expectation_repetition", "workspace_selection", "value_separation")


@dataclass(frozen=True)
class BehaviorRun:
    substrate: str
    probe: str
    episode: Episode
    metrics: dict[str, float | int | str]


def common_behaviors_report_spec(
    *,
    dt: float = 1.0,
    fps: int = 4,
    include_assets: bool = True,
) -> ProducerReportSpec:
    def build_episode() -> Episode:
        return build_common_behavior_episode("cave", "expectation_repetition", dt=dt)

    extra_assets = ()
    if include_assets:
        extra_assets = (
            ReportExtraAsset(
                id="common_behavior_metrics",
                title="Common Behavior Metrics JSON",
                filename="common_behavior_metrics.json",
                writer=lambda episode, output: write_common_behavior_metrics_json(
                    output,
                    dt=dt,
                ),
            ),
        )

    return ProducerReportSpec(
        id="common-behaviors",
        title="Common Behavior Suite",
        episode_factory=build_episode,
        input_summary="Cave, CaveNet, and minimal_subject through shared behavior probes",
        description=(
            "Runs the same small behavior probes through multiple substrates. "
            "The goal is not to require identical mechanisms, but to ask whether "
            "each substrate emits a normal Episode with comparable evidence for "
            "expectation, selection, and value behavior."
        ),
        views=default_views(),
        view_assets=core_scenario_assets() if include_assets else (),
        extra_assets=extra_assets,
        checks=(lambda episode: check_common_behaviors(dt=dt),),
        frame_time=2.0,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "common_behavior_suite",
            "scenario": "common_behaviors",
            "dt": dt,
            "fps": fps,
            "substrates": list(SUBSTRATES),
            "probes": list(PROBES),
            "vocabulary": list(COMMON_BEHAVIOR_VOCABULARY),
        },
        sections=(
            ReportSection(
                title="Purpose",
                body=(
                    "Cave is the reference behavior vocabulary. CaveNet and "
                    "minimal_subject are substrate-side bridges. This suite checks "
                    "whether the common Episode contract is sufficient to compare "
                    "basic Cave-like behaviors across them."
                ),
            ),
            ReportSection(
                title="Shared Probes",
                body=(
                    "The probes are deliberately small: repeated input followed by "
                    "a violation, a compression/selection event with a distractor, "
                    "and positive/negative valued events. Each substrate is scored "
                    "with the same metric names where possible."
                ),
                asset_ids=("common_behavior_metrics",),
            ),
        ),
    )


def build_common_behavior_episode(
    substrate: str,
    probe: str,
    *,
    dt: float = 1.0,
) -> Episode:
    sequence = common_behavior_sequence(probe)
    if substrate == "minimal_subject":
        return run_minimal_subject(
            sequence,
            vocabulary=list(COMMON_BEHAVIOR_VOCABULARY),
            preference_vector=_preference_vector(),
            config=MinimalSubjectConfig(
                workspace_capacity=2,
                diagnostic_features=("cue", "expected", "good", "bad"),
            ),
            source_name=f"minimal:{probe}",
        )
    if substrate == "cave":
        model = model_for_sequence(
            sequence,
            params=common_behavior_params(probe),
            vocabulary=list(COMMON_BEHAVIOR_VOCABULARY),
        )
        return CaveProducer(model, name=f"cave:{probe}").run(dt=dt)
    if substrate == "cavenet":
        model = model_for_sequence(
            sequence,
            params=common_behavior_params(probe),
            vocabulary=list(COMMON_BEHAVIOR_VOCABULARY),
        )
        cavenet = CaveNet.from_subject_state(
            sequence=model.sequence,
            subject_state=model.subject_state,
            params=model.params,
            vocabulary=model.vocabulary,
            sensorium=model.sensorium,
        )
        return CaveNetProducer(cavenet, name=f"cavenet:{probe}").run(dt=dt)
    raise ValueError(f"unsupported substrate: {substrate}")


def common_behavior_runs(*, dt: float = 1.0) -> tuple[BehaviorRun, ...]:
    runs: list[BehaviorRun] = []
    for probe in PROBES:
        for substrate in SUBSTRATES:
            episode = build_common_behavior_episode(substrate, probe, dt=dt)
            runs.append(
                BehaviorRun(
                    substrate=substrate,
                    probe=probe,
                    episode=episode,
                    metrics=_metrics_for_probe(probe, episode),
                )
            )
    return tuple(runs)


def common_behavior_sequence(probe: str) -> InputSequence:
    return _sequence_for_probe(probe)


def common_behavior_params(probe: str) -> ModelParams:
    return _params_for_probe(probe)


def expectation_repetition_metrics(episode: Episode) -> dict[str, float | int | str]:
    return _expectation_metrics(episode)


def workspace_selection_metrics(episode: Episode) -> dict[str, float | int | str]:
    return _selection_metrics(episode)


def check_common_behaviors(*, dt: float = 1.0) -> dict[str, object]:
    runs = common_behavior_runs(dt=dt)
    metrics = _metrics_payload(runs)
    roles = _role_payload(runs)
    errors: list[str] = []

    for substrate in SUBSTRATES:
        expectation = roles["expectation_repetition"][substrate]
        if not expectation["surprise_drop"] > 0.0:
            errors.append(f"{substrate} did not reduce surprise on repeated input")
        if not expectation["violation_margin"] > 0.0:
            errors.append(f"{substrate} did not raise surprise on violation")

        selection = roles["workspace_selection"][substrate]
        if not selection["active_feature_count"] <= 2:
            errors.append(f"{substrate} did not keep selection within capacity")
        if not selection["dropped_mass"] > 0.0:
            errors.append(f"{substrate} did not drop any distractor mass")

        value = roles["value_separation"][substrate]
        if not value["positive_signal"] > 0.0:
            errors.append(f"{substrate} did not expose a positive value signal")
        if not value["negative_signal"] > 0.0:
            errors.append(f"{substrate} did not expose a negative value signal")
        if not value["utility_contrast"] > 0.0:
            errors.append(f"{substrate} did not separate positive and negative utility")

    equivalence = _cavenet_cave_equivalence(runs)
    if not equivalence["max_actual_distance"] <= 1e-12:
        errors.append("CaveNet and Cave diverged on actual inputs in common probes")
    if not equivalence["max_memory_distance"] <= 1e-12:
        errors.append("CaveNet and Cave diverged on memory states in common probes")

    return {
        "id": "common_behaviors",
        "ok": not errors,
        "errors": errors,
        "metrics": metrics,
        "roles": roles,
        "equivalence": equivalence,
    }


def write_common_behavior_metrics_json(output: Path, *, dt: float = 1.0) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    result = check_common_behaviors(dt=dt)
    output.write_text(json.dumps(encode_value(result), indent=2) + "\n", encoding="utf-8")


def _metrics_payload(runs: tuple[BehaviorRun, ...]) -> dict[str, dict[str, dict[str, object]]]:
    payload: dict[str, dict[str, dict[str, object]]] = {}
    for run in runs:
        payload.setdefault(run.probe, {})[run.substrate] = dict(run.metrics)
    return payload


def _role_payload(runs: tuple[BehaviorRun, ...]) -> dict[str, dict[str, dict[str, float | int]]]:
    by_key = {(run.probe, run.substrate): run.metrics for run in runs}
    roles: dict[str, dict[str, dict[str, float | int]]] = {
        "expectation_repetition": {},
        "workspace_selection": {},
        "value_separation": {},
    }
    for substrate in SUBSTRATES:
        expectation = by_key[("expectation_repetition", substrate)]
        roles["expectation_repetition"][substrate] = {
            "first_repeat_surprise": float(expectation["first_repeat_surprise"]),
            "late_repeat_surprise": float(expectation["late_repeat_surprise"]),
            "violation_surprise": float(expectation["violation_surprise"]),
            "surprise_drop": float(expectation["surprise_drop"]),
            "violation_margin": float(expectation["violation_margin"]),
        }
        selection = by_key[("workspace_selection", substrate)]
        roles["workspace_selection"][substrate] = {
            "active_feature_count": int(selection["active_feature_count"]),
            "dropped_mass": float(selection["dropped_mass"]),
            "compression_cost": float(selection["compression_cost"]),
        }
        value = by_key[("value_separation", substrate)]
        roles["value_separation"][substrate] = {
            "positive_signal": float(value["positive_signal"]),
            "negative_signal": float(value["negative_signal"]),
            "positive_utility": float(value["positive_utility"]),
            "negative_utility": float(value["negative_utility"]),
            "utility_contrast": float(value["utility_contrast"]),
        }
    return roles


def _metrics_for_probe(probe: str, episode: Episode) -> dict[str, float | int | str]:
    if probe == "expectation_repetition":
        return _expectation_metrics(episode)
    if probe == "workspace_selection":
        return _selection_metrics(episode)
    if probe == "value_separation":
        return _value_metrics(episode)
    raise ValueError(f"unsupported probe: {probe}")


def _expectation_metrics(episode: Episode) -> dict[str, float | int | str]:
    by_input = _observations_by_input(episode)
    first = by_input["repeat_0"][0]
    late = by_input["repeat_2"][0]
    violation = by_input["violation"][0]
    return {
        "first_repeat_surprise": float(first.surprise),
        "late_repeat_surprise": float(late.surprise),
        "violation_surprise": float(violation.surprise),
        "surprise_drop": float(first.surprise - late.surprise),
        "violation_margin": float(violation.surprise - late.surprise),
        "observation_count": len(episode.observations),
        "adapter": str(episode.metadata.get("adapter", "")),
    }


def _selection_metrics(episode: Episode) -> dict[str, float | int | str]:
    observation = episode.observations[0]
    source_mass = sum(float(np.sum(np.abs(features))) for features in observation.input_features.values())
    actual_mass = float(np.sum(np.abs(observation.actual)))
    active_feature_count = int(np.count_nonzero(np.abs(observation.actual) > 1e-12))
    workspace_metadata = observation.metadata.get("workspace", {})
    compression_cost = float(workspace_metadata.get("compression_cost", 0.0))
    attended_input = observation.metadata.get("attended_input")
    if attended_input is not None:
        source_mass = float(np.sum(np.abs(np.asarray(attended_input, dtype=float))))
    if "minimal_subject" in observation.metadata:
        weights = observation.metadata["minimal_subject"].get("workspace_weights", {})
        active_feature_count = len(weights)
        source_mass = float(np.sum(np.abs(next(iter(observation.input_features.values())))))
        actual_mass = float(np.sum(np.abs(observation.actual)))
        compression_cost = 0.0 if source_mass <= 1e-12 else max(0.0, source_mass - actual_mass) / source_mass
    return {
        "source_mass": source_mass,
        "actual_mass": actual_mass,
        "active_feature_count": active_feature_count,
        "dropped_mass": max(0.0, source_mass - actual_mass),
        "compression_cost": compression_cost,
        "adapter": str(episode.metadata.get("adapter", "")),
    }


def _value_metrics(episode: Episode) -> dict[str, float | int | str]:
    by_input = _observations_by_input(episode)
    positive = by_input["positive_value"][0]
    negative = by_input["negative_value"][0]
    positive_value = _value_signal(positive, positive=True)
    negative_value = _value_signal(negative, positive=False)
    positive_utility = _utility(positive)
    negative_utility = _utility(negative)
    return {
        "positive_signal": positive_value,
        "negative_signal": negative_value,
        "positive_utility": positive_utility,
        "negative_utility": negative_utility,
        "utility_contrast": positive_utility - negative_utility,
        "adapter": str(episode.metadata.get("adapter", "")),
    }


def _value_signal(observation, *, positive: bool) -> float:
    minimal = observation.metadata.get("minimal_subject")
    if isinstance(minimal, dict):
        value = float(minimal.get("preference_value", 0.0))
        return max(0.0, value if positive else -value)
    valence = observation.metadata.get("valence", {})
    key = "pleasure" if positive else "pain"
    return float(valence.get(key, 0.0))


def _utility(observation) -> float:
    minimal = observation.metadata.get("minimal_subject")
    if isinstance(minimal, dict):
        return float(minimal.get("utility", 0.0))
    objective = observation.metadata.get("objective", {})
    return float(objective.get("utility", 0.0))


def _observations_by_input(episode: Episode) -> dict[str, list]:
    by_input: dict[str, list] = {}
    for observation in episode.observations:
        for input_id in observation.active_inputs:
            by_input.setdefault(input_id, []).append(observation)
    return by_input


def _cavenet_cave_equivalence(runs: tuple[BehaviorRun, ...]) -> dict[str, float]:
    max_actual = 0.0
    max_memory = 0.0
    for probe in PROBES:
        cave = next(run.episode for run in runs if run.probe == probe and run.substrate == "cave")
        cavenet = next(run.episode for run in runs if run.probe == probe and run.substrate == "cavenet")
        for cave_obs, cavenet_obs in zip(cave.observations, cavenet.observations):
            max_actual = max(max_actual, _norm(cave_obs.actual - cavenet_obs.actual))
            max_memory = max(max_memory, _norm(cave_obs.memory_state - cavenet_obs.memory_state))
    return {
        "max_actual_distance": max_actual,
        "max_memory_distance": max_memory,
    }


def _sequence_for_probe(probe: str) -> InputSequence:
    if probe == "expectation_repetition":
        return InputSequence(
            [
                _event("repeat_0", 0.0, 0, {"expected": 1.0}),
                _event("repeat_1", 1.0, 1, {"expected": 1.0}),
                _event("repeat_2", 2.0, 2, {"expected": 1.0}),
                _event("violation", 3.0, 3, {"violation": 1.0}),
            ]
        )
    if probe == "workspace_selection":
        return InputSequence(
            [
                _event(
                    "selection_event",
                    0.0,
                    0,
                    {"cue": 1.0, "good": 0.7, "distractor": 0.4},
                )
            ]
        )
    if probe == "value_separation":
        return InputSequence(
            [
                _event(
                    "positive_value",
                    0.0,
                    0,
                    {"good": 1.0},
                    metadata={"affect": {"pleasure": 1.0}},
                ),
                _event(
                    "negative_value",
                    1.0,
                    1,
                    {"bad": 1.0},
                    metadata={"affect": {"pain": 1.0}},
                ),
            ]
        )
    raise ValueError(f"unsupported probe: {probe}")


def _params_for_probe(probe: str) -> ModelParams:
    params = ModelParams(
        memory=MemoryParams(retention=0.0, decay_tau=4.0, max_age=8.0),
        attention=AttentionProfile(mode="constant", level=1.0),
        topology=SubjectiveTopologyParams(
            feature_x="expected",
            feature_y="violation",
            prior=SubjectiveTopologyPrior(),
        ),
    )
    if probe == "workspace_selection":
        params = dataclass_replace(
            params,
            workspace_compressor=TopKWorkspaceCompressor(capacity=2),
            workspace_input_mode="workspace",
        )
    return params


def _event(
    id: str,
    start: float,
    order: int,
    features: dict[str, float],
    *,
    metadata: dict[str, object] | None = None,
) -> ExperienceObject:
    return ExperienceObject(
        id=id,
        temporal_extent=TemporalExtent(start=start, end=start + 1.0, order_index=order),
        features=FeatureVector(features),
        salience=1.0,
        metadata={} if metadata is None else dict(metadata),
    )


def _preference_vector() -> np.ndarray:
    return np.array([0.0, 1.0, 0.0, 0.0, 1.0, -1.0], dtype=float)


def _norm(value) -> float:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.linalg.norm(array.ravel()) / np.sqrt(array.size))
