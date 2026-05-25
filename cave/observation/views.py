from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from cave.commitments.attention import INTERNAL_EXPECTATION_CHANNEL, attention_effect
from cave.observation.episodes import EpisodeInput
from cave.observation.experience import FeatureVector, Presentation, visual_presentation_from_features
from cave.observation.structural import EpisodeFrame
from cave.commitments.topology import SubjectiveTopologyState


@dataclass(frozen=True)
class BaseViewState:
    name: str
    title: str
    t: float


class ExperienceView(Protocol):
    name: str
    title: str
    preferred_aspect: float

    def project(self, frame: EpisodeFrame) -> BaseViewState:
        ...


@dataclass(frozen=True)
class PresentationItem:
    source_id: str
    kind: str
    presentation: Presentation
    phase: float
    salience: float
    opacity: float


@dataclass(frozen=True)
class PresentationViewState(BaseViewState):
    items: list[PresentationItem]


@dataclass(frozen=True)
class MemoryItemView:
    source_id: str
    kind: str
    presentation: Presentation
    age: float
    strength: float
    x: float
    y: float
    scale: float
    depth: float


@dataclass(frozen=True)
class MemoryLookbackViewState(BaseViewState):
    items: list[MemoryItemView]
    max_age: float


@dataclass(frozen=True)
class TimelineInterval:
    source_id: str
    kind: str
    presentation: Presentation
    start: float
    end: float
    active: bool


@dataclass(frozen=True)
class AttentionPoint:
    t: float
    value: float


@dataclass(frozen=True)
class TimelineViewState(BaseViewState):
    duration: float
    pointer_t: float
    intervals: list[TimelineInterval]
    attention_points: list[AttentionPoint]
    pointer_attention: float
    channel_attention_points: dict[str, list[AttentionPoint]]
    pointer_channel_attention: dict[str, float]


@dataclass(frozen=True)
class ObserverTrailPoint:
    t: float
    x: float
    y: float
    attention: float


@dataclass(frozen=True)
class ObserverViewState(BaseViewState):
    openness: float
    error: float
    surprise: float
    pupil_scale: float
    gaze_x: float
    gaze_y: float
    gaze_label: str
    utility: float
    attention: float
    trail_points: list[ObserverTrailPoint]


@dataclass(frozen=True)
class SubjectSurfaceTrailPoint:
    t: float
    aperture: float
    carry: float
    utility: float


@dataclass(frozen=True)
class SubjectSurfaceViewState(BaseViewState):
    aperture: float
    carry: float
    utility: float
    valence: float
    surprise: float
    pressure: float
    input_label: str
    active_channels: tuple[str, ...]
    mode: str
    trail_points: list[SubjectSurfaceTrailPoint]


@dataclass(frozen=True)
class SubjectiveTopologyViewState(BaseViewState):
    topology: SubjectiveTopologyState
    grid_x: np.ndarray | None = None
    grid_y: np.ndarray | None = None
    density: np.ndarray | None = None
    expected_density: np.ndarray | None = None
    actual_density: np.ndarray | None = None
    expected_attention: float = 1.0
    actual_attention: float = 1.0


@dataclass(frozen=True)
class ExpectationActualViewState(BaseViewState):
    vocabulary: list[str]
    expected_before: np.ndarray
    actual: np.ndarray
    error: np.ndarray
    expected_after: np.ndarray
    surprise: float
    learning_rate: float
    expected_attention: float = 1.0
    actual_attention: float = 1.0


@dataclass(frozen=True)
class CorrectionViewState(BaseViewState):
    feature_x: str
    feature_y: str
    bounds: tuple[float, float]
    experience_times: list[float]
    experience_labels: list[str]
    expected_point: np.ndarray | None
    actual_point: np.ndarray | None
    after_point: np.ndarray | None
    surprise: float
    learning_rate: float
    expected_attention: float = 1.0
    actual_attention: float = 1.0
    normalized: bool = False
    series_times: np.ndarray | None = None
    expected_series: np.ndarray | None = None
    actual_series: np.ndarray | None = None
    after_series: np.ndarray | None = None
    expected_attention_series: np.ndarray | None = None
    actual_attention_series: np.ndarray | None = None


@dataclass(frozen=True)
class AffectPoint:
    t: float
    pain: float
    pleasure: float
    net: float
    surprise: float
    attention: float
    utility: float


@dataclass(frozen=True)
class AffectViewState(BaseViewState):
    duration: float
    points: list[AffectPoint]
    current: AffectPoint


@dataclass(frozen=True)
class ActionPoint:
    t: float
    kind: str
    target_id: str | None
    target_channel: str | None
    strength: float
    expected_utility_delta: float
    exposure: float


@dataclass(frozen=True)
class ActionViewState(BaseViewState):
    duration: float
    points: list[ActionPoint]
    current: ActionPoint


@dataclass(frozen=True)
class PresentationView:
    name: str = "presentation"
    title: str = "Presentation / Wall POV"
    preferred_aspect: float = 4.0

    def project(self, frame: EpisodeFrame) -> PresentationViewState:
        inputs_by_id = frame.episode.input_by_id()
        items = []
        presentation_mode = frame.episode.metadata.get("presentation_mode")
        if presentation_mode == "current_conversation_segment":
            segment_id = frame.observation.metadata.get("segment_id")
            presentation_ids = [segment_id] if isinstance(segment_id, str) else []
        elif presentation_mode == "current_text":
            presentation_ids = [
                input_id
                for input_id in frame.observation.active_inputs
                if abs(inputs_by_id[input_id].start - frame.observation.t) < 1e-9
            ]
        else:
            presentation_ids = list(frame.observation.active_inputs)
        for input_id in presentation_ids:
            item = inputs_by_id.get(input_id)
            if item is None:
                continue
            phase = (
                0.5
                if presentation_mode in {
                    "current_text",
                    "current_conversation_segment",
                }
                else (frame.observation.t - item.start) / item.duration
            )
            items.append(
                PresentationItem(
                    source_id=item.id,
                    kind=item.kind,
                    presentation=presentation_for_episode_input(item, frame.episode.vocabulary),
                    phase=min(1.0, max(0.0, phase)),
                    salience=item.salience,
                    opacity=_channel_attention_for_observation(
                        frame.observation,
                        item.modality,
                    ),
                )
            )
        return PresentationViewState(
            name=self.name,
            title=self.title,
            t=frame.observation.t,
            items=items,
        )


@dataclass(frozen=True)
class MemoryLookbackView:
    max_age: float | None = None
    min_strength: float = 0.05
    name: str = "memory"
    title: str = "Memory / Lookback"
    preferred_aspect: float = 4.0

    def project(self, frame: EpisodeFrame) -> MemoryLookbackViewState:
        lookback_mode = frame.episode.metadata.get("lookback_mode")
        if lookback_mode == "attention_context":
            return self._project_attention_context(
                frame,
                title="Context / Temporal Lookback",
            )
        if lookback_mode == "conversation_mock_memory":
            return self._project_attention_context(
                frame,
                title="Conversation / Mock Memory",
            )
        max_age = self.max_age if self.max_age is not None else float(
            frame.episode.metadata.get("memory_max_age", 6.0)
        )
        items = []
        for item in frame.topology_frame.memory_items:
            age = item.age(frame.observation.t)
            strength = float(item.strength)
            if age <= max_age and strength >= self.min_strength:
                depth = min(1.0, max(0.0, age / max_age)) if max_age > 0.0 else 0.0
                x = 0.5 + 0.18 * (item.source.order_index % 2 * 2 - 1) * depth
                y = 0.36 + 0.42 * depth
                scale = 1.0 - 0.72 * depth
                items.append(
                    MemoryItemView(
                        source_id=item.source.id,
                        kind=item.source.kind,
                        presentation=presentation_for_episode_input(
                            item.source,
                            frame.episode.vocabulary,
                        ),
                        age=age,
                        strength=strength,
                        x=x,
                        y=y,
                        scale=scale,
                        depth=depth,
                    )
                )
        return MemoryLookbackViewState(
            name=self.name,
            title=self.title,
            t=frame.observation.t,
            items=items,
            max_age=max_age,
        )

    def _project_attention_context(
        self,
        frame: EpisodeFrame,
        *,
        title: str,
    ) -> MemoryLookbackViewState:
        inputs_by_id = frame.episode.input_by_id()
        weighted_ids = [
            (input_id, float(weight))
            for input_id, weight in frame.observation.attention_weights.items()
            if weight >= self.min_strength
        ]
        weighted_ids.sort(key=lambda item: inputs_by_id[item[0]].order_index)
        items = []
        current_t = frame.observation.t
        for input_id, strength in weighted_ids:
            item = inputs_by_id.get(input_id)
            if item is None:
                continue
            distance = max(0.0, current_t - item.start)
            depth = min(1.0, distance / max(1.0, current_t))
            x = 0.18 + 0.64 * (item.start / max(1.0, current_t))
            y = 0.5
            scale = 1.0
            items.append(
                MemoryItemView(
                    source_id=item.id,
                    kind=item.kind,
                    presentation=presentation_for_episode_input(
                        item,
                        frame.episode.vocabulary,
                    ),
                    age=distance,
                    strength=strength,
                    x=x,
                    y=y,
                    scale=scale,
                    depth=depth,
                )
            )
        return MemoryLookbackViewState(
            name=self.name,
            title=title,
            t=frame.observation.t,
            items=items,
            max_age=max(1.0, current_t),
        )


@dataclass(frozen=True)
class TimelineView:
    name: str = "timeline"
    title: str = "Timeline / Tape"
    preferred_aspect: float = 4.0

    def project(self, frame: EpisodeFrame) -> TimelineViewState:
        intervals = [
            TimelineInterval(
                source_id=item.id,
                kind=item.kind,
                presentation=presentation_for_episode_input(item, frame.episode.vocabulary),
                start=item.start,
                end=item.end,
                active=item.id in frame.observation.active_inputs,
            )
            for item in frame.episode.inputs
        ]
        attention_curve = frame.episode.metadata.get("attention_curve")
        if attention_curve is None:
            attention_points = [
                AttentionPoint(t=obs.t, value=obs.attention)
                for obs in frame.episode.observations
            ]
        else:
            attention_points = [
                AttentionPoint(t=float(point["t"]), value=float(point["value"]))
                for point in attention_curve
            ]
        channel_attention_points = _timeline_channel_attention_points(
            frame.episode.observations
        )
        return TimelineViewState(
            name=self.name,
            title=self.title,
            t=frame.observation.t,
            duration=frame.episode.duration,
            pointer_t=frame.observation.t,
            intervals=intervals,
            attention_points=attention_points,
            pointer_attention=frame.observation.attention,
            channel_attention_points=channel_attention_points,
            pointer_channel_attention=_attention_channel_impacts(frame.observation),
        )


@dataclass(frozen=True)
class ObserverView:
    name: str = "observer"
    title: str = "Observer"
    preferred_aspect: float = 1.0

    def project(self, frame: EpisodeFrame) -> ObserverViewState:
        observation = frame.observation
        evolved = observation.metadata.get("evolved_subject", {})
        objective = observation.metadata.get("objective", {})
        if not isinstance(evolved, dict):
            evolved = {}
        if not isinstance(objective, dict):
            objective = {}
        openness = float(evolved.get("exposure", observation.attention))
        utility = float(objective.get("utility", 0.0))
        if evolved:
            error = abs(float(evolved.get("outcome_value", 0.0)))
            surprise = abs(float(evolved.get("utility", utility)))
        else:
            error = float(np.linalg.norm(observation.error))
            surprise = float(observation.surprise)
        gaze_x, gaze_y = _observer_gaze(observation.actual)
        gaze_label = _observer_gaze_label(frame)
        trail_points = _observer_trail_points(frame)
        return ObserverViewState(
            name=self.name,
            title=self.title,
            t=observation.t,
            openness=float(np.clip(openness, 0.0, 1.0)),
            error=error,
            surprise=surprise,
            pupil_scale=float(np.clip(max(error, surprise), 0.0, 1.0)),
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            gaze_label=gaze_label,
            utility=utility,
            attention=float(observation.attention),
            trail_points=trail_points,
        )


@dataclass(frozen=True)
class SubjectSurfaceView:
    name: str = "subject_surface"
    title: str = "Subject Surface"
    preferred_aspect: float = 1.0

    def project(self, frame: EpisodeFrame) -> SubjectSurfaceViewState:
        observation = frame.observation
        evolved = observation.metadata.get("evolved_subject", {})
        objective = observation.metadata.get("objective", {})
        if not isinstance(evolved, dict):
            evolved = {}
        if not isinstance(objective, dict):
            objective = {}

        carry_series = _subject_surface_carry_series(frame)
        utility = float(evolved.get("utility", objective.get("utility", 0.0)))
        valence = _subject_surface_valence(evolved, utility)
        pressure = max(abs(valence), min(1.0, abs(utility)))
        return SubjectSurfaceViewState(
            name=self.name,
            title=self.title,
            t=observation.t,
            aperture=float(np.clip(evolved.get("exposure", observation.attention), 0.0, 1.0)),
            carry=float(carry_series[frame.index]) if len(carry_series) > frame.index else 0.0,
            utility=utility,
            valence=valence,
            surprise=float(np.clip(max(observation.surprise, np.linalg.norm(observation.error)), 0.0, 1.0)),
            pressure=float(np.clip(pressure, 0.0, 1.0)),
            input_label=_observer_gaze_label(frame),
            active_channels=_subject_surface_channels(frame),
            mode="evolved" if evolved else "cave",
            trail_points=[
                SubjectSurfaceTrailPoint(
                    t=obs.t,
                    aperture=_subject_surface_observation_aperture(obs),
                    carry=float(carry_series[index]) if len(carry_series) > index else 0.0,
                    utility=_subject_surface_observation_utility(obs),
                )
                for index, obs in enumerate(frame.episode.observations[: frame.index + 1])
            ],
        )


@dataclass(frozen=True)
class SubjectiveTopologyView:
    grid_resolution: int = 60
    name: str = "subjective_topology"
    title: str = "Subjective State Topology"
    preferred_aspect: float = 1.0

    def project(self, frame: EpisodeFrame) -> SubjectiveTopologyViewState:
        topology = frame.topology_frame.topology
        grid_x, grid_y, density = topology.grid(self.grid_resolution)
        _, _, expected_density = topology.expected_grid(self.grid_resolution)
        _, _, actual_density = topology.actual_grid(self.grid_resolution)
        return SubjectiveTopologyViewState(
            name=self.name,
            title=self.title,
            t=frame.observation.t,
            topology=topology,
            grid_x=grid_x,
            grid_y=grid_y,
            density=density,
            expected_density=expected_density,
            actual_density=actual_density,
            expected_attention=_internal_expectation_attention(frame.observation),
            actual_attention=_external_input_attention(frame.observation),
        )


@dataclass(frozen=True)
class ExpectationActualView:
    name: str = "expectation_actual"
    title: str = "Expectation / Actual"
    preferred_aspect: float = 2.2

    def project(self, frame: EpisodeFrame) -> ExpectationActualViewState:
        return ExpectationActualViewState(
            name=self.name,
            title=self.title,
            t=frame.observation.t,
            vocabulary=list(frame.episode.vocabulary),
            expected_before=frame.observation.expected.copy(),
            actual=frame.observation.actual.copy(),
            error=frame.observation.error.copy(),
            expected_after=frame.observation.memory_state.copy(),
            surprise=frame.observation.surprise,
            learning_rate=frame.observation.learning_rate,
            expected_attention=_internal_expectation_attention(frame.observation),
            actual_attention=_external_input_attention(frame.observation),
        )


@dataclass(frozen=True)
class CorrectionView:
    name: str = "correction"
    title: str = "Correction / Feature Plane"
    preferred_aspect: float = 1.0

    def project(self, frame: EpisodeFrame) -> CorrectionViewState:
        topology = frame.topology_frame.topology
        correction = frame.topology_frame.correction
        if correction is None:
            expected_point = actual_point = after_point = None
            surprise = frame.observation.surprise
            learning_rate = frame.observation.learning_rate
        else:
            expected_point = correction.expected_point.copy()
            actual_point = correction.actual_point.copy()
            after_point = correction.after_point.copy()
            surprise = correction.surprise
            learning_rate = correction.learning_rate
        return CorrectionViewState(
            name=self.name,
            title=self.title,
            t=frame.observation.t,
            feature_x=topology.feature_x,
            feature_y=topology.feature_y,
            bounds=topology.bounds,
            experience_times=[item.start for item in frame.episode.inputs],
            experience_labels=[item.kind for item in frame.episode.inputs],
            expected_point=expected_point,
            actual_point=actual_point,
            after_point=after_point,
            surprise=surprise,
            learning_rate=learning_rate,
            expected_attention=_internal_expectation_attention(frame.observation),
            actual_attention=_external_input_attention(frame.observation),
        )


@dataclass(frozen=True)
class AffectView:
    name: str = "affect"
    title: str = "Affect / Objective"
    preferred_aspect: float = 2.2

    def project(self, frame: EpisodeFrame) -> AffectViewState:
        points = [
            _affect_point_for_observation(observation)
            for observation in frame.episode.observations
        ]
        current = _affect_point_for_observation(frame.observation)
        return AffectViewState(
            name=self.name,
            title=self.title,
            t=frame.observation.t,
            duration=frame.episode.duration,
            points=points,
            current=current,
        )


@dataclass(frozen=True)
class ActionView:
    name: str = "action"
    title: str = "Action / Exposure"
    preferred_aspect: float = 2.2

    def project(self, frame: EpisodeFrame) -> ActionViewState:
        points = [
            _action_point_for_observation(observation)
            for observation in frame.episode.observations
        ]
        current = _action_point_for_observation(frame.observation)
        return ActionViewState(
            name=self.name,
            title=self.title,
            t=frame.observation.t,
            duration=frame.episode.duration,
            points=points,
            current=current,
        )


def default_views() -> list[ExperienceView]:
    return [
        PresentationView(),
        MemoryLookbackView(),
        TimelineView(),
        ExpectationActualView(),
        CorrectionView(),
        SubjectiveTopologyView(),
    ]


def observer_views() -> list[ExperienceView]:
    return [
        SubjectSurfaceView(),
        TimelineView(),
        SubjectiveTopologyView(),
    ]


def _observer_gaze(actual: np.ndarray) -> tuple[float, float]:
    if actual.size == 0 or float(np.linalg.norm(actual)) <= 1e-12:
        return 0.0, 0.0
    x = float(actual[0])
    y = float(actual[1]) if actual.size > 1 else 0.0
    norm = max(1.0, float(np.linalg.norm([x, y])))
    return float(np.clip(x / norm, -1.0, 1.0)), float(np.clip(y / norm, -1.0, 1.0))


def _internal_expectation_attention(observation) -> float:
    effective = observation.metadata.get("effective_attention", {})
    if isinstance(effective, dict) and "internal_expectation" in effective:
        return float(np.clip(effective["internal_expectation"], 0.0, 1.0))
    channels = observation.metadata.get("attention_channels")
    if isinstance(channels, dict):
        return float(
            np.clip(
                _attention_effect_for_observation(observation)
                * float(channels.get(INTERNAL_EXPECTATION_CHANNEL, 0.0)),
                0.0,
                1.0,
            )
        )
    return 1.0


def _external_input_attention(observation) -> float:
    effective = observation.metadata.get("effective_attention", {})
    if isinstance(effective, dict) and "external_input" in effective:
        return float(np.clip(effective["external_input"], 0.0, 1.0))
    channels = observation.metadata.get("attention_channels")
    if isinstance(channels, dict):
        external = sum(
            max(0.0, float(weight))
            for channel, weight in channels.items()
            if channel != INTERNAL_EXPECTATION_CHANNEL
        )
        return float(
            np.clip(
                _attention_effect_for_observation(observation) * external,
                0.0,
                1.0,
            )
        )
    return 1.0


def _channel_attention_for_observation(observation, channel: str) -> float:
    impacts = observation.metadata.get("attention_channel_impacts")
    if isinstance(impacts, dict) and channel in impacts:
        return float(np.clip(impacts[channel], 0.0, 1.0))
    channels = observation.metadata.get("attention_channels")
    if isinstance(channels, dict):
        return float(
            np.clip(
                _attention_effect_for_observation(observation)
                * float(channels.get(channel, 0.0)),
                0.0,
                1.0,
            )
        )
    return _attention_effect_for_observation(observation)


def _attention_effect_for_observation(observation) -> float:
    return float(
        np.clip(
            observation.metadata.get(
                "attention_effect",
                attention_effect(observation.attention),
            ),
            0.0,
            1.0,
        )
    )


def _attention_channel_impacts(observation) -> dict[str, float]:
    impacts = observation.metadata.get("attention_channel_impacts")
    if isinstance(impacts, dict):
        return {
            str(channel): float(np.clip(value, 0.0, 1.0))
            for channel, value in impacts.items()
        }
    channels = observation.metadata.get("attention_channels")
    if isinstance(channels, dict):
        effect = _attention_effect_for_observation(observation)
        return {
            str(channel): float(np.clip(effect * float(weight), 0.0, 1.0))
            for channel, weight in channels.items()
        }
    return {}


def _timeline_channel_attention_points(observations) -> dict[str, list[AttentionPoint]]:
    channel_names: set[str] = set()
    impacts_by_observation = []
    for observation in observations:
        impacts = _attention_channel_impacts(observation)
        impacts_by_observation.append((observation, impacts))
        channel_names.update(impacts)
    return {
        channel: [
            AttentionPoint(
                t=observation.t,
                value=float(impacts.get(channel, 0.0)),
            )
            for observation, impacts in impacts_by_observation
        ]
        for channel in sorted(channel_names)
    }


def _observer_gaze_label(frame: EpisodeFrame) -> str:
    action = frame.observation.metadata.get("action", {})
    if isinstance(action, dict) and isinstance(action.get("target_id"), str):
        return str(action["target_id"])
    if frame.observation.active_inputs:
        return frame.observation.active_inputs[0]
    return "none"


def _observer_trail_points(frame: EpisodeFrame) -> list[ObserverTrailPoint]:
    prefix = frame.episode.observations[: frame.index + 1]
    if not prefix:
        return []
    memory = [np.asarray(obs.memory_state, dtype=float) for obs in prefix]
    if not memory or memory[0].size == 0:
        return []
    xs = np.array([state[0] if state.size > 0 else 0.0 for state in memory], dtype=float)
    ys = np.array([state[1] if state.size > 1 else 0.0 for state in memory], dtype=float)
    scale = max(1.0, float(np.max(np.abs(np.concatenate([xs, ys])))))
    return [
        ObserverTrailPoint(
            t=float(obs.t),
            x=float(np.clip(x / scale, -1.0, 1.0)),
            y=float(np.clip(y / scale, -1.0, 1.0)),
            attention=float(obs.attention),
        )
        for obs, x, y in zip(prefix, xs, ys)
    ]


def _subject_surface_observation_aperture(observation) -> float:
    evolved = observation.metadata.get("evolved_subject", {})
    if isinstance(evolved, dict) and "exposure" in evolved:
        return float(np.clip(evolved.get("exposure", 0.0), 0.0, 1.0))
    return float(np.clip(observation.attention, 0.0, 1.0))


def _subject_surface_observation_utility(observation) -> float:
    evolved = observation.metadata.get("evolved_subject", {})
    if isinstance(evolved, dict) and "utility" in evolved:
        return float(evolved.get("utility", 0.0))
    objective = observation.metadata.get("objective", {})
    if isinstance(objective, dict):
        return float(objective.get("utility", 0.0))
    return 0.0


def _subject_surface_valence(evolved: dict, utility: float) -> float:
    outcome_value = float(evolved.get("outcome_value", 0.0))
    if abs(outcome_value) > 1e-9:
        return 1.0 if outcome_value > 0.0 else -1.0
    future = str(evolved.get("future_outcome", "neutral"))
    if future == "good":
        return 1.0
    if future == "bad":
        return -1.0
    if abs(utility) > 1e-9:
        return 1.0 if utility > 0.0 else -1.0
    return 0.0


def _subject_surface_channels(frame: EpisodeFrame) -> tuple[str, ...]:
    weights = frame.observation.attention_weights
    if weights:
        channels = sorted(weights.items(), key=lambda item: abs(float(item[1])), reverse=True)
        return tuple(str(name) for name, _ in channels[:3])
    return tuple(frame.observation.active_inputs[:3])


def _subject_surface_carry_series(frame: EpisodeFrame) -> np.ndarray:
    observations = frame.episode.observations
    if not observations:
        return np.zeros(0, dtype=float)
    if any(_subject_surface_has_evolved_metadata(obs) for obs in observations):
        charged = _subject_surface_evolved_charge_series(observations)
        if charged is not None:
            return charged

    norms = np.array([float(np.linalg.norm(obs.memory_state)) for obs in observations], dtype=float)
    scale = float(np.max(np.abs(norms))) if norms.size else 0.0
    if scale <= 1e-12:
        return np.zeros(len(observations), dtype=float)
    return norms / scale


def _subject_surface_has_evolved_metadata(observation) -> bool:
    evolved = observation.metadata.get("evolved_subject", {})
    return isinstance(evolved, dict) and bool(evolved)


def _subject_surface_evolved_charge_series(observations) -> np.ndarray | None:
    hidden = [np.asarray(obs.memory_state, dtype=float) for obs in observations]
    if not hidden or hidden[0].size == 0:
        return np.zeros(len(observations), dtype=float)
    try:
        matrix = np.stack(hidden, axis=0)
    except ValueError:
        return None

    labels = []
    for obs in observations:
        evolved = obs.metadata.get("evolved_subject", {})
        if not isinstance(evolved, dict):
            labels.append("neutral")
            continue
        outcome_value = float(evolved.get("outcome_value", 0.0))
        if abs(outcome_value) > 1e-9:
            labels.append("good" if outcome_value > 0.0 else "bad")
        else:
            labels.append(str(evolved.get("future_outcome", "neutral")))

    labels_array = np.asarray(labels, dtype=object)
    good = matrix[labels_array == "good"]
    bad = matrix[labels_array == "bad"]
    if good.size == 0 or bad.size == 0:
        return np.zeros(len(observations), dtype=float)
    axis_vector = np.mean(good, axis=0) - np.mean(bad, axis=0)
    norm = float(np.linalg.norm(axis_vector))
    if norm <= 1e-12:
        return np.zeros(len(observations), dtype=float)
    raw = matrix @ (axis_vector / norm)
    scale = float(np.max(np.abs(raw)))
    if scale <= 1e-12:
        return np.zeros(len(observations), dtype=float)
    return raw / scale


def _affect_point_for_observation(observation) -> AffectPoint:
    valence = observation.metadata.get("valence", {})
    objective = observation.metadata.get("objective", {})
    if not isinstance(valence, dict):
        valence = {}
    if not isinstance(objective, dict):
        objective = {}
    return AffectPoint(
        t=float(observation.t),
        pain=float(valence.get("pain", 0.0)),
        pleasure=float(valence.get("pleasure", 0.0)),
        net=float(valence.get("net", 0.0)),
        surprise=float(observation.surprise),
        attention=float(observation.attention),
        utility=float(objective.get("utility", 0.0)),
    )


def _action_point_for_observation(observation) -> ActionPoint:
    action = observation.metadata.get("action", {})
    if not isinstance(action, dict):
        action = {}
    exposure_by_id = action.get("object_exposure", {})
    if not isinstance(exposure_by_id, dict):
        exposure_by_id = {}
    target_id = action.get("target_id")
    exposure = (
        float(exposure_by_id.get(target_id, 1.0))
        if isinstance(target_id, str)
        else 1.0
    )
    return ActionPoint(
        t=float(observation.t),
        kind=str(action.get("kind", "maintain")),
        target_id=None if target_id is None else str(target_id),
        target_channel=(
            None
            if action.get("target_channel") is None
            else str(action.get("target_channel"))
        ),
        strength=float(action.get("strength", 0.0)),
        expected_utility_delta=float(action.get("expected_utility_delta", 0.0)),
        exposure=exposure,
    )


def presentation_for_episode_input(
    item: EpisodeInput,
    vocabulary: list[str],
) -> Presentation:
    if item.presentation is not None:
        return item.presentation
    features = FeatureVector(
        {
            key: float(item.features[index])
            for index, key in enumerate(vocabulary)
            if index < item.features.size
        }
    )
    return visual_presentation_from_features(features, item.metadata)
