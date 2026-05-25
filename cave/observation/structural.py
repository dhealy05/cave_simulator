from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cave.observation.episodes import Episode, EpisodeInput, EpisodeObservation
from cave.observation.experience import ExperienceObject, FeatureVector, TemporalExtent
from cave.commitments.memory import memory_strength
from cave.commitments.topology import (
    SubjectiveTopologyCorrection,
    SubjectiveTopologyParams,
    SubjectiveTopologyPrior,
    SubjectiveTopologyState,
    SubjectiveTopologyWell,
)


Array = np.ndarray


@dataclass(frozen=True)
class EpisodeMemoryItem:
    source: EpisodeInput
    ended_t: float
    base_strength: float
    strength: float

    def age(self, t: float) -> float:
        return max(0.0, t - self.ended_t)


@dataclass(frozen=True)
class EpisodeCorrection:
    expected_point: Array
    actual_point: Array
    after_point: Array
    surprise: float
    learning_rate: float


@dataclass(frozen=True)
class EpisodeTopologyFrame:
    t: float
    topology: SubjectiveTopologyState
    correction: EpisodeCorrection | None
    memory_items: list[EpisodeMemoryItem]


@dataclass(frozen=True)
class EpisodeStructuralState:
    episode: Episode
    topology_frames: list[EpisodeTopologyFrame]

    def frame_at_index(self, index: int) -> EpisodeTopologyFrame:
        return self.topology_frames[index]


@dataclass(frozen=True)
class EpisodeFrame:
    episode: Episode
    observation: EpisodeObservation
    structural: EpisodeStructuralState
    index: int

    @property
    def topology_frame(self) -> EpisodeTopologyFrame:
        return self.structural.frame_at_index(self.index)


@dataclass(frozen=True)
class _MemoryAccumulator:
    source: EpisodeInput
    attention_total: float = 0.0
    samples: int = 0

    def add(self, value: float) -> "_MemoryAccumulator":
        return _MemoryAccumulator(
            source=self.source,
            attention_total=self.attention_total + value,
            samples=self.samples + 1,
        )

    @property
    def average_attention(self) -> float:
        if self.samples <= 0:
            return 0.0
        return self.attention_total / self.samples


def structural_state_for_episode(
    episode: Episode,
    *,
    topology_params: SubjectiveTopologyParams | None = None,
    decay_tau: float = 2.0,
    max_age: float = 6.0,
) -> EpisodeStructuralState:
    params = (
        topology_params
        or episode.metadata.get("topology_params")
        or SubjectiveTopologyParams(prior=SubjectiveTopologyPrior())
    )
    decay_tau = float(episode.metadata.get("memory_decay_tau", decay_tau))
    max_age = float(episode.metadata.get("memory_max_age", max_age))
    topology = SubjectiveTopologyState.initial(
        feature_x=params.feature_x,
        feature_y=params.feature_y,
        bounds=params.bounds,
        resolution=params.resolution,
        prior=params.prior,
    )
    inputs_by_id = episode.input_by_id()
    active: dict[str, _MemoryAccumulator] = {}
    finalized: dict[str, tuple[EpisodeInput, float, float]] = {}
    frames: list[EpisodeTopologyFrame] = []

    for observation in episode.observations:
        current_ids = set(observation.active_inputs)
        next_active: dict[str, _MemoryAccumulator] = {}
        for input_id in observation.active_inputs:
            source = inputs_by_id.get(input_id)
            if source is None:
                continue
            accumulator = active.get(input_id, _MemoryAccumulator(source=source))
            next_active[input_id] = accumulator.add(
                float(observation.attention_weights.get(input_id, observation.attention))
            )

        for input_id, accumulator in active.items():
            if input_id in current_ids:
                continue
            if accumulator.average_attention > 0.0:
                finalized[input_id] = (
                    accumulator.source,
                    accumulator.source.end,
                    accumulator.average_attention,
                )

        memory_items, finalized = _visible_memory_items(
            finalized,
            observation.t,
            decay_tau=decay_tau,
            max_age=max_age,
        )
        topology = _update_topology(
            topology,
            episode,
            observation,
            memory_items,
            params,
        )
        correction = _episode_correction(topology)
        frames.append(
            EpisodeTopologyFrame(
                t=observation.t,
                topology=topology,
                correction=correction,
                memory_items=memory_items,
            )
        )
        active = next_active

    return EpisodeStructuralState(episode=episode, topology_frames=frames)


def episode_frames(
    episode: Episode,
    structural: EpisodeStructuralState | None = None,
) -> list[EpisodeFrame]:
    structural = structural or structural_state_for_episode(episode)
    return [
        EpisodeFrame(
            episode=episode,
            observation=observation,
            structural=structural,
            index=index,
        )
        for index, observation in enumerate(episode.observations)
    ]


def frame_for_time(
    episode: Episode,
    t: float,
    structural: EpisodeStructuralState | None = None,
) -> EpisodeFrame:
    frames = episode_frames(episode, structural)
    if not frames:
        raise ValueError("episode has no observations")
    index = min(range(len(frames)), key=lambda i: abs(frames[i].observation.t - t))
    return frames[index]


def memory_trajectory(episode: Episode) -> Array:
    _require_observations(episode)
    return np.stack([obs.memory_state for obs in episode.observations], axis=0)


def actual_trajectory(episode: Episode) -> Array:
    _require_observations(episode)
    return np.stack([obs.actual for obs in episode.observations], axis=0)


def expected_trajectory(episode: Episode) -> Array:
    _require_observations(episode)
    return np.stack([obs.expected for obs in episode.observations], axis=0)


def prediction_error_trajectory(episode: Episode) -> Array:
    _require_observations(episode)
    return np.stack([obs.error for obs in episode.observations], axis=0)


def surprise_learning_trajectory(episode: Episode) -> Array:
    _require_observations(episode)
    return np.array(
        [
            [obs.surprise, obs.learning_rate]
            for obs in episode.observations
        ],
        dtype=float,
    )


def active_context_trajectory(episode: Episode) -> Array:
    _require_observations(episode)
    inputs_by_id = episode.input_by_id()
    duration = max(float(episode.duration), 1e-12)
    total_inputs = max(len(episode.inputs), 1)
    rows = []
    for observation in episode.observations:
        active_inputs = [
            inputs_by_id[input_id]
            for input_id in observation.active_inputs
            if input_id in inputs_by_id
        ]
        active_count = len(active_inputs)
        available_count = max(
            sum(1 for item in episode.inputs if item.start <= observation.t),
            1,
        )
        retained_mass = float(observation.metadata.get("retained_attention_mass", 1.0))
        if active_count == 0:
            rows.append(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    float(observation.attention),
                    retained_mass,
                ]
            )
            continue

        weights = np.array(
            [
                float(observation.attention_weights.get(item.id, 0.0))
                for item in active_inputs
            ],
            dtype=float,
        )
        weight_total = float(np.sum(weights))
        if weight_total <= 1e-12:
            weights = np.full(active_count, 1.0 / active_count, dtype=float)
        else:
            weights = weights / weight_total

        starts = np.array([float(item.start) for item in active_inputs], dtype=float)
        lookbacks = np.maximum(0.0, float(observation.t) - starts) / duration
        mean_lookback = float(np.sum(weights * lookbacks))
        spread = float(np.sqrt(np.sum(weights * (lookbacks - mean_lookback) ** 2)))
        span = float((np.max(starts) - np.min(starts)) / duration)
        current_included = float(
            any(item.contains(observation.t) for item in active_inputs)
        )
        rows.append(
            [
                active_count / total_inputs,
                active_count / available_count,
                mean_lookback,
                spread,
                span,
                current_included,
                float(observation.attention),
                retained_mass,
            ]
        )
    return np.array(rows, dtype=float)


def final_memory_embedding(episode: Episode) -> Array:
    _require_observations(episode)
    return episode.observations[-1].memory_state.copy()


def memory_trajectory_embedding(
    episode: Episode,
    *,
    samples: int = 128,
) -> Array:
    return resample_trajectory(memory_trajectory(episode), samples=samples)


def actual_trajectory_embedding(
    episode: Episode,
    *,
    samples: int = 128,
) -> Array:
    return resample_trajectory(actual_trajectory(episode), samples=samples)


def expected_trajectory_embedding(
    episode: Episode,
    *,
    samples: int = 128,
) -> Array:
    return resample_trajectory(expected_trajectory(episode), samples=samples)


def prediction_error_trajectory_embedding(
    episode: Episode,
    *,
    samples: int = 128,
) -> Array:
    return resample_trajectory(prediction_error_trajectory(episode), samples=samples)


def surprise_learning_trajectory_embedding(
    episode: Episode,
    *,
    samples: int = 128,
) -> Array:
    return resample_trajectory(surprise_learning_trajectory(episode), samples=samples)


def active_context_embedding(
    episode: Episode,
    *,
    samples: int = 128,
) -> Array:
    return resample_trajectory(active_context_trajectory(episode), samples=samples)


def subjective_trajectory_embedding(
    episode: Episode,
    *,
    samples: int = 128,
) -> Array:
    expected = expected_trajectory_embedding(episode, samples=samples)
    error = prediction_error_trajectory_embedding(episode, samples=samples)
    surprise_learning = surprise_learning_trajectory_embedding(episode, samples=samples)
    return np.concatenate([expected, error, surprise_learning], axis=1)


def state_effect_embedding(
    episode: Episode,
    *,
    samples: int = 128,
    baseline: Array | None = None,
) -> Array:
    observed = memory_trajectory(episode)
    baseline = no_input_memory_baseline(episode) if baseline is None else baseline
    return resample_trajectory(observed - baseline, samples=samples)


def internal_experience_embedding(
    episode: Episode,
    *,
    samples: int = 128,
) -> Array:
    return subjective_trajectory_embedding(episode, samples=samples)


def experience_effect_embedding(
    episode: Episode,
    *,
    samples: int = 128,
    baseline: Array | None = None,
) -> Array:
    return state_effect_embedding(episode, samples=samples, baseline=baseline)


def no_input_memory_baseline(episode: Episode) -> Array:
    _require_observations(episode)
    first = episode.observations[0].memory_state
    return np.zeros((len(episode.observations), first.size), dtype=float)


def resample_trajectory(trajectory: Array, *, samples: int = 128) -> Array:
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 2:
        raise ValueError("trajectory must have shape (time, dimensions)")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if trajectory.shape[0] == 0:
        raise ValueError("trajectory must contain at least one timestep")
    if trajectory.shape[0] == samples:
        return trajectory.copy()
    if trajectory.shape[0] == 1:
        return np.repeat(trajectory, samples, axis=0)

    source_x = np.linspace(0.0, 1.0, trajectory.shape[0])
    target_x = np.linspace(0.0, 1.0, samples)
    return np.stack(
        [
            np.interp(target_x, source_x, trajectory[:, dim])
            for dim in range(trajectory.shape[1])
        ],
        axis=1,
    )


def _visible_memory_items(
    finalized: dict[str, tuple[EpisodeInput, float, float]],
    t: float,
    *,
    decay_tau: float,
    max_age: float,
) -> tuple[list[EpisodeMemoryItem], dict[str, tuple[EpisodeInput, float, float]]]:
    items = []
    updated: dict[str, tuple[EpisodeInput, float, float]] = {}
    for source, ended_t, base_strength in finalized.values():
        age = max(0.0, t - ended_t)
        strength = base_strength * memory_strength(age, decay_tau)
        if age <= max_age and strength > 0.0:
            updated[source.id] = (source, ended_t, strength)
            items.append(
                EpisodeMemoryItem(
                    source=source,
                    ended_t=ended_t,
                    base_strength=base_strength,
                    strength=strength,
                )
            )
    return sorted(items, key=lambda item: item.source.order_index), updated


def _update_topology(
    previous: SubjectiveTopologyState,
    episode: Episode,
    observation: EpisodeObservation,
    memory_items: list[EpisodeMemoryItem],
    params: SubjectiveTopologyParams,
) -> SubjectiveTopologyState:
    expected_density = previous._diffuse(
        np.array(previous.expected_density, dtype=float) * params.decay,
        params.diffusion,
    )
    actual_density = previous._diffuse(
        np.array(previous.actual_density, dtype=float) * params.decay,
        params.diffusion,
    )
    untagged_density = np.clip(
        np.array(previous.density, dtype=float)
        - np.array(previous.expected_density, dtype=float)
        - np.array(previous.actual_density, dtype=float),
        0.0,
        None,
    )
    untagged_density = previous._diffuse(
        untagged_density * params.decay,
        params.diffusion,
    )
    inputs_by_id = episode.input_by_id()
    current_centers = []

    for input_id in observation.active_inputs:
        source = inputs_by_id.get(input_id)
        if source is None:
            continue
        features = observation.input_features.get(input_id, source.features)
        center = previous.center_for_vector(
            features,
            episode.vocabulary,
            params.feature_x,
            params.feature_y,
        )
        current_centers.append(center)
        weight = (
            float(observation.attention_weights.get(input_id, observation.attention))
            * source.salience
            * source.learning_weight
        )
        actual_density += (
            weight
            * params.deposit_strength
            * previous._gaussian_grid(center, params.deposit_width)
        )

    if previous.last_focus is not None and current_centers:
        transition_impact = float(
            observation.metadata.get("attention_effect", observation.attention)
        )
        for center in current_centers:
            actual_density += (
                transition_impact
                * params.transition_strength
                * previous._transition_grid(
                    previous.last_focus,
                    center,
                    params.deposit_width,
                )
            )

    expected_attention = _internal_expectation_attention(observation)
    expected_width = (
        params.expectation_deposit_width
        if params.expectation_deposit_width is not None
        else params.deposit_width
    )
    expected_density += (
        expected_attention
        * params.expectation_deposit_strength
        * previous._gaussian_grid(
            previous.center_for_vector(
                observation.expected,
                episode.vocabulary,
                params.feature_x,
                params.feature_y,
            ),
            expected_width,
        )
    )

    expected_density = np.clip(expected_density, 0.0, params.max_density)
    actual_density = np.clip(actual_density, 0.0, params.max_density)
    density = np.clip(
        untagged_density + expected_density + actual_density,
        0.0,
        params.max_density,
    )
    wells = [
        SubjectiveTopologyWell(
            source=_experience_object_for_input(item.source, episode.vocabulary),
            center=previous.center_for_vector(
                item.source.features,
                episode.vocabulary,
                params.feature_x,
                params.feature_y,
            ),
            width=params.deposit_width,
            weight=float(item.strength),
        )
        for item in memory_items
    ]
    last_focus = previous.last_focus
    if current_centers:
        last_focus = np.mean(np.array(current_centers), axis=0)

    correction = SubjectiveTopologyCorrection(
        expected_point=previous.center_for_vector(
            observation.expected,
            episode.vocabulary,
            params.feature_x,
            params.feature_y,
        ),
        actual_point=previous.center_for_vector(
            observation.actual,
            episode.vocabulary,
            params.feature_x,
            params.feature_y,
        ),
        after_point=previous.center_for_vector(
            observation.memory_state,
            episode.vocabulary,
            params.feature_x,
            params.feature_y,
        ),
        surprise=observation.surprise,
        learning_rate=observation.learning_rate,
    )
    return SubjectiveTopologyState(
        density=density,
        wells=wells,
        expected_density=expected_density,
        actual_density=actual_density,
        correction=correction,
        bounds=params.bounds,
        feature_x=params.feature_x,
        feature_y=params.feature_y,
        last_focus=None if last_focus is None else np.array(last_focus, dtype=float),
    )


def _internal_expectation_attention(observation: EpisodeObservation) -> float:
    effective = observation.metadata.get("effective_attention", {})
    if isinstance(effective, dict) and "internal_expectation" in effective:
        return float(np.clip(effective["internal_expectation"], 0.0, 1.0))
    channels = observation.metadata.get("attention_channels")
    if isinstance(channels, dict):
        attention_effect = float(
            observation.metadata.get("attention_effect", observation.attention)
        )
        return float(
            np.clip(
                attention_effect * float(channels.get("internal_expectation", 0.0)),
                0.0,
                1.0,
            )
        )
    return 1.0


def _episode_correction(topology: SubjectiveTopologyState) -> EpisodeCorrection | None:
    correction = topology.correction
    if correction is None:
        return None
    return EpisodeCorrection(
        expected_point=correction.expected_point.copy(),
        actual_point=correction.actual_point.copy(),
        after_point=correction.after_point.copy(),
        surprise=correction.surprise,
        learning_rate=correction.learning_rate,
    )


def _experience_object_for_input(
    item: EpisodeInput,
    vocabulary: list[str],
) -> ExperienceObject:
    return ExperienceObject(
        id=item.id,
        temporal_extent=TemporalExtent(item.start, item.end, item.order_index),
        features=FeatureVector(
            {
                key: float(item.features[index])
                for index, key in enumerate(vocabulary)
                if index < item.features.size
            }
        ),
        kind=item.kind,
        presentation=item.presentation,
        salience=item.salience,
        learning_weight=item.learning_weight,
        modality=item.modality,
        metadata=dict(item.metadata),
    )


def _require_observations(episode: Episode) -> None:
    if not episode.observations:
        raise ValueError("episode has no observations")
