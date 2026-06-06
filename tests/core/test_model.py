from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from cave.demonstrations.examples import (
    DEFAULT_VOCABULARY,
    default_model_params,
    demo_model,
    demo_sequence,
    model_for_sequence,
    random_experience_model,
    random_experience_sequence,
)
from cave.observation.episodes import CaveProducer, Episode, EpisodeObservation
from cave.substrates.cavenet import (
    CaveNet,
    CaveNetAdaptationPolicy,
    CaveNetConfig,
    CaveNetProducer,
    compare_cavenet_to_cave,
)
from cave.substrates.minimal_subject import MinimalSubjectConfig, emergence_metrics, run_minimal_subject
from cave.commitments.attention import (
    AttentionChannelCurve,
    AttentionProfile,
    AttentionState,
    INTERNAL_EXPECTATION_CHANNEL,
    ObjectiveAdaptiveAttentionPolicy,
    SurpriseAdaptiveAttentionPolicy,
    attention_effect,
    balanced_attention_profile,
    external_only_attention_profile,
    internal_only_attention_profile,
    zero_attention_profile,
)
from cave.commitments.agency import PreferenceActionPolicy, PreferenceProfile
from cave.commitments.affect import MetadataValenceEvaluator
from cave.observation.experience import (
    ExperienceObject,
    ExperienceQualities,
    ExperienceQualityResolver,
    FeatureProjection,
    FeatureVector,
    INTERNAL_EXPERIENCE_CHANNEL,
    InputSequence,
    InternalExperienceGenerator,
    ShapePresentation,
    TemporalExtent,
    feature_axis_value,
    generate_internal_experiences,
    load_experience_document,
    resolve_experience_effects,
    visual_presentation_from_features,
)
from cave.commitments.learning import ImportanceWeightedLearningRule
from cave.commitments.memory import (
    MemoryTrace,
    memory_strength,
)
from cave.commitments.objective import LinearObjectiveEvaluator
from cave.demonstrations.state import SubjectState
from cave.commitments.topology import (
    SubjectiveTopologyParams,
    SubjectiveTopologyPrior,
    SubjectiveTopologyState,
)
from cave.commitments.workspace import TopKWorkspaceCompressor
from cave.observation.pipeline import episode_payload, run_payload, views_from_names
from cave.observation.projections import TimelineProjection, WallPOVProjection, project_all
from cave.presentation.renderers import (
    LayoutSpec,
    MatplotlibRenderer,
    flatten_topology_state,
    save_topology_state_surface,
    topology_state_surface,
)
from cave.presentation.renderers.matplotlib_renderer import (
    available_styles,
    normalize_correction_series,
    resolve_style,
)
from cave.observation.sensing import FeatureSensor, Sensorium
from cave.observation.structural import episode_frames, frame_for_time, structural_state_for_episode
from cave.observation.views import (
    AffectView,
    ActionView,
    CorrectionView,
    CorrectionViewState,
    ExpectationActualView,
    ExpectationActualViewState,
    MemoryLookbackView,
    PresentationView,
    SubjectSurfaceView,
    SubjectSurfaceViewState,
    SubjectiveTopologyView,
    SubjectiveTopologyViewState,
    TimelineView,
    default_views,
)


def episode_frame_for_model(model, t: float, *, dt: float = 0.1):
    episode = CaveProducer(model).run(dt=dt)
    structural = structural_state_for_episode(episode)
    return frame_for_time(episode, t, structural)


def test_input_sequence_activation_and_features() -> None:
    sequence = demo_sequence()

    assert [obj.id for obj in sequence.active_at(0.6)] == ["evt_triangle"]
    assert [obj.id for obj in sequence.active_at(1.3)] == []
    assert [obj.id for obj in sequence.active_at(1.5)] == ["evt_circle"]

    features = sequence.features_at(0.6, list(DEFAULT_VOCABULARY))

    np.testing.assert_allclose(
        features,
        np.array([0.0, 0.558, 0.117, 0.468, 0.711, 0.9, 0.0, 0.765, 0.495]),
    )


def test_visual_presentation_is_decoded_from_feature_vector() -> None:
    sequence = demo_sequence()
    triangle = sequence.active_at(0.6)[0]
    circle = sequence.active_at(1.5)[0]

    assert triangle.presentation is None

    triangle_presentation = visual_presentation_from_features(triangle.features)
    circle_presentation = visual_presentation_from_features(circle.features)

    assert isinstance(triangle_presentation, ShapePresentation)
    assert triangle_presentation.shape_type == "polygon"
    assert len(triangle_presentation.points) == 3
    assert float(triangle_presentation.style["size"]) > 0.9
    assert circle_presentation.shape_type == "circle"


def test_experience_quality_resolver_derives_effects_from_qualities() -> None:
    painful = resolve_experience_effects(ExperienceQualities(pain=1.0))
    pleasurable = resolve_experience_effects(ExperienceQualities(pleasure=1.0))
    novel = resolve_experience_effects(ExperienceQualities(novelty=1.0))

    assert painful.salience > pleasurable.salience
    assert painful.learning_weight > pleasurable.learning_weight
    assert novel.pleasure > 0.0


def test_experience_quality_resolver_turns_overload_into_pain() -> None:
    effects = ExperienceQualityResolver().resolve(
        ExperienceQualities(magnitude=1.0, overload=1.0)
    )

    assert effects.pain > 0.0
    assert effects.salience > 0.0


def test_feature_projection_supports_weighted_landscape_axes() -> None:
    features = FeatureVector({"angularity": 1.0, "symmetry": 0.5, "sides": 0.25})
    axis = FeatureProjection(
        name="form",
        weights={"angularity": 0.5, "symmetry": 0.25, "sides": 0.25},
    )

    assert feature_axis_value(features, axis) == 0.6875


def test_random_experience_sequence_is_seeded_and_vector_authored() -> None:
    first = random_experience_sequence(count=5, seed=11)
    second = random_experience_sequence(count=5, seed=11)
    different = random_experience_sequence(count=5, seed=12)

    assert len(first.objects) == 5
    assert [obj.temporal_extent.start for obj in first.objects] == sorted(
        obj.temporal_extent.start for obj in first.objects
    )
    assert [obj.features.values for obj in first.objects] == [
        obj.features.values for obj in second.objects
    ]
    assert first.objects[0].features.values != different.objects[0].features.values

    for obj in first.objects:
        assert obj.presentation is None
        assert set(DEFAULT_VOCABULARY).issubset(obj.features.values)
        assert all(0.0 <= obj.features.values[key] <= 1.0 for key in DEFAULT_VOCABULARY)


def test_random_experience_model_runs_through_views() -> None:
    model = random_experience_model(count=4, seed=3)
    frame = episode_frame_for_model(model, 0.1)
    presentation = PresentationView().project(frame)

    assert model.sequence.duration > 0.0
    assert len(model.vocabulary) == len(DEFAULT_VOCABULARY)
    assert presentation.items
    assert isinstance(presentation.items[0].presentation, ShapePresentation)


def test_memory_keeps_vector_and_object_trace() -> None:
    sequence = demo_sequence()
    memory = MemoryTrace(
        vector=np.zeros(len(DEFAULT_VOCABULARY)),
        retention=0.5,
        decay_tau=1.0,
        max_age=2.0,
    )

    current_objects = sequence.active_at(0.3)
    u_t = sequence.features_at(0.3, list(DEFAULT_VOCABULARY))
    memory.update(0.3, u_t, current_objects)

    assert memory.items == []
    assert "evt_triangle" in memory.active
    np.testing.assert_allclose(memory.vector, 0.5 * u_t)

    memory.update(1.3, np.zeros(len(DEFAULT_VOCABULARY)), [])

    assert len(memory.items) == 1
    assert memory.items[0].source.id == "evt_triangle"
    assert memory.items[0].ended_t == 1.2
    assert memory.items[0].strength == 0.5 * memory_strength(0.1, 1.0)


def test_memory_update_accepts_variable_learning_rate() -> None:
    memory = MemoryTrace(
        vector=np.array([0.0]),
        retention=0.5,
        decay_tau=1.0,
        max_age=2.0,
    )

    memory.update(
        0.0,
        np.array([1.0]),
        [],
        learning_rate=0.8,
    )

    np.testing.assert_allclose(memory.vector, np.array([0.8]))


def test_memory_keeps_expectation_trace_separate_from_evidence_vector() -> None:
    memory = MemoryTrace(
        vector=np.array([0.0, 0.0]),
        retention=0.5,
        decay_tau=1.0,
        max_age=2.0,
    )

    memory.update(
        0.0,
        np.array([0.0, 0.0]),
        [],
        AttentionState(channel_weights={INTERNAL_EXPECTATION_CHANNEL: 1.0}),
        expected_input=np.array([1.0, 0.25]),
    )

    np.testing.assert_allclose(memory.vector, np.array([0.0, 0.0]))
    np.testing.assert_allclose(memory.expectation_vector, np.array([1.0, 0.25]))
    assert memory.expectation_strength == pytest.approx(1.0)


def test_topology_deposits_attended_expectation_without_actual_input() -> None:
    params = SubjectiveTopologyParams(
        feature_x="angularity",
        feature_y="roundness",
        resolution=20,
        prior=SubjectiveTopologyPrior(),
        expectation_deposit_strength=0.5,
    )
    topology = SubjectiveTopologyState.initial(
        feature_x=params.feature_x,
        feature_y=params.feature_y,
        resolution=params.resolution,
        prior=params.prior,
    )
    memory = MemoryTrace(vector=np.zeros(2))
    prediction = np.array([1.0, 1.0])

    updated = topology.update(
        memory,
        [],
        params,
        current_attention=AttentionState(
            channel_weights={INTERNAL_EXPECTATION_CHANNEL: 1.0}
        ),
        vocabulary=["angularity", "roundness"],
        expected_input=prediction,
        actual_input=np.zeros(2),
        after_input=np.zeros(2),
    )

    assert float(np.sum(updated.expected_density)) > 0.0
    assert float(np.sum(updated.actual_density)) == pytest.approx(0.0)
    np.testing.assert_allclose(updated.density, updated.expected_density)


def test_model_step_returns_snapshot_scene_state() -> None:
    model = demo_model(seed=1)
    first = model.step(0.2)
    first_density = first.subject_state.topology.density.copy()

    model.step(0.4)

    np.testing.assert_allclose(first.subject_state.topology.density, first_density)


def test_projections_put_coordinates_in_view_state_only() -> None:
    model = demo_model(seed=1)
    frame = episode_frame_for_model(model, 0.2)
    views = project_all(frame)

    assert set(views) == {"wall_pov", "lookback", "timeline", "subjective_topology"}
    wall_object = views["wall_pov"].rendered_objects[0]

    assert wall_object.transform.x > 0.0
    assert not hasattr(frame.episode.inputs[0], "x")
    assert not hasattr(frame.episode.inputs[0], "wall_position")


def test_timeline_projection_can_use_state_sequence() -> None:
    model = demo_model(seed=1)
    frame = episode_frame_for_model(model, 0.2)

    timeline = TimelineProjection().project(frame)

    assert any(obj.role == "event_interval" for obj in timeline.rendered_objects)
    assert any(obj.role == "timeline_pointer" for obj in timeline.rendered_objects)


def test_wall_projection_is_pure() -> None:
    model = demo_model(seed=1)
    frame = episode_frame_for_model(model, 0.2)
    before = frame.topology_frame.topology.density.copy()

    WallPOVProjection().project(frame)

    np.testing.assert_allclose(frame.topology_frame.topology.density, before)


def test_typed_views_project_semantic_state() -> None:
    model = demo_model(seed=1)
    episode = CaveProducer(model).run(dt=0.1)
    structural = structural_state_for_episode(episode)
    frame = frame_for_time(episode, 0.2, structural)

    presentation = PresentationView().project(frame)
    timeline = TimelineView().project(frame)
    topology = SubjectiveTopologyView(grid_resolution=12).project(frame)

    assert presentation.items[0].source_id == "evt_triangle"
    assert timeline.intervals[0].active
    assert isinstance(topology, SubjectiveTopologyViewState)
    assert topology.density is not None
    assert topology.density.shape == (12, 12)

    memory_frame = frame_for_time(episode, 1.3, structural)
    memory = MemoryLookbackView().project(memory_frame)
    assert memory.items[0].source_id == "evt_triangle"


def test_subjective_topology_is_stateful_density_plane() -> None:
    model = demo_model(seed=1)
    triangle_state = model.step(0.2)
    circle_state = model.run(start=0.4, end=1.5, dt=0.1)[-1]

    triangle = triangle_state.current_objects[0]
    circle = circle_state.current_objects[0]
    topology = circle_state.subject_state.topology
    triangle_center = topology.center_for_object(triangle)
    circle_center = topology.center_for_object(circle)
    midpoint = 0.5 * (triangle_center + circle_center)

    assert topology.intensity_at(triangle_center) > 0.0
    assert topology.intensity_at(circle_center) > 0.0
    assert topology.intensity_at(midpoint) > 0.0


def test_subjective_topology_default_initial_state_is_flat() -> None:
    topology = SubjectiveTopologyState.initial(resolution=12)

    assert float(np.max(topology.density)) == 0.0


def test_subjective_topology_prior_seeds_initial_density() -> None:
    flat = SubjectiveTopologyState.initial(resolution=12)
    seeded = SubjectiveTopologyState.initial(
        resolution=12,
        prior=SubjectiveTopologyPrior(
            mode="random_wells",
            strength=0.2,
            width=0.25,
            seed=3,
            well_count=3,
        ),
    )
    repeated = SubjectiveTopologyState.initial(
        resolution=12,
        prior=SubjectiveTopologyPrior(
            mode="random_wells",
            strength=0.2,
            width=0.25,
            seed=3,
            well_count=3,
        ),
    )

    assert float(np.max(seeded.density)) > float(np.max(flat.density))
    np.testing.assert_allclose(seeded.density, repeated.density)


def test_subject_state_uses_topology_prior_params() -> None:
    trace = MemoryTrace(vector=np.zeros(len(DEFAULT_VOCABULARY)))
    subject_state = SubjectState.initial(
        trace,
        SubjectiveTopologyParams(
            resolution=12,
            prior=SubjectiveTopologyPrior(mode="basin", strength=0.2, width=0.35),
        ),
    )

    assert float(np.max(subject_state.topology.density)) > 0.0


def test_attention_controls_current_input_memory_and_topology() -> None:
    model = demo_model(seed=1)
    model.params = replace(
        model.params,
        attention=AttentionProfile(mode="constant", level=0.0),
    )
    before = model.subject_state.topology.density.copy()
    expected = model.subject_state.topology._diffuse(
        before * model.params.topology.decay,
        model.params.topology.diffusion,
    )

    state = model.step(0.2)

    np.testing.assert_allclose(state.input_vector, np.zeros(len(DEFAULT_VOCABULARY)))
    assert state.subject_state.memory.items == []
    np.testing.assert_allclose(state.subject_state.topology.density, expected)

    episode_model = demo_model(seed=1)
    episode_model.params = replace(
        episode_model.params,
        attention=AttentionProfile(mode="constant", level=0.0),
    )
    presentation = PresentationView().project(episode_frame_for_model(episode_model, 0.2))
    assert presentation.items[0].opacity == 0.0


def test_prediction_state_is_computed_before_memory_update() -> None:
    model = demo_model(seed=1)

    state = model.step(0.2)

    np.testing.assert_allclose(
        state.prediction.expected_input,
        np.zeros(len(DEFAULT_VOCABULARY)),
    )
    np.testing.assert_allclose(
        state.prediction.prediction_error,
        state.input_vector,
    )
    assert state.prediction.surprise > 0.0


def test_cavenet_matches_symbolic_cave_core_readouts() -> None:
    cave_model = demo_model(seed=1)
    cavenet_model = demo_model(seed=1)

    cave_episode = CaveProducer(cave_model).run(dt=0.2)
    cavenet = CaveNet.from_subject_state(
        sequence=cavenet_model.sequence,
        subject_state=cavenet_model.subject_state,
        params=cavenet_model.params,
        vocabulary=cavenet_model.vocabulary,
        sensorium=cavenet_model.sensorium,
    )
    cavenet_episode = CaveNetProducer(cavenet).run(dt=0.2)
    comparison = compare_cavenet_to_cave(cave_episode, cavenet_episode)

    assert comparison.ok
    assert cavenet_episode.metadata["adapter"] == "CaveNet"
    assert cavenet_episode.metadata["cavenet"]["representation"] == "fixed_network_form"
    assert comparison.metrics["max_actual_distance"] <= 1e-12
    assert comparison.metrics["max_memory_distance"] <= 1e-12


def test_cavenet_parameter_gain_perturbs_attention_block() -> None:
    cavenet_model = demo_model(seed=1)
    cavenet = CaveNet.from_subject_state(
        sequence=cavenet_model.sequence,
        subject_state=cavenet_model.subject_state,
        params=cavenet_model.params,
        vocabulary=cavenet_model.vocabulary,
        sensorium=cavenet_model.sensorium,
        config=CaveNetConfig(attention_gain=0.0),
    )
    episode = CaveNetProducer(cavenet).run(dt=0.2)

    assert episode.metadata["cavenet_config"]["attention_gain"] == 0.0
    assert all(np.allclose(observation.actual, 0.0) for observation in episode.observations)


def test_cavenet_expectation_readout_uses_internal_attention_channel() -> None:
    sequence = InputSequence(
        objects=[
            ExperienceObject(
                id="visual",
                temporal_extent=TemporalExtent(0.0, 1.0, 0),
                features=FeatureVector({"energy": 1.0}),
            )
        ]
    )
    params = replace(
        default_model_params(),
        attention=AttentionProfile(
            mode="constant",
            level=1.0,
            channel_weights={INTERNAL_EXPECTATION_CHANNEL: 1.0},
        ),
        topology=SubjectiveTopologyParams(
            feature_x="energy",
            feature_y="energy",
            prior=SubjectiveTopologyPrior(),
        ),
    )
    subject_state = SubjectState.initial(
        MemoryTrace(vector=np.array([0.8])),
        params.topology,
    )
    cavenet = CaveNet.from_subject_state(
        sequence=sequence,
        subject_state=subject_state,
        params=params,
        vocabulary=["energy"],
    )

    episode = CaveNetProducer(cavenet).run(dt=0.5)

    np.testing.assert_allclose(episode.observations[0].actual, np.array([0.0]))
    np.testing.assert_allclose(episode.observations[0].expected, np.array([0.8]))


def test_cavenet_adaptation_policy_moves_gains_under_pressure() -> None:
    config = CaveNetConfig(attention_gain=0.4, learning_rate_gain=0.2)
    policy = CaveNetAdaptationPolicy(
        enabled=True,
        surprise_threshold=0.1,
        learning_gain_rate=0.5,
        attention_gain_rate=0.25,
    )

    adapted = policy.adapt(
        config,
        surprise=0.5,
        utility=-0.2,
        compression_cost=0.1,
    )

    assert adapted.learning_rate_gain > config.learning_rate_gain
    assert adapted.attention_gain > config.attention_gain


def test_minimal_subject_develops_preference_weighted_memory_readout() -> None:
    vocabulary = ["cue", "preferred", "distractor"]
    objects = []
    t = 0.0
    for index in range(8):
        objects.append(
            ExperienceObject(
                id=f"cue_{index}",
                temporal_extent=TemporalExtent(t, t + 1.0, 2 * index),
                features=FeatureVector({"cue": 1.0, "distractor": 0.8}),
            )
        )
        t += 1.0
        objects.append(
            ExperienceObject(
                id=f"preferred_{index}",
                temporal_extent=TemporalExtent(t, t + 1.0, 2 * index + 1),
                features=FeatureVector({"preferred": 1.0, "distractor": 0.8}),
            )
        )
        t += 1.0
    episode = run_minimal_subject(
        InputSequence(objects),
        vocabulary=vocabulary,
        preference_vector=np.array([0.0, 1.0, 0.0]),
        config=MinimalSubjectConfig(
            workspace_capacity=2,
            diagnostic_features=("cue",),
        ),
    )

    metrics = emergence_metrics(episode)

    assert metrics["skill_gain"] > 0.0
    assert metrics["late_memory_strength"] > 0.0
    assert "minimal_subject" in episode.observations[-1].metadata


def test_learning_weight_increases_memory_update_rate() -> None:
    base = ExperienceObject(
        id="base",
        temporal_extent=TemporalExtent(0.0, 1.0, 0),
        features=FeatureVector({"energy": 1.0}),
        learning_weight=1.0,
    )
    important = ExperienceObject(
        id="important",
        temporal_extent=TemporalExtent(0.0, 1.0, 0),
        features=FeatureVector({"energy": 1.0}),
        learning_weight=2.0,
    )
    base_model = model_for_sequence(InputSequence([base]), vocabulary=["energy"])
    important_model = model_for_sequence(InputSequence([important]), vocabulary=["energy"])

    base_state = base_model.step(0.5)
    important_state = important_model.step(0.5)

    assert base_state.learning_rate == 1.0 - base_model.params.memory.retention
    assert important_state.learning_rate > base_state.learning_rate
    assert important_state.subject_state.memory.vector[0] > base_state.subject_state.memory.vector[0]


def test_surprise_weighted_learning_rule_raises_rate() -> None:
    rule = ImportanceWeightedLearningRule(surprise_gain=1.0)
    attention = AttentionState(capacity=1.0)

    low = rule.learning_rate(
        base_rate=0.1,
        attention=attention,
        importance=1.0,
        surprise=0.0,
    )
    high = rule.learning_rate(
        base_rate=0.1,
        attention=attention,
        importance=1.0,
        surprise=2.0,
    )

    assert low == 0.1
    assert high == pytest.approx(0.3)


def test_attention_effect_preserves_high_band_variation() -> None:
    assert attention_effect(0.0) == 0.0
    assert attention_effect(1.0) == 1.0
    assert attention_effect(0.25) == 0.25
    assert attention_effect(0.5) == 0.5
    assert 0.5 < attention_effect(0.75) < 0.75
    assert 0.75 < attention_effect(0.9) < 0.9


def test_attention_state_normalizes_channel_distribution() -> None:
    attention = AttentionState(
        channel_weights={"visual": 2.0, "audio": 1.0},
        capacity=0.75,
    )

    assert attention.scalar == 0.75
    assert attention.channel_weight("visual") == 2.0 / 3.0
    assert attention.channel_weight("audio") == 1.0 / 3.0
    assert attention.channel_weight("imaginal") == 0.0


def test_default_attention_splits_external_and_internal_experience() -> None:
    attention = AttentionState(capacity=1.0)

    assert attention.channel_weight("visual") == pytest.approx(0.5)
    assert attention.channel_weight(INTERNAL_EXPECTATION_CHANNEL) == pytest.approx(0.5)
    assert attention.object_impact(
        ExperienceObject(
            id="visual",
            temporal_extent=TemporalExtent(0.0, 1.0, 0),
            features=FeatureVector({"energy": 1.0}),
        )
    ) == pytest.approx(0.5)
    assert attention.internal_expectation_impact() == pytest.approx(0.5)


def test_named_attention_profiles_make_capacity_and_allocation_explicit() -> None:
    balanced = balanced_attention_profile().state_at(0.0, 1.0)
    zero = zero_attention_profile().state_at(0.0, 1.0)
    external = external_only_attention_profile().state_at(0.0, 1.0)
    internal = internal_only_attention_profile().state_at(0.0, 1.0)

    assert balanced.capacity == 1.0
    assert balanced.channel_weight("visual") == pytest.approx(0.5)
    assert balanced.channel_weight(INTERNAL_EXPECTATION_CHANNEL) == pytest.approx(0.5)
    assert zero.capacity == 0.0
    assert external.channel_weight("visual") == pytest.approx(1.0)
    assert external.channel_weight(INTERNAL_EXPECTATION_CHANNEL) == 0.0
    assert internal.channel_weight("visual") == 0.0
    assert internal.channel_weight(INTERNAL_EXPECTATION_CHANNEL) == pytest.approx(1.0)


def test_attention_profile_can_schedule_channel_weights() -> None:
    profile = AttentionProfile(
        mode="constant",
        level=1.0,
        channel_weights={"visual": 1.0, INTERNAL_EXPECTATION_CHANNEL: 1.0},
        channel_curves={
            "visual": AttentionChannelCurve(
                mode="sine",
                level=0.5,
                amplitude=0.5,
                phase=0.0,
            ),
            INTERNAL_EXPECTATION_CHANNEL: AttentionChannelCurve(
                mode="sine",
                level=0.5,
                amplitude=0.5,
                phase=np.pi,
            ),
        },
    )

    early = profile.state_at(0.25, 1.0)
    late = profile.state_at(0.75, 1.0)

    assert early.channel_weight("visual") > early.channel_weight(
        INTERNAL_EXPECTATION_CHANNEL
    )
    assert late.channel_weight(INTERNAL_EXPECTATION_CHANNEL) > late.channel_weight(
        "visual"
    )


def test_attention_distribution_weights_input_channels() -> None:
    sequence = InputSequence(
        objects=[
            ExperienceObject(
                id="visual",
                temporal_extent=TemporalExtent(0.0, 1.0, 0),
                features=FeatureVector({"energy": 1.0}),
                salience=1.0,
                modality="visual",
            ),
            ExperienceObject(
                id="audio",
                temporal_extent=TemporalExtent(0.0, 1.0, 1),
                features=FeatureVector({"energy": 1.0}),
                salience=1.0,
                modality="audio",
            ),
        ]
    )
    attention = AttentionState(
        channel_weights={"visual": 0.25, "audio": 0.75},
        capacity=1.0,
    )

    attended = sequence.attended_features_at(0.5, ["energy"], attention)

    np.testing.assert_allclose(attended, np.array([1.0]))

    visual_attention = AttentionState(
        channel_weights={"visual": 1.0, "audio": 0.0},
        capacity=0.5,
    )
    attended_visual = sequence.attended_features_at(0.5, ["energy"], visual_attention)

    np.testing.assert_allclose(attended_visual, np.array([0.5]))


def test_internal_expectation_channel_gates_prediction_readout() -> None:
    sequence = InputSequence(
        objects=[
            ExperienceObject(
                id="visual",
                temporal_extent=TemporalExtent(0.0, 1.0, 0),
                features=FeatureVector({"energy": 1.0}),
            )
        ]
    )
    params = replace(
        default_model_params(),
        attention=AttentionProfile(
            mode="constant",
            level=1.0,
            channel_weights={"visual": 0.5, INTERNAL_EXPECTATION_CHANNEL: 0.5},
        ),
        topology=SubjectiveTopologyParams(
            feature_x="energy",
            feature_y="energy",
            prior=SubjectiveTopologyPrior(),
        ),
    )
    model = model_for_sequence(sequence, params=params, vocabulary=["energy"])
    model.subject_state.memory.vector = np.array([0.8])

    state = model.step(0.0)

    np.testing.assert_allclose(state.input_vector, np.array([0.5]))
    np.testing.assert_allclose(state.prediction.expected_input, np.array([0.4]))
    np.testing.assert_allclose(state.prediction.prediction_error, np.array([0.1]))


def test_internal_only_attention_can_experience_expectation_without_external_input() -> None:
    sequence = InputSequence(
        objects=[
            ExperienceObject(
                id="visual",
                temporal_extent=TemporalExtent(0.0, 1.0, 0),
                features=FeatureVector({"energy": 1.0}),
            )
        ]
    )
    params = replace(
        default_model_params(),
        attention=AttentionProfile(
            mode="constant",
            level=1.0,
            channel_weights={INTERNAL_EXPECTATION_CHANNEL: 1.0},
        ),
        topology=SubjectiveTopologyParams(
            feature_x="energy",
            feature_y="energy",
            prior=SubjectiveTopologyPrior(),
        ),
    )
    model = model_for_sequence(sequence, params=params, vocabulary=["energy"])
    model.subject_state.memory.vector = np.array([0.8])

    state = model.step(0.0)

    np.testing.assert_allclose(state.input_vector, np.array([0.0]))
    np.testing.assert_allclose(state.prediction.expected_input, np.array([0.8]))
    np.testing.assert_allclose(state.prediction.prediction_error, np.array([-0.8]))


def test_internal_experience_generator_discretizes_expected_input() -> None:
    sequence = InputSequence(
        objects=[
            ExperienceObject(
                id="visual",
                temporal_extent=TemporalExtent(0.0, 1.0, 0),
                features=FeatureVector({"energy": 1.0}),
            )
        ]
    )
    params = replace(
        default_model_params(),
        attention=AttentionProfile(
            mode="constant",
            level=1.0,
            channel_weights={INTERNAL_EXPECTATION_CHANNEL: 1.0},
        ),
        topology=SubjectiveTopologyParams(
            feature_x="energy",
            feature_y="energy",
            prior=SubjectiveTopologyPrior(),
        ),
    )
    model = model_for_sequence(sequence, params=params, vocabulary=["energy"])
    model.subject_state.memory.vector = np.array([0.8])
    episode = CaveProducer(model).run(dt=0.5)

    internal_sequence = generate_internal_experiences(
        episode,
        generator=InternalExperienceGenerator(surprise_gain=0.0),
    )

    assert internal_sequence.objects
    first = internal_sequence.objects[0]
    assert first.kind == "internal expectation"
    assert first.modality == INTERNAL_EXPERIENCE_CHANNEL
    assert first.features.value("energy") == pytest.approx(0.8)
    assert first.salience == pytest.approx(0.8)
    assert first.metadata["source"] == "generated_internal_experience"


def test_internal_experience_generator_weights_pain_above_pleasure() -> None:
    base = {
        "t_normalized": 0.0,
        "expected": np.array([0.0]),
        "actual": np.array([0.0]),
        "memory_state": np.array([0.0]),
        "surprise": 0.0,
        "learning_rate": 0.0,
        "attention": 1.0,
        "attention_weights": {},
        "active_inputs": [],
        "input_features": {},
    }
    episode = Episode(
        source_name="test",
        vocabulary=["energy"],
        inputs=[],
        observations=[
            EpisodeObservation(
                t=0.0,
                metadata={"valence": {"pain": 1.0, "pleasure": 0.0}},
                **base,
            ),
            EpisodeObservation(
                t=1.0,
                metadata={"valence": {"pain": 0.0, "pleasure": 1.0}},
                **base,
            ),
        ],
        duration=2.0,
    )

    internal_sequence = InternalExperienceGenerator().generate(episode)

    assert len(internal_sequence.objects) == 2
    pain_event, pleasure_event = internal_sequence.objects
    assert pain_event.salience > pleasure_event.salience
    assert pain_event.learning_weight > pleasure_event.learning_weight


def test_default_sensorium_preserves_visual_model_input() -> None:
    model = demo_model(seed=1)
    state = model.step(0.3)
    expected = model.sequence.attended_features_at(
        0.3,
        model.vocabulary,
        state.attention_state,
    )

    assert set(state.sensor_responses) == {"visual"}
    np.testing.assert_allclose(state.input_vector, expected)


def test_default_sensorium_ignores_unsensed_audio_objects() -> None:
    sequence = InputSequence(
        objects=[
            ExperienceObject(
                id="audio",
                temporal_extent=TemporalExtent(0.0, 1.0, 0),
                features=FeatureVector({"energy": 1.0}),
                modality="audio",
            )
        ]
    )
    model = model_for_sequence(sequence, vocabulary=["energy"])
    state = model.step(0.5)

    np.testing.assert_allclose(state.input_vector, np.array([0.0]))


def test_sensorium_can_add_audio_sense_channel() -> None:
    sequence = InputSequence(
        objects=[
            ExperienceObject(
                id="audio",
                temporal_extent=TemporalExtent(0.0, 1.0, 0),
                features=FeatureVector({"energy": 1.0}),
                modality="audio",
            )
        ]
    )
    model = model_for_sequence(sequence, vocabulary=["energy"])
    model.params = replace(
        model.params,
        attention=AttentionProfile(
            mode="constant",
            level=1.0,
            channel_weights={"audio": 1.0},
        ),
    )
    model.sensorium = Sensorium(
        sensors=(FeatureSensor(modality="audio", channel="audio"),)
    )
    state = model.step(0.5)

    np.testing.assert_allclose(state.input_vector, np.array([1.0]))


def test_adaptive_attention_shifts_toward_surprising_sensor_channel() -> None:
    sequence = InputSequence(
        objects=[
            ExperienceObject(
                id="audio",
                temporal_extent=TemporalExtent(0.0, 2.0, 0),
                features=FeatureVector({"energy": 1.0}),
                modality="audio",
            )
        ]
    )
    model = model_for_sequence(sequence, vocabulary=["energy"])
    model.params = replace(
        model.params,
        attention=AttentionProfile(
            mode="constant",
            level=1.0,
            channel_weights={"visual": 1.0},
        ),
        attention_policy=SurpriseAdaptiveAttentionPolicy(learning_rate=0.5),
    )
    model.sensorium = Sensorium(
        sensors=(FeatureSensor(modality="audio", channel="audio"),)
    )

    first = model.step(0.0)
    second = model.step(0.5)

    np.testing.assert_allclose(first.input_vector, np.array([0.0]))
    assert first.next_attention_channel_weights["audio"] == 0.5
    assert first.next_attention_channel_weights["visual"] == 0.5
    assert second.attention_state.channel_weight("audio") == 0.5
    np.testing.assert_allclose(second.input_vector, np.array([0.5]))


def test_affect_metadata_enters_valence_state_without_surprise_pain() -> None:
    sequence = InputSequence(
        objects=[
            ExperienceObject(
                id="painful",
                temporal_extent=TemporalExtent(0.0, 1.0, 0),
                features=FeatureVector({"threat": 1.0}),
                metadata={"affect": {"pain": 0.8}},
            )
        ]
    )
    params = replace(
        default_model_params(),
        attention=AttentionProfile(mode="constant", level=1.0),
        topology=SubjectiveTopologyParams(
            feature_x="threat",
            feature_y="threat",
            prior=SubjectiveTopologyPrior(),
        ),
        valence_evaluator=MetadataValenceEvaluator(),
        objective_evaluator=LinearObjectiveEvaluator(prediction_weight=0.0),
    )
    model = model_for_sequence(sequence, params=params, vocabulary=["threat"])
    state = model.step(0.0)

    assert state.prediction.surprise > 0.0
    assert state.valence.pain == pytest.approx(0.4)
    assert state.valence.components["surprise_pain"] == 0.0
    assert state.objective.pain_cost == pytest.approx(0.4)
    assert state.objective.utility == pytest.approx(-0.4)


def test_surprise_can_create_pain_when_configured() -> None:
    sequence = InputSequence(
        objects=[
            ExperienceObject(
                id="neutral",
                temporal_extent=TemporalExtent(0.0, 1.0, 0),
                features=FeatureVector({"energy": 1.0}),
            )
        ]
    )
    params = replace(
        default_model_params(),
        attention=AttentionProfile(mode="constant", level=1.0),
        topology=SubjectiveTopologyParams(
            feature_x="energy",
            feature_y="energy",
            prior=SubjectiveTopologyPrior(),
        ),
        valence_evaluator=MetadataValenceEvaluator(surprise_pain_gain=0.5),
    )
    model = model_for_sequence(sequence, params=params, vocabulary=["energy"])
    state = model.step(0.0)

    assert state.valence.pain == pytest.approx(0.5 * state.prediction.surprise)
    assert state.valence.components["object_pain"] == 0.0


def test_objective_attention_policy_shifts_toward_painful_channel() -> None:
    sequence = InputSequence(
        objects=[
            ExperienceObject(
                id="visual",
                temporal_extent=TemporalExtent(0.0, 1.0, 0),
                features=FeatureVector({"visual_signal": 1.0, "audio_signal": 0.0}),
                modality="visual",
            ),
            ExperienceObject(
                id="audio",
                temporal_extent=TemporalExtent(0.0, 1.0, 1),
                features=FeatureVector({"visual_signal": 0.0, "audio_signal": 1.0}),
                modality="audio",
                metadata={"affect": {"pain": 1.0}},
            ),
        ]
    )
    params = replace(
        default_model_params(),
        attention=AttentionProfile(
            mode="constant",
            level=1.0,
            channel_weights={"visual": 0.8, "audio": 0.2},
        ),
        attention_policy=ObjectiveAdaptiveAttentionPolicy(
            learning_rate=0.75,
            signal_gain=0.1,
            pain_gain=4.0,
        ),
        topology=SubjectiveTopologyParams(
            feature_x="visual_signal",
            feature_y="audio_signal",
            prior=SubjectiveTopologyPrior(),
        ),
        valence_evaluator=MetadataValenceEvaluator(),
    )
    model = model_for_sequence(
        sequence,
        params=params,
        vocabulary=["visual_signal", "audio_signal"],
    )
    model.sensorium = Sensorium(
        sensors=(
            FeatureSensor(modality="visual", channel="visual"),
            FeatureSensor(modality="audio", channel="audio"),
        )
    )

    first = model.step(0.0)
    second = model.step(0.5)

    assert first.valence.pain > 0.0
    assert first.next_attention_channel_weights["audio"] > 0.2
    assert second.attention_state.channel_weight("audio") == pytest.approx(
        first.next_attention_channel_weights["audio"]
    )


def test_episode_observation_exposes_affect_metadata_and_view() -> None:
    sequence = InputSequence(
        objects=[
            ExperienceObject(
                id="pleasant",
                temporal_extent=TemporalExtent(0.0, 1.0, 0),
                features=FeatureVector({"comfort": 1.0}),
                metadata={"affect": {"pleasure": 0.7}},
            )
        ]
    )
    params = replace(
        default_model_params(),
        attention=AttentionProfile(mode="constant", level=1.0),
        topology=SubjectiveTopologyParams(
            feature_x="comfort",
            feature_y="comfort",
            prior=SubjectiveTopologyPrior(),
        ),
        valence_evaluator=MetadataValenceEvaluator(),
    )
    model = model_for_sequence(sequence, params=params, vocabulary=["comfort"])
    episode = CaveProducer(model).run(dt=0.5)
    observation = episode.observations[0]

    assert observation.metadata["valence"]["pleasure"] == pytest.approx(0.35)
    assert "objective" in observation.metadata

    structural = structural_state_for_episode(episode)
    frame = frame_for_time(episode, 0.0, structural)
    view_state = AffectView().project(frame)

    assert view_state.current.pleasure == pytest.approx(0.35)
    assert view_state.points


def test_workspace_compression_can_replace_state_input() -> None:
    sequence = InputSequence(
        objects=[
            ExperienceObject(
                id="complex",
                temporal_extent=TemporalExtent(0.0, 1.0, 0),
                features=FeatureVector({"a": 1.0, "b": 0.5, "c": 0.25}),
            )
        ]
    )
    params = replace(
        default_model_params(),
        attention=AttentionProfile(mode="constant", level=1.0),
        topology=SubjectiveTopologyParams(
            feature_x="a",
            feature_y="b",
            prior=SubjectiveTopologyPrior(),
        ),
        workspace_compressor=TopKWorkspaceCompressor(capacity=1),
        workspace_input_mode="workspace",
    )
    model = model_for_sequence(sequence, params=params, vocabulary=["a", "b", "c"])
    state = model.step(0.0)

    np.testing.assert_allclose(
        state.attended_input_vector,
        np.array([0.5, 0.25, 0.125]),
    )
    np.testing.assert_allclose(state.input_vector, np.array([0.5, 0.0, 0.0]))
    assert state.workspace.active_features == ["a"]
    assert state.workspace.compression_cost > 0.0
    assert state.objective.compression_cost == pytest.approx(
        state.workspace.compression_cost
    )


def test_preference_action_modulates_exposure_without_changing_sequence() -> None:
    sequence = InputSequence(
        objects=[
            ExperienceObject(
                id="warm",
                temporal_extent=TemporalExtent(0.0, 1.0, 0),
                features=FeatureVector({"warmth": 1.0, "threat": 0.0}),
            ),
            ExperienceObject(
                id="threat",
                temporal_extent=TemporalExtent(0.0, 1.0, 1),
                features=FeatureVector({"warmth": 0.0, "threat": 1.0}),
            ),
        ]
    )
    params = replace(
        default_model_params(),
        attention=AttentionProfile(mode="constant", level=1.0),
        topology=SubjectiveTopologyParams(
            feature_x="warmth",
            feature_y="threat",
            prior=SubjectiveTopologyPrior(),
        ),
        action_policy=PreferenceActionPolicy(
            PreferenceProfile(
                feature_rewards={"warmth": 1.0},
                feature_aversions={"threat": 0.1},
                approach_gain=0.5,
            )
        ),
    )
    model = model_for_sequence(
        sequence,
        params=params,
        vocabulary=["warmth", "threat"],
    )
    episode = CaveProducer(model).run(dt=0.5)
    first = episode.observations[0]

    assert [obj.id for obj in sequence.objects] == ["warm", "threat"]
    assert first.metadata["action"]["kind"] == "approach"
    assert first.metadata["action"]["target_id"] == "warm"
    assert first.metadata["action"]["object_exposure"]["warm"] > 1.0
    np.testing.assert_allclose(first.actual, np.array([0.75, 0.5]))

    structural = structural_state_for_episode(episode)
    frame = frame_for_time(episode, 0.0, structural)
    action_view = ActionView().project(frame)
    assert action_view.current.kind == "approach"
    assert action_view.current.exposure > 1.0


def test_matplotlib_renderer_accepts_any_view_list(tmp_path) -> None:
    model = demo_model(seed=1)
    frame = episode_frame_for_model(model, 0.2)
    output = tmp_path / "selected_views.png"

    renderer = MatplotlibRenderer(layout=LayoutSpec(columns=2))
    renderer.save_frame(frame, [PresentationView(), TimelineView()], output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_matplotlib_renderer_accepts_named_style(tmp_path) -> None:
    model = demo_model(seed=1)
    frame = episode_frame_for_model(model, 0.2)
    output = tmp_path / "crt_frame.png"

    renderer = MatplotlibRenderer(layout=LayoutSpec(columns=2), style="crt")
    renderer.save_frame(frame, [PresentationView(), TimelineView()], output)

    assert "crt" in available_styles()
    assert resolve_style("no-style").name == "default"
    assert renderer.style.name == "crt"
    assert output.exists()
    assert output.stat().st_size > 0


def test_correction_series_is_normalized_for_animation() -> None:
    first = CorrectionViewState(
        name="correction",
        title="Correction",
        t=0.0,
        feature_x="x",
        feature_y="y",
        bounds=(-1.0, 1.0),
        experience_times=[0.0, 1.0],
        experience_labels=["first", "second"],
        expected_point=np.array([2.0, 10.0]),
        actual_point=np.array([4.0, 12.0]),
        after_point=np.array([6.0, 14.0]),
        surprise=0.0,
        learning_rate=0.0,
    )
    second = CorrectionViewState(
        name="correction",
        title="Correction",
        t=1.0,
        feature_x="x",
        feature_y="y",
        bounds=(-1.0, 1.0),
        experience_times=[0.0, 1.0],
        experience_labels=["first", "second"],
        expected_point=np.array([10.0, 20.0]),
        actual_point=np.array([10.0, 22.0]),
        after_point=np.array([10.0, 24.0]),
        surprise=0.0,
        learning_rate=0.0,
    )

    normalized = normalize_correction_series([[first], [second]])
    normalized_first = normalized[0][0]
    normalized_second = normalized[1][0]

    assert isinstance(normalized_first, CorrectionViewState)
    assert isinstance(normalized_second, CorrectionViewState)
    assert normalized_first.normalized
    assert normalized_first.bounds == (0.0, 1.0)
    assert normalized_first.title == "Prediction Correction Over Time"
    assert normalized_first.experience_times == [0.0, 1.0]
    assert normalized_first.experience_labels == ["first", "second"]
    np.testing.assert_allclose(normalized_first.series_times, np.array([0.0]))
    np.testing.assert_allclose(normalized_second.series_times, np.array([0.0, 1.0]))
    np.testing.assert_allclose(normalized_first.expected_point, np.array([0.0, 0.0]))
    np.testing.assert_allclose(normalized_second.after_point, np.array([1.0, 1.0]))
    np.testing.assert_allclose(
        normalized_second.actual_series,
        np.array([[0.25, 1.0 / 7.0], [1.0, 6.0 / 7.0]]),
    )


def test_topology_state_surface_flattens_topology_density() -> None:
    model = random_experience_model(count=3, seed=5)
    episode = CaveProducer(model).run(dt=0.2)
    structural = structural_state_for_episode(episode)
    frames = episode_frames(episode, structural)
    surface = topology_state_surface(episode, resolution=16)

    assert surface.times.shape == (len(frames),)
    assert surface.coordinates.shape == (16 * 16, 2)
    assert surface.density.shape == (len(frames), 16 * 16)
    assert surface.expected_density.shape == (len(frames), 16 * 16)
    assert surface.actual_density.shape == (len(frames), 16 * 16)
    assert float(np.sum(surface.expected_density)) > 0.0
    assert float(np.sum(surface.actual_density)) > 0.0
    np.testing.assert_allclose(
        surface.density[-1],
        flatten_topology_state(frames[-1].topology_frame.topology, 16).density,
    )


def test_topology_state_surface_requires_states() -> None:
    empty = Episode(
        source_name="empty",
        vocabulary=[],
        inputs=[],
        observations=[],
        duration=0.0,
    )
    with pytest.raises(ValueError, match="at least one frame is required"):
        topology_state_surface(empty)


@pytest.mark.slow
def test_save_topology_state_surface(tmp_path) -> None:
    model = random_experience_model(count=3, seed=5)
    episode = CaveProducer(model).run(dt=0.2)
    output = tmp_path / "topology_state_surface.png"

    save_topology_state_surface(episode, output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_default_views_are_renderer_neutral() -> None:
    views = default_views()

    assert [view.name for view in views] == [
        "presentation",
        "memory",
        "timeline",
        "expectation_actual",
        "correction",
        "subjective_topology",
    ]


def test_subject_surface_view_projects_default_cave_frame() -> None:
    model = demo_model(seed=1)
    frame = episode_frame_for_model(model, 0.2)

    view_state = SubjectSurfaceView().project(frame)

    assert isinstance(view_state, SubjectSurfaceViewState)
    assert 0.0 <= view_state.aperture <= 1.0
    assert -1.0 <= view_state.carry <= 1.0
    assert view_state.mode == "cave"
    assert view_state.trail_points


def test_expectation_actual_view_projects_correction_state() -> None:
    model = demo_model(seed=1)
    frame = episode_frame_for_model(model, 0.2)

    view_state = ExpectationActualView().project(frame)
    effective = frame.observation.metadata["effective_attention"]

    assert isinstance(view_state, ExpectationActualViewState)
    assert view_state.vocabulary == list(DEFAULT_VOCABULARY)
    np.testing.assert_allclose(view_state.expected_before, frame.observation.expected)
    np.testing.assert_allclose(view_state.actual, frame.observation.actual)
    np.testing.assert_allclose(view_state.error, frame.observation.error)
    np.testing.assert_allclose(view_state.expected_after, frame.observation.memory_state)
    assert view_state.expected_attention == pytest.approx(
        effective["internal_expectation"]
    )
    assert view_state.actual_attention == pytest.approx(effective["external_input"])


def test_correction_view_projects_feature_plane_points() -> None:
    model = demo_model(seed=1)
    frame = episode_frame_for_model(model, 0.2)

    view_state = CorrectionView().project(frame)
    effective = frame.observation.metadata["effective_attention"]

    assert isinstance(view_state, CorrectionViewState)
    topology = frame.topology_frame.topology
    correction = frame.topology_frame.correction
    assert correction is not None
    assert view_state.feature_x == topology.feature_x
    assert view_state.feature_y == topology.feature_y
    assert view_state.bounds == topology.bounds
    assert view_state.experience_times == [
        item.start
        for item in frame.episode.inputs
    ]
    assert view_state.experience_labels == [
        item.kind
        for item in frame.episode.inputs
    ]
    np.testing.assert_allclose(view_state.expected_point, correction.expected_point)
    np.testing.assert_allclose(view_state.actual_point, correction.actual_point)
    np.testing.assert_allclose(view_state.after_point, correction.after_point)
    assert view_state.surprise == correction.surprise
    assert view_state.learning_rate == correction.learning_rate
    assert view_state.expected_attention == pytest.approx(
        effective["internal_expectation"]
    )
    assert view_state.actual_attention == pytest.approx(effective["external_input"])


def test_load_experience_document_from_json(tmp_path) -> None:
    path = tmp_path / "experience.json"
    path.write_text(
        json.dumps(
            {
                "name": "minimal",
                "vocabulary": ["energy", "hue"],
                "objects": [
                    {
                        "id": "evt_a",
                        "start": 0.0,
                        "end": 1.0,
                        "features": {"energy": 0.75, "hue": 0.2},
                        "salience": 0.8,
                        "learning_weight": 1.7,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    document = load_experience_document(path)

    assert document.name == "minimal"
    assert document.vocabulary == ["energy", "hue"]
    assert document.sequence.objects[0].id == "evt_a"
    assert document.sequence.objects[0].features.value("energy") == 0.75
    assert document.sequence.objects[0].learning_weight == 1.7


def test_load_experience_document_resolves_authored_qualities(tmp_path) -> None:
    path = tmp_path / "experience.json"
    path.write_text(
        json.dumps(
            {
                "vocabulary": ["energy"],
                "objects": [
                    {
                        "id": "evt_painful",
                        "start": 0.0,
                        "end": 1.0,
                        "features": {"energy": 1.0},
                        "qualities": {"pain": 1.0},
                    },
                    {
                        "id": "evt_pleasant",
                        "start": 1.0,
                        "end": 2.0,
                        "features": {"energy": 1.0},
                        "qualities": {"pleasure": 1.0},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    document = load_experience_document(path)
    painful, pleasant = document.sequence.objects

    assert painful.salience > pleasant.salience
    assert painful.learning_weight > pleasant.learning_weight
    assert painful.metadata["qualities"] == {"pain": 1.0}
    assert painful.metadata["affect"]["pain"] == pytest.approx(1.0)
    assert pleasant.metadata["affect"]["pleasure"] == pytest.approx(1.0)


def test_load_experience_document_keeps_explicit_effect_overrides(tmp_path) -> None:
    path = tmp_path / "experience.json"
    path.write_text(
        json.dumps(
            {
                "vocabulary": ["energy"],
                "objects": [
                    {
                        "id": "evt_authored_override",
                        "start": 0.0,
                        "end": 1.0,
                        "features": {"energy": 1.0},
                        "qualities": {"pain": 1.0},
                        "salience": 0.2,
                        "learning_weight": 1.05,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    obj = load_experience_document(path).sequence.objects[0]

    assert obj.salience == pytest.approx(0.2)
    assert obj.learning_weight == pytest.approx(1.05)
    assert obj.metadata["resolved_effects"]["salience"] > obj.salience


def test_pipeline_payload_accepts_loaded_experience(tmp_path) -> None:
    path = tmp_path / "experience.json"
    path.write_text(
        json.dumps(
            {
                "vocabulary": ["energy"],
                "objects": [
                    {
                        "id": "evt_energy",
                        "start": 0.0,
                        "end": 0.4,
                        "features": {"energy": 1.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    document = load_experience_document(path)
    model = model_for_sequence(
        document.sequence,
        vocabulary=document.vocabulary,
    )

    episode = CaveProducer(model).run(dt=0.2)
    payload = run_payload(episode)

    assert payload["vocabulary"] == ["energy"]
    assert payload["inputs"][0]["id"] == "evt_energy"
    assert payload["frames"][0]["active_input_ids"] == ["evt_energy"]


def test_cave_episode_source_maps_scene_state_to_episode() -> None:
    episode = CaveProducer(demo_model(seed=1)).run(dt=0.2)

    assert episode.source_name == "cave"
    assert [item.id for item in episode.inputs] == [
        "evt_triangle",
        "evt_circle",
        "evt_square",
        "evt_gap",
    ]
    assert episode.observations
    first = episode.observations[0]
    assert first.active_inputs == ["evt_triangle"]
    assert first.metadata["effective_attention"]["external_input"] == pytest.approx(
        first.metadata["attention_effect"]
        * first.metadata["attention_channels"]["visual"]
    )
    assert first.metadata["effective_attention"]["internal_expectation"] == pytest.approx(
        first.metadata["attention_effect"]
        * first.metadata["attention_channels"][INTERNAL_EXPECTATION_CHANNEL]
    )
    np.testing.assert_allclose(
        first.actual,
        episode.inputs[0].features
        * episode.inputs[0].salience
        * attention_effect(first.attention)
        * first.metadata["attention_channels"]["visual"],
    )
    np.testing.assert_allclose(first.error, first.actual - first.expected)


def test_episode_structural_state_recomputes_topology_and_views() -> None:
    episode = CaveProducer(demo_model(seed=1)).run(dt=0.2)
    structural = structural_state_for_episode(episode)
    frames = episode_frames(episode, structural)

    assert len(frames) == len(episode.observations)
    assert structural.topology_frames[0].topology.density.shape == (72, 72)
    view_state = PresentationView().project(frames[0])

    assert view_state.items
    assert view_state.items[0].source_id == "evt_triangle"
    assert view_state.items[0].opacity == pytest.approx(
        frames[0].observation.metadata["attention_channel_impacts"]["visual"]
    )


def test_episode_payload_contains_episode_centered_frames() -> None:
    episode = CaveProducer(demo_model(seed=1)).run(dt=0.5)
    payload = episode_payload(episode)

    assert payload["source_name"] == "cave"
    assert payload["inputs"][0]["id"] == "evt_triangle"
    assert "actual" in payload["frames"][0]
    assert "topology" in payload["frames"][0]
    assert "expected_density" in payload["frames"][0]["topology"]
    assert "actual_density" in payload["frames"][0]["topology"]
    assert "views" in payload["frames"][0]


def test_views_from_names_selects_requested_pipeline_views() -> None:
    views = views_from_names("presentation,timeline")

    assert [view.name for view in views] == ["presentation", "timeline"]


def test_views_from_names_rejects_removed_subject_surface_renderer() -> None:
    with pytest.raises(ValueError, match="unsupported view 'subject_surface'"):
        views_from_names("subject_surface")
