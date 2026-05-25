from __future__ import annotations

from dataclasses import replace

import numpy as np

from cave.demonstrations.examples import (
    DEFAULT_VOCABULARY,
    default_model_params,
    random_experience_sequence,
)
from cave.commitments.attention import AttentionProfile
from cave.commitments.memory import (
    MemoryTrace,
)
from cave.demonstrations.state import SubjectState
from cave.commitments.topology import SubjectiveTopologyParams, SubjectiveTopologyPrior
from cave.observation.experience import ExperienceObject, FeatureVector, InputSequence, TemporalExtent
from cave.observation.sensing import FeatureSensor, Sensorium
from cave.demonstrations.subjects import (
    embedding_distance,
    state_effect_embedding,
    expected_input_trajectory_embedding,
    subjective_trajectory_embedding,
    classical_mds,
    make_subject,
    memory_trajectory_embedding,
    nearest_neighbors,
    pairwise_distance_matrix,
    prediction_error_trajectory_embedding,
    run_subject,
    save_subject_comparison_dashboard,
    save_episode_comparison_dashboard,
    surprise_learning_trajectory_embedding,
    threshold_clusters,
)
from cave.observation.episode_runs import LabeledEpisode
from cave.observation.structural import (
    state_effect_embedding as episode_state_effect_embedding,
    memory_trajectory_embedding as episode_memory_trajectory_embedding,
)


def test_subject_sensorium_controls_domain_response() -> None:
    params = replace(
        default_model_params(),
        attention=AttentionProfile(
            mode="constant",
            level=1.0,
            channel_weights={"audio": 1.0},
        ),
        topology=SubjectiveTopologyParams(prior=SubjectiveTopologyPrior()),
    )
    sequence = InputSequence(
        objects=[
            ExperienceObject(
                id="doorbell",
                temporal_extent=TemporalExtent(0.0, 1.0, 0),
                features=FeatureVector({"energy": 1.0}),
                modality="audio",
            )
        ]
    )
    visual_only = run_subject(
        sequence,
        make_subject("visual", params=params, vocabulary=["energy"]),
        dt=0.5,
    )
    audio_subject = make_subject(
        "audio",
        params=params,
        vocabulary=["energy"],
        sensorium=Sensorium(
            sensors=(FeatureSensor(modality="audio", channel="audio"),)
        ),
    )
    audio_run = run_subject(sequence, audio_subject, dt=0.5)

    np.testing.assert_allclose(visual_only.episode.observations[0].actual, np.array([0.0]))
    np.testing.assert_allclose(audio_run.episode.observations[0].actual, np.array([1.0]))


def test_zero_attention_collapses_different_sequences_to_same_state_effect() -> None:
    params = replace(
        default_model_params(),
        attention=AttentionProfile(mode="constant", level=0.0),
        topology=SubjectiveTopologyParams(prior=SubjectiveTopologyPrior()),
    )
    subject = make_subject("zero", params=params)
    first = run_subject(
        random_experience_sequence(count=5, seed=1),
        subject,
        dt=0.2,
        end=3.0,
    )
    second = run_subject(
        random_experience_sequence(count=5, seed=2),
        subject,
        dt=0.2,
        end=3.0,
    )

    distance = embedding_distance(
        state_effect_embedding(first),
        state_effect_embedding(second),
    )

    assert distance == 0.0


def test_same_sequence_with_different_attention_diverges() -> None:
    sequence = random_experience_sequence(count=5, seed=3)
    zero_params = replace(
        default_model_params(),
        attention=AttentionProfile(mode="constant", level=0.0),
        topology=SubjectiveTopologyParams(prior=SubjectiveTopologyPrior()),
    )
    full_params = replace(
        default_model_params(),
        attention=AttentionProfile(mode="constant", level=1.0),
        topology=SubjectiveTopologyParams(prior=SubjectiveTopologyPrior()),
    )
    zero_run = run_subject(sequence, make_subject("zero", params=zero_params), dt=0.2)
    full_run = run_subject(sequence, make_subject("full", params=full_params), dt=0.2)

    distance = embedding_distance(
        state_effect_embedding(zero_run),
        state_effect_embedding(full_run),
    )

    assert distance > 0.0


def test_different_priors_affect_observed_output_but_not_zero_attention_effect() -> None:
    params = replace(
        default_model_params(),
        attention=AttentionProfile(mode="constant", level=0.0),
        topology=SubjectiveTopologyParams(prior=SubjectiveTopologyPrior()),
    )
    vector = np.zeros(len(DEFAULT_VOCABULARY), dtype=float)
    prior_vector = np.linspace(0.1, 0.9, len(DEFAULT_VOCABULARY))
    flat_subject = make_subject(
        "flat",
        params=params,
        initial_state=SubjectState.initial(
            MemoryTrace(vector=vector.copy()),
            params.topology,
        ),
    )
    prior_subject = make_subject(
        "prior",
        params=params,
        initial_state=SubjectState.initial(
            MemoryTrace(vector=prior_vector.copy()),
            params.topology,
        ),
    )
    sequence = random_experience_sequence(count=5, seed=4)
    flat_run = run_subject(sequence, flat_subject, dt=0.2, end=3.0)
    prior_run = run_subject(sequence, prior_subject, dt=0.2, end=3.0)

    observed_distance = embedding_distance(
        memory_trajectory_embedding(flat_run),
        memory_trajectory_embedding(prior_run),
    )
    effect_distance = embedding_distance(
        state_effect_embedding(flat_run),
        state_effect_embedding(prior_run),
    )

    assert observed_distance > 0.0
    assert effect_distance <= 1e-12


def test_pairwise_distances_and_nearest_neighbors() -> None:
    params = replace(
        default_model_params(),
        attention=AttentionProfile(mode="constant", level=1.0),
        topology=SubjectiveTopologyParams(prior=SubjectiveTopologyPrior()),
    )
    subject = make_subject("subject", params=params)
    runs = [
        run_subject(random_experience_sequence(count=4, seed=seed), subject, dt=0.2)
        for seed in [10, 11, 12]
    ]

    distances = pairwise_distance_matrix(
        runs,
        lambda run: state_effect_embedding(run, samples=32),
    )
    neighbors = nearest_neighbors(
        runs,
        lambda run: state_effect_embedding(run, samples=32),
        k=1,
    )

    assert distances.shape == (3, 3)
    np.testing.assert_allclose(np.diag(distances), np.zeros(3))
    assert len(neighbors) == 3
    assert all(len(row) == 1 for row in neighbors)


def test_subjective_trajectory_embeddings_include_prediction_state() -> None:
    params = replace(
        default_model_params(),
        attention=AttentionProfile(mode="constant", level=1.0),
        topology=SubjectiveTopologyParams(prior=SubjectiveTopologyPrior()),
    )
    run = run_subject(
        random_experience_sequence(count=3, seed=14),
        make_subject("subject", params=params),
        dt=0.2,
    )

    expected = expected_input_trajectory_embedding(run, samples=8)
    error = prediction_error_trajectory_embedding(run, samples=8)
    surprise_learning = surprise_learning_trajectory_embedding(run, samples=8)
    internal = subjective_trajectory_embedding(run, samples=8)

    assert expected.shape == (8, len(DEFAULT_VOCABULARY))
    assert error.shape == (8, len(DEFAULT_VOCABULARY))
    assert surprise_learning.shape == (8, 2)
    assert internal.shape == (8, 2 * len(DEFAULT_VOCABULARY) + 2)
    np.testing.assert_allclose(
        internal,
        np.concatenate([expected, error, surprise_learning], axis=1),
    )


def test_threshold_clusters_groups_identical_state_effects() -> None:
    zero_params = replace(
        default_model_params(),
        attention=AttentionProfile(mode="constant", level=0.0),
        topology=SubjectiveTopologyParams(prior=SubjectiveTopologyPrior()),
    )
    full_params = replace(
        default_model_params(),
        attention=AttentionProfile(mode="constant", level=1.0),
        topology=SubjectiveTopologyParams(prior=SubjectiveTopologyPrior()),
    )
    zero_subject = make_subject("zero", params=zero_params)
    full_subject = make_subject("full", params=full_params)
    runs = [
        run_subject(random_experience_sequence(count=4, seed=20), zero_subject, dt=0.2),
        run_subject(random_experience_sequence(count=4, seed=21), zero_subject, dt=0.2),
        run_subject(random_experience_sequence(count=4, seed=20), full_subject, dt=0.2),
    ]

    clusters = threshold_clusters(
        runs,
        lambda run: state_effect_embedding(run, samples=32),
        threshold=1e-12,
    )

    assert [0, 1] in clusters
    assert [2] in clusters


def test_classical_mds_returns_two_dimensional_coordinates() -> None:
    distances = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )

    coords = classical_mds(distances)

    assert coords.shape == (3, 2)


def test_save_subject_comparison_dashboard(tmp_path) -> None:
    params = replace(
        default_model_params(),
        attention=AttentionProfile(mode="constant", level=1.0),
        topology=SubjectiveTopologyParams(prior=SubjectiveTopologyPrior()),
    )
    subject = make_subject("subject", params=params)
    runs = [
        run_subject(random_experience_sequence(count=3, seed=seed), subject, dt=0.2)
        for seed in [30, 31]
    ]
    labels = ["Q0-subject", "Q1-subject"]
    output = tmp_path / "dashboard.png"

    save_subject_comparison_dashboard(
        runs,
        labels,
        output,
        effect_embedding=lambda run: state_effect_embedding(run, samples=16),
        observed_embedding=lambda run: memory_trajectory_embedding(run, samples=16),
        internal_embedding=lambda run: subjective_trajectory_embedding(run, samples=16),
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_episode_comparison_dashboard_accepts_labeled_episodes(tmp_path) -> None:
    params = replace(
        default_model_params(),
        attention=AttentionProfile(mode="constant", level=1.0),
        topology=SubjectiveTopologyParams(prior=SubjectiveTopologyPrior()),
    )
    subject = make_subject("subject", params=params)
    subject_runs = [
        run_subject(random_experience_sequence(count=3, seed=seed), subject, dt=0.2)
        for seed in [40, 41]
    ]
    runs = [
        LabeledEpisode(
            id=run.id,
            episode=run.episode,
            label=f"episode-{index}",
            group="generic",
        )
        for index, run in enumerate(subject_runs)
    ]
    output = tmp_path / "episode_dashboard.png"

    save_episode_comparison_dashboard(
        runs,
        output,
        effect_embedding=lambda run: episode_state_effect_embedding(
            run.episode,
            samples=16,
        ),
        observed_embedding=lambda run: episode_memory_trajectory_embedding(
            run.episode,
            samples=16,
        ),
    )

    assert output.exists()
    assert output.stat().st_size > 0
