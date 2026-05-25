from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cave.observation.experience import TextPresentation
from cave.observation.pipeline import episode_payload
from cave.observation.producers.sources.gpt2 import (
    GPT2Producer,
    attention_concentration,
    build_gpt2_episode,
    fit_episode_projection,
    select_active_context,
)
from cave.observation.structural import episode_frames, structural_state_for_episode
from cave.observation.views import MemoryLookbackView


def test_episode_projection_names_and_normalizes_features() -> None:
    vectors = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    projection = fit_episode_projection(vectors, feature_count=4)
    projected = projection.project(vectors)

    assert projection.feature_names == ["pc1", "pc2", "pc3", "pc4"]
    assert projected.shape == (3, 4)
    assert np.all(projected >= 0.0)
    assert np.all(projected <= 1.0)
    np.testing.assert_allclose(projected[:, 3], np.full(3, 0.5))


def test_attention_concentration_tracks_focus() -> None:
    focused = attention_concentration(np.array([0.98, 0.01, 0.01]))
    diffuse = attention_concentration(np.array([1.0, 1.0, 1.0]))

    assert focused > diffuse
    assert diffuse == pytest.approx(0.0)


def test_attended_top_k_selects_and_renormalizes_context() -> None:
    selected = select_active_context(
        np.array([0.1, 0.6, 0.2, 0.1]),
        3,
        mode="attended_top_k",
        top_k=2,
    )

    assert selected.positions == [1, 2]
    assert selected.retained_mass == pytest.approx(0.8)
    np.testing.assert_allclose(selected.weights, np.array([0.75, 0.25]))


def test_gpt2_episode_builder_maps_fake_forward_pass() -> None:
    token_ids = np.array([0, 1, 2], dtype=int)
    token_texts = ["Hello", " Paul", "!"]
    embedding_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    logits = np.array(
        [
            [0.0, 3.0, 1.0, -1.0],
            [0.0, -1.0, 4.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    hidden_states = np.array(
        [
            [0.2, 0.1, 0.0],
            [0.0, 0.5, 0.2],
            [0.1, 0.0, 0.7],
        ],
        dtype=float,
    )
    attentions = np.array(
        [
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.8, 0.2, 0.0],
                    [0.2, 0.3, 0.5],
                ],
                [
                    [1.0, 0.0, 0.0],
                    [0.6, 0.4, 0.0],
                    [0.1, 0.7, 0.2],
                ],
            ]
        ],
        dtype=float,
    )

    episode = build_gpt2_episode(
        source_name="gpt2",
        token_ids=token_ids,
        token_texts=token_texts,
        embedding_matrix=embedding_matrix,
        logits=logits,
        hidden_states=hidden_states,
        attentions=attentions,
        feature_count=2,
        active_top_k=2,
        top_prediction_k=2,
        decode_token=lambda token_id: f"tok{token_id}",
    )

    assert episode.source_name == "gpt2"
    assert episode.vocabulary == ["pc1", "pc2"]
    assert episode.metadata["presentation_mode"] == "current_text"
    assert episode.metadata["lookback_mode"] == "attention_context"
    assert [item.id for item in episode.inputs] == ["tok:0", "tok:1", "tok:2"]
    assert isinstance(episode.inputs[1].presentation, TextPresentation)
    assert episode.inputs[1].presentation.text == "Paul"
    assert len(episode.observations) == 2
    first = episode.observations[0]
    assert first.t == 1.0
    assert first.metadata["predicted_from_position"] == 0
    assert first.metadata["token_id"] == 1
    assert first.metadata["top_predictions"][0]["token_id"] == 1
    assert first.metadata["actual_token_probability"] > 0.0
    assert sum(first.attention_weights.values()) == pytest.approx(1.0)

    structural = structural_state_for_episode(episode)
    frames = episode_frames(episode, structural)
    payload = episode_payload(episode)

    assert len(frames) == 2
    assert payload["source_name"] == "gpt2"
    assert payload["frames"][0]["views"]

    lookback = MemoryLookbackView(min_strength=0.0).project(frames[0])
    assert lookback.title == "Context / Temporal Lookback"
    assert lookback.items
    assert all(0.0 <= item.strength <= 1.0 for item in lookback.items)


def test_gpt2_episode_builder_requires_two_tokens() -> None:
    with pytest.raises(ValueError, match="at least two tokens"):
        build_gpt2_episode(
            source_name="gpt2",
            token_ids=[0],
            token_texts=["Hello"],
            embedding_matrix=np.ones((1, 2)),
            logits=np.ones((1, 1)),
            hidden_states=np.ones((1, 2)),
            attentions=np.ones((1, 1, 1, 1)),
            feature_count=2,
        )


def test_local_gpt2_source_smoke() -> None:
    model_path = Path("lib/models/gpt2")
    if not model_path.exists():
        pytest.skip("local GPT-2 model is not installed")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    episode = GPT2Producer(model_path, feature_count=4, active_top_k=3).run(
        "Hello, my name is Paul and I like to "
    )
    structural = structural_state_for_episode(episode)
    frames = episode_frames(episode, structural)
    payload = episode_payload(episode)

    assert episode.source_name == "gpt2"
    assert episode.observations
    assert frames
    assert payload["source_name"] == "gpt2"
