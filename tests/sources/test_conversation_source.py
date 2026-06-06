from __future__ import annotations

import numpy as np
import pytest

from cave.observation.experience import TextPresentation
from cave.observation.pipeline import episode_payload
from cave.observation.producers.sources.conversation import (
    ConversationSegment,
    build_conversation_episode,
    select_active_segments,
)
from cave.observation.structural import episode_frames, structural_state_for_episode
from cave.observation.views import MemoryLookbackView, PresentationView


def test_attended_top_k_selects_prior_segments() -> None:
    selected = select_active_segments(
        np.array([0.1, 0.7, 0.2]),
        3,
        mode="attended_top_k",
        top_k=2,
    )

    assert selected.positions == [1, 2]
    assert selected.retained_mass == pytest.approx(0.9)
    np.testing.assert_allclose(selected.weights, np.array([0.7 / 0.9, 0.2 / 0.9]))


def test_conversation_episode_builder_maps_turns_to_mock_memory_context() -> None:
    episode = fake_conversation_episode()

    assert episode.source_name == "conversation"
    assert episode.metadata["adapter"] == "ConversationProducer"
    assert episode.metadata["memory_interpretation"] == "mock_prior_context"
    assert episode.metadata["presentation_mode"] == "current_conversation_segment"
    assert episode.metadata["lookback_mode"] == "conversation_mock_memory"
    assert [item.id for item in episode.inputs] == ["turn:0", "turn:1", "turn:2"]
    assert episode.inputs[1].kind.startswith("assistant:")
    assert isinstance(episode.inputs[1].presentation, TextPresentation)
    assert "Protocol memory" in episode.inputs[1].presentation.text

    assert len(episode.observations) == 2
    first = episode.observations[0]
    assert first.t == 1.0
    assert first.metadata["segment_id"] == "turn:1"
    assert first.metadata["segment_role"] == "assistant"
    assert first.active_inputs == ["turn:0"]
    assert sum(first.attention_weights.values()) == pytest.approx(1.0)
    assert first.metadata["top_predictions"]
    assert first.surprise > 0.0
    np.testing.assert_allclose(first.error, first.actual - first.expected)

    second = episode.observations[1]
    assert second.metadata["segment_id"] == "turn:2"
    assert set(second.active_inputs).issubset({"turn:0", "turn:1"})

    structural = structural_state_for_episode(episode)
    frames = episode_frames(episode, structural)
    payload = episode_payload(episode)

    assert len(frames) == 2
    assert payload["source_name"] == "conversation"
    assert payload["frames"][0]["views"]

    presentation = PresentationView().project(frames[0])
    assert [item.source_id for item in presentation.items] == ["turn:1"]

    lookback = MemoryLookbackView(min_strength=0.0).project(frames[0])
    assert lookback.title == "Conversation / Mock Memory"
    assert [item.source_id for item in lookback.items] == ["turn:0"]


def test_conversation_episode_builder_requires_two_segments() -> None:
    with pytest.raises(ValueError, match="at least two segments"):
        build_conversation_episode(
            source_name="conversation",
            segments=[
                ConversationSegment(
                    id="turn:0",
                    role="user",
                    text="Only one.",
                    formatted_text="User: Only one.\n",
                    start_token=0,
                    end_token=1,
                    order_index=0,
                )
            ],
            token_ids=[0],
            embedding_matrix=np.ones((2, 2)),
            logits=np.ones((1, 2)),
            hidden_states=np.ones((1, 2)),
            attentions=np.ones((1, 1, 1, 1)),
            feature_count=2,
        )


def fake_conversation_episode():
    segments = [
        ConversationSegment(
            id="turn:0",
            role="user",
            text="Prior context?",
            formatted_text="User: Prior context?\n",
            start_token=0,
            end_token=2,
            order_index=0,
        ),
        ConversationSegment(
            id="turn:1",
            role="assistant",
            text="Protocol memory.",
            formatted_text="Assistant: Protocol memory.\n",
            start_token=2,
            end_token=4,
            order_index=1,
        ),
        ConversationSegment(
            id="turn:2",
            role="user",
            text="Expected actual?",
            formatted_text="User: Expected actual?\n",
            start_token=4,
            end_token=6,
            order_index=2,
        ),
    ]
    token_ids = np.array([0, 1, 2, 3, 1, 4], dtype=int)
    embedding_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    logits = np.array(
        [
            [0.0, 3.0, 1.0, 0.0, -1.0],
            [0.0, 1.0, 4.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 4.0, -1.0],
            [0.0, 4.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 4.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    hidden_states = np.array(
        [
            [0.3, 0.1, 0.0, 0.0],
            [0.2, 0.4, 0.0, 0.0],
            [0.0, 0.2, 0.6, 0.0],
            [0.0, 0.1, 0.3, 0.8],
            [0.5, 0.5, 0.1, 0.0],
            [0.6, 0.7, 0.0, 0.2],
        ],
        dtype=float,
    )
    attentions = np.zeros((1, 2, 6, 6), dtype=float)
    for head in range(2):
        for query in range(6):
            attentions[0, head, query, : query + 1] = 1.0 / (query + 1)
    attentions[0, 0, 4, 2:4] += 0.2
    attentions[0, 1, 5, 2:4] += 0.2

    return build_conversation_episode(
        source_name="conversation",
        backend_name="fake-gpt2",
        segments=segments,
        token_ids=token_ids,
        embedding_matrix=embedding_matrix,
        logits=logits,
        hidden_states=hidden_states,
        attentions=attentions,
        feature_count=3,
        context_selection="attended_top_k",
        context_top_k=2,
        top_prediction_k=2,
        decode_token=lambda token_id: f"tok{token_id}",
    )
