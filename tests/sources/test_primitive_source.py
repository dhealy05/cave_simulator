from __future__ import annotations

import numpy as np
import pytest

from cave.observation.producers.sources.primitive import (
    PRIMITIVE_ETA,
    PRIMITIVE_MEMORY_INITIAL,
    PrimitiveProducer,
    primitive_input_sequence,
    rollout_vectors,
)


def test_rollout_vectors_runs_primitive_recurrence() -> None:
    rows = rollout_vectors(
        [(1.0, 0.0), (0.0, 1.0)],
        eta=0.25,
        memory_initial=(0.0, 0.0),
    )

    assert len(rows) == 2
    assert rows[0].expected == (0.0, 0.0)
    assert rows[0].error == (1.0, 0.0)
    assert rows[0].memory == (0.25, 0.0)
    assert rows[1].expected == (0.25, 0.0)
    assert rows[1].error == (-0.25, 1.0)
    assert rows[1].memory == pytest.approx((0.1875, 0.25))


def test_primitive_input_sequence_authors_experience_objects() -> None:
    sequence = primitive_input_sequence(("tree", "snake"))

    assert [obj.kind for obj in sequence.objects] == ["tree", "snake"]
    assert sequence.objects[0].modality == "primitive"
    assert sequence.objects[0].metadata["label"] == "Tree"
    assert sequence.objects[1].metadata["label"] == "Snake"


def test_primitive_producer_emits_common_episode_contract() -> None:
    sequence = primitive_input_sequence(("tree", "snake"))
    episode = PrimitiveProducer(sequence).run()

    assert episode.source_name == "primitive"
    assert episode.vocabulary == ["primitive_x", "primitive_y"]
    assert len(episode.inputs) == 2
    assert len(episode.observations) == 2
    assert episode.metadata["adapter"] == "PrimitiveProducer"

    first = episode.observations[0]
    np.testing.assert_allclose(first.expected, PRIMITIVE_MEMORY_INITIAL)
    np.testing.assert_allclose(first.actual, (0.18, 0.74))
    np.testing.assert_allclose(first.memory_state, (0.191, 0.773))
    assert first.learning_rate == pytest.approx(PRIMITIVE_ETA)
    assert first.active_inputs == ["primitive_000_tree"]

    second = episode.observations[1]
    np.testing.assert_allclose(second.expected, first.memory_state)
    np.testing.assert_allclose(second.error, second.actual - second.expected)
    assert second.surprise > first.surprise
