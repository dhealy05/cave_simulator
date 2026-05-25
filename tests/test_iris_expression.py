from __future__ import annotations

import numpy as np

from cave.presentation.renderers.matplotlib_renderer.iris import (
    build_iris_frames,
    save_iris_expression_animation,
)
from cave.substrates.evolved_subject import (
    EvolvedSubject,
    EvolvedSubjectConfig,
    evolved_episode_from_run,
    exposure_control_sequence,
    genome_size,
    run_evolved_subject,
)


def _cheap_evolved_episode():
    # A random (un-evolved) genome is enough to exercise the renderer without
    # paying for evolution; the smoke test only checks that frames + gif build.
    config = EvolvedSubjectConfig()
    rng = np.random.default_rng(0)
    genome = rng.normal(0.0, 0.5, size=genome_size(config))
    subject = EvolvedSubject(genome=genome, config=config)
    sequence = exposure_control_sequence(cycles=4, seed=1, structured=True)
    run = run_evolved_subject(subject, sequence)
    return evolved_episode_from_run(run)


def test_build_iris_frames_maps_phases_and_aperture() -> None:
    episode = _cheap_evolved_episode()
    frames = build_iris_frames(episode)

    assert len(frames) == len(episode.observations)
    assert {frame.phase for frame in frames} >= {"cue", "gap", "outcome"}
    assert all(0.0 <= frame.aperture <= 1.0 for frame in frames)
    assert all(-1.0 <= frame.charge <= 1.0 for frame in frames)
    # outcome frames should carry a signed external valence
    assert any(frame.phase == "outcome" and frame.valence != 0.0 for frame in frames)


def test_save_iris_expression_animation_writes_gif(tmp_path) -> None:
    episode = _cheap_evolved_episode()
    output = tmp_path / "iris_expression.gif"
    save_iris_expression_animation(episode, output, fps=4, max_frames=12)

    assert output.exists()
    assert output.stat().st_size > 0
