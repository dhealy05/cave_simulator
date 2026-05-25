"""Smoke tests for the hard 'needle in the delay' exposure world.

Covers the vocab/world-factory refactor (backward compatibility) and the new
dissociated_exposure_sequence world generator. The discriminating probes and
report wiring are added in a later step; this only asserts the world is built
correctly and the substrate can evolve on it through the existing GA.
"""
from __future__ import annotations

import numpy as np

from cave.substrates.evolved_subject import (
    EVOLVED_HARD_VOCABULARY,
    EVOLVED_VOCABULARY,
    EvolutionConfig,
    EvolvedSubjectConfig,
    dissociated_exposure_sequence,
    evolve_subject,
    evolved_episode_from_run,
    genome_size,
    run_evolved_subject,
)


def _hard_world(seed: int):
    return dissociated_exposure_sequence(cycles=10, seed=seed, delay_steps=3)


def test_default_vocabulary_is_backward_compatible():
    cfg = EvolvedSubjectConfig(hidden_dim=5)
    assert tuple(cfg.vocabulary) == tuple(EVOLVED_VOCABULARY)
    # 5*input + 5*5 + 5 + 5 + 1 with a 5-channel vocab -> unchanged genome size
    assert genome_size(cfg) == 5 * 5 + 5 * 5 + 5 + 5 + 1


def test_genome_size_tracks_configured_vocabulary():
    hard = EvolvedSubjectConfig(hidden_dim=6, vocabulary=tuple(EVOLVED_HARD_VOCABULARY))
    hidden, inp = 6, len(EVOLVED_HARD_VOCABULARY)
    assert genome_size(hard) == hidden * inp + hidden * hidden + hidden + hidden + 1


def test_dissociated_world_structure():
    seq = dissociated_exposure_sequence(cycles=60, seed=1, delay_steps=3)
    classes = {o.metadata.get("cue_class") for o in seq.objects if "cue" in o.id}
    assert classes == {"rare", "common"}

    # distractors are present and salient in the delay
    assert any(
        o.id.startswith("delay_") and o.features.value("distractor") > 0.0
        for o in seq.objects
    )

    # frequency is anti-correlated with value: rare carries a larger |outcome|
    rare = [abs(o.metadata["outcome_value"]) for o in seq.objects
            if o.metadata.get("cue_class") == "rare" and o.metadata["outcome_value"] != 0.0]
    common = [abs(o.metadata["outcome_value"]) for o in seq.objects
              if o.metadata.get("cue_class") == "common" and o.metadata["outcome_value"] != 0.0]
    assert rare and common
    assert min(rare) > max(common)
    # ...and common is the more frequent cue
    n_rare = sum(1 for o in seq.objects if o.metadata.get("cue_class") == "rare" and "cue" in o.id)
    n_common = sum(1 for o in seq.objects if o.metadata.get("cue_class") == "common" and "cue" in o.id)
    assert n_common > n_rare


def test_controls_degrade_the_world():
    # no-distractor control: delay slots carry no distractor channel
    clean = dissociated_exposure_sequence(cycles=10, seed=2, delay_steps=3, distractors=False)
    assert all(o.features.value("distractor") == 0.0 for o in clean.objects)


def test_evolve_on_hard_world_runs_and_serializes():
    subject_config = EvolvedSubjectConfig(
        hidden_dim=6, recurrent=True, vocabulary=tuple(EVOLVED_HARD_VOCABULARY)
    )
    result = evolve_subject(
        subject_config=subject_config,
        evolution_config=EvolutionConfig(
            seed=3, generations=4, population_size=8, elite_count=2, world_count=4
        ),
        world_factory=_hard_world,
    )
    assert result.subject.genome.shape == (genome_size(subject_config),)
    assert np.isfinite(result.fitness_history[-1])

    run = run_evolved_subject(result.subject, _hard_world(99))
    episode = evolved_episode_from_run(run)
    assert episode.vocabulary == list(EVOLVED_HARD_VOCABULARY)
    assert len(episode.observations) == len(run.sequence.objects)
    assert np.isfinite(run.total_utility)

    # cue_class flows through to observation metadata for the per-cue probes
    seen = {
        obs.metadata["evolved_subject"]["cue_class"] for obs in episode.observations
    }
    assert {"rare", "common"} & seen


def test_check_runs_and_controls_collapse():
    from cave.pressure.tests.evolved_dissociation import check_evolved_dissociation

    res = check_evolved_dissociation(
        generations=12, population_size=16, world_count=6, eval_world_count=12
    )
    assert set(res["roles"]) == {
        "value_retention",
        "exposure_regulation",
        "control_collapse",
        "magnitude_selectivity",
    }
    # Mechanistic invariants that hold even when tiny smoke params under-train:
    # the frequency-counter is strongly negative so any learning beats it, and
    # the capacity-removing controls structurally collapse. (The full retention
    # threshold, rare_future_accuracy >= 0.8, is validated by the suite run at
    # real generations, not by this fast smoke test.)
    assert res["roles"]["value_retention"]["utility_gain_over_frequency_counter"] > 0.0
    assert res["roles"]["control_collapse"]["reset_rare_contrast"] < 0.1
    assert res["roles"]["control_collapse"]["nonrecurrent_rare_accuracy"] <= 0.7
    # the magnitude-selectivity readout is reported descriptively
    assert "common_exposure_contrast" in res["roles"]["magnitude_selectivity"]
