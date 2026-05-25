from __future__ import annotations

from cave.observation.episodes import (
    CaveEpisodeSource,
    CaveProducer,
    Episode,
    EpisodeProducer,
    EpisodeSource,
)
from cave.presentation.runs import ExperienceRun, slugify
from cave.observation.producers.sources.gpt2 import GPT2EpisodeSource, GPT2Producer


def test_experience_run_output_dir_groups_by_source_and_run_id() -> None:
    episode = Episode(
        source_name="gpt2",
        vocabulary=["pc1", "pc2"],
        inputs=[],
        observations=[],
        duration=0.0,
    )
    run = ExperienceRun(id="Hello, Paul!", episode=episode)

    assert slugify("Hello, Paul!") == "hello-paul"
    assert run.output_dir("out/episodes").as_posix() == "out/episodes/gpt2/hello-paul"


def test_producer_names_keep_source_aliases_compatible() -> None:
    assert CaveProducer is CaveEpisodeSource
    assert GPT2Producer is GPT2EpisodeSource
    assert EpisodeProducer is EpisodeSource
