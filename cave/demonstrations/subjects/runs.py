from __future__ import annotations

from dataclasses import dataclass

from cave.observation.episode_runs import LabeledEpisode
from cave.observation.episodes import Episode, episode_from_cave_states
from cave.observation.experience import InputSequence
from cave.demonstrations.simulation import ExperienceModel
from cave.demonstrations.subjects.profiles import SubjectProfile


@dataclass(frozen=True)
class SubjectRun:
    id: str
    subject: SubjectProfile
    sequence: InputSequence
    episode: Episode

    def as_labeled_episode(self, label: str | None = None) -> LabeledEpisode:
        return LabeledEpisode(
            id=self.id,
            episode=self.episode,
            label=label,
            group=self.subject.id,
            metadata={
                "subject_id": self.subject.id,
                "sequence_length": len(self.sequence.objects),
            },
        )


def run_subject(
    sequence: InputSequence,
    subject: SubjectProfile,
    *,
    dt: float = 0.1,
    start: float = 0.0,
    end: float | None = None,
    run_id: str | None = None,
) -> SubjectRun:
    end = sequence.duration if end is None else end
    model = ExperienceModel(
        sequence=sequence,
        subject_state=subject.fresh_state(),
        params=subject.params,
        vocabulary=list(subject.vocabulary),
        sensorium=subject.sensorium,
    )
    states = model.run(start=start, end=end, dt=dt)
    episode = episode_from_cave_states(
        subject.id,
        sequence,
        list(subject.vocabulary),
        states,
        metadata={
            "source": "cave.demonstrations.simulation",
            "subject_id": subject.id,
            "memory_decay_tau": subject.params.memory.decay_tau,
            "memory_max_age": subject.params.memory.max_age,
            "memory_retention": subject.params.memory.retention,
            "topology_params": subject.params.topology,
        },
    )
    return SubjectRun(
        id=run_id or f"{subject.id}:{len(sequence.objects)}:{start:g}-{end:g}",
        subject=subject,
        sequence=sequence,
        episode=episode,
    )
