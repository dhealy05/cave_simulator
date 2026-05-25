from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Literal

import numpy as np

from cave.observation.episodes import Episode, EpisodeInput, EpisodeObservation
from cave.observation.experience import ExperienceObject, FeatureVector, InputSequence, TemporalExtent, presentation_for_object


Array = np.ndarray

EVOLVED_VOCABULARY = ["cue_good", "cue_bad", "good", "bad", "neutral"]
EVOLVED_HARD_VOCABULARY = [
    "cue_rare_good",
    "cue_rare_bad",
    "cue_common_good",
    "cue_common_bad",
    "distractor",
    "good",
    "bad",
    "neutral",
]
OutcomeLabel = Literal["good", "bad", "neutral"]


@dataclass(frozen=True)
class EvolvedSubjectConfig:
    hidden_dim: int = 5
    exposure_cost: float = 0.05
    action_change_cost: float = 0.02
    recurrent: bool = True
    vocabulary: tuple[str, ...] = tuple(EVOLVED_VOCABULARY)

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.exposure_cost < 0.0:
            raise ValueError("exposure_cost must be non-negative")
        if self.action_change_cost < 0.0:
            raise ValueError("action_change_cost must be non-negative")


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = 48
    elite_count: int = 8
    generations: int = 80
    mutation_sigma: float = 0.18
    world_count: int = 20
    cycles_per_world: int = 8
    seed: int = 17

    def __post_init__(self) -> None:
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if not 0 < self.elite_count <= self.population_size:
            raise ValueError("elite_count must be in 1..population_size")
        if self.generations <= 0:
            raise ValueError("generations must be positive")
        if self.mutation_sigma < 0.0:
            raise ValueError("mutation_sigma must be non-negative")
        if self.world_count <= 0:
            raise ValueError("world_count must be positive")
        if self.cycles_per_world <= 0:
            raise ValueError("cycles_per_world must be positive")


@dataclass(frozen=True)
class EvolvedSubject:
    genome: Array
    config: EvolvedSubjectConfig

    @property
    def genome_size(self) -> int:
        return genome_size(self.config)

    def weights(self) -> dict[str, Array]:
        return decode_genome(self.genome, self.config)

    def initial_hidden(self) -> Array:
        return np.zeros(self.config.hidden_dim, dtype=float)

    def exposure(self, hidden: Array) -> float:
        weights = self.weights()
        return _sigmoid(float(np.dot(weights["w_a"], hidden) + weights["b_a"]))

    def update_hidden(self, hidden: Array, observation: Array) -> Array:
        weights = self.weights()
        recurrent = 0.0 if not self.config.recurrent else np.dot(weights["w_h"], hidden)
        return np.tanh(np.dot(weights["w_x"], observation) + recurrent + weights["b_h"])


@dataclass(frozen=True)
class EvolvedStep:
    t: float
    input_id: str
    observation: Array
    hidden_before: Array
    hidden_after: Array
    exposure: float
    next_exposure: float
    outcome_value: float
    utility: float
    future_outcome: OutcomeLabel
    cue_class: str = "none"


@dataclass(frozen=True)
class EvolvedRun:
    subject: EvolvedSubject
    sequence: InputSequence
    steps: tuple[EvolvedStep, ...]
    total_utility: float
    metadata: dict[str, object]


@dataclass(frozen=True)
class EvolutionResult:
    subject: EvolvedSubject
    fitness_history: tuple[float, ...]
    random_baseline_fitness: float
    config: EvolutionConfig
    subject_config: EvolvedSubjectConfig


def genome_size(config: EvolvedSubjectConfig) -> int:
    input_dim = len(config.vocabulary)
    hidden = config.hidden_dim
    return hidden * input_dim + hidden * hidden + hidden + hidden + 1


def decode_genome(genome: Array, config: EvolvedSubjectConfig) -> dict[str, Array]:
    genome = np.asarray(genome, dtype=float)
    expected = genome_size(config)
    if genome.shape != (expected,):
        raise ValueError("genome length does not match config")
    input_dim = len(config.vocabulary)
    hidden = config.hidden_dim
    cursor = 0
    w_x = genome[cursor : cursor + hidden * input_dim].reshape(hidden, input_dim)
    cursor += hidden * input_dim
    w_h = genome[cursor : cursor + hidden * hidden].reshape(hidden, hidden)
    cursor += hidden * hidden
    b_h = genome[cursor : cursor + hidden]
    cursor += hidden
    w_a = genome[cursor : cursor + hidden]
    cursor += hidden
    b_a = float(genome[cursor])
    return {"w_x": w_x, "w_h": w_h, "b_h": b_h, "w_a": w_a, "b_a": b_a}


def evolve_subject(
    *,
    subject_config: EvolvedSubjectConfig | None = None,
    evolution_config: EvolutionConfig | None = None,
    world_factory: Callable[[int], InputSequence] | None = None,
) -> EvolutionResult:
    subject_config = subject_config or EvolvedSubjectConfig()
    evolution_config = evolution_config or EvolutionConfig()
    rng = np.random.default_rng(evolution_config.seed)
    size = genome_size(subject_config)
    population = rng.normal(0.0, 0.5, size=(evolution_config.population_size, size))
    if world_factory is None:
        def world_factory(seed: int) -> InputSequence:
            return exposure_control_sequence(
                cycles=evolution_config.cycles_per_world,
                seed=seed,
                structured=True,
            )
    worlds = [
        world_factory(evolution_config.seed + 1000 + index)
        for index in range(evolution_config.world_count)
    ]
    history: list[float] = []
    best_genome = population[0].copy()
    best_fitness = -float("inf")
    for generation in range(evolution_config.generations):
        fitness = np.array(
            [
                evaluate_genome(genome, subject_config, worlds)
                for genome in population
            ],
            dtype=float,
        )
        order = np.argsort(-fitness, kind="stable")
        elites = population[order[: evolution_config.elite_count]]
        if float(fitness[order[0]]) > best_fitness:
            best_fitness = float(fitness[order[0]])
            best_genome = population[order[0]].copy()
        history.append(best_fitness)
        sigma = evolution_config.mutation_sigma * (0.25 + 0.75 * (1.0 - generation / max(1, evolution_config.generations - 1)))
        next_population = [best_genome.copy()]
        while len(next_population) < evolution_config.population_size:
            parent = elites[int(rng.integers(0, len(elites)))]
            next_population.append(parent + rng.normal(0.0, sigma, size=size))
        population = np.asarray(next_population, dtype=float)

    random_subject = EvolvedSubject(
        genome=np.zeros(size, dtype=float),
        config=subject_config,
    )
    return EvolutionResult(
        subject=EvolvedSubject(best_genome, subject_config),
        fitness_history=tuple(history),
        random_baseline_fitness=evaluate_subject(random_subject, worlds),
        config=evolution_config,
        subject_config=subject_config,
    )


def evaluate_genome(
    genome: Array,
    subject_config: EvolvedSubjectConfig,
    worlds: list[InputSequence],
    *,
    reset_each_step: bool = False,
) -> float:
    return evaluate_subject(
        EvolvedSubject(np.asarray(genome, dtype=float), subject_config),
        worlds,
        reset_each_step=reset_each_step,
    )


def evaluate_subject(
    subject: EvolvedSubject,
    worlds: list[InputSequence],
    *,
    reset_each_step: bool = False,
) -> float:
    if not worlds:
        return 0.0
    return float(
        np.mean(
            [
                run_evolved_subject(
                    subject,
                    world,
                    reset_each_step=reset_each_step,
                ).total_utility
                for world in worlds
            ]
        )
    )


def run_evolved_subject(
    subject: EvolvedSubject,
    sequence: InputSequence,
    *,
    reset_each_step: bool = False,
) -> EvolvedRun:
    hidden = subject.initial_hidden()
    previous_exposure = subject.exposure(hidden)
    steps: list[EvolvedStep] = []
    total_utility = 0.0
    for obj in sequence.objects:
        if reset_each_step:
            hidden = subject.initial_hidden()
            previous_exposure = subject.exposure(hidden)
        observation = obj.features.to_array(list(subject.config.vocabulary)) * obj.salience
        exposure = previous_exposure
        outcome_value = float(obj.metadata.get("outcome_value", 0.0))
        utility = (
            exposure * outcome_value
            - subject.config.exposure_cost * exposure
            - subject.config.action_change_cost * abs(exposure - previous_exposure)
        )
        hidden_after = subject.update_hidden(hidden, observation)
        next_exposure = subject.exposure(hidden_after)
        steps.append(
            EvolvedStep(
                t=obj.temporal_extent.start,
                input_id=obj.id,
                observation=observation,
                hidden_before=hidden.copy(),
                hidden_after=hidden_after.copy(),
                exposure=exposure,
                next_exposure=next_exposure,
                outcome_value=outcome_value,
                utility=float(utility),
                future_outcome=str(obj.metadata.get("future_outcome", "neutral")),  # type: ignore[arg-type]
                cue_class=str(obj.metadata.get("cue_class", "none")),
            )
        )
        total_utility += float(utility)
        hidden = hidden_after
        previous_exposure = next_exposure
    return EvolvedRun(
        subject=subject,
        sequence=sequence,
        steps=tuple(steps),
        total_utility=total_utility,
        metadata={
            "total_utility": total_utility,
            "reset_each_step": reset_each_step,
            "subject_config": subject.config,
        },
    )


def evolved_episode_from_run(
    run: EvolvedRun,
    *,
    source_name: str = "evolved-subject",
    metadata: dict[str, object] | None = None,
) -> Episode:
    vocab = list(run.subject.config.vocabulary)
    inputs = [
        EpisodeInput(
            id=obj.id,
            kind=obj.kind,
            start=obj.temporal_extent.start,
            end=obj.temporal_extent.end,
            order_index=obj.temporal_extent.order_index,
            features=obj.features.to_array(vocab),
            modality=obj.modality,
            salience=obj.salience,
            learning_weight=obj.learning_weight,
            presentation=presentation_for_object(obj),
            metadata=dict(obj.metadata),
        )
        for obj in run.sequence.objects
    ]
    duration = run.sequence.duration
    observations = [
        EpisodeObservation(
            t=step.t,
            t_normalized=0.0 if duration <= 0.0 else step.t / duration,
            expected=np.zeros(len(vocab), dtype=float),
            actual=step.observation.copy(),
            memory_state=step.hidden_before.copy(),
            surprise=0.0,
            learning_rate=0.0,
            attention=step.exposure,
            attention_weights={step.input_id: step.exposure},
            active_inputs=[step.input_id],
            input_features={step.input_id: step.observation.copy()},
            metadata={
                "evolved_subject": {
                    "hidden_before": step.hidden_before.copy(),
                    "hidden_after": step.hidden_after.copy(),
                    "exposure": step.exposure,
                    "next_exposure": step.next_exposure,
                    "outcome_value": step.outcome_value,
                    "utility": step.utility,
                    "future_outcome": step.future_outcome,
                    "cue_class": step.cue_class,
                },
                "objective": {
                    "utility": step.utility,
                    "outcome_value": step.outcome_value,
                    "exposure_cost": run.subject.config.exposure_cost * step.exposure,
                    "attention_cost": run.subject.config.exposure_cost * step.exposure,
                    "compression_cost": 0.0,
                    "prediction_cost": 0.0,
                },
            },
        )
        for step in run.steps
    ]
    return Episode(
        source_name=source_name,
        vocabulary=vocab,
        inputs=inputs,
        observations=observations,
        duration=duration,
        metadata={
            "source": "cave.substrates.evolved_subject",
            "adapter": "EvolvedSubject",
            "total_utility": run.total_utility,
            **({} if metadata is None else dict(metadata)),
        },
    )


def exposure_control_sequence(
    *,
    cycles: int = 8,
    seed: int = 0,
    structured: bool = True,
    direct_outcome_visible: bool = False,
) -> InputSequence:
    rng = np.random.default_rng(seed)
    objects: list[ExperienceObject] = []
    t = 0.0
    order = 0
    outcomes = ["good", "bad"]
    for cycle in range(cycles):
        cue_label = outcomes[int(rng.integers(0, 2))]
        outcome_label = cue_label if structured else outcomes[int(rng.integers(0, 2))]
        cue_features = {
            f"cue_{cue_label}": 1.0,
            "good" if direct_outcome_visible and outcome_label == "good" else "neutral": 0.0,
            "bad" if direct_outcome_visible and outcome_label == "bad" else "neutral": 0.0,
        }
        cue_features = {key: value for key, value in cue_features.items() if value != 0.0}
        objects.append(
            _event(
                f"cue_{cue_label}_{cycle}",
                t,
                order,
                cue_features,
                future_outcome=outcome_label,
            )
        )
        t += 1.0
        order += 1
        objects.append(
            _event(
                f"delay_{cycle}",
                t,
                order,
                {"neutral": 1.0},
                future_outcome=outcome_label,
            )
        )
        t += 1.0
        order += 1
        value = 1.0 if outcome_label == "good" else -1.0
        objects.append(
            _event(
                f"{outcome_label}_outcome_{cycle}",
                t,
                order,
                {outcome_label: 1.0},
                outcome_value=value,
            )
        )
        t += 1.0
        order += 1
    return InputSequence(objects)


def dissociated_exposure_sequence(
    *,
    cycles: int = 12,
    seed: int = 0,
    delay_steps: int = 4,
    rare_prob: float = 0.15,
    rare_value: float = 1.0,
    common_value: float = 0.05,
    distractors: bool = True,
    structured: bool = True,
) -> InputSequence:
    """A hard exposure world that frequency- and last-thing-seen reflexes fail.

    Frequency is anti-correlated with value: the ``common`` cue appears most often
    but predicts a near-zero outcome (``common_value``), while the ``rare`` cue is
    infrequent but carries the real outcome (``rare_value``). The delay between
    cue and outcome is filled with salient, non-predictive ``distractor`` events,
    so the predictive cue is never the most recent observation. Winning therefore
    requires selectively retaining the *valuable* cue across noise rather than
    counting frequency or reacting to the last input.

    Each cue/delay/outcome event tags ``metadata['cue_class']`` in
    ``{'rare', 'common'}`` so readouts can be conditioned per class. Use
    ``distractors=False`` for the no-distractor control and ``structured=False``
    for the shuffled-temporal control (cue no longer predicts outcome).
    """
    rng = np.random.default_rng(seed)
    objects: list[ExperienceObject] = []
    t = 0.0
    order = 0
    signs = ["good", "bad"]
    for cycle in range(cycles):
        cue_class = "rare" if bool(rng.random() < rare_prob) else "common"
        sign = signs[int(rng.integers(0, 2))]
        magnitude = rare_value if cue_class == "rare" else common_value
        outcome_label = sign if structured else signs[int(rng.integers(0, 2))]
        outcome_value = magnitude if outcome_label == "good" else -magnitude
        channel = f"cue_{cue_class}_{sign}"
        objects.append(
            _event(
                f"{channel}_{cycle}",
                t,
                order,
                {channel: 1.0},
                future_outcome=outcome_label,
                cue_class=cue_class,
            )
        )
        t += 1.0
        order += 1
        for step in range(delay_steps):
            features = (
                {"distractor": float(rng.uniform(0.6, 1.0))}
                if distractors
                else {"neutral": 1.0}
            )
            objects.append(
                _event(
                    f"delay_{cycle}_{step}",
                    t,
                    order,
                    features,
                    future_outcome=outcome_label,
                    cue_class=cue_class,
                )
            )
            t += 1.0
            order += 1
        objects.append(
            _event(
                f"{outcome_label}_outcome_{cycle}",
                t,
                order,
                {outcome_label: 1.0},
                outcome_value=outcome_value,
                cue_class=cue_class,
            )
        )
        t += 1.0
        order += 1
    return InputSequence(objects)


def evolved_behavior_metrics(episode: Episode) -> dict[str, float]:
    outcome_observations = [
        obs for obs in episode.observations
        if obs.metadata.get("evolved_subject", {}).get("outcome_value", 0.0) != 0.0
    ]
    good = [
        obs.attention for obs in outcome_observations
        if obs.metadata["evolved_subject"]["outcome_value"] > 0.0
    ]
    bad = [
        obs.attention for obs in outcome_observations
        if obs.metadata["evolved_subject"]["outcome_value"] < 0.0
    ]
    utility = [
        float(obs.metadata.get("evolved_subject", {}).get("utility", 0.0))
        for obs in episode.observations
    ]
    return {
        "utility": float(np.sum(utility)),
        "good_exposure": float(np.mean(good)) if good else 0.0,
        "bad_exposure": float(np.mean(bad)) if bad else 0.0,
        "exposure_contrast": (float(np.mean(good)) if good else 0.0) - (float(np.mean(bad)) if bad else 0.0),
    }


def latent_future_outcome_accuracy(episode: Episode) -> float:
    hidden = []
    labels = []
    for obs in episode.observations:
        meta = obs.metadata.get("evolved_subject", {})
        outcome_value = float(meta.get("outcome_value", 0.0))
        if outcome_value == 0.0:
            continue
        future = "good" if outcome_value > 0.0 else "bad"
        hidden.append(np.asarray(obs.memory_state, dtype=float))
        labels.append(1 if future == "good" else 0)
    if len(set(labels)) < 2 or len(labels) < 4:
        return 0.0
    vectors = np.stack(hidden, axis=0)
    labels_array = np.asarray(labels, dtype=int)
    predictions = []
    for index in range(len(labels)):
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[index] = False
        positive = vectors[train_mask & (labels_array == 1)]
        negative = vectors[train_mask & (labels_array == 0)]
        if positive.size == 0 or negative.size == 0:
            predictions.append(0)
            continue
        positive_center = np.mean(positive, axis=0)
        negative_center = np.mean(negative, axis=0)
        vector = vectors[index]
        pred = int(np.linalg.norm(vector - positive_center) <= np.linalg.norm(vector - negative_center))
        predictions.append(pred)
    return float(np.mean(np.asarray(predictions, dtype=int) == labels_array))


def _loo_nearest_centroid_accuracy(hidden: list[Array], labels: list[int]) -> float:
    if len(set(labels)) < 2 or len(labels) < 4:
        return 0.0
    vectors = np.stack(hidden, axis=0)
    labels_array = np.asarray(labels, dtype=int)
    predictions = []
    for index in range(len(labels)):
        mask = np.ones(len(labels), dtype=bool)
        mask[index] = False
        positive = vectors[mask & (labels_array == 1)]
        negative = vectors[mask & (labels_array == 0)]
        if positive.size == 0 or negative.size == 0:
            predictions.append(0)
            continue
        vector = vectors[index]
        pred = int(
            np.linalg.norm(vector - positive.mean(axis=0))
            <= np.linalg.norm(vector - negative.mean(axis=0))
        )
        predictions.append(pred)
    return float(np.mean(np.asarray(predictions, dtype=int) == labels_array))


def conditioned_future_outcome_accuracy(episodes: list[Episode], cue_class: str) -> float:
    """Decode future outcome (good/bad) from hidden state at outcome steps,
    pooled across episodes and restricted to one cue class.

    For a value-tracker this should be high for the valuable ``rare`` cue and
    near chance for the worthless ``common`` cue. A frequency-counter shows the
    opposite signature (it encodes the cue it sees most, regardless of value)."""
    hidden: list[Array] = []
    labels: list[int] = []
    for episode in episodes:
        for obs in episode.observations:
            meta = obs.metadata.get("evolved_subject", {})
            if meta.get("cue_class") != cue_class:
                continue
            outcome_value = float(meta.get("outcome_value", 0.0))
            if outcome_value == 0.0:
                continue
            hidden.append(np.asarray(obs.memory_state, dtype=float))
            labels.append(1 if outcome_value > 0.0 else 0)
    return _loo_nearest_centroid_accuracy(hidden, labels)


def conditioned_exposure_contrast(episodes: list[Episode], cue_class: str) -> float:
    """Good-minus-bad exposure at outcome steps for one cue class, pooled."""
    good: list[float] = []
    bad: list[float] = []
    for episode in episodes:
        for obs in episode.observations:
            meta = obs.metadata.get("evolved_subject", {})
            if meta.get("cue_class") != cue_class:
                continue
            outcome_value = float(meta.get("outcome_value", 0.0))
            if outcome_value > 0.0:
                good.append(obs.attention)
            elif outcome_value < 0.0:
                bad.append(obs.attention)
    g = float(np.mean(good)) if good else 0.0
    b = float(np.mean(bad)) if bad else 0.0
    return g - b


def frequency_counter_utility(
    sequences: list[InputSequence],
    *,
    exposure_cost: float = 0.05,
    action_change_cost: float = 0.02,
) -> float:
    """Decoy baseline: open in proportion to how often the current cue class has
    been seen, ignoring outcome value/sign. On the dissociated world this backs
    the frequent-but-worthless ``common`` cue and under-opens for the valuable
    ``rare`` cue, so a real value-tracker should beat it."""
    totals: list[float] = []
    for sequence in sequences:
        counts: dict[str, int] = {}
        seen = 0
        exposure = 0.5
        previous = 0.5
        total = 0.0
        for obj in sequence.objects:
            outcome_value = float(obj.metadata.get("outcome_value", 0.0))
            cue_class = obj.metadata.get("cue_class")
            if "cue" in obj.id and cue_class:
                counts[cue_class] = counts.get(cue_class, 0) + 1
                seen += 1
                exposure = counts[cue_class] / seen
            total += (
                exposure * outcome_value
                - exposure_cost * exposure
                - action_change_cost * abs(exposure - previous)
            )
            previous = exposure
        totals.append(total)
    return float(np.mean(totals)) if totals else 0.0


@lru_cache(maxsize=8)
def cached_evolution_result(
    *,
    seed: int = 17,
    generations: int = 80,
    population_size: int = 48,
    world_count: int = 20,
    hidden_dim: int = 5,
) -> EvolutionResult:
    return evolve_subject(
        subject_config=EvolvedSubjectConfig(hidden_dim=hidden_dim, recurrent=True),
        evolution_config=EvolutionConfig(
            seed=seed,
            generations=generations,
            population_size=population_size,
            elite_count=max(2, min(8, population_size // 4)),
            world_count=world_count,
        ),
    )


@lru_cache(maxsize=8)
def cached_nonrecurrent_evolution_result(
    *,
    seed: int = 18,
    generations: int = 80,
    population_size: int = 48,
    world_count: int = 20,
    hidden_dim: int = 5,
) -> EvolutionResult:
    return evolve_subject(
        subject_config=EvolvedSubjectConfig(hidden_dim=hidden_dim, recurrent=False),
        evolution_config=EvolutionConfig(
            seed=seed,
            generations=generations,
            population_size=population_size,
            elite_count=max(2, min(8, population_size // 4)),
            world_count=world_count,
        ),
    )


def _event(
    id: str,
    start: float,
    order: int,
    features: dict[str, float],
    *,
    outcome_value: float = 0.0,
    future_outcome: str = "neutral",
    cue_class: str | None = None,
) -> ExperienceObject:
    metadata: dict[str, object] = {
        "outcome_value": outcome_value,
        "future_outcome": future_outcome,
    }
    if cue_class is not None:
        metadata["cue_class"] = cue_class
    return ExperienceObject(
        id=id,
        temporal_extent=TemporalExtent(start=start, end=start + 1.0, order_index=order),
        features=FeatureVector(features),
        metadata=metadata,
    )


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0))))
