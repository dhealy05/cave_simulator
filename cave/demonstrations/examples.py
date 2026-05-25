from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cave.commitments.attention import AttentionProfile
from cave.observation.experience import (
    ExperienceObject,
    FeatureProjection,
    FeatureVector,
    InputSequence,
    TemporalExtent,
)
from cave.commitments.memory import (
    MemoryParams,
    MemoryTrace,
)
from cave.demonstrations.simulation import ExperienceModel, ModelParams
from cave.demonstrations.state import SubjectState
from cave.commitments.topology import SubjectiveTopologyParams, SubjectiveTopologyPrior


DEFAULT_VOCABULARY = [
    "sides",
    "size",
    "hue",
    "saturation",
    "lightness",
    "angularity",
    "roundness",
    "symmetry",
    "novelty",
]


@dataclass(frozen=True)
class RandomExperienceSpec:
    count: int = 8
    seed: int = 7
    min_duration: float = 0.45
    max_duration: float = 1.4
    min_gap: float = 0.08
    max_gap: float = 0.42
    min_salience: float = 0.35
    max_salience: float = 1.0

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("count must be positive")
        if self.min_duration <= 0.0 or self.max_duration < self.min_duration:
            raise ValueError("duration bounds must be positive and ordered")
        if self.min_gap < 0.0 or self.max_gap < self.min_gap:
            raise ValueError("gap bounds must be non-negative and ordered")
        if self.min_salience < 0.0 or self.max_salience < self.min_salience:
            raise ValueError("salience bounds must be non-negative and ordered")


def demo_sequence() -> InputSequence:
    return InputSequence(
        objects=[
            ExperienceObject(
                id="evt_triangle",
                temporal_extent=TemporalExtent(start=0.0, end=1.2, order_index=0),
                features=FeatureVector(
                    {
                        "sides": 0.0,
                        "size": 0.62,
                        "hue": 0.13,
                        "saturation": 0.52,
                        "lightness": 0.79,
                        "angularity": 1.0,
                        "roundness": 0.0,
                        "symmetry": 0.85,
                        "novelty": 0.55,
                    }
                ),
                kind="experience",
                salience=0.9,
            ),
            ExperienceObject(
                id="evt_circle",
                temporal_extent=TemporalExtent(start=1.4, end=2.7, order_index=1),
                features=FeatureVector(
                    {
                        "sides": 1.0,
                        "size": 0.7,
                        "hue": 0.58,
                        "saturation": 0.86,
                        "lightness": 0.78,
                        "angularity": 0.0,
                        "roundness": 1.0,
                        "symmetry": 1.0,
                        "novelty": 0.42,
                    }
                ),
                kind="experience",
                salience=0.8,
            ),
            ExperienceObject(
                id="evt_square",
                temporal_extent=TemporalExtent(start=2.9, end=4.1, order_index=2),
                features=FeatureVector(
                    {
                        "sides": 0.08,
                        "size": 0.82,
                        "hue": 0.03,
                        "saturation": 0.65,
                        "lightness": 0.68,
                        "angularity": 0.8,
                        "roundness": 0.05,
                        "symmetry": 0.95,
                        "novelty": 0.35,
                    }
                ),
                kind="experience",
                salience=0.85,
            ),
            ExperienceObject(
                id="evt_gap",
                temporal_extent=TemporalExtent(start=4.4, end=5.1, order_index=3),
                features=FeatureVector(
                    {
                        "sides": 0.0,
                        "size": 0.22,
                        "hue": 0.0,
                        "saturation": 0.0,
                        "lightness": 0.82,
                        "angularity": 0.05,
                        "roundness": 0.05,
                        "symmetry": 0.4,
                        "novelty": 0.1,
                    }
                ),
                kind="experience",
                salience=0.35,
            ),
        ]
    )


def random_experience_sequence(
    spec: RandomExperienceSpec | None = None,
    *,
    count: int | None = None,
    seed: int | None = None,
) -> InputSequence:
    spec = spec or RandomExperienceSpec()
    if count is not None or seed is not None:
        spec = RandomExperienceSpec(
            count=spec.count if count is None else count,
            seed=spec.seed if seed is None else seed,
            min_duration=spec.min_duration,
            max_duration=spec.max_duration,
            min_gap=spec.min_gap,
            max_gap=spec.max_gap,
            min_salience=spec.min_salience,
            max_salience=spec.max_salience,
        )

    rng = np.random.default_rng(spec.seed)
    objects: list[ExperienceObject] = []
    t = 0.0
    previous_features: dict[str, float] | None = None

    for index in range(spec.count):
        duration = float(rng.uniform(spec.min_duration, spec.max_duration))
        start = t
        end = start + duration
        features = random_feature_values(rng, previous_features)
        previous_features = features

        objects.append(
            ExperienceObject(
                id=f"evt_random_{index:03d}",
                temporal_extent=TemporalExtent(
                    start=start,
                    end=end,
                    order_index=index,
                ),
                features=FeatureVector(features),
                kind="experience",
                salience=float(rng.uniform(spec.min_salience, spec.max_salience)),
            )
        )
        t = end + float(rng.uniform(spec.min_gap, spec.max_gap))

    return InputSequence(objects=objects)


def random_feature_values(
    rng: np.random.Generator,
    previous: dict[str, float] | None = None,
) -> dict[str, float]:
    sides = float(rng.beta(1.35, 1.35))
    roundness = _clamp01(0.08 + 0.86 * sides + float(rng.normal(0.0, 0.1)))
    angularity = _clamp01(1.0 - 0.78 * roundness + float(rng.normal(0.0, 0.12)))
    symmetry = _clamp01(0.35 + 0.55 * rng.random() + 0.1 * (1.0 - abs(0.5 - sides)))

    hue = float(rng.random())
    if previous is not None and rng.random() < 0.45:
        hue = _wrap01(previous["hue"] + float(rng.normal(0.0, 0.08)))

    novelty_base = float(rng.beta(1.2, 2.0))
    if previous is None:
        novelty = max(0.55, novelty_base)
    else:
        delta = np.array(
            [
                sides - previous["sides"],
                roundness - previous["roundness"],
                angularity - previous["angularity"],
                hue - previous["hue"],
            ],
            dtype=float,
        )
        novelty = _clamp01(
            0.25 * novelty_base + 0.75 * min(1.0, float(np.linalg.norm(delta)))
        )

    return {
        "sides": sides,
        "size": float(rng.uniform(0.2, 0.95)),
        "hue": hue,
        "saturation": float(rng.uniform(0.25, 0.9)),
        "lightness": float(rng.uniform(0.42, 0.82)),
        "angularity": angularity,
        "roundness": roundness,
        "symmetry": symmetry,
        "novelty": novelty,
    }


def default_model_params() -> ModelParams:
    return ModelParams(
        memory=MemoryParams(retention=0.82, decay_tau=1.6, max_age=4.0),
        attention=AttentionProfile(mode="sine", level=0.5, amplitude=0.5),
        topology=SubjectiveTopologyParams(
            feature_x=FeatureProjection(
                name="form",
                weights={
                    "angularity": 0.5,
                    "symmetry": 0.25,
                    "sides": 0.25,
                },
            ),
            feature_y=FeatureProjection(
                name="sensory tone",
                weights={
                    "roundness": 0.4,
                    "hue": 0.2,
                    "saturation": 0.2,
                    "novelty": 0.2,
                },
            ),
            prior=SubjectiveTopologyPrior(
                mode="random_wells",
                strength=0.18,
                width=0.28,
                seed=19,
                well_count=5,
            ),
        ),
    )


def model_for_sequence(
    sequence: InputSequence,
    params: ModelParams | None = None,
    vocabulary: list[str] | None = None,
) -> ExperienceModel:
    vocabulary = list(DEFAULT_VOCABULARY if vocabulary is None else vocabulary)
    params = params or default_model_params()
    trace = MemoryTrace(
        vector=np.zeros(len(vocabulary), dtype=float),
        retention=params.memory.retention,
        decay_tau=params.memory.decay_tau,
        max_age=params.memory.max_age,
    )

    return ExperienceModel(
        sequence=sequence,
        subject_state=SubjectState.initial(trace, params.topology),
        params=params,
        vocabulary=vocabulary,
    )


def demo_model(seed: int = 7) -> ExperienceModel:
    return model_for_sequence(demo_sequence())


def random_experience_model(
    spec: RandomExperienceSpec | None = None,
    *,
    count: int | None = None,
    seed: int | None = None,
) -> ExperienceModel:
    return model_for_sequence(
        random_experience_sequence(spec, count=count, seed=seed)
    )


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _wrap01(value: float) -> float:
    return float(value % 1.0)
