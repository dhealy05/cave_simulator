from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cave.demonstrations.examples import DEFAULT_VOCABULARY
from cave.observation.experience import (
    ExperienceObject,
    FeatureVector,
    InputSequence,
    TemporalExtent,
)
from cave.observation.episodes import CaveProducer
from cave.commitments.memory import (
    MemoryParams,
    MemoryTrace,
)
from cave.presentation.renderers.matplotlib_renderer import LayoutSpec, MatplotlibRenderer, available_styles
from cave.demonstrations.simulation import ExperienceModel, ModelParams
from cave.demonstrations.state import SubjectState
from cave.observation.views import PresentationView


def triangle_sequence() -> InputSequence:
    return InputSequence(
        objects=[
            ExperienceObject(
                id="evt_triangle",
                temporal_extent=TemporalExtent(start=0.0, end=2.4, order_index=0),
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
        ]
    )


def triangle_model(seed: int = 7) -> ExperienceModel:
    vocabulary = list(DEFAULT_VOCABULARY)
    sequence = triangle_sequence()
    params = ModelParams(
        memory=MemoryParams(retention=0.82, decay_tau=1.6, max_age=4.0),
    )
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("triangle_pov.gif"))
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--style", default="default", choices=available_styles())
    args = parser.parse_args()

    renderer = MatplotlibRenderer(
        layout=LayoutSpec(columns=1, figsize_per_cell=(6.0, 6.0)),
        style=args.style,
    )
    model = triangle_model()
    episode = CaveProducer(model).run(dt=args.dt)
    renderer.save_animation(
        episode,
        [PresentationView()],
        args.output,
        fps=args.fps,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
