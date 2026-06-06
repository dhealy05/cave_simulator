from __future__ import annotations

import argparse
from pathlib import Path

from cave.observation.episodes import CaveProducer
from cave.demonstrations.examples import demo_model
from cave.presentation.renderers.matplotlib_renderer import (
    LayoutSpec,
    MatplotlibRenderer,
    available_styles,
)
from cave.presentation.renderers.topology_surface_renderer import save_topology_state_surface
from cave.observation.views import (
    CorrectionView,
    ExpectationActualView,
    MemoryLookbackView,
    PresentationView,
    SubjectiveTopologyView,
    TimelineView,
    default_views,
)
from cave.demonstrations.subjects import (
    state_effect_embedding,
    subjective_trajectory_embedding,
    memory_trajectory_embedding,
    save_subject_comparison_dashboard,
    threshold_clusters,
)
from cave.demonstrations.subjects.subject_dashboard import controlled_subject_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate README paper assets.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/results/reference"))
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--style",
        default="default",
        choices=available_styles(),
        help="Named renderer style.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    render_animation(
        [PresentationView()],
        output_dir / "01_presentation_wall.gif",
        columns=1,
        dt=args.dt,
        fps=args.fps,
        seed=args.seed,
        style=args.style,
    )
    render_animation(
        [MemoryLookbackView()],
        output_dir / "02_memory_lookback.gif",
        columns=1,
        dt=args.dt,
        fps=args.fps,
        seed=args.seed,
        style=args.style,
    )
    render_animation(
        [TimelineView()],
        output_dir / "03_timeline_tape.gif",
        columns=1,
        dt=args.dt,
        fps=args.fps,
        seed=args.seed,
        style=args.style,
    )
    render_animation(
        [ExpectationActualView()],
        output_dir / "04_expectation_actual.gif",
        columns=1,
        dt=args.dt,
        fps=args.fps,
        seed=args.seed,
        style=args.style,
    )
    render_animation(
        [CorrectionView()],
        output_dir / "05_prediction_correction_over_time.gif",
        columns=1,
        dt=args.dt,
        fps=args.fps,
        seed=args.seed,
        style=args.style,
    )
    render_animation(
        [SubjectiveTopologyView()],
        output_dir / "06_subjective_topology.gif",
        columns=1,
        dt=args.dt,
        fps=args.fps,
        seed=args.seed,
        style=args.style,
    )
    render_animation(
        default_views(),
        output_dir / "07_multi_view_state.gif",
        columns=2,
        dt=args.dt,
        fps=args.fps,
        seed=args.seed,
        style=args.style,
    )
    render_topology_surface(
        output_dir / "08_topology_state_surface.png",
        dt=args.dt,
        seed=args.seed,
    )
    render_subject_comparison(output_dir / "09_subject_comparison.png")


def render_animation(
    views,
    output: Path,
    *,
    columns: int,
    dt: float,
    fps: int,
    seed: int,
    style: str,
) -> None:
    renderer = MatplotlibRenderer(
        layout=LayoutSpec(columns=columns, figsize_per_cell=(6.0, 6.0)),
        style=style,
    )
    episode = CaveProducer(demo_model(seed=seed)).run(dt=dt)
    renderer.save_animation(
        episode,
        views,
        output,
        dt=dt,
        fps=fps,
    )
    print(f"wrote {output}")


def render_topology_surface(output: Path, *, dt: float, seed: int) -> None:
    episode = CaveProducer(demo_model(seed=seed)).run(dt=dt)
    save_topology_state_surface(episode, output)
    print(f"wrote {output}")


def render_subject_comparison(output: Path) -> None:
    runs, labels = controlled_subject_runs(
        sequence_count=3,
        event_count=5,
        seed=101,
        dt=0.2,
        end=3.0,
    )
    effect = lambda run: state_effect_embedding(run, samples=48)
    observed = lambda run: memory_trajectory_embedding(run, samples=48)
    internal = lambda run: subjective_trajectory_embedding(run, samples=48)
    save_subject_comparison_dashboard(
        runs,
        labels,
        output,
        effect_embedding=effect,
        observed_embedding=observed,
        internal_embedding=internal,
        title="Subject Comparison: 3 sequences x 5 subjects",
    )
    print(f"wrote {output}")
    print("experience-effect clusters:")
    for cluster in threshold_clusters(runs, effect, threshold=1e-12):
        print([labels[index] for index in cluster])


if __name__ == "__main__":
    main()
