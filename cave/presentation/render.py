from __future__ import annotations

import argparse
from pathlib import Path

from cave.observation.pipeline import add_experience_source_args, producer_from_source_args, views_from_names
from cave.presentation.renderers.matplotlib_renderer import LayoutSpec, MatplotlibRenderer, available_styles
from cave.observation.structural import frame_for_time, structural_state_for_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Cave experience as a frame or animation.")
    add_experience_source_args(parser)
    parser.add_argument("--output", type=Path, default=Path("animation.gif"))
    parser.add_argument("--views", default="all", help="Comma-separated views or 'all'.")
    parser.add_argument("--frame", action="store_true", help="Render one still frame instead of an animation.")
    parser.add_argument(
        "--filmstrip",
        choices=("intervals", "blur", "shared-axes"),
        default=None,
        help="Render the animation frames as a static filmstrip effect.",
    )
    parser.add_argument(
        "--filmstrip-max-frames",
        type=int,
        default=12,
        help="Maximum rendered frames used by --filmstrip.",
    )
    parser.add_argument("--time", type=float, default=2.4, help="Frame time for --frame.")
    parser.add_argument("--start", type=float, default=0.0, help="Animation start time.")
    parser.add_argument("--end", type=float, default=None, help="Animation end time. Defaults to sequence duration.")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument(
        "--style",
        default="default",
        choices=available_styles(),
        help="Named renderer style.",
    )
    args = parser.parse_args()

    producer = producer_from_source_args(args)
    episode = producer.run(start=args.start, end=args.end, dt=args.dt)
    try:
        views = views_from_names(args.views)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    renderer = MatplotlibRenderer(
        layout=LayoutSpec(columns=args.columns),
        style=args.style,
    )
    if args.frame:
        structural = structural_state_for_episode(episode)
        frame = frame_for_time(episode, args.time, structural)
        renderer.save_frame(frame, views, args.output)
    elif args.filmstrip:
        renderer.save_filmstrip(
            episode,
            views,
            args.output,
            mode=args.filmstrip,
            start=args.start,
            end=args.end,
            dt=args.dt,
            max_frames=args.filmstrip_max_frames,
        )
    else:
        renderer.save_animation(
            episode,
            views,
            args.output,
            start=args.start,
            end=args.end,
            dt=args.dt,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
