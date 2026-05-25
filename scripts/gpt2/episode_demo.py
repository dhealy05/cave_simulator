from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cave.observation.pipeline import views_from_names
from cave.presentation.renderers.matplotlib_renderer import available_styles
from cave.presentation.runs import ExperienceRun, slugify
from cave.observation.producers.sources.gpt2 import GPT2Producer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Cave episode from a GPT-2 forward pass.",
    )
    parser.add_argument(
        "--text",
        default="Hello, my name is Paul and I like to ",
        help="Input text to adapt into an episode.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("lib/models/gpt2"),
        help="Local GPT-2 model directory.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Legacy JSON payload output path. Prefer --output-root.",
    )
    parser.add_argument(
        "--output-frame",
        type=Path,
        default=None,
        help="Legacy frame image output path. Prefer --output-root.",
    )
    parser.add_argument(
        "--output-gif",
        type=Path,
        default=None,
        help="Legacy animation GIF output path. Prefer --output-root.",
    )
    parser.add_argument("--run-id", default=None, help="Run folder id.")
    parser.add_argument("--output-root", type=Path, default=Path("out/episodes"))
    parser.add_argument("--time", type=float, default=1.0, help="Frame time.")
    parser.add_argument(
        "--views",
        default="presentation,memory,timeline,expectation_actual,correction,subjective_topology",
    )
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--style", default="default", choices=available_styles())
    parser.add_argument("--feature-count", type=int, default=8)
    parser.add_argument("--active-top-k", type=int, default=8)
    args = parser.parse_args()

    producer = GPT2Producer(
        args.model_path,
        feature_count=args.feature_count,
        active_top_k=args.active_top_k,
    )
    episode = producer.run(args.text)
    views = views_from_names(args.views)
    run = ExperienceRun(
        id=args.run_id or slugify(args.text[:48]),
        episode=episode,
        input_summary=args.text,
        config={
            "model_path": str(args.model_path),
            "feature_count": args.feature_count,
            "active_top_k": args.active_top_k,
            "views": args.views,
            "style": args.style,
        },
    )
    outputs = run.write_outputs(
        root=args.output_root,
        views=views,
        frame_time=args.time,
        fps=args.fps,
        columns=args.columns,
        style=args.style,
    )
    print(f"wrote {outputs.episode_json}")
    print(f"wrote {outputs.metadata_json}")
    if outputs.frame_png is not None:
        print(f"wrote {outputs.frame_png}")
    if outputs.animation_gif is not None:
        print(f"wrote {outputs.animation_gif}")

    if args.output_json is not None:
        run.write_json(args.output_json)
        print(f"wrote {args.output_json}")
    if args.output_frame is not None:
        run.render_frame(args.output_frame, views, frame_time=args.time, style=args.style)
        print(f"wrote {args.output_frame}")
    if args.output_gif is not None:
        run.render_animation(args.output_gif, views, fps=args.fps, style=args.style)
        print(f"wrote {args.output_gif}")


if __name__ == "__main__":
    main()
