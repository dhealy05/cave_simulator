from __future__ import annotations

import argparse
from pathlib import Path

from cave.observation.episodes import CaveProducer
from cave.demonstrations.examples import demo_model, random_experience_model
from cave.observation.pipeline import episode_payload, write_json_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cave experience model demo.")
    parser.add_argument("--dt", type=float, default=0.1, help="Simulation timestep.")
    parser.add_argument(
        "--random",
        action="store_true",
        help="Generate a seeded random experience sequence instead of the fixed demo.",
    )
    parser.add_argument("--count", type=int, default=8, help="Random sequence length.")
    parser.add_argument("--seed", type=int, default=7, help="Random sequence seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Prints to stdout when omitted.",
    )
    args = parser.parse_args()

    model = (
        random_experience_model(count=args.count, seed=args.seed)
        if args.random
        else demo_model(seed=args.seed)
    )
    episode = CaveProducer(model).run(dt=args.dt)
    write_json_payload(episode_payload(episode), args.output)


if __name__ == "__main__":
    main()
