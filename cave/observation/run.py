from __future__ import annotations

import argparse
from pathlib import Path

from cave.observation.pipeline import (
    add_experience_source_args,
    episode_payload,
    producer_from_source_args,
    run_episode,
    write_json_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Cave experience pipeline and export JSON frames.")
    add_experience_source_args(parser)
    parser.add_argument("--dt", type=float, default=0.1, help="Simulation timestep.")
    parser.add_argument("--start", type=float, default=0.0, help="Simulation start time.")
    parser.add_argument("--end", type=float, default=None, help="Simulation end time. Defaults to sequence duration.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Prints to stdout when omitted.",
    )
    args = parser.parse_args()

    producer = producer_from_source_args(args)
    episode = run_episode(producer, start=args.start, end=args.end, dt=args.dt)
    write_json_payload(episode_payload(episode), args.output)


if __name__ == "__main__":
    main()
