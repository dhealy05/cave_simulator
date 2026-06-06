"""Generate static storybook panels for the Cave Walker explainer.

Two modes, both rendered with the existing Cave Walker renderer (no new visual
code, no animation):

  default          regenerate the canonical six-page storybook into
                   ``assets/storybook/`` (the panels embedded in docs/storybook.md).

  --random         generate an "infinite" random scenario: a seeded episode of
                   trees / rocks / snakes, run through the same primitive
                   recurrence, rendered to its own folder with an auto-written
                   ``scenario.md`` (captions + real numbers per page).

Each page lays out three panels:

  left   - the object world Jimmy walks through (what he sees)
  middle - the two-feature plane: expected (orange), actual (blue), memory
           (green) -- the correction geometry
  right  - the accumulated topology-like field from the trajectory so far

Examples:

    python notebooks/demos/primitive_demo/scripts/generate_storybook.py
    python notebooks/demos/primitive_demo/scripts/generate_storybook.py --random --seed 7 --length 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "cave_walker"))

from cave_walker_demo import (  # noqa: E402
    NATIVE_HEIGHT,
    NATIVE_WIDTH,
    _draw_internal_map,
    _draw_walker_topology,
    _render_gba_viewport,
    _surprise_band,
    build_walker_rollout,
    nearest_prototype,
    save_walker_filmstrip,
    save_walker_filmstrip_blur,
)
from cave_walker_objects import (  # noqa: E402
    EPISODE,
    ETA,
    MEMORY_INITIAL,
    OBJECTS,
    prototype_features,
)
from cave.observation.producers.sources.primitive import (  # noqa: E402
    DEFAULT_PRIMITIVE_WEIGHTS,
    primitive_jittered_features,
    primitive_random_episode,
    rollout_vectors,
)


DEFAULT_WEIGHTS = DEFAULT_PRIMITIVE_WEIGHTS


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def _scroll_for(surprise: float) -> float:
    # Hold a surprising encounter inside its error phase so the alert pose shows.
    return 0.42 if surprise > 0.45 else 0.30


def save_page(rows, index: int, output: Path, episode: tuple[str, ...]) -> None:
    row = rows[index]
    fig = plt.figure(figsize=(16.0, 4.6), dpi=130)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.3, 1.0, 1.0), wspace=0.16)
    world_axis = fig.add_subplot(grid[0, 0])
    plane_axis = fig.add_subplot(grid[0, 1])
    topology_axis = fig.add_subplot(grid[0, 2])

    viewport = _render_gba_viewport(row, index, episode, _scroll_for(row.surprise))
    world_axis.imshow(
        viewport,
        interpolation="nearest",
        extent=(0, NATIVE_WIDTH, 0, NATIVE_HEIGHT),
    )
    world_axis.set_title("What Jimmy sees", loc="left", fontsize=12, fontweight="bold")
    world_axis.set_xticks([])
    world_axis.set_yticks([])
    world_axis.set_aspect("equal", adjustable="box")
    for spine in world_axis.spines.values():
        spine.set_visible(False)

    _draw_internal_map(plane_axis, rows, index, episode)
    plane_axis.set_title("Expect -> see -> correct", loc="left", fontsize=12, fontweight="bold")

    _draw_walker_topology(topology_axis, rows, index)
    topology_axis.set_title("Accumulated topology field", loc="left", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Canonical six-page storybook (feeds docs/storybook.md)
# --------------------------------------------------------------------------- #
def generate_canonical() -> None:
    rows = build_walker_rollout()
    out_dir = BASE / "assets" / "storybook"
    out_dir.mkdir(parents=True, exist_ok=True)
    for index in range(len(rows)):
        path = out_dir / f"page_{index + 1}_{EPISODE[index]}.png"
        save_page(rows, index, path, EPISODE)
        print(f"page {index + 1} ({EPISODE[index]}, surprise={rows[index].surprise:.3f}): {path}")

    # Closing-page filmstrips: the whole walk laid out as one static object.
    # These are committed assets, unlike the cave_walker out/ copies, so the
    # storybooks and walkthrough can embed them directly.
    filmstrip_path = out_dir / "filmstrip.png"
    filmstrip_blur_path = out_dir / "filmstrip_blur.png"
    save_walker_filmstrip(rows, filmstrip_path, episode=EPISODE)
    save_walker_filmstrip_blur(rows, filmstrip_blur_path, episode=EPISODE)
    print(f"filmstrip: {filmstrip_path}")
    print(f"filmstrip (blur): {filmstrip_blur_path}")


# --------------------------------------------------------------------------- #
# Random "infinite" scenarios
# --------------------------------------------------------------------------- #
def random_episode(rng: np.random.Generator, length: int) -> tuple[str, ...]:
    return primitive_random_episode(rng, length, weights=DEFAULT_WEIGHTS)


def scenario_features(
    episode: tuple[str, ...],
    rng: np.random.Generator,
    jitter: float,
) -> list[tuple[float, float]]:
    """Per-encounter feature vectors: tree variants + small seeded jitter so
    that even repeated objects vary slightly (as real encounters would)."""
    return primitive_jittered_features(episode, rng, jitter)


def caption(rows, index: int, episode: tuple[str, ...]) -> str:
    row = rows[index]
    seen = OBJECTS[episode[index]].label
    expected = OBJECTS[nearest_prototype(row.expected, prototype_features())[0]].label
    memory = OBJECTS[nearest_prototype(row.memory, prototype_features())[0]].label
    band = _surprise_band(row.surprise)
    lead = (
        f"Jimmy sets out expecting something {expected}-ish."
        if index == 0
        else f"Jimmy expected something {expected}-ish."
    )
    react = {
        "low": f"It's a {seen} — almost exactly his guess, barely a ripple",
        "medium": f"It's a {seen} — a fair miss",
        "high": f"It's a {seen} — a big surprise",
    }[band]
    return f"{lead} {react} (surprise {row.surprise:.3f}). Afterward his memory sits closest to {memory}."


def numbers_block(row) -> str:
    E = row.expected
    U = row.actual
    M = row.memory
    P = (U[0] - E[0], U[1] - E[1])
    return (
        "```text\n"
        f"E_t = ({E[0]:.3f}, {E[1]:.3f})\n"
        f"U_t = ({U[0]:.3f}, {U[1]:.3f})\n"
        f"P_t = ({P[0]:+.3f}, {P[1]:+.3f})    surprise = {row.surprise:.3f}\n"
        f"M_t = E_t + {ETA}*P_t = ({M[0]:.3f}, {M[1]:.3f})\n"
        "```"
    )


def write_scenario(seed: int, length: int, jitter: float, output_dir: Path) -> None:
    rng = np.random.default_rng(seed)
    episode = random_episode(rng, length)
    features = scenario_features(episode, rng, jitter)
    rows = rollout_vectors(features, eta=ETA, memory_initial=MEMORY_INITIAL)

    output_dir.mkdir(parents=True, exist_ok=True)
    page_files: list[str] = []
    for index in range(len(rows)):
        name = f"page_{index + 1}_{episode[index]}.png"
        save_page(rows, index, output_dir / name, episode)
        page_files.append(name)
        print(f"page {index + 1} ({episode[index]}, surprise={rows[index].surprise:.3f})")

    (output_dir / "scenario.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "length": length,
                "jitter": jitter,
                "eta": ETA,
                "memory_initial": list(MEMORY_INITIAL),
                "episode": list(episode),
                "rollout": [
                    {
                        "t": r.t,
                        "object": episode[i],
                        "expected": list(r.expected),
                        "actual": list(r.actual),
                        "memory": list(r.memory),
                        "surprise": r.surprise,
                    }
                    for i, r in enumerate(rows)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pretty = " → ".join(OBJECTS[obj_id].label for obj_id in episode)
    lines = [
        f"# Jimmy's walk — scenario seed {seed}",
        "",
        "*Auto-generated by `generate_storybook.py --random`. One of infinitely "
        "many walks Jimmy can take through a world of trees, rocks, and snakes.*",
        "",
        f"**This walk:** {pretty}",
        "",
        "The loop is the same on every page — only the world changes:",
        "",
        "```text",
        "E_t = M_{t-1}                 expectation IS the last memory",
        "P_t = U_t - E_t               error: actual minus expected",
        "surprise = |P_t|",
        f"M_t = E_t + η·P_t   (η = {ETA})   memory closes a fraction η of the gap",
        "```",
        "",
        "Middle panel dots: 🟠 expected `E_t` · 🔵 actual `U_t` · 🟢 memory `M_t`; "
        "the pink arrow is the error `P_t`. For the full explanation see "
        "[the walkthrough](walkthrough.md).",
        "",
        "---",
        "",
    ]
    for index, name in enumerate(page_files):
        lines += [
            f"## Page {index + 1} — {OBJECTS[episode[index]].label}",
            "",
            f"![Page {index + 1}]({name})",
            "",
            caption(rows, index, episode),
            "",
            numbers_block(rows[index]),
            "",
        ]
    (output_dir / "scenario.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nscenario.md: {output_dir / 'scenario.md'}")
    print(f"episode: {pretty}")


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--random", action="store_true", help="generate a random scenario instead of the canonical storybook")
    parser.add_argument("--seed", type=int, default=0, help="random seed (determines the scenario)")
    parser.add_argument("--length", type=int, default=8, help="number of encounters in the random walk")
    parser.add_argument("--jitter", type=float, default=0.03, help="per-encounter feature jitter so repeats vary")
    parser.add_argument("--output-dir", type=Path, default=None, help="output folder (default: generated/scenarios/seed_<seed>)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.random:
        generate_canonical()
        return
    if args.length < 1:
        raise SystemExit("--length must be at least 1")
    output_dir = args.output_dir or (BASE / "generated" / "scenarios" / f"seed_{args.seed}")
    write_scenario(args.seed, args.length, args.jitter, output_dir)


if __name__ == "__main__":
    main()
