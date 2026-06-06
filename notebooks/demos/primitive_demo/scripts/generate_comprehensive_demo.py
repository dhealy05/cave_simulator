"""Generate still panels for the comprehensive primitive-demo storyboard."""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import ImageDraw

BASE = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BASE / "cave_walker"))

from cave.observation.producers.sources.primitive import (  # noqa: E402
    PrimitiveVectorStep,
    rollout_vectors,
)
from cave_walker_demo import (  # noqa: E402
    COLORS,
    DEFAULT_SUBJECT_SPRITE_DIR,
    ENCOUNTER_SCREEN_X,
    GROUND_Y,
    NATIVE_HEIGHT,
    NATIVE_WIDTH,
    OBJECTS,
    OBJECT_SPRITE_DIR,
    SPRITE_ANIMATOR,
    _draw_clouds,
    _draw_image,
    _draw_internal_map,
    _draw_native_text,
    _draw_tilemap,
    _draw_walker_topology,
    _gba_asset,
    _paste_grounded_sprite,
    _paste_looped_layer,
    _render_gba_viewport,
    _rgb,
    _style_panel,
)
from cave_walker_objects import ETA, MEMORY_INITIAL, prototype_features  # noqa: E402


OUT_DIR = BASE / "assets" / "comprehensive"


@dataclass(frozen=True)
class ObserverSpec:
    name: str
    subject_sprite_dir: str
    outward_attention: float
    inward_attention: float
    eta: float
    memory_initial: tuple[float, float] = MEMORY_INITIAL


def _gated_step(object_id: str, observer: ObserverSpec, *, t: int = 1) -> PrimitiveVectorStep:
    sensed = np.array(OBJECTS[object_id].features, dtype=float)
    prior = np.array(observer.memory_initial, dtype=float)
    actual = observer.outward_attention * sensed
    expected = observer.inward_attention * prior
    error = actual - expected
    memory = prior + observer.eta * error
    return PrimitiveVectorStep(
        t=t,
        actual=tuple(float(value) for value in actual),
        expected=tuple(float(value) for value in expected),
        error=tuple(float(value) for value in error),
        surprise=float(np.linalg.norm(error)),
        memory_previous=tuple(float(value) for value in prior),
        memory=tuple(float(value) for value in memory),
    )


def _save(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(output)


def _wrap(text: str, width: int) -> str:
    """Wrap each line to ``width`` while preserving intentional line/paragraph breaks."""
    lines = []
    for line in text.split("\n"):
        lines.append(textwrap.fill(line, width=width) if line.strip() else line)
    return "\n".join(lines)


def _caption(axis: Axes, title: str, body: str, *, width: int = 30) -> None:
    _style_panel(axis, title)
    axis.text(0.06, 0.80, _wrap(body, width), transform=axis.transAxes, fontsize=13, color=COLORS["text"], va="top")


def _feature_bar(axis: Axes, label: str, value: float, y: float, color: str) -> None:
    """Draw a labelled feature bar with the label on its own line above a full-width bar."""
    axis.text(0.08, y + 0.06, label, transform=axis.transAxes, fontsize=12, fontweight="bold", color=COLORS["text"])
    axis.add_patch(
        Rectangle((0.08, y - 0.025), float(value) * 0.60, 0.045, transform=axis.transAxes, color=color, alpha=0.85)
    )
    axis.text(
        0.08 + float(value) * 0.60 + 0.03,
        y,
        f"{value:.2f}",
        transform=axis.transAxes,
        va="center",
        fontsize=11,
        color=COLORS["text"],
    )


def _empty_viewport() -> np.ndarray:
    camera_x = 46.0
    canvas = SPRITE_ANIMATOR.canvas("sky")
    _paste_looped_layer(canvas, _gba_asset("background/hills_far_bubbly.png"), camera_x, factor=0.18, y=42)
    _paste_looped_layer(canvas, _gba_asset("background/hills_near_bubbly.png"), camera_x, factor=0.44, y=56)
    _draw_clouds(canvas, camera_x)
    _draw_tilemap(canvas, camera_x)
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle((ENCOUNTER_SCREEN_X - 5, 25, ENCOUNTER_SCREEN_X + 5, GROUND_Y + 2), fill=(*_rgb("memory"), 20))
    draw.line((ENCOUNTER_SCREEN_X, 25, ENCOUNTER_SCREEN_X, GROUND_Y + 2), fill=(*_rgb("memory"), 70), width=1)
    _paste_grounded_sprite(canvas, _gba_asset(f"{DEFAULT_SUBJECT_SPRITE_DIR}/subject_walk_1.png"), 72, GROUND_Y)
    _draw_native_text(draw, "no salient object", (122, 32), anchor="mm", fill="#1F2430")
    return np.asarray(canvas.convert("RGBA"))


def _draw_viewport(axis: Axes, image: np.ndarray, title: str) -> None:
    axis.imshow(image, interpolation="nearest", extent=(0, NATIVE_WIDTH, 0, NATIVE_HEIGHT))
    axis.set_title(title, loc="left", fontsize=12, fontweight="bold", color=COLORS["text"])
    axis.set_xlim(0, NATIVE_WIDTH)
    axis.set_ylim(0, NATIVE_HEIGHT)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_aspect("equal", adjustable="box")


def render_empty_world(output: Path) -> None:
    fig = plt.figure(figsize=(12.0, 4.8), dpi=140)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.45, 1.0), wspace=0.16)
    world_axis = fig.add_subplot(grid[0, 0])
    state_axis = fig.add_subplot(grid[0, 1])
    _draw_viewport(world_axis, _empty_viewport(), "0. Empty wasteland")
    _caption(
        state_axis,
        "Empty experience",
        "world: empty\nsensed object: none\nU_t: near zero\nP_t: near zero\nM_t: unchanged\n\nNothing is admitted strongly enough to update the observer.",
    )
    _save(fig, output)


def render_object_world(output: Path) -> None:
    row = rollout_vectors([OBJECTS["snake"].features], eta=ETA, memory_initial=MEMORY_INITIAL)[0]
    viewport = _render_gba_viewport(row, 0, ("snake",), 0.30)
    fig = plt.figure(figsize=(12.0, 4.8), dpi=140)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.45, 1.0), wspace=0.16)
    world_axis = fig.add_subplot(grid[0, 0])
    state_axis = fig.add_subplot(grid[0, 1])
    _draw_viewport(world_axis, viewport, "1. Object in the world")
    _caption(
        state_axis,
        "Not yet experience",
        "The snake is externally available.\n\nIt has not become an admitted input until sensing and attention pass it into the observer.",
    )
    _save(fig, output)


def render_snake_vector(output: Path) -> None:
    snake = OBJECTS["snake"]
    fig = plt.figure(figsize=(12.0, 4.8), dpi=140)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.2), wspace=0.20)
    sprite_axis = fig.add_subplot(grid[0, 0])
    vector_axis = fig.add_subplot(grid[0, 1])
    _style_panel(sprite_axis, "2. Snake as object")
    sprite_axis.set_xlim(0, 1)
    sprite_axis.set_ylim(0, 1)
    _draw_image(sprite_axis, _gba_asset(f"{OBJECT_SPRITE_DIR}/{snake.sprite}"), (0.50, 0.52), 0.36, zorder=3)
    sprite_axis.text(0.50, 0.18, "world object: Snake", ha="center", fontsize=13, color=COLORS["text"])
    _style_panel(vector_axis, "Snake as numbers")
    vector_axis.text(0.08, 0.86, "feature vector", transform=vector_axis.transAxes, fontsize=12, fontweight="bold")
    _feature_bar(vector_axis, "danger / salience", snake.features[0], 0.62, COLORS["actual"])
    _feature_bar(vector_axis, "height / liveliness", snake.features[1], 0.42, COLORS["actual"])
    vector_axis.text(
        0.08,
        0.24,
        _wrap("The primitive update receives admitted numbers derived from the encounter, not the drawing itself.", 44),
        transform=vector_axis.transAxes,
        fontsize=12,
        color=COLORS["text"],
        va="top",
    )
    _save(fig, output)


def render_observer(output: Path) -> None:
    fig = plt.figure(figsize=(11.5, 5.2), dpi=140)
    grid = fig.add_gridspec(1, 2, width_ratios=(0.92, 1.08), wspace=0.20)
    sprite_axis = fig.add_subplot(grid[0, 0])
    readout_axis = fig.add_subplot(grid[0, 1])
    _style_panel(sprite_axis, "3. Observer")
    sprite_axis.set_xlim(0, 1)
    sprite_axis.set_ylim(0, 1)
    _draw_image(
        sprite_axis,
        _gba_asset(f"{DEFAULT_SUBJECT_SPRITE_DIR}/subject_thinking.png"),
        (0.50, 0.58),
        0.30,
        zorder=3,
    )
    sprite_axis.text(0.50, 0.25, "a subject with state", ha="center", fontsize=13, color=COLORS["text"])
    _caption(
        readout_axis,
        "Observer parameters",
        "memory M_{t-1}\nlearning rate eta\noutward attention schedule\ninward attention schedule\n\nThe observer is not just a sprite.\nIt is the state and schedule that decide\nhow the world enters.",
    )
    _save(fig, output)


def render_attention_gates(output: Path) -> None:
    fig = plt.figure(figsize=(11.5, 5.2), dpi=140)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.12, 1.0), wspace=0.22)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    _style_panel(axes[0], "4. Attention gates")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    gates = [
        ("world\nvector", 0.18, COLORS["actual"]),
        ("outward\ngate", 0.48, COLORS["memory"]),
        ("admitted\nU_t", 0.78, COLORS["actual"]),
    ]
    for label, x, color in gates:
        axes[0].add_patch(Rectangle((x - 0.10, 0.55), 0.20, 0.15, facecolor=color, alpha=0.18, edgecolor=color, lw=1.5))
        axes[0].text(x, 0.625, label, ha="center", va="center", fontsize=9, color=COLORS["text"])
    for start, end in ((0.29, 0.37), (0.59, 0.68)):
        axes[0].add_patch(FancyArrowPatch((start, 0.62), (end, 0.62), arrowstyle="->", mutation_scale=14, color=COLORS["muted"]))
    axes[0].text(0.13, 0.32, "outward_attention[t] = 0.90", fontsize=10, color=COLORS["text"])
    axes[0].text(0.13, 0.22, "inward_attention[t] = 0.55", fontsize=10, color=COLORS["text"])
    _style_panel(axes[1], "Generated expectation")
    axes[1].text(
        0.06,
        0.80,
        "The observer also admits some amount\nof its own prior expectation.\n\nThat inward gate decides how strongly\nE_t enters the comparison.",
        transform=axes[1].transAxes,
        fontsize=12,
        color=COLORS["text"],
        va="top",
    )
    _save(fig, output)


def render_one_encounter(output: Path) -> None:
    observer = ObserverSpec("Observer", "sprites/subjects/default", 1.0, 1.0, ETA)
    row = _gated_step("snake", observer)
    rows = [row]
    fig = plt.figure(figsize=(16.0, 4.8), dpi=140)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.35, 1.0, 1.0), wspace=0.16)
    world_axis = fig.add_subplot(grid[0, 0])
    map_axis = fig.add_subplot(grid[0, 1])
    topology_axis = fig.add_subplot(grid[0, 2])
    _draw_viewport(world_axis, _render_gba_viewport(row, 0, ("snake",), 0.42), "5. One encounter")
    _draw_internal_map(map_axis, rows, 0, ("snake",))
    _draw_walker_topology(topology_axis, rows, 0)
    _save(fig, output)


def render_two_observers(output: Path) -> None:
    observers = [
        ObserverSpec("Low prior fear", "sprites/subjects/sleepy", 0.88, 0.55, 0.35),
        ObserverSpec("High prior pull", "sprites/subjects/excited", 0.88, 1.20, 0.65),
    ]
    rows = [_gated_step("snake", observer) for observer in observers]
    fig = plt.figure(figsize=(15.5, 7.0), dpi=140)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.14, 0.98, 1.0), hspace=0.30, wspace=0.36)
    for row_index, (observer, row) in enumerate(zip(observers, rows, strict=True)):
        world_axis = fig.add_subplot(grid[row_index, 0])
        map_axis = fig.add_subplot(grid[row_index, 1])
        readout_axis = fig.add_subplot(grid[row_index, 2])
        _draw_viewport(
            world_axis,
            _render_gba_viewport(row, 0, ("snake",), 0.42, subject_sprite_dir=observer.subject_sprite_dir),
            f"6. Same snake: {observer.name}",
        )
        _draw_internal_map(map_axis, [row], 0, ("snake",))
        map_axis.set_title("Subject-side update", loc="left", fontsize=11, fontweight="bold", color=COLORS["text"])
        _style_panel(readout_axis, "Observer schedule")
        readout_axis.text(0.08, 0.78, f"outward attention: {observer.outward_attention:.2f}", transform=readout_axis.transAxes, fontsize=11)
        readout_axis.text(0.08, 0.66, f"inward attention:  {observer.inward_attention:.2f}", transform=readout_axis.transAxes, fontsize=11)
        readout_axis.text(0.08, 0.54, f"eta:               {observer.eta:.2f}", transform=readout_axis.transAxes, fontsize=11)
        readout_axis.text(0.08, 0.34, f"surprise:          {row.surprise:.3f}", transform=readout_axis.transAxes, fontsize=12, color=COLORS["error"], fontweight="bold")
        readout_axis.text(0.08, 0.22, f"M_t: ({row.memory[0]:.3f}, {row.memory[1]:.3f})", transform=readout_axis.transAxes, fontsize=11, color=COLORS["memory"])
    fig.suptitle("Same external snake; different subject-side update", fontsize=15, fontweight="bold", color=COLORS["text"])
    _save(fig, output)


def render_sequence(output: Path) -> None:
    episode = ("rock", "snake", "tree", "snake")
    rows = rollout_vectors([OBJECTS[object_id].features for object_id in episode], eta=ETA, memory_initial=MEMORY_INITIAL)
    fig = plt.figure(figsize=(15.5, 8.5), dpi=140)
    grid = fig.add_gridspec(2, 2, height_ratios=(0.85, 1.1), hspace=0.26, wspace=0.20)
    world_axis = fig.add_subplot(grid[0, :])
    map_axis = fig.add_subplot(grid[1, 0])
    topology_axis = fig.add_subplot(grid[1, 1])
    _draw_sequence_strip(world_axis, rows, episode)
    _draw_internal_map(map_axis, rows, len(rows) - 1, episode)
    _draw_walker_topology(topology_axis, rows, len(rows) - 1)
    _save(fig, output)


def _draw_sequence_strip(axis: Axes, rows: Sequence[PrimitiveVectorStep], episode: tuple[str, ...]) -> None:
    _style_panel(axis, "7. Sequence, not isolated events")
    axis.set_xlim(-0.6, len(rows) - 0.4)
    axis.set_ylim(0.0, 1.0)
    axis.plot([-0.45, len(rows) - 0.55], [0.42, 0.42], color="#7C8A5F", linewidth=3.0, alpha=0.5)
    for index, (row, object_id) in enumerate(zip(rows, episode, strict=True)):
        x = float(index)
        axis.axvline(x, color=COLORS["error"] if row.surprise > 0.45 else COLORS["memory"], alpha=0.10, linewidth=14)
        _draw_image(axis, _gba_asset(f"{OBJECT_SPRITE_DIR}/{OBJECTS[object_id].sprite}"), (x + 0.16, 0.55), 0.20)
        _draw_image(axis, _gba_asset(f"{DEFAULT_SUBJECT_SPRITE_DIR}/subject_walk_1.png"), (x - 0.16, 0.54), 0.18)
        axis.text(x, 0.18, f"t={row.t}\n{OBJECTS[object_id].label}\nsurprise {row.surprise:.2f}", ha="center", va="top", fontsize=9, color=COLORS["text"])


def render_all(output_dir: Path = OUT_DIR) -> dict[str, Path]:
    outputs = {
        "empty_world": output_dir / "00_empty_wasteland.png",
        "object_world": output_dir / "01_object_in_world.png",
        "snake_vector": output_dir / "02_object_becomes_numbers.png",
        "observer": output_dir / "03_observer.png",
        "attention_gates": output_dir / "04_attention_gates.png",
        "one_encounter": output_dir / "05_one_encounter.png",
        "two_observers": output_dir / "06_same_snake_different_observers.png",
        "sequence_topology": output_dir / "07_sequence_topology.png",
    }
    render_empty_world(outputs["empty_world"])
    render_object_world(outputs["object_world"])
    render_snake_vector(outputs["snake_vector"])
    render_observer(outputs["observer"])
    render_attention_gates(outputs["attention_gates"])
    render_one_encounter(outputs["one_encounter"])
    render_two_observers(outputs["two_observers"])
    render_sequence(outputs["sequence_topology"])
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_all(args.output_dir)


if __name__ == "__main__":
    main()
