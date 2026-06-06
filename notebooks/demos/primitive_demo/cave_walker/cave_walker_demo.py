from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from cave.observation.producers.sources.primitive import (
    PrimitiveVectorStep,
    nearest_prototype,
    rollout_vectors,
)
from cave.presentation.filmstrip import draw_image_on_axis
from cave.presentation.sprites import walker_rendering as shared_walker
from cave.presentation.sprites import (
    GBA_SPRITE_ASSET_DIR,
    PixelSpriteAnimator,
    SpriteAssetStore,
    SpriteClip,
    eased_camera_x,
    object_world_x,
    scroll_frame_tuples,
    scroll_frames as sprite_scroll_frames,
)

from cave_walker_objects import (
    EPISODE,
    ETA,
    MEMORY_INITIAL,
    OBJECTS,
    TREE_VARIANTS,
    episode_features,
    prototype_features,
    tree_variant_for_episode_index,
)


COLORS = {
    "actual": "#0072B2",
    "expected": "#D55E00",
    "error": "#CC79A7",
    "memory": "#009E73",
    "grid": "#D1D5DB",
    "panel": "#F8FAFC",
    "text": "#111827",
    "muted": "#6B7280",
    "world": "#E7F0DC",
    "ground": "#B7C39B",
}

ASSET_DIR = GBA_SPRITE_ASSET_DIR
NATIVE_WIDTH = 240
NATIVE_HEIGHT = 160
SUBJECT_SCREEN_X = 72
ENCOUNTER_SCREEN_X = 110
OBJECT_WORLD_START = 130
OBJECT_SPACING = 74
GROUND_Y = 136
OBJECT_SPRITE_DIR = "sprites/objects"
DEFAULT_SUBJECT_SPRITE_DIR = "sprites/subjects/default"

SPRITE_PALETTE = {
    "outline": "#1F2430",
    "sky": "#9DD7F5",
    "expectation": "#E87518",
    "error": "#D83A68",
    "memory": "#129B63",
    "white": "#FFFFFF",
}
SPRITE_ASSETS = SpriteAssetStore(ASSET_DIR, palette=SPRITE_PALETTE)
SPRITE_ANIMATOR = PixelSpriteAnimator(
    SPRITE_ASSETS,
    width=NATIVE_WIDTH,
    height=NATIVE_HEIGHT,
    default_ground_y=GROUND_Y,
)
SUBJECT_CLIPS = {
    "expect": SpriteClip("subject_expect", 4, "subject_thinking.png"),
    "notice": SpriteClip("subject_notice", 3, "subject_idle.png"),
    "surprised": SpriteClip("subject_surprised", 4, "subject_surprised.png"),
    "update": SpriteClip("subject_update", 4, "subject_update.png"),
    "recover": SpriteClip("subject_recover", 3, "subject_idle.png"),
    "walk": SpriteClip("subject_walk", 6, "subject_walk_1.png"),
}


def build_walker_rollout(
    episode: tuple[str, ...] = EPISODE,
    *,
    eta: float = ETA,
    memory_initial: tuple[float, float] = MEMORY_INITIAL,
) -> list[PrimitiveVectorStep]:
    return rollout_vectors(
        episode_features(episode),
        eta=eta,
        memory_initial=memory_initial,
    )


def save_walker_frame(
    rows: Sequence[PrimitiveVectorStep],
    output: Path,
    *,
    episode: tuple[str, ...] = EPISODE,
    index: int | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if index is None:
        index = max(range(len(rows)), key=lambda row_index: rows[row_index].surprise)
    fig, axes = _make_figure()
    _draw_walker_dashboard(fig, axes, rows, index, episode=episode, scroll_progress=0.38)
    fig.savefig(output)
    plt.close(fig)


def save_walker_animation(
    rows: Sequence[PrimitiveVectorStep],
    output: Path,
    *,
    episode: tuple[str, ...] = EPISODE,
    fps: int = 2,
    hold_frames: int = 8,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = _scroll_frames(len(rows), hold_frames=hold_frames)
    fig, axes = _make_figure()

    def update(frame: tuple[int, float]):
        index, scroll_progress = frame
        _draw_walker_dashboard(fig, axes, rows, index, episode=episode, scroll_progress=scroll_progress)
        return axes

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 / max(1, fps),
        blit=False,
    )
    anim.save(output, writer=animation.PillowWriter(fps=fps), dpi=130)
    plt.close(fig)


def save_walker_filmstrip(
    rows: Sequence[PrimitiveVectorStep],
    output: Path,
    *,
    episode: tuple[str, ...] = EPISODE,
) -> None:
    """Render the primitive trajectory as one time-offset static object."""

    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16.0, 7.2), dpi=140)
    grid = fig.add_gridspec(2, 1, height_ratios=(0.72, 1.45), hspace=0.22)
    world_axis = fig.add_subplot(grid[0, 0])
    trajectory_axis = fig.add_subplot(grid[1, 0])
    fig.suptitle(
        "Cave Walker Filmstrip: the primitive trajectory as one object",
        fontsize=16,
        fontweight="bold",
        color=COLORS["text"],
    )
    _draw_world_filmstrip(world_axis, rows, episode)
    _draw_subjective_filmstrip(trajectory_axis, rows, episode)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def save_walker_filmstrip_blur(
    rows: Sequence[PrimitiveVectorStep],
    output: Path,
    *,
    episode: tuple[str, ...] = EPISODE,
    hold_frames: int = 18,
) -> None:
    """Render the same filmstrip with every GIF-like subframe accumulated."""

    output.parent.mkdir(parents=True, exist_ok=True)
    samples = _scroll_frames(len(rows), hold_frames=hold_frames)
    fig = plt.figure(figsize=(16.0, 7.2), dpi=140)
    grid = fig.add_gridspec(2, 1, height_ratios=(0.72, 1.45), hspace=0.22)
    world_axis = fig.add_subplot(grid[0, 0])
    trajectory_axis = fig.add_subplot(grid[1, 0])
    fig.suptitle(
        "Cave Walker Filmstrip: dense afterimage of the whole GIF",
        fontsize=16,
        fontweight="bold",
        color=COLORS["text"],
    )
    _draw_world_filmstrip_blur(world_axis, rows, episode, samples)
    _draw_subjective_filmstrip_blur(trajectory_axis, rows, episode, samples)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def run_demo(output_dir: Path, *, fps: int = 2) -> dict[str, str]:
    rows = build_walker_rollout()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "rollout_json": output_dir / "cave_walker_rollout.json",
        "frame": output_dir / "cave_walker_frame.png",
        "animation": output_dir / "cave_walker.gif",
        "filmstrip": output_dir / "cave_walker_filmstrip.png",
        "filmstrip_blur": output_dir / "cave_walker_filmstrip_blur.png",
    }
    paths["rollout_json"].write_text(
        json.dumps(
            {
                "episode": list(EPISODE),
                "eta": ETA,
                "memory_initial": MEMORY_INITIAL,
                "objects": {key: asdict(value) for key, value in OBJECTS.items()},
                "tree_variants": [asdict(variant) for variant in TREE_VARIANTS],
                "rollout": [_row_payload(row, object_id) for row, object_id in zip(rows, EPISODE)],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_walker_frame(rows, paths["frame"])
    save_walker_animation(rows, paths["animation"], fps=fps)
    save_walker_filmstrip(rows, paths["filmstrip"])
    save_walker_filmstrip_blur(rows, paths["filmstrip_blur"])
    return {name: str(path) for name, path in paths.items()}


def _make_figure():
    fig = plt.figure(figsize=(13.6, 9.4), dpi=130)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.065, top=0.91, wspace=0.18, hspace=0.34)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.22), width_ratios=(1.18, 1.0))
    world_axis = fig.add_subplot(grid[0, 0])
    hud_axis = fig.add_subplot(grid[0, 1])
    trajectory_axis = fig.add_subplot(grid[1, 0])
    topology_axis = fig.add_subplot(grid[1, 1])
    return fig, [world_axis, hud_axis, trajectory_axis, topology_axis]


def _draw_walker_dashboard(
    fig,
    axes: Sequence[Axes],
    rows: Sequence[PrimitiveVectorStep],
    index: int,
    *,
    episode: tuple[str, ...],
    scroll_progress: float = 0.0,
) -> None:
    fig.suptitle(
        "Cave Walker: object world over the primitive recurrence",
        fontsize=16,
        fontweight="bold",
        color=COLORS["text"],
    )
    for axis in axes:
        axis.clear()
    row = rows[index]
    _draw_world_strip(axes[0], row, index, episode, scroll_progress)
    _draw_hud(axes[1], row, episode[index])
    _draw_internal_map(axes[2], rows, index, episode)
    _draw_walker_topology(axes[3], rows, index)


def _draw_world_strip(
    axis: Axes,
    row: PrimitiveVectorStep,
    index: int,
    episode: tuple[str, ...],
    scroll_progress: float,
) -> None:
    axis.set_title("Side-scrolling world: continuous motion, discrete encounter", loc="left", fontsize=12, fontweight="bold")
    viewport = _render_gba_viewport(row, index, episode, scroll_progress)
    axis.imshow(viewport, interpolation="nearest", extent=(0, NATIVE_WIDTH, 0, NATIVE_HEIGHT))
    axis.set_xlim(0, NATIVE_WIDTH)
    axis.set_ylim(0, NATIVE_HEIGHT)
    axis.set_facecolor("#9DD7F5")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_aspect("equal", adjustable="box")


def _render_gba_viewport(
    row: PrimitiveVectorStep,
    index: int,
    episode: tuple[str, ...],
    scroll_progress: float,
    *,
    subject_sprite_dir: str = DEFAULT_SUBJECT_SPRITE_DIR,
) -> np.ndarray:
    camera_x = _camera_x(index, scroll_progress, len(episode))
    canvas = SPRITE_ANIMATOR.canvas("sky")
    _paste_looped_layer(canvas, _gba_asset("background/hills_far_bubbly.png"), camera_x, factor=0.18, y=42)
    _paste_looped_layer(canvas, _gba_asset("background/hills_near_bubbly.png"), camera_x, factor=0.44, y=56)
    _draw_clouds(canvas, camera_x)
    _draw_tilemap(canvas, camera_x)
    draw = ImageDraw.Draw(canvas, "RGBA")
    _draw_encounter_zone(draw, row, scroll_progress)
    for tile_index, object_id in enumerate(episode):
        screen_x = int(round(_object_world_x(tile_index) - camera_x))
        if screen_x < -40 or screen_x > NATIVE_WIDTH + 40:
            continue
        obj = OBJECTS[object_id]
        sprite_name = _gba_object_sprite(object_id, is_current=tile_index == index, row=row, scroll_progress=scroll_progress)
        sprite = _scaled_sprite(_gba_asset(f"{OBJECT_SPRITE_DIR}/{sprite_name}"), _object_scale(object_id, episode, tile_index))
        _paste_grounded_sprite(canvas, sprite, screen_x, GROUND_Y)
        label_y = GROUND_Y - sprite.height - 6
        _draw_native_text(draw, obj.label, (screen_x, label_y), anchor="mm", fill=_hex("outline"))
        if tile_index == index:
            color = _hex("error") if row.surprise > 0.45 and abs(screen_x - ENCOUNTER_SCREEN_X) < 11 else _hex("memory")
            half_width = max(18, sprite.width // 2 + 2)
            draw.rectangle((screen_x - half_width, GROUND_Y - sprite.height - 2, screen_x + half_width, GROUND_Y + 2), outline=color, width=2)
    subject_dir, subject_sprite = _subject_sprite(row, scroll_progress, subject_sprite_dir=subject_sprite_dir)
    bob = int(round(np.sin((index + scroll_progress) * np.pi * 2) * 2))
    if row.surprise > 0.45 and scroll_progress < 0.25:
        draw.ellipse((SUBJECT_SCREEN_X - 18, GROUND_Y - 36, SUBJECT_SCREEN_X + 18, GROUND_Y), fill=(*_rgb("error"), 48))
    _paste_grounded_sprite(canvas, _gba_asset(f"{subject_dir}/{subject_sprite}"), SUBJECT_SCREEN_X, GROUND_Y + bob)
    draw.line((SUBJECT_SCREEN_X + 12, GROUND_Y - 8, ENCOUNTER_SCREEN_X - 12, GROUND_Y - 8), fill=(*_rgb("outline"), 150), width=1)
    draw.polygon(
        [(ENCOUNTER_SCREEN_X - 12, GROUND_Y - 8), (ENCOUNTER_SCREEN_X - 17, GROUND_Y - 11), (ENCOUNTER_SCREEN_X - 17, GROUND_Y - 5)],
        fill=(*_rgb("outline"), 150),
    )
    _draw_native_text(draw, f"t={row.t}", (SUBJECT_SCREEN_X, GROUND_Y + 13), anchor="mm", fill=_hex("outline"))
    _draw_expectation_bubble(draw, row)
    return np.asarray(canvas.convert("RGBA"))


def _draw_encounter_zone(draw: ImageDraw.ImageDraw, row: PrimitiveVectorStep, scroll_progress: float) -> None:
    fill_color = _rgb("error") if _is_error_phase(row, scroll_progress) else _rgb("memory")
    draw.rectangle((ENCOUNTER_SCREEN_X - 5, 25, ENCOUNTER_SCREEN_X + 5, GROUND_Y + 2), fill=(*fill_color, 42))
    draw.line((ENCOUNTER_SCREEN_X, 25, ENCOUNTER_SCREEN_X, GROUND_Y + 2), fill=(*fill_color, 125), width=1)
    _draw_native_text(draw, "zone", (ENCOUNTER_SCREEN_X + 14, 18), anchor="lm", fill=_hex("outline"))


def _draw_expectation_bubble(draw: ImageDraw.ImageDraw, row: PrimitiveVectorStep) -> None:
    expected_id, _ = nearest_prototype(row.expected, prototype_features())
    label = f"expects {OBJECTS[expected_id].label}"
    x0, y0, x1, y1 = 18, 12, 96, 31
    draw.rounded_rectangle((x0, y0, x1, y1), radius=3, fill=(*_rgb("white"), 220), outline=_hex("expectation"), width=1)
    _draw_native_text(draw, label, ((x0 + x1) // 2, (y0 + y1) // 2), anchor="mm", fill=_hex("outline"))


def _draw_clouds(canvas: Image.Image, camera_x: float) -> None:
    cloud = _gba_asset("background/cloud_bubbly.png")
    for world_x, y in ((28, 18), (158, 10), (298, 24), (430, 15)):
        x = int(round(world_x - (camera_x * 0.14) % 260))
        for offset in (-260, 0, 260):
            canvas.alpha_composite(cloud, (x + offset, y))


def _draw_tilemap(canvas: Image.Image, camera_x: float) -> None:
    grass = _gba_asset("tiles/grass_tile.png")
    dirt = _gba_asset("tiles/dirt_tile.png")
    edge = _gba_asset("tiles/platform_edge.png")
    start_tile = int(camera_x // 16) - 2
    for tile in range(start_tile, start_tile + 20):
        screen_x = int(tile * 16 - camera_x)
        canvas.alpha_composite(edge, (screen_x, GROUND_Y - 16))
        canvas.alpha_composite(grass, (screen_x, GROUND_Y))
        for y in range(GROUND_Y + 16, NATIVE_HEIGHT, 16):
            canvas.alpha_composite(dirt, (screen_x, y))


def _paste_looped_layer(canvas: Image.Image, layer: Image.Image, camera_x: float, *, factor: float, y: int) -> None:
    SPRITE_ANIMATOR.paste_looped_layer(canvas, layer, camera_x, factor=factor, y=y)


def _subject_sprite(
    row: PrimitiveVectorStep,
    scroll_progress: float,
    *,
    subject_sprite_dir: str = DEFAULT_SUBJECT_SPRITE_DIR,
) -> tuple[str, str]:
    progress = float(np.clip(scroll_progress, 0.0, 1.0))
    if progress < 0.18:
        return _subject_clip_frame(SUBJECT_CLIPS["expect"], progress / 0.18, subject_sprite_dir=subject_sprite_dir)
    if progress < 0.32:
        return _subject_clip_frame(SUBJECT_CLIPS["notice"], (progress - 0.18) / 0.14, subject_sprite_dir=subject_sprite_dir)
    if row.surprise > 0.45 and progress < 0.52:
        return _subject_clip_frame(SUBJECT_CLIPS["surprised"], (progress - 0.32) / 0.20, subject_sprite_dir=subject_sprite_dir)
    update_start = 0.52 if row.surprise > 0.45 else 0.32
    if progress < update_start + 0.20:
        return _subject_clip_frame(SUBJECT_CLIPS["update"], (progress - update_start) / 0.20, subject_sprite_dir=subject_sprite_dir)
    if progress < update_start + 0.36:
        return _subject_clip_frame(SUBJECT_CLIPS["recover"], (progress - update_start - 0.20) / 0.16, subject_sprite_dir=subject_sprite_dir)
    return _subject_clip_frame(SUBJECT_CLIPS["walk"], progress, subject_sprite_dir=subject_sprite_dir)


def _subject_clip_frame(
    clip: SpriteClip,
    progress: float,
    *,
    subject_sprite_dir: str,
) -> tuple[str, str]:
    frame = int(np.clip(progress, 0.0, 0.999) * clip.frame_count)
    frame_name = f"{clip.prefix}_{frame}.png"
    for directory, name in (
        (subject_sprite_dir, frame_name),
        (DEFAULT_SUBJECT_SPRITE_DIR, frame_name),
        (subject_sprite_dir, clip.fallback),
        (DEFAULT_SUBJECT_SPRITE_DIR, clip.fallback),
    ):
        if SPRITE_ASSETS.has_image(Path(directory) / name):
            return directory, name
    return DEFAULT_SUBJECT_SPRITE_DIR, clip.fallback


def _gba_object_sprite(object_id: str, *, is_current: bool, row: PrimitiveVectorStep, scroll_progress: float) -> str:
    if object_id == "snake":
        if is_current and _is_error_phase(row, scroll_progress):
            return "object_snake_alert.png"
        return "object_snake_1.png" if int(scroll_progress * 8) % 2 == 0 else "object_snake_2.png"
    if object_id == "rock":
        return "object_rock.png"
    return "object_tree.png"


def _is_error_phase(row: PrimitiveVectorStep, scroll_progress: float) -> bool:
    return row.surprise > 0.45 and 0.32 <= scroll_progress < 0.52


def _animation_frame(prefix: str, frame_count: int, progress: float, *, fallback: str) -> str:
    return SPRITE_ASSETS.animation_frame(prefix, frame_count, progress, fallback=fallback)


def _object_scale(object_id: str, episode: tuple[str, ...], index: int) -> float:
    if object_id == "tree":
        return tree_variant_for_episode_index(episode, index).scale
    return 1.0


def _object_display_width(base_width: float, object_id: str, episode: tuple[str, ...], index: int) -> float:
    return base_width * _object_scale(object_id, episode, index)


def _scaled_sprite(sprite: Image.Image, scale: float) -> Image.Image:
    return SPRITE_ANIMATOR.scaled(sprite, scale)


def _paste_grounded_sprite(canvas: Image.Image, sprite: Image.Image, center_x: int, ground_y: int) -> None:
    SPRITE_ANIMATOR.paste_grounded(canvas, sprite, center_x, ground_y)


def _draw_native_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    *,
    anchor: str,
    fill: str | tuple[int, int, int, int],
) -> None:
    SPRITE_ANIMATOR.native_text(draw, text, xy, anchor=anchor, fill=fill)


def _gba_asset(path: str) -> Image.Image:
    return SPRITE_ASSETS.image(path)


def _rgb(name: str) -> tuple[int, int, int]:
    return SPRITE_ASSETS.rgb(name)


def _hex(name: str) -> str:
    return SPRITE_ASSETS.color(name)


def _scroll_frames(step_count: int, *, hold_frames: int) -> list[tuple[int, float]]:
    return scroll_frame_tuples(sprite_scroll_frames(step_count, hold_frames=hold_frames))


def _object_world_x(index: int) -> float:
    return object_world_x(index, start=OBJECT_WORLD_START, spacing=OBJECT_SPACING)


def _camera_x(index: int, scroll_progress: float, step_count: int) -> float:
    return eased_camera_x(
        index,
        scroll_progress,
        step_count,
        object_start=OBJECT_WORLD_START,
        object_spacing=OBJECT_SPACING,
        encounter_x=ENCOUNTER_SCREEN_X,
    )


def _draw_hud(axis: Axes, row: PrimitiveVectorStep, actual_id: str) -> None:
    _style_panel(axis, "Subject HUD: current primitive update")
    prototypes = prototype_features()
    expected_id, expected_distance = nearest_prototype(row.expected, prototypes)
    memory_id, _ = nearest_prototype(row.memory, prototypes)
    actual_label = OBJECTS[actual_id].label
    expected_label = OBJECTS[expected_id].label
    memory_label = OBJECTS[memory_id].label
    band = _surprise_band(row.surprise)
    lines = [
        ("expects", f"{expected_label}-like", COLORS["expected"]),
        ("actual", actual_label, COLORS["actual"]),
        ("error", f"{band} ({row.surprise:.3f})", COLORS["error"]),
        ("memory", f"shifting toward {memory_label}", COLORS["memory"]),
    ]
    y = 0.86
    for label, value, color in lines:
        axis.text(0.08, y, label, transform=axis.transAxes, fontsize=10, color=COLORS["muted"])
        axis.text(0.30, y, value, transform=axis.transAxes, fontsize=12, color=color, fontweight="bold")
        y -= 0.16
    _draw_vector_bar(axis, "E_t", row.expected, 0.28, COLORS["expected"])
    _draw_vector_bar(axis, "U_t", row.actual, 0.18, COLORS["actual"])
    _draw_vector_bar(axis, "M_t", row.memory, 0.08, COLORS["memory"])
    axis.text(
        0.08,
        0.50,
        f"nearest expectation distance: {expected_distance:.3f}",
        transform=axis.transAxes,
        fontsize=9,
        color=COLORS["muted"],
    )


def _draw_internal_map(
    axis: Axes,
    rows: Sequence[PrimitiveVectorStep],
    index: int,
    episode: tuple[str, ...],
) -> None:
    _style_panel(axis, "Subjective trajectory: E_t -> U_t -> M_t")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("feature x: danger / salience")
    axis.set_ylabel("feature y: height / liveliness")
    axis.grid(True, color=COLORS["grid"], linewidth=0.8, alpha=0.8)
    for obj in OBJECTS.values():
        x, y = obj.features
        axis.scatter([x], [y], marker="s", s=85, color="#E5E7EB", edgecolor=COLORS["text"], linewidth=0.8, zorder=1)
        axis.text(x + 0.015, y + 0.015, obj.label, fontsize=8, color=COLORS["text"])
    memories = np.array([row.memory for row in rows[: index + 1]], dtype=float)
    if len(memories):
        axis.plot(memories[:, 0], memories[:, 1], color=COLORS["memory"], linewidth=2.0, label="memory path", zorder=2)
    row = rows[index]
    expected = np.array(row.expected)
    actual = np.array(row.actual)
    memory = np.array(row.memory)
    axis.scatter([expected[0]], [expected[1]], s=105, color=COLORS["expected"], label="E_t expected", zorder=4)
    axis.scatter([actual[0]], [actual[1]], s=105, color=COLORS["actual"], label="U_t actual", zorder=5)
    axis.scatter([memory[0]], [memory[1]], s=105, color=COLORS["memory"], label="M_t memory", zorder=6)
    axis.add_patch(
        FancyArrowPatch(
            expected,
            actual,
            arrowstyle="->",
            color=COLORS["error"],
            linewidth=2.4,
            mutation_scale=15,
            zorder=3,
        )
    )
    axis.plot([actual[0], memory[0]], [actual[1], memory[1]], color=COLORS["memory"], linewidth=1.5, alpha=0.65)
    axis.text(0.03, 0.94, f"world object: {OBJECTS[episode[index]].label}", transform=axis.transAxes, fontsize=10)
    axis.legend(loc="lower right", fontsize=8, frameon=False)


def _draw_walker_topology(axis: Axes, rows: Sequence[PrimitiveVectorStep], index: int) -> None:
    _style_panel(axis, "Accumulated topology-like field L_t(x, y)")
    shown_rows = rows[: index + 1]
    xx, yy, density, mass = _walker_topology_density(shown_rows)
    axis.contourf(xx, yy, density, levels=14, cmap="viridis")
    axis.contour(xx, yy, density, levels=7, colors="white", linewidths=0.4, alpha=0.55)
    memories = np.array([row.memory for row in shown_rows], dtype=float)
    actuals = np.array([row.actual for row in shown_rows], dtype=float)
    expecteds = np.array([row.expected for row in shown_rows], dtype=float)
    axis.plot(memories[:, 0], memories[:, 1], color="white", linewidth=2.0, alpha=0.95)
    axis.scatter(expecteds[:, 0], expecteds[:, 1], s=30, color=COLORS["expected"], edgecolor="white", linewidth=0.4)
    axis.scatter(actuals[:, 0], actuals[:, 1], s=30, color=COLORS["actual"], edgecolor="white", linewidth=0.4)
    axis.scatter(memories[:, 0], memories[:, 1], s=36, color=COLORS["memory"], edgecolor="white", linewidth=0.5)
    for obj in OBJECTS.values():
        x, y = obj.features
        axis.text(x + 0.015, y + 0.015, obj.label, fontsize=8, color="white")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("feature x: danger / salience")
    axis.set_ylabel("feature y: height / liveliness")
    axis.grid(True, color="white", linewidth=0.55, alpha=0.42)
    axis.text(
        0.03,
        0.94,
        f"density mass = {mass[-1]:.1f}",
        transform=axis.transAxes,
        fontsize=8,
        color="white",
        bbox={"facecolor": "#111827", "alpha": 0.72, "edgecolor": "none", "pad": 3},
    )


def _walker_topology_density(
    rows: Sequence[PrimitiveVectorStep],
    *,
    grid_size: int = 90,
    sigma: float = 0.055,
    decay: float = 0.94,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    grid = np.linspace(0.0, 1.0, grid_size)
    xx, yy = np.meshgrid(grid, grid)
    density = np.zeros_like(xx)
    mass: list[float] = []
    weights = {"expected": 0.55, "actual": 1.00, "memory": 0.82}
    for row in rows:
        density *= decay
        density += weights["expected"] * _gaussian_2d(xx, yy, row.expected, sigma)
        density += weights["actual"] * _gaussian_2d(xx, yy, row.actual, sigma)
        density += weights["memory"] * _gaussian_2d(xx, yy, row.memory, sigma)
        mass.append(float(density.sum()))
    return xx, yy, density, mass


def _draw_world_filmstrip(
    axis: Axes,
    rows: Sequence[PrimitiveVectorStep],
    episode: tuple[str, ...],
) -> None:
    _style_panel(axis, "World strip: the same walk laid out as one continuous path")
    spacing = 1.28
    xs = np.arange(len(rows), dtype=float) * spacing
    axis.set_xlim(-0.55, xs[-1] + 0.65)
    axis.set_ylim(0.0, 1.0)
    axis.plot(
        [xs[0] - 0.18, xs[-1] + 0.18],
        [0.34, 0.34],
        color="#7C8A5F",
        linewidth=3.2,
        alpha=0.55,
    )
    axis.plot(
        xs,
        np.full_like(xs, 0.34),
        color=COLORS["memory"],
        linewidth=1.5,
        alpha=0.55,
    )
    for index, (row, object_id) in enumerate(zip(rows, episode)):
        x = xs[index]
        band_color = COLORS["error"] if row.surprise > 0.45 else COLORS["memory"]
        axis.axvline(x, ymin=0.05, ymax=0.92, color=band_color, alpha=0.10, linewidth=10)
        subject_sprite = "subject_surprised.png" if row.surprise > 0.45 else "subject_walk_1.png"
        _draw_image(
            axis,
            _gba_asset(f"{DEFAULT_SUBJECT_SPRITE_DIR}/{subject_sprite}"),
            (x - 0.18, 0.48),
            0.26,
            zorder=5,
        )
        object_sprite = _gba_asset(
            f"{OBJECT_SPRITE_DIR}/{_gba_object_sprite(object_id, is_current=True, row=row, scroll_progress=0.42)}"
        )
        _draw_image(
            axis,
            object_sprite,
            (x + 0.18, 0.47),
            _object_display_width(0.28, object_id, episode, index),
            zorder=6,
        )
        axis.text(
            x,
            0.11,
            f"t={row.t}",
            ha="center",
            va="center",
            fontsize=8,
            color=COLORS["muted"],
        )
        axis.text(
            x,
            0.84,
            OBJECTS[object_id].label,
            ha="center",
            va="center",
            fontsize=9,
            color=COLORS["text"],
        )
    axis.set_xticks([])
    axis.set_yticks([])


def _draw_world_filmstrip_blur(
    axis: Axes,
    rows: Sequence[PrimitiveVectorStep],
    episode: tuple[str, ...],
    samples: Sequence[tuple[int, float]],
) -> None:
    _style_panel(axis, "World strip: every GIF subframe accumulated as an afterimage")
    spacing = 1.28
    xs = np.arange(len(rows), dtype=float) * spacing
    axis.set_xlim(-0.55, xs[-1] + 0.65)
    axis.set_ylim(0.0, 1.0)
    axis.plot(
        [xs[0] - 0.18, xs[-1] + 0.18],
        [0.34, 0.34],
        color="#7C8A5F",
        linewidth=3.2,
        alpha=0.55,
    )
    for index, object_id in enumerate(episode):
        row = rows[index]
        x = xs[index] + 0.18
        band_color = COLORS["error"] if row.surprise > 0.45 else COLORS["memory"]
        axis.axvline(xs[index], ymin=0.05, ymax=0.92, color=band_color, alpha=0.08, linewidth=10)
        object_sprite = _gba_asset(f"{OBJECT_SPRITE_DIR}/{OBJECTS[object_id].sprite}")
        _draw_image(
            axis,
            object_sprite,
            (x, 0.47),
            _object_display_width(0.28, object_id, episode, index),
            alpha=0.94,
            zorder=8,
        )
        axis.text(xs[index], 0.11, f"t={row.t}", ha="center", va="center", fontsize=8, color=COLORS["muted"])
        axis.text(xs[index], 0.84, OBJECTS[object_id].label, ha="center", va="center", fontsize=9, color=COLORS["text"])

    for sample_index, (index, progress) in enumerate(samples):
        row = rows[index]
        x = (index + progress) * spacing - 0.18
        if index >= len(rows) - 1:
            x = xs[index] - 0.18
        subject_dir, sprite = _subject_sprite(row, progress)
        alpha = 0.16
        if sample_index == 0 or sample_index == len(samples) - 1:
            alpha = 0.34
        _draw_image(axis, _gba_asset(f"{subject_dir}/{sprite}"), (x, 0.48), 0.25, alpha=alpha, zorder=5)
    axis.plot(
        [(index + progress) * spacing - 0.18 for index, progress in samples],
        [0.48 for _ in samples],
        color=COLORS["memory"],
        linewidth=1.0,
        alpha=0.45,
        zorder=4,
    )
    axis.set_xticks([])
    axis.set_yticks([])


def _draw_subjective_filmstrip(
    axis: Axes,
    rows: Sequence[PrimitiveVectorStep],
    episode: tuple[str, ...],
) -> None:
    _style_panel(
        axis,
        "Subjective trajectory filmstrip: each slice keeps feature geometry, then shifts by time",
    )
    stride = 1.34
    xx, yy, density = _filmstrip_density(rows, time_stride=stride)
    axis.contourf(xx, yy, density, levels=16, cmap="viridis", alpha=0.72)
    axis.contour(xx, yy, density, levels=8, colors="white", linewidths=0.35, alpha=0.40)

    memory_points = np.array(
        [
            [index * stride + row.memory[0], row.memory[1]]
            for index, row in enumerate(rows)
        ],
        dtype=float,
    )
    axis.plot(
        memory_points[:, 0],
        memory_points[:, 1],
        color="white",
        linewidth=4.2,
        alpha=0.62,
        zorder=3,
    )
    axis.plot(
        memory_points[:, 0],
        memory_points[:, 1],
        color=COLORS["memory"],
        linewidth=2.0,
        label="memory path",
        zorder=4,
    )

    for index, (row, object_id) in enumerate(zip(rows, episode)):
        offset = index * stride
        axis.axvline(offset, color="white", linewidth=0.7, alpha=0.32, zorder=1)
        expected = np.array([offset + row.expected[0], row.expected[1]], dtype=float)
        actual = np.array([offset + row.actual[0], row.actual[1]], dtype=float)
        memory = np.array([offset + row.memory[0], row.memory[1]], dtype=float)
        axis.add_patch(
            FancyArrowPatch(
                expected,
                actual,
                arrowstyle="->",
                color=COLORS["error"],
                linewidth=2.0,
                mutation_scale=13,
                alpha=0.88,
                zorder=5,
            )
        )
        axis.plot(
            [actual[0], memory[0]],
            [actual[1], memory[1]],
            color=COLORS["memory"],
            linewidth=1.5,
            alpha=0.78,
            zorder=5,
        )
        _draw_image(
            axis,
            _gba_asset(f"{OBJECT_SPRITE_DIR}/{OBJECTS[object_id].sprite}"),
            tuple(actual),
            _object_display_width(0.16, object_id, episode, index),
            alpha=0.92,
            zorder=7,
        )
        axis.scatter(
            [expected[0]],
            [expected[1]],
            s=84,
            color=COLORS["expected"],
            edgecolor="white",
            linewidth=0.7,
            zorder=8,
        )
        axis.scatter(
            [actual[0]],
            [actual[1]],
            s=58,
            color=COLORS["actual"],
            edgecolor="white",
            linewidth=0.7,
            zorder=8,
        )
        axis.scatter(
            [memory[0]],
            [memory[1]],
            s=84,
            color=COLORS["memory"],
            edgecolor="white",
            linewidth=0.7,
            zorder=8,
        )
        axis.text(offset + 0.03, 0.04, f"t={row.t}", fontsize=8, color="white", zorder=9)

    axis.set_xlim(-0.08, (len(rows) - 1) * stride + 1.08)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("auto")
    axis.set_xlabel("time offset + feature x")
    axis.set_ylabel("feature y: height / liveliness")
    axis.text(
        0.015,
        0.94,
        "orange E_t, blue/sprite U_t, green M_t; density is the same primitive deposits laid out through time",
        transform=axis.transAxes,
        fontsize=9,
        color="white",
        bbox={"facecolor": "#111827", "alpha": 0.72, "edgecolor": "none", "pad": 4},
        zorder=10,
    )


def _draw_subjective_filmstrip_blur(
    axis: Axes,
    rows: Sequence[PrimitiveVectorStep],
    episode: tuple[str, ...],
    samples: Sequence[tuple[int, float]],
) -> None:
    _style_panel(
        axis,
        "Subjective trajectory filmstrip: update anchors plus every GIF subframe blurred together",
    )
    stride = 1.34
    xx, yy, density = _filmstrip_blur_density(rows, samples, time_stride=stride)
    axis.contourf(xx, yy, density, levels=18, cmap="viridis", alpha=0.74)
    axis.contour(xx, yy, density, levels=9, colors="white", linewidths=0.28, alpha=0.35)

    sampled_memory = np.array(
        [
            [(index + progress) * stride + rows[index].memory[0], rows[index].memory[1]]
            for index, progress in samples
        ],
        dtype=float,
    )
    axis.plot(sampled_memory[:, 0], sampled_memory[:, 1], color="white", linewidth=5.0, alpha=0.28, zorder=3)
    axis.plot(sampled_memory[:, 0], sampled_memory[:, 1], color=COLORS["memory"], linewidth=1.35, alpha=0.55, zorder=4)

    for index, progress in samples:
        row = rows[index]
        offset = (index + progress) * stride
        expected = np.array([offset + row.expected[0], row.expected[1]], dtype=float)
        actual = np.array([offset + row.actual[0], row.actual[1]], dtype=float)
        memory = np.array([offset + row.memory[0], row.memory[1]], dtype=float)
        axis.add_patch(
            FancyArrowPatch(
                expected,
                actual,
                arrowstyle="->",
                color=COLORS["error"],
                linewidth=0.75,
                mutation_scale=7,
                alpha=0.18,
                zorder=5,
            )
        )
        axis.plot(
            [actual[0], memory[0]],
            [actual[1], memory[1]],
            color=COLORS["memory"],
            linewidth=0.8,
            alpha=0.20,
            zorder=5,
        )

    for index, (row, object_id) in enumerate(zip(rows, episode)):
        offset = index * stride
        expected = np.array([offset + row.expected[0], row.expected[1]], dtype=float)
        actual = np.array([offset + row.actual[0], row.actual[1]], dtype=float)
        memory = np.array([offset + row.memory[0], row.memory[1]], dtype=float)
        axis.axvline(offset, color="white", linewidth=0.7, alpha=0.30, zorder=1)
        _draw_image(
            axis,
            _gba_asset(f"{OBJECT_SPRITE_DIR}/{OBJECTS[object_id].sprite}"),
            tuple(actual),
            _object_display_width(0.16, object_id, episode, index),
            alpha=0.78,
            zorder=8,
        )
        axis.scatter([expected[0]], [expected[1]], s=76, color=COLORS["expected"], edgecolor="white", linewidth=0.7, zorder=9)
        axis.scatter([actual[0]], [actual[1]], s=52, color=COLORS["actual"], edgecolor="white", linewidth=0.7, zorder=9)
        axis.scatter([memory[0]], [memory[1]], s=76, color=COLORS["memory"], edgecolor="white", linewidth=0.7, zorder=9)
        axis.text(offset + 0.03, 0.04, f"t={row.t}", fontsize=8, color="white", zorder=10)

    axis.set_xlim(-0.08, (len(rows) - 1) * stride + 1.08)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("auto")
    axis.set_xlabel("time offset + feature x")
    axis.set_ylabel("feature y: height / liveliness")
    axis.text(
        0.015,
        0.94,
        "faint strokes are GIF subframes; bright dots are encounter/update anchors",
        transform=axis.transAxes,
        fontsize=9,
        color="white",
        bbox={"facecolor": "#111827", "alpha": 0.72, "edgecolor": "none", "pad": 4},
        zorder=10,
    )


def _filmstrip_density(
    rows: Sequence[PrimitiveVectorStep],
    *,
    time_stride: float,
    grid_size_x: int = 360,
    grid_size_y: int = 96,
    sigma: float = 0.055,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_axis = np.linspace(-0.08, (len(rows) - 1) * time_stride + 1.08, grid_size_x)
    y_axis = np.linspace(0.0, 1.0, grid_size_y)
    xx, yy = np.meshgrid(x_axis, y_axis)
    density = np.zeros_like(xx)
    weights = {"expected": 0.55, "actual": 1.00, "memory": 0.82}
    for index, row in enumerate(rows):
        offset = index * time_stride
        density += weights["expected"] * _gaussian_2d(
            xx,
            yy,
            (offset + row.expected[0], row.expected[1]),
            sigma,
        )
        density += weights["actual"] * _gaussian_2d(
            xx,
            yy,
            (offset + row.actual[0], row.actual[1]),
            sigma,
        )
        density += weights["memory"] * _gaussian_2d(
            xx,
            yy,
            (offset + row.memory[0], row.memory[1]),
            sigma,
        )
    max_density = float(np.max(density))
    if max_density > 0.0:
        density = density / max_density
    return xx, yy, density


def _filmstrip_blur_density(
    rows: Sequence[PrimitiveVectorStep],
    samples: Sequence[tuple[int, float]],
    *,
    time_stride: float,
    grid_size_x: int = 420,
    grid_size_y: int = 96,
    sigma: float = 0.055,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_axis = np.linspace(-0.08, (len(rows) - 1) * time_stride + 1.08, grid_size_x)
    y_axis = np.linspace(0.0, 1.0, grid_size_y)
    xx, yy = np.meshgrid(x_axis, y_axis)
    density = np.zeros_like(xx)
    weights = {"expected": 0.55, "actual": 1.00, "memory": 0.82}
    sample_weight = 1.0 / max(1, len(samples))
    for index, progress in samples:
        row = rows[index]
        offset = (index + progress) * time_stride
        density += sample_weight * weights["expected"] * _gaussian_2d(
            xx,
            yy,
            (offset + row.expected[0], row.expected[1]),
            sigma,
        )
        density += sample_weight * weights["actual"] * _gaussian_2d(
            xx,
            yy,
            (offset + row.actual[0], row.actual[1]),
            sigma,
        )
        density += sample_weight * weights["memory"] * _gaussian_2d(
            xx,
            yy,
            (offset + row.memory[0], row.memory[1]),
            sigma,
        )
    max_density = float(np.max(density))
    if max_density > 0.0:
        density = density / max_density
    return xx, yy, density


def _draw_image(
    axis: Axes,
    image: Image.Image,
    center: tuple[float, float],
    width: float,
    *,
    alpha: float = 1.0,
    zorder: int = 1,
) -> None:
    draw_image_on_axis(axis, image, center, width, alpha=alpha, zorder=zorder)


def _gaussian_2d(xx: np.ndarray, yy: np.ndarray, center: Sequence[float], sigma: float) -> np.ndarray:
    cx, cy = center
    distance_squared = (xx - cx) ** 2 + (yy - cy) ** 2
    return np.exp(-distance_squared / (2 * sigma**2))


def _draw_vector_bar(axis: Axes, label: str, values: Sequence[float], y: float, color: str) -> None:
    x0 = 0.30
    axis.text(0.08, y, label, transform=axis.transAxes, fontsize=9, color=COLORS["muted"])
    for offset, value in enumerate(values):
        axis.add_patch(
            Rectangle(
                (x0 + offset * 0.23, y - 0.015),
                float(value) * 0.18,
                0.035,
                transform=axis.transAxes,
                color=color,
                alpha=0.85,
            )
        )
        axis.text(
            x0 + offset * 0.23 + 0.19,
            y,
            f"{value:.2f}",
            transform=axis.transAxes,
            va="center",
            fontsize=8,
            color=COLORS["text"],
        )


def _style_panel(axis: Axes, title: str) -> None:
    axis.set_title(title, loc="left", fontsize=12, fontweight="bold", color=COLORS["text"])
    axis.set_facecolor(COLORS["panel"])
    for spine in axis.spines.values():
        spine.set_color("#CBD5E1")
    axis.set_xticks([])
    axis.set_yticks([])


def _surprise_band(value: float) -> str:
    if value >= 0.55:
        return "high"
    if value >= 0.25:
        return "medium"
    return "low"


def _row_payload(row: PrimitiveVectorStep, object_id: str) -> dict[str, object]:
    payload = asdict(row)
    payload["object_id"] = object_id
    payload["object_label"] = OBJECTS[object_id].label
    payload["expected_label"] = OBJECTS[nearest_prototype(row.expected, prototype_features())[0]].label
    payload["memory_label"] = OBJECTS[nearest_prototype(row.memory, prototype_features())[0]].label
    return payload


# Keep the notebook demo's historical private names, but route the reusable
# side-scroller implementation through the package-owned module.
build_walker_rollout = shared_walker.build_walker_rollout
_render_gba_viewport = shared_walker.render_gba_viewport
_scroll_frames = shared_walker.walker_scroll_frames
_draw_clouds = shared_walker._draw_clouds
_draw_tilemap = shared_walker._draw_tilemap
_paste_looped_layer = shared_walker._paste_looped_layer
_subject_sprite = shared_walker._subject_sprite
_gba_object_sprite = shared_walker._gba_object_sprite
_object_scale = shared_walker._object_scale
_paste_grounded_sprite = shared_walker._paste_grounded_sprite
_draw_native_text = shared_walker._draw_native_text
_gba_asset = shared_walker._gba_asset
_rgb = shared_walker._rgb
_hex = shared_walker._hex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Cave Walker over the primitive recurrence.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "out",
        help="Directory for generated Cave Walker frame, GIF, and rollout JSON.",
    )
    parser.add_argument("--fps", type=int, default=2, help="GIF frames per second.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_demo(args.output_dir, fps=args.fps)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
