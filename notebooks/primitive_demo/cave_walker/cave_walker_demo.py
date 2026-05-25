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
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from primitive_engine import PrimitiveVectorStep, nearest_prototype, rollout_vectors

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

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "gba"
NATIVE_WIDTH = 240
NATIVE_HEIGHT = 160
SUBJECT_SCREEN_X = 72
ENCOUNTER_SCREEN_X = 110
OBJECT_WORLD_START = 130
OBJECT_SPACING = 74
GROUND_Y = 136


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


def run_demo(output_dir: Path, *, fps: int = 2) -> dict[str, str]:
    rows = build_walker_rollout()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "rollout_json": output_dir / "cave_walker_rollout.json",
        "frame": output_dir / "cave_walker_frame.png",
        "animation": output_dir / "cave_walker.gif",
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
) -> np.ndarray:
    camera_x = _camera_x(index, scroll_progress, len(episode))
    canvas = Image.new("RGBA", (NATIVE_WIDTH, NATIVE_HEIGHT), _hex("sky"))
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
        sprite = _scaled_sprite(_gba_asset(f"sprites/{sprite_name}"), _object_scale(object_id, episode, tile_index))
        _paste_grounded_sprite(canvas, sprite, screen_x, GROUND_Y)
        label_y = GROUND_Y - sprite.height - 6
        _draw_native_text(draw, obj.label, (screen_x, label_y), anchor="mm", fill=_hex("outline"))
        if tile_index == index:
            color = _hex("error") if row.surprise > 0.45 and abs(screen_x - ENCOUNTER_SCREEN_X) < 11 else _hex("memory")
            half_width = max(18, sprite.width // 2 + 2)
            draw.rectangle((screen_x - half_width, GROUND_Y - sprite.height - 2, screen_x + half_width, GROUND_Y + 2), outline=color, width=2)
    subject_sprite = _subject_sprite(row, scroll_progress)
    bob = int(round(np.sin((index + scroll_progress) * np.pi * 2) * 2))
    if row.surprise > 0.45 and scroll_progress < 0.25:
        draw.ellipse((SUBJECT_SCREEN_X - 18, GROUND_Y - 36, SUBJECT_SCREEN_X + 18, GROUND_Y), fill=(*_rgb("error"), 48))
    _paste_grounded_sprite(canvas, _gba_asset(f"sprites/{subject_sprite}"), SUBJECT_SCREEN_X, GROUND_Y + bob)
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
    width = layer.width
    offset = int(round(-(camera_x * factor) % width))
    for x in range(offset - width, NATIVE_WIDTH + width, width):
        canvas.alpha_composite(layer, (x, y))


def _subject_sprite(row: PrimitiveVectorStep, scroll_progress: float) -> str:
    progress = float(np.clip(scroll_progress, 0.0, 1.0))
    if progress < 0.18:
        return _animation_frame("subject_expect", 4, progress / 0.18, fallback="subject_thinking.png")
    if progress < 0.32:
        return _animation_frame("subject_notice", 3, (progress - 0.18) / 0.14, fallback="subject_idle.png")
    if row.surprise > 0.45 and progress < 0.52:
        return _animation_frame("subject_surprised", 4, (progress - 0.32) / 0.20, fallback="subject_surprised.png")
    update_start = 0.52 if row.surprise > 0.45 else 0.32
    if progress < update_start + 0.20:
        return _animation_frame(
            "subject_update",
            4,
            (progress - update_start) / 0.20,
            fallback="subject_update.png",
        )
    if progress < update_start + 0.36:
        return _animation_frame(
            "subject_recover",
            3,
            (progress - update_start - 0.20) / 0.16,
            fallback="subject_idle.png",
        )
    return _animation_frame("subject_walk", 6, progress, fallback="subject_walk_1.png")


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
    frame = int(np.clip(progress, 0.0, 0.999) * frame_count)
    name = f"{prefix}_{frame}.png"
    if (ASSET_DIR / "sprites" / name).exists():
        return name
    return fallback


def _object_scale(object_id: str, episode: tuple[str, ...], index: int) -> float:
    if object_id == "tree":
        return tree_variant_for_episode_index(episode, index).scale
    return 1.0


def _scaled_sprite(sprite: Image.Image, scale: float) -> Image.Image:
    if scale == 1.0:
        return sprite
    size = (int(round(sprite.width * scale)), int(round(sprite.height * scale)))
    return sprite.resize(size, Image.Resampling.NEAREST)


def _paste_grounded_sprite(canvas: Image.Image, sprite: Image.Image, center_x: int, ground_y: int) -> None:
    canvas.alpha_composite(sprite, (int(round(center_x - sprite.width / 2)), ground_y - sprite.height + 1))


def _draw_native_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    *,
    anchor: str,
    fill: str | tuple[int, int, int, int],
) -> None:
    draw.text(xy, text, anchor=anchor, fill=fill, font=ImageFont.load_default())


def _gba_asset(path: str) -> Image.Image:
    return Image.open(ASSET_DIR / path).convert("RGBA")


def _rgb(name: str) -> tuple[int, int, int]:
    value = _hex(name).lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _hex(name: str) -> str:
    palette = {
        "outline": "#1F2430",
        "sky": "#9DD7F5",
        "expectation": "#E87518",
        "error": "#D83A68",
        "memory": "#129B63",
        "white": "#FFFFFF",
    }
    return palette[name]


def _scroll_frames(step_count: int, *, hold_frames: int) -> list[tuple[int, float]]:
    frames: list[tuple[int, float]] = []
    for index in range(step_count):
        if index == step_count - 1:
            frames.extend((index, 0.0) for _ in range(hold_frames))
            continue
        for subframe in range(hold_frames):
            progress = subframe / max(1, hold_frames)
            frames.append((index, progress))
    return frames


def _object_world_x(index: int) -> float:
    return OBJECT_WORLD_START + index * OBJECT_SPACING


def _camera_x(index: int, scroll_progress: float, step_count: int) -> float:
    current = _object_world_x(index) - ENCOUNTER_SCREEN_X
    if index >= step_count - 1:
        return current
    next_camera = _object_world_x(index + 1) - ENCOUNTER_SCREEN_X
    eased = 0.5 - 0.5 * np.cos(np.clip(scroll_progress, 0.0, 1.0) * np.pi)
    return current + (next_camera - current) * eased


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
