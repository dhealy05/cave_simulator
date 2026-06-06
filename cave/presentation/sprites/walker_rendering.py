"""Shared Cave Walker rollout and side-scroller viewport rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from cave.observation.producers.sources.primitive import (
    PRIMITIVE_EPISODE,
    PRIMITIVE_ETA,
    PRIMITIVE_MEMORY_INITIAL,
    PRIMITIVE_OBJECTS,
    PrimitiveVectorStep,
    nearest_prototype,
    primitive_episode_features,
    primitive_prototype_features,
    primitive_tree_variant,
    rollout_vectors,
)
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
SPRITE_ASSETS = SpriteAssetStore(GBA_SPRITE_ASSET_DIR, palette=SPRITE_PALETTE)
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
    episode: Sequence[str] = PRIMITIVE_EPISODE,
    *,
    eta: float = PRIMITIVE_ETA,
    memory_initial: Sequence[float] = PRIMITIVE_MEMORY_INITIAL,
) -> list[PrimitiveVectorStep]:
    """Run the primitive walker recurrence for a symbolic object episode."""

    return rollout_vectors(
        primitive_episode_features(episode),
        eta=eta,
        memory_initial=memory_initial,
    )


def render_gba_viewport(
    row: PrimitiveVectorStep,
    index: int,
    episode: Sequence[str],
    scroll_progress: float,
    *,
    subject_sprite_dir: str = DEFAULT_SUBJECT_SPRITE_DIR,
) -> np.ndarray:
    """Render one HUD-free GBA-style Cave Walker viewport frame."""

    episode = tuple(episode)
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
        obj = PRIMITIVE_OBJECTS[object_id]
        sprite_name = _gba_object_sprite(
            object_id,
            is_current=tile_index == index,
            row=row,
            scroll_progress=scroll_progress,
        )
        sprite = _scaled_sprite(
            _gba_asset(f"{OBJECT_SPRITE_DIR}/{sprite_name}"),
            _object_scale(object_id, episode, tile_index),
        )
        _paste_grounded_sprite(canvas, sprite, screen_x, GROUND_Y)
        label_y = GROUND_Y - sprite.height - 6
        _draw_native_text(draw, obj.label, (screen_x, label_y), anchor="mm", fill=_hex("outline"))
        if tile_index == index:
            color = _hex("error") if row.surprise > 0.45 and abs(screen_x - ENCOUNTER_SCREEN_X) < 11 else _hex("memory")
            half_width = max(18, sprite.width // 2 + 2)
            draw.rectangle(
                (screen_x - half_width, GROUND_Y - sprite.height - 2, screen_x + half_width, GROUND_Y + 2),
                outline=color,
                width=2,
            )
    subject_dir, subject_sprite = _subject_sprite(row, scroll_progress, subject_sprite_dir=subject_sprite_dir)
    bob = int(round(np.sin((index + scroll_progress) * np.pi * 2) * 2))
    if row.surprise > 0.45 and scroll_progress < 0.25:
        draw.ellipse((SUBJECT_SCREEN_X - 18, GROUND_Y - 36, SUBJECT_SCREEN_X + 18, GROUND_Y), fill=(*_rgb("error"), 48))
    _paste_grounded_sprite(canvas, _gba_asset(f"{subject_dir}/{subject_sprite}"), SUBJECT_SCREEN_X, GROUND_Y + bob)
    draw.line(
        (SUBJECT_SCREEN_X + 12, GROUND_Y - 8, ENCOUNTER_SCREEN_X - 12, GROUND_Y - 8),
        fill=(*_rgb("outline"), 150),
        width=1,
    )
    draw.polygon(
        [
            (ENCOUNTER_SCREEN_X - 12, GROUND_Y - 8),
            (ENCOUNTER_SCREEN_X - 17, GROUND_Y - 11),
            (ENCOUNTER_SCREEN_X - 17, GROUND_Y - 5),
        ],
        fill=(*_rgb("outline"), 150),
    )
    _draw_native_text(draw, f"t={row.t}", (SUBJECT_SCREEN_X, GROUND_Y + 13), anchor="mm", fill=_hex("outline"))
    _draw_expectation_bubble(draw, row)
    return np.asarray(canvas.convert("RGBA"))


def walker_scroll_frames(step_count: int, *, hold_frames: int) -> list[tuple[int, float]]:
    """Return side-scroller frame indices and per-step progress values."""

    return scroll_frame_tuples(sprite_scroll_frames(step_count, hold_frames=hold_frames))


def _draw_encounter_zone(draw: ImageDraw.ImageDraw, row: PrimitiveVectorStep, scroll_progress: float) -> None:
    fill_color = _rgb("error") if _is_error_phase(row, scroll_progress) else _rgb("memory")
    draw.rectangle((ENCOUNTER_SCREEN_X - 5, 25, ENCOUNTER_SCREEN_X + 5, GROUND_Y + 2), fill=(*fill_color, 42))
    draw.line((ENCOUNTER_SCREEN_X, 25, ENCOUNTER_SCREEN_X, GROUND_Y + 2), fill=(*fill_color, 125), width=1)
    _draw_native_text(draw, "zone", (ENCOUNTER_SCREEN_X + 14, 18), anchor="lm", fill=_hex("outline"))


def _draw_expectation_bubble(draw: ImageDraw.ImageDraw, row: PrimitiveVectorStep) -> None:
    expected_id, _ = nearest_prototype(row.expected, primitive_prototype_features())
    label = f"expects {PRIMITIVE_OBJECTS[expected_id].label}"
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
        return _subject_clip_frame(
            SUBJECT_CLIPS["notice"],
            (progress - 0.18) / 0.14,
            subject_sprite_dir=subject_sprite_dir,
        )
    if row.surprise > 0.45 and progress < 0.52:
        return _subject_clip_frame(
            SUBJECT_CLIPS["surprised"],
            (progress - 0.32) / 0.20,
            subject_sprite_dir=subject_sprite_dir,
        )
    update_start = 0.52 if row.surprise > 0.45 else 0.32
    if progress < update_start + 0.20:
        return _subject_clip_frame(
            SUBJECT_CLIPS["update"],
            (progress - update_start) / 0.20,
            subject_sprite_dir=subject_sprite_dir,
        )
    if progress < update_start + 0.36:
        return _subject_clip_frame(
            SUBJECT_CLIPS["recover"],
            (progress - update_start - 0.20) / 0.16,
            subject_sprite_dir=subject_sprite_dir,
        )
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


def _object_scale(object_id: str, episode: Sequence[str], index: int) -> float:
    if object_id == "tree":
        occurrence_index = sum(1 for prior_id in episode[:index] if prior_id == "tree")
        return primitive_tree_variant(occurrence_index).scale
    return 1.0


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


_render_gba_viewport = render_gba_viewport
_scroll_frames = walker_scroll_frames
