from __future__ import annotations

from pathlib import Path

from cave.presentation.sprites.sprite_animation import (
    PixelSpriteAnimator,
    ScrollFrame,
    SpriteAssetStore,
    SpriteClip,
    cosine_ease,
    eased_camera_x,
    object_world_x,
    scroll_frame_tuples,
    scroll_frames,
)


SPRITE_ASSET_DIR = Path(__file__).resolve().parent / "assets"
GBA_SPRITE_ASSET_DIR = SPRITE_ASSET_DIR / "gba"

__all__ = [
    "GBA_SPRITE_ASSET_DIR",
    "PixelSpriteAnimator",
    "SPRITE_ASSET_DIR",
    "ScrollFrame",
    "SpriteAssetStore",
    "SpriteClip",
    "cosine_ease",
    "eased_camera_x",
    "object_world_x",
    "scroll_frame_tuples",
    "scroll_frames",
]
