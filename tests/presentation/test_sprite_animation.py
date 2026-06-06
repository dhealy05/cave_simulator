from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from cave.presentation.sprites import (
    PixelSpriteAnimator,
    SpriteAssetStore,
    SpriteClip,
    eased_camera_x,
    object_world_x,
    scroll_frame_tuples,
    scroll_frames,
)


def _write_sprite(path, color=(255, 0, 0, 255), size=(4, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", size, color).save(path)


def test_sprite_asset_store_selects_clip_frame_or_fallback(tmp_path) -> None:
    _write_sprite(tmp_path / "sprites" / "walk_1.png")
    assets = SpriteAssetStore(tmp_path)
    clip = SpriteClip("walk", 4, "idle.png")

    assert clip.frame_name(assets, 0.35) == "walk_1.png"
    assert clip.frame_name(assets, 0.90) == "idle.png"


def test_pixel_sprite_animator_composes_grounded_sprite(tmp_path) -> None:
    _write_sprite(tmp_path / "sprites" / "subject.png", size=(2, 3))
    assets = SpriteAssetStore(tmp_path, palette={"sky": "#0000FF"})
    animator = PixelSpriteAnimator(assets, width=12, height=10, default_ground_y=8)

    canvas = animator.canvas("sky")
    sprite = animator.scaled(animator.sprite("subject.png"), 2.0)
    animator.paste_grounded(canvas, sprite, center_x=6)
    rgba = animator.rgba_array(canvas)

    assert rgba.shape == (10, 12, 4)
    np.testing.assert_array_equal(rgba[3, 4], np.array([255, 0, 0, 255]))
    np.testing.assert_array_equal(rgba[0, 0], np.array([0, 0, 255, 255]))


def test_scroll_frames_and_camera_helpers_match_side_scroller_math() -> None:
    frames = scroll_frame_tuples(scroll_frames(3, hold_frames=4))

    assert frames[:5] == [(0, 0.0), (0, 0.25), (0, 0.5), (0, 0.75), (1, 0.0)]
    assert frames[-4:] == [(2, 0.0), (2, 0.0), (2, 0.0), (2, 0.0)]
    assert object_world_x(2, start=130.0, spacing=74.0) == pytest.approx(278.0)
    assert eased_camera_x(
        0,
        0.5,
        3,
        object_start=130.0,
        object_spacing=74.0,
        encounter_x=110.0,
    ) == pytest.approx(57.0)
