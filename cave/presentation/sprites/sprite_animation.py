from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ColorValue = str | tuple[int, int, int] | tuple[int, int, int, int]


@dataclass(frozen=True)
class SpriteClip:
    """A numbered sprite sequence such as ``subject_walk_0.png``."""

    prefix: str
    frame_count: int
    fallback: str
    directory: str = "sprites"

    def frame_name(self, assets: SpriteAssetStore, progress: float) -> str:
        return assets.animation_frame(
            self.prefix,
            self.frame_count,
            progress,
            fallback=self.fallback,
            directory=self.directory,
        )


@dataclass(frozen=True)
class ScrollFrame:
    index: int
    progress: float


@dataclass
class SpriteAssetStore:
    """File-backed asset loader for pixel-art sprite scenes."""

    root: Path | str
    palette: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def image(self, path: str | Path) -> Image.Image:
        return Image.open(self.root / path).convert("RGBA")

    def sprite(self, name: str, *, directory: str = "sprites") -> Image.Image:
        return self.image(Path(directory) / name)

    def has_image(self, path: str | Path) -> bool:
        return (self.root / path).exists()

    def animation_frame(
        self,
        prefix: str,
        frame_count: int,
        progress: float,
        *,
        fallback: str,
        directory: str = "sprites",
    ) -> str:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        frame = int(np.clip(progress, 0.0, 0.999) * frame_count)
        name = f"{prefix}_{frame}.png"
        if self.has_image(Path(directory) / name):
            return name
        return fallback

    def color(self, name: str) -> str:
        return self.palette[name]

    def rgb(self, name: str) -> tuple[int, int, int]:
        value = self.color(name).lstrip("#")
        if len(value) != 6:
            raise ValueError(f"palette color {name!r} must be a #RRGGBB value")
        return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


@dataclass
class PixelSpriteAnimator:
    """Reusable helpers for composing small pixel-art sprite scenes."""

    assets: SpriteAssetStore
    width: int
    height: int
    default_ground_y: int | None = None

    def canvas(self, background: ColorValue | str) -> Image.Image:
        fill = (
            self.assets.color(background)
            if isinstance(background, str) and background in self.assets.palette
            else background
        )
        return Image.new("RGBA", (self.width, self.height), fill)

    def draw(self, canvas: Image.Image) -> ImageDraw.ImageDraw:
        return ImageDraw.Draw(canvas, "RGBA")

    def image(self, path: str | Path) -> Image.Image:
        return self.assets.image(path)

    def sprite(self, name: str, *, directory: str = "sprites") -> Image.Image:
        return self.assets.sprite(name, directory=directory)

    def clip_frame(self, clip: SpriteClip, progress: float) -> str:
        return clip.frame_name(self.assets, progress)

    def scaled(self, image: Image.Image, scale: float) -> Image.Image:
        if scale == 1.0:
            return image
        if scale <= 0.0:
            raise ValueError("scale must be positive")
        size = (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        )
        return image.resize(size, Image.Resampling.NEAREST)

    def paste_grounded(
        self,
        canvas: Image.Image,
        sprite: Image.Image,
        center_x: float,
        ground_y: int | None = None,
    ) -> None:
        if ground_y is None:
            if self.default_ground_y is None:
                raise ValueError("ground_y is required when no default_ground_y is configured")
            ground_y = self.default_ground_y
        canvas.alpha_composite(
            sprite,
            (int(round(center_x - sprite.width / 2)), ground_y - sprite.height + 1),
        )

    def paste_looped_layer(
        self,
        canvas: Image.Image,
        layer: Image.Image,
        camera_x: float,
        *,
        factor: float,
        y: int,
    ) -> None:
        width = layer.width
        if width <= 0:
            raise ValueError("layer width must be positive")
        offset = int(round(-(camera_x * factor) % width))
        for x in range(offset - width, self.width + width, width):
            canvas.alpha_composite(layer, (x, y))

    def native_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        xy: tuple[int, int],
        *,
        anchor: str,
        fill: ColorValue,
    ) -> None:
        draw.text(xy, text, anchor=anchor, fill=fill, font=ImageFont.load_default())

    def rgba_array(self, canvas: Image.Image) -> np.ndarray:
        return np.asarray(canvas.convert("RGBA"))


def scroll_frames(step_count: int, *, hold_frames: int) -> list[ScrollFrame]:
    if step_count < 0:
        raise ValueError("step_count must be non-negative")
    if hold_frames <= 0:
        raise ValueError("hold_frames must be positive")
    frames: list[ScrollFrame] = []
    for index in range(step_count):
        if index == step_count - 1:
            frames.extend(ScrollFrame(index, 0.0) for _ in range(hold_frames))
            continue
        for subframe in range(hold_frames):
            frames.append(ScrollFrame(index, subframe / hold_frames))
    return frames


def object_world_x(index: int, *, start: float, spacing: float) -> float:
    return start + index * spacing


def eased_camera_x(
    index: int,
    progress: float,
    step_count: int,
    *,
    object_start: float,
    object_spacing: float,
    encounter_x: float,
    easing: Callable[[float], float] | None = None,
) -> float:
    current = (
        object_world_x(index, start=object_start, spacing=object_spacing)
        - encounter_x
    )
    if index >= step_count - 1:
        return current
    next_camera = (
        object_world_x(index + 1, start=object_start, spacing=object_spacing)
        - encounter_x
    )
    eased = (easing or cosine_ease)(float(np.clip(progress, 0.0, 1.0)))
    return current + (next_camera - current) * eased


def cosine_ease(progress: float) -> float:
    return float(0.5 - 0.5 * np.cos(np.clip(progress, 0.0, 1.0) * np.pi))


def scroll_frame_tuples(frames: Sequence[ScrollFrame]) -> list[tuple[int, float]]:
    return [(frame.index, frame.progress) for frame in frames]
