from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FrameLabeler = Callable[[int], str]


@dataclass(frozen=True)
class FilmstripSettings:
    """Settings for turning rendered animation frames into one still image."""

    frame_width: int = 240
    frame_gap: int = 10
    blur_frame_step: int | None = None
    padding: int = 14
    label_height: int = 18
    background: str = "#FFFFFF"
    interval_alpha: float = 1.0
    blur_alpha: float = 0.18
    endpoint_alpha: float = 0.42
    max_interval_frames: int = 12


def interval_frame_indices(frame_count: int, max_frames: int) -> list[int]:
    if frame_count <= 0:
        return []
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if frame_count <= max_frames:
        return list(range(frame_count))
    return [int(index) for index in np.linspace(0, frame_count - 1, max_frames)]


def save_interval_filmstrip(
    frames: Sequence[Image.Image],
    output: str | Path,
    *,
    settings: FilmstripSettings | None = None,
    labels: Sequence[str] | FrameLabeler | None = None,
) -> None:
    image = interval_filmstrip(frames, settings=settings, labels=labels)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def save_blur_filmstrip(
    frames: Sequence[Image.Image],
    output: str | Path,
    *,
    settings: FilmstripSettings | None = None,
    labels: Sequence[str] | FrameLabeler | None = None,
) -> None:
    image = blur_filmstrip(frames, settings=settings, labels=labels)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def save_shared_axes_filmstrip(
    frames: Sequence[Image.Image],
    output: str | Path,
    *,
    settings: FilmstripSettings | None = None,
) -> None:
    image = shared_axes_filmstrip(frames, settings=settings)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def interval_filmstrip(
    frames: Sequence[Image.Image],
    *,
    settings: FilmstripSettings | None = None,
    labels: Sequence[str] | FrameLabeler | None = None,
) -> Image.Image:
    settings = settings or FilmstripSettings()
    selected_indices = interval_frame_indices(
        len(frames),
        settings.max_interval_frames,
    )
    selected = [frames[index] for index in selected_indices]
    if not selected:
        raise ValueError("filmstrip requires at least one frame")

    prepared = [_fit_frame(frame, settings.frame_width) for frame in selected]
    frame_height = max(frame.height for frame in prepared)
    width = (
        2 * settings.padding
        + len(prepared) * settings.frame_width
        + max(0, len(prepared) - 1) * settings.frame_gap
    )
    height = 2 * settings.padding + frame_height + settings.label_height
    canvas = Image.new("RGBA", (width, height), settings.background)
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = ImageFont.load_default()

    for strip_index, (source_index, frame) in enumerate(zip(selected_indices, prepared)):
        x = settings.padding + strip_index * (settings.frame_width + settings.frame_gap)
        y = settings.padding
        _alpha_composite(canvas, frame, (x, y), settings.interval_alpha)
        label = _label_for(labels, source_index, default=f"{source_index}")
        if label:
            draw.text(
                (x + settings.frame_width // 2, y + frame_height + 5),
                label,
                anchor="mt",
                fill="#111827",
                font=font,
            )
    return canvas


def shared_axes_filmstrip(
    frames: Sequence[Image.Image],
    *,
    settings: FilmstripSettings | None = None,
    diff_threshold: float = 18.0,
) -> Image.Image:
    """Move changing chart marks through time without repeating static axes."""

    settings = settings or FilmstripSettings()
    if not frames:
        raise ValueError("filmstrip requires at least one frame")

    prepared = [_fit_frame(frame, settings.frame_width) for frame in frames]
    arrays = np.stack(
        [np.asarray(frame.convert("RGBA"), dtype=float) for frame in prepared],
        axis=0,
    )
    background = arrays[0]
    height, width = background.shape[:2]
    frame_step = (
        settings.blur_frame_step
        if settings.blur_frame_step is not None
        else max(1, settings.frame_width // 3)
    )
    if frame_step <= 0:
        raise ValueError("blur_frame_step must be positive")
    strip_width = width + max(0, len(arrays) - 1) * frame_step
    strip_height = height
    strip = Image.new("RGBA", (strip_width, strip_height), settings.background)

    reference = Image.fromarray(
        np.clip(background, 0.0, 255.0).astype(np.uint8),
        mode="RGBA",
    )
    clean_reference = Image.fromarray(
        _clean_static_background(background),
        mode="RGBA",
    )
    for x in range(0, strip_width, max(1, frame_step)):
        _alpha_composite(strip, clean_reference, (x, 0), 0.95)
    _alpha_composite(strip, reference, (0, 0), 0.88)

    for index, frame_array in enumerate(arrays):
        diff = np.linalg.norm(frame_array[..., :3] - background[..., :3], axis=2)
        mask = np.clip((diff - diff_threshold) / max(diff_threshold, 1.0), 0.0, 1.0)
        mask *= _visible_ink_mask(frame_array)
        alpha = max(settings.blur_alpha, 0.46)
        if index == 0 or index == len(arrays) - 1:
            alpha = max(settings.endpoint_alpha, 0.78)
        source_alpha = (frame_array[..., 3] / 255.0) * mask * alpha
        layer = np.zeros_like(frame_array)
        layer[..., :3] = frame_array[..., :3]
        layer[..., 3] = 255.0 * source_alpha
        image = Image.fromarray(np.clip(layer, 0.0, 255.0).astype(np.uint8), mode="RGBA")
        strip.alpha_composite(image, (index * frame_step, 0))

    if settings.padding <= 0 and settings.label_height <= 0:
        return strip
    output = Image.new(
        "RGBA",
        (
            strip_width + 2 * settings.padding,
            strip_height + 2 * settings.padding + settings.label_height,
        ),
        settings.background,
    )
    output.alpha_composite(strip, (settings.padding, settings.padding))
    if settings.label_height > 0:
        draw = ImageDraw.Draw(output, "RGBA")
        draw.text(
            (output.width // 2, settings.padding + strip_height + 5),
            f"0 -> {len(frames) - 1}",
            anchor="mt",
            fill="#111827",
            font=ImageFont.load_default(),
        )
    return output


def _visible_ink_mask(frame_array: np.ndarray) -> np.ndarray:
    rgb = frame_array[..., :3]
    luminance = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )
    saturation = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    return np.where((luminance < 246.0) | (saturation > 24.0), 1.0, 0.0)


def _clean_static_background(frame_array: np.ndarray) -> np.ndarray:
    cleaned = np.asarray(frame_array, dtype=float).copy()
    mask = _visible_ink_mask(cleaned) > 0.0
    unmasked = cleaned[~mask]
    if len(unmasked) == 0:
        fill = np.array([255.0, 255.0, 255.0, 255.0])
    else:
        fill = np.median(unmasked, axis=0)
    cleaned[mask] = fill
    return np.clip(cleaned, 0.0, 255.0).astype(np.uint8)


def blur_filmstrip(
    frames: Sequence[Image.Image],
    *,
    settings: FilmstripSettings | None = None,
    labels: Sequence[str] | FrameLabeler | None = None,
) -> Image.Image:
    settings = settings or FilmstripSettings()
    if not frames:
        raise ValueError("filmstrip requires at least one frame")

    prepared = [_fit_frame(frame, settings.frame_width) for frame in frames]
    frame_height = max(frame.height for frame in prepared)
    frame_step = (
        settings.blur_frame_step
        if settings.blur_frame_step is not None
        else max(1, settings.frame_width // 3)
    )
    if frame_step <= 0:
        raise ValueError("blur_frame_step must be positive")
    width = (
        2 * settings.padding
        + settings.frame_width
        + max(0, len(prepared) - 1) * frame_step
    )
    height = 2 * settings.padding + frame_height + settings.label_height
    canvas = Image.new("RGBA", (width, height), settings.background)
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = ImageFont.load_default()

    for index, frame in enumerate(prepared):
        alpha = settings.blur_alpha
        if index == 0 or index == len(prepared) - 1:
            alpha = settings.endpoint_alpha
        x = settings.padding + index * frame_step
        _alpha_composite(canvas, frame, (x, settings.padding), alpha)

    first_label = _label_for(labels, 0, default="0")
    last_label = _label_for(labels, len(frames) - 1, default=f"{len(frames) - 1}")
    label = f"{first_label} -> {last_label}" if first_label or last_label else ""
    if label:
        draw.text(
            (width // 2, settings.padding + frame_height + 5),
            label,
            anchor="mt",
            fill="#111827",
            font=font,
        )
    return canvas


def draw_image_on_axis(
    axis,
    image: Image.Image,
    center: tuple[float, float],
    width: float,
    *,
    alpha: float = 1.0,
    zorder: int = 1,
    interpolation: str = "nearest",
) -> None:
    rgba = np.asarray(image.convert("RGBA"), dtype=float)
    rgba[..., 3] *= float(np.clip(alpha, 0.0, 1.0))
    rgba = np.clip(rgba, 0.0, 255.0).astype(np.uint8)
    cx, cy = center
    height = width * image.height / max(1, image.width)
    axis.imshow(
        rgba,
        extent=(cx - width / 2, cx + width / 2, cy - height / 2, cy + height / 2),
        interpolation=interpolation,
        zorder=zorder,
    )


def _fit_frame(frame: Image.Image, width: int) -> Image.Image:
    if width <= 0:
        raise ValueError("frame_width must be positive")
    frame = frame.convert("RGBA")
    if frame.width == width:
        return frame
    height = max(1, int(round(frame.height * width / max(1, frame.width))))
    return frame.resize((width, height), Image.Resampling.LANCZOS)


def _alpha_composite(
    canvas: Image.Image,
    image: Image.Image,
    xy: tuple[int, int],
    alpha: float,
) -> None:
    layer = image.copy()
    layer_alpha = layer.getchannel("A")
    layer_alpha = layer_alpha.point(lambda value: int(value * np.clip(alpha, 0.0, 1.0)))
    layer.putalpha(layer_alpha)
    canvas.alpha_composite(layer, xy)


def _label_for(
    labels: Sequence[str] | FrameLabeler | None,
    index: int,
    *,
    default: str,
) -> str:
    if labels is None:
        return default
    if callable(labels):
        return labels(index)
    if 0 <= index < len(labels):
        return labels[index]
    return default
