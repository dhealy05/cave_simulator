from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass, field
from typing import Any

from cave.observation.experience.features import FeatureVector


@dataclass(frozen=True)
class Presentation:
    modality: str = "visual"
    style: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ShapePresentation(Presentation):
    shape_type: str = "triangle"
    points: list[tuple[float, float]] = field(default_factory=list)


@dataclass(frozen=True)
class TextPresentation(Presentation):
    text: str = ""
    font_role: str = "default"


@dataclass(frozen=True)
class ImagePresentation(Presentation):
    image_ref: str = ""
    alt: str = ""


@dataclass(frozen=True)
class AudioPresentation(Presentation):
    modality: str = "audio"
    waveform_ref: str | None = None
    label: str = ""


def visual_presentation_from_features(
    features: FeatureVector,
    metadata: dict[str, Any] | None = None,
) -> ShapePresentation:
    metadata = metadata or {}
    roundness = _clamp01(features.value("roundness"))
    sides_value = features.value("shape_sides", 3.0 + 13.0 * _clamp01(features.value("sides", 0.0)))
    sides = max(3, min(16, int(round(sides_value))))
    size = 0.45 + 0.75 * _clamp01(features.value("size", 0.55))
    rotation = 2.0 * math.pi * _clamp01(features.value("rotation", 0.0))

    hue = _clamp01(features.value("hue", features.value("color_hue", 0.12)))
    saturation = _clamp01(features.value("saturation", 0.55))
    lightness = _clamp01(features.value("lightness", 0.72))
    fill = _hsl_to_hex(hue, saturation, lightness)
    stroke = str(metadata.get("stroke", "#212529"))

    if roundness >= 0.9 and sides_value >= 8.0:
        return ShapePresentation(
            shape_type="circle",
            style={"fill": fill, "stroke": stroke, "size": size},
        )

    return ShapePresentation(
        shape_type="polygon",
        points=_regular_polygon_points(sides, rotation),
        style={"fill": fill, "stroke": stroke, "size": size},
    )


def _regular_polygon_points(
    sides: int,
    rotation: float = 0.0,
) -> list[tuple[float, float]]:
    return [
        (
            math.cos(rotation + 2.0 * math.pi * index / sides),
            math.sin(rotation + 2.0 * math.pi * index / sides),
        )
        for index in range(sides)
    ]


def _hsl_to_hex(hue: float, saturation: float, lightness: float) -> str:
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
