from __future__ import annotations

from PIL import Image

from cave.presentation.filmstrip import (
    FilmstripSettings,
    blur_filmstrip,
    interval_filmstrip,
    interval_frame_indices,
    shared_axes_filmstrip,
)


def test_interval_frame_indices_samples_endpoints() -> None:
    assert interval_frame_indices(0, 4) == []
    assert interval_frame_indices(3, 4) == [0, 1, 2]
    assert interval_frame_indices(10, 4) == [0, 3, 6, 9]


def test_interval_filmstrip_writes_horizontal_strip() -> None:
    frames = [
        Image.new("RGBA", (8, 4), color)
        for color in ("#FF0000", "#00FF00", "#0000FF")
    ]

    image = interval_filmstrip(
        frames,
        settings=FilmstripSettings(
            frame_width=8,
            frame_gap=2,
            padding=1,
            label_height=0,
        ),
        labels=["a", "b", "c"],
    )

    assert image.size == (30, 6)
    assert image.getpixel((1, 1))[:3] == (255, 0, 0)
    assert image.getpixel((11, 1))[:3] == (0, 255, 0)
    assert image.getpixel((21, 1))[:3] == (0, 0, 255)


def test_blur_filmstrip_composites_frames_into_one_panel() -> None:
    frames = [
        Image.new("RGBA", (8, 4), "#FF0000"),
        Image.new("RGBA", (8, 4), "#0000FF"),
    ]

    image = blur_filmstrip(
        frames,
        settings=FilmstripSettings(
            frame_width=8,
            padding=1,
            label_height=0,
            blur_frame_step=4,
            blur_alpha=0.5,
            endpoint_alpha=0.5,
        ),
    )

    assert image.size == (14, 6)
    red, green, blue, _ = image.getpixel((1, 1))
    assert red > 0
    assert red > green
    assert red > blue
    overlap = image.getpixel((5, 1))
    assert overlap[0] > 0
    assert overlap[2] > 0


def test_shared_axes_filmstrip_keeps_one_frame_extent() -> None:
    base = Image.new("RGBA", (8, 4), "#FFFFFF")
    red = base.copy()
    blue = base.copy()
    red.putpixel((2, 2), (255, 0, 0, 255))
    blue.putpixel((5, 2), (0, 0, 255, 255))

    image = shared_axes_filmstrip(
        [red, blue],
        settings=FilmstripSettings(
            frame_width=8,
            padding=0,
            label_height=0,
            blur_frame_step=4,
            blur_alpha=1.0,
            endpoint_alpha=1.0,
        ),
        diff_threshold=1.0,
    )

    assert image.size == (12, 4)
    red = image.getpixel((2, 2))[:3]
    blue = image.getpixel((9, 2))[:3]
    assert red[0] > red[1]
    assert red[0] > red[2]
    assert blue[2] > blue[0]
    assert blue[2] > blue[1]
