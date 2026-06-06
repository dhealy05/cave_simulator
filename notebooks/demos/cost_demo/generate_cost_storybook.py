"""Copy committed cost figures into the cost storybook asset directory."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts" / "results" / "readme"
DEST = Path(__file__).resolve().parent / "storybook_assets"

ASSETS = {
    "20_primitive_compression_scorecard.png": "01_primitive_compression_scorecard.png",
    "22_primitive_compression_coupling.png": "02_primitive_compression_coupling.png",
    "23_primitive_compression_overlay.png": "03_primitive_compression_overlay.png",
    "24_primitive_compression_timeline.png": "04_primitive_compression_timeline.png",
    "26_compression_clamp_stream.png": "05_compression_clamp_stream.png",
    "28_compression_clamp_governance.gif": "06_compression_clamp_governance.gif",
    "27_compression_clamp_controls.png": "07_compression_clamp_controls.png",
}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for src_name, dest_name in ASSETS.items():
        src = SOURCE / src_name
        dest = DEST / dest_name
        if not src.exists():
            raise FileNotFoundError(f"Missing source asset: {src}")
        shutil.copy2(src, dest)
        print(f"{src.relative_to(ROOT)} -> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
