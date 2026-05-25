from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WalkerObject:
    id: str
    label: str
    sprite: str
    features: tuple[float, float]
    value: float = 0.0


@dataclass(frozen=True)
class TreeVariant:
    scale: float
    features: tuple[float, float]


ASSET_DIR = Path(__file__).resolve().parent / "assets" / "gba"
SPRITE_DIR = ASSET_DIR / "sprites"

OBJECTS: dict[str, WalkerObject] = {
    "tree": WalkerObject(
        id="tree",
        label="Tree",
        sprite="object_tree.png",
        features=(0.20, 0.80),
        value=0.20,
    ),
    "rock": WalkerObject(
        id="rock",
        label="Rock",
        sprite="object_rock.png",
        features=(0.50, 0.30),
        value=0.00,
    ),
    "snake": WalkerObject(
        id="snake",
        label="Snake",
        sprite="object_snake_1.png",
        features=(0.90, 0.90),
        value=-1.00,
    ),
}

EPISODE = ("tree", "tree", "tree", "snake", "rock", "tree")
MEMORY_INITIAL = OBJECTS["tree"].features
ETA = 0.45

TREE_VARIANTS: tuple[TreeVariant, ...] = (
    TreeVariant(scale=1.28, features=(0.18, 0.74)),
    TreeVariant(scale=1.48, features=(0.22, 0.86)),
    TreeVariant(scale=1.16, features=(0.17, 0.79)),
    TreeVariant(scale=1.38, features=(0.24, 0.82)),
)


def episode_features(episode: tuple[str, ...] = EPISODE) -> list[tuple[float, float]]:
    features: list[tuple[float, float]] = []
    tree_count = 0
    for object_id in episode:
        if object_id == "tree":
            variant = tree_variant(tree_count)
            features.append(variant.features)
            tree_count += 1
        else:
            features.append(OBJECTS[object_id].features)
    return features


def prototype_features() -> dict[str, tuple[float, float]]:
    return {object_id: obj.features for object_id, obj in OBJECTS.items()}


def tree_variant(occurrence_index: int) -> TreeVariant:
    return TREE_VARIANTS[occurrence_index % len(TREE_VARIANTS)]


def tree_variant_for_episode_index(episode: tuple[str, ...], index: int) -> TreeVariant:
    if episode[index] != "tree":
        raise ValueError("episode index is not a tree")
    occurrence_index = sum(1 for object_id in episode[:index] if object_id == "tree")
    return tree_variant(occurrence_index)


def sprite_path(name: str) -> Path:
    return SPRITE_DIR / name
