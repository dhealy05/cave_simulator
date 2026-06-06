from __future__ import annotations

from pathlib import Path

from cave.observation.producers.sources.primitive import (
    PRIMITIVE_EPISODE,
    PRIMITIVE_ETA,
    PRIMITIVE_MEMORY_INITIAL,
    PRIMITIVE_OBJECTS,
    PrimitiveTreeVariant,
    PrimitiveWorldObject,
    primitive_episode_features,
    primitive_prototype_features,
    primitive_tree_variant,
)
from cave.presentation.sprites import GBA_SPRITE_ASSET_DIR


WalkerObject = PrimitiveWorldObject
TreeVariant = PrimitiveTreeVariant

ASSET_DIR = GBA_SPRITE_ASSET_DIR
SPRITE_DIR = ASSET_DIR / "sprites"
OBJECT_SPRITE_DIR = SPRITE_DIR / "objects"

OBJECTS = PRIMITIVE_OBJECTS
EPISODE = PRIMITIVE_EPISODE
MEMORY_INITIAL = PRIMITIVE_MEMORY_INITIAL
ETA = PRIMITIVE_ETA
TREE_VARIANTS = tuple(primitive_tree_variant(index) for index in range(4))


def episode_features(episode: tuple[str, ...] = EPISODE) -> list[tuple[float, float]]:
    return primitive_episode_features(episode)


def prototype_features() -> dict[str, tuple[float, float]]:
    return primitive_prototype_features()


def tree_variant(occurrence_index: int) -> TreeVariant:
    return primitive_tree_variant(occurrence_index)


def tree_variant_for_episode_index(episode: tuple[str, ...], index: int) -> TreeVariant:
    if episode[index] != "tree":
        raise ValueError("episode index is not a tree")
    occurrence_index = sum(1 for object_id in episode[:index] if object_id == "tree")
    return tree_variant(occurrence_index)


def sprite_path(name: str) -> Path:
    if name.startswith("object_"):
        return OBJECT_SPRITE_DIR / name
    if name.startswith("subject_"):
        return SPRITE_DIR / "subjects" / "default" / name
    return SPRITE_DIR / name
