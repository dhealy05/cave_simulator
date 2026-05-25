from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cave.commitments.attention.state import AttentionState, attention_effect, coerce_attention_state
from cave.observation.experience.features import (
    Array,
    FeatureAxis,
    FeatureVector,
    feature_axis_label,
    feature_axis_value,
)
from cave.observation.experience.objects import ExperienceObject
from cave.commitments.memory.trace import MemoryTrace


@dataclass(frozen=True)
class SubjectiveTopologyWell:
    source: ExperienceObject
    center: Array
    width: float
    weight: float

    def __post_init__(self) -> None:
        if self.width <= 0.0:
            raise ValueError("SubjectiveTopologyWell.width must be positive")
        object.__setattr__(self, "center", np.array(self.center, dtype=float))


@dataclass(frozen=True)
class SubjectiveTopologyPrior:
    mode: str = "flat"
    strength: float = 0.0
    width: float = 0.45
    seed: int = 7
    well_count: int = 4

    def __post_init__(self) -> None:
        if self.strength < 0.0:
            raise ValueError("SubjectiveTopologyPrior.strength must be non-negative")
        if self.width <= 0.0:
            raise ValueError("SubjectiveTopologyPrior.width must be positive")
        if self.well_count < 0:
            raise ValueError("SubjectiveTopologyPrior.well_count must be non-negative")


@dataclass(frozen=True)
class SubjectiveTopologyParams:
    feature_x: FeatureAxis = "angularity"
    feature_y: FeatureAxis = "roundness"
    bounds: tuple[float, float] = (-1.0, 1.0)
    resolution: int = 72
    prior: SubjectiveTopologyPrior = field(default_factory=SubjectiveTopologyPrior)
    decay: float = 0.94
    diffusion: float = 0.18
    deposit_strength: float = 0.22
    expectation_deposit_strength: float = 0.14
    deposit_width: float = 0.18
    expectation_deposit_width: float | None = None
    transition_strength: float = 0.08
    max_density: float = 1.0
    attention_gamma: float = 2.0


@dataclass(frozen=True)
class SubjectiveTopologyCorrection:
    expected_point: Array
    actual_point: Array
    after_point: Array
    surprise: float
    learning_rate: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected_point", np.array(self.expected_point, dtype=float))
        object.__setattr__(self, "actual_point", np.array(self.actual_point, dtype=float))
        object.__setattr__(self, "after_point", np.array(self.after_point, dtype=float))
        object.__setattr__(self, "surprise", float(self.surprise))
        object.__setattr__(self, "learning_rate", float(self.learning_rate))


@dataclass(frozen=True)
class SubjectiveTopologyState:
    density: Array
    wells: list[SubjectiveTopologyWell]
    correction: SubjectiveTopologyCorrection | None = None
    bounds: tuple[float, float] = (-1.0, 1.0)
    feature_x: FeatureAxis = "angularity"
    feature_y: FeatureAxis = "roundness"
    last_focus: Array | None = None
    expected_density: Array | None = None
    actual_density: Array | None = None

    def __post_init__(self) -> None:
        density = np.asarray(self.density, dtype=float)
        expected_density = (
            np.zeros_like(density, dtype=float)
            if self.expected_density is None
            else np.asarray(self.expected_density, dtype=float)
        )
        actual_density = (
            np.zeros_like(density, dtype=float)
            if self.actual_density is None
            else np.asarray(self.actual_density, dtype=float)
        )
        if expected_density.shape != density.shape:
            raise ValueError("expected_density must match density shape")
        if actual_density.shape != density.shape:
            raise ValueError("actual_density must match density shape")
        object.__setattr__(self, "density", density)
        object.__setattr__(self, "expected_density", expected_density)
        object.__setattr__(self, "actual_density", actual_density)

    @classmethod
    def initial(
        cls,
        *,
        feature_x: FeatureAxis = "angularity",
        feature_y: FeatureAxis = "roundness",
        bounds: tuple[float, float] = (-1.0, 1.0),
        resolution: int = 72,
        prior: SubjectiveTopologyPrior | None = None,
    ) -> "SubjectiveTopologyState":
        if resolution <= 1:
            raise ValueError("resolution must be greater than 1")
        prior = prior or SubjectiveTopologyPrior()
        density = initial_topology_density(
            resolution=resolution,
            bounds=bounds,
            prior=prior,
        )
        return SubjectiveTopologyState(
            density=density,
            wells=[],
            expected_density=np.zeros_like(density, dtype=float),
            actual_density=np.zeros_like(density, dtype=float),
            bounds=bounds,
            feature_x=feature_x,
            feature_y=feature_y,
        )

    def update(
        self,
        memory: MemoryTrace,
        current_objects: list[ExperienceObject],
        params: SubjectiveTopologyParams,
        current_attention: AttentionState | float | None = None,
        vocabulary: list[str] | None = None,
        expected_input: Array | None = None,
        actual_input: Array | None = None,
        after_input: Array | None = None,
        surprise: float = 0.0,
        learning_rate: float = 0.0,
    ) -> "SubjectiveTopologyState":
        current_attention = coerce_attention_state(current_attention)
        expected_density = self._diffuse(
            np.array(self.expected_density, dtype=float) * params.decay,
            params.diffusion,
        )
        actual_density = self._diffuse(
            np.array(self.actual_density, dtype=float) * params.decay,
            params.diffusion,
        )
        untagged_density = np.clip(
            np.array(self.density, dtype=float)
            - np.array(self.expected_density, dtype=float)
            - np.array(self.actual_density, dtype=float),
            0.0,
            None,
        )
        untagged_density = self._diffuse(
            untagged_density * params.decay,
            params.diffusion,
        )

        current_centers = [
            self.center_for_object(obj, params.feature_x, params.feature_y)
            for obj in current_objects
        ]
        for obj, center in zip(current_objects, current_centers):
            actual_density += (
                current_attention.object_impact(obj)
                * obj.salience
                * obj.learning_weight
                * params.deposit_strength
                * self._gaussian_grid(center, params.deposit_width)
            )

        if self.last_focus is not None and current_centers:
            transition_impact = attention_effect(
                current_attention.capacity,
                params.attention_gamma,
            )
            for center in current_centers:
                actual_density += (
                    transition_impact
                    * params.transition_strength
                    * self._transition_grid(
                        self.last_focus,
                        center,
                        params.deposit_width,
                    )
                )

        if vocabulary is not None and expected_input is not None:
            expected_center = self.center_for_vector(
                expected_input,
                vocabulary,
                params.feature_x,
                params.feature_y,
            )
            expected_width = (
                params.expectation_deposit_width
                if params.expectation_deposit_width is not None
                else params.deposit_width
            )
            expected_density += (
                current_attention.internal_expectation_impact()
                * params.expectation_deposit_strength
                * self._gaussian_grid(expected_center, expected_width)
            )

        expected_density = np.clip(expected_density, 0.0, params.max_density)
        actual_density = np.clip(actual_density, 0.0, params.max_density)
        density = np.clip(
            untagged_density + expected_density + actual_density,
            0.0,
            params.max_density,
        )
        wells = []
        for item in memory.items:
            obj = item.source
            wells.append(
                SubjectiveTopologyWell(
                    source=obj,
                    center=self.center_for_object(obj, params.feature_x, params.feature_y),
                    width=params.deposit_width,
                    weight=float(item.strength),
                )
            )

        last_focus = self.last_focus
        if current_centers:
            last_focus = np.mean(np.array(current_centers), axis=0)

        correction = None
        if (
            vocabulary is not None
            and expected_input is not None
            and actual_input is not None
            and after_input is not None
        ):
            correction = SubjectiveTopologyCorrection(
                expected_point=self.center_for_vector(
                    expected_input,
                    vocabulary,
                    params.feature_x,
                    params.feature_y,
                ),
                actual_point=self.center_for_vector(
                    actual_input,
                    vocabulary,
                    params.feature_x,
                    params.feature_y,
                ),
                after_point=self.center_for_vector(
                    after_input,
                    vocabulary,
                    params.feature_x,
                    params.feature_y,
                ),
                surprise=surprise,
                learning_rate=learning_rate,
            )

        return SubjectiveTopologyState(
            density=density,
            wells=wells,
            expected_density=expected_density,
            actual_density=actual_density,
            correction=correction,
            bounds=params.bounds,
            feature_x=params.feature_x,
            feature_y=params.feature_y,
            last_focus=None if last_focus is None else np.array(last_focus, dtype=float),
        )

    def center_for_object(
        self,
        obj: ExperienceObject,
        feature_x: FeatureAxis | None = None,
        feature_y: FeatureAxis | None = None,
    ) -> Array:
        fx_name = feature_x or self.feature_x
        fy_name = feature_y or self.feature_y
        fx = feature_axis_value(obj.features, fx_name)
        fy = feature_axis_value(obj.features, fy_name)
        return np.array([2.0 * fx - 1.0, 2.0 * fy - 1.0], dtype=float)

    def center_for_vector(
        self,
        vector: Array,
        vocabulary: list[str],
        feature_x: FeatureAxis | None = None,
        feature_y: FeatureAxis | None = None,
    ) -> Array:
        vector = np.asarray(vector, dtype=float)
        values = {
            key: float(vector[index])
            for index, key in enumerate(vocabulary)
            if index < vector.size
        }
        features = FeatureVector(values)
        fx_name = feature_x or self.feature_x
        fy_name = feature_y or self.feature_y
        fx = feature_axis_value(features, fx_name)
        fy = feature_axis_value(features, fy_name)
        return np.array([2.0 * fx - 1.0, 2.0 * fy - 1.0], dtype=float)

    def axis_labels(self) -> tuple[str, str]:
        return feature_axis_label(self.feature_x), feature_axis_label(self.feature_y)

    def intensity_at(self, z: Array) -> float:
        x_axis, y_axis = self.axes()
        z = np.array(z, dtype=float)
        fx = float(np.interp(float(z[0]), x_axis, np.arange(x_axis.size)))
        fy = float(np.interp(float(z[1]), y_axis, np.arange(y_axis.size)))
        i = max(0, min(x_axis.size - 1, int(round(fx))))
        j = max(0, min(y_axis.size - 1, int(round(fy))))
        return float(self.density[j, i])

    def axes(self) -> tuple[Array, Array]:
        lower, upper = self.bounds
        resolution = self.density.shape[0]
        axis = np.linspace(lower, upper, resolution)
        return axis, axis

    def grid(self, resolution: int | None = None) -> tuple[Array, Array, Array]:
        return self._grid_for_density(self.density, resolution)

    def expected_grid(self, resolution: int | None = None) -> tuple[Array, Array, Array]:
        return self._grid_for_density(self.expected_density, resolution)

    def actual_grid(self, resolution: int | None = None) -> tuple[Array, Array, Array]:
        return self._grid_for_density(self.actual_density, resolution)

    def _grid_for_density(
        self,
        density: Array,
        resolution: int | None = None,
    ) -> tuple[Array, Array, Array]:
        if resolution is None:
            x_axis, y_axis = self.axes()
            grid_x, grid_y = np.meshgrid(x_axis, y_axis)
            return grid_x, grid_y, np.asarray(density, dtype=float).copy()
        if resolution <= 1:
            raise ValueError("resolution must be greater than 1")
        lower, upper = self.bounds
        axis = np.linspace(lower, upper, resolution)
        grid_x, grid_y = np.meshgrid(axis, axis)
        source_x, source_y = self.axes()
        rows = [
            np.interp(axis, source_x, row)
            for row in np.asarray(density, dtype=float)
        ]
        row_resampled = np.array(rows, dtype=float)
        density = np.array(
            [
                np.interp(axis, source_y, row_resampled[:, col])
                for col in range(row_resampled.shape[1])
            ],
            dtype=float,
        ).T
        return grid_x, grid_y, density

    def _gaussian_grid(self, center: Array, width: float) -> Array:
        x_axis, y_axis = self.axes()
        grid_x, grid_y = np.meshgrid(x_axis, y_axis)
        cx, cy = float(center[0]), float(center[1])
        return np.exp(
            -((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
            / (2.0 * width**2)
        )

    def _transition_grid(self, start: Array, end: Array, width: float) -> Array:
        x_axis, y_axis = self.axes()
        grid_x, grid_y = np.meshgrid(x_axis, y_axis)
        start = np.array(start, dtype=float)
        end = np.array(end, dtype=float)
        segment = end - start
        length_sq = float(np.dot(segment, segment))
        if length_sq < 1e-12:
            return self._gaussian_grid(end, width)
        points = np.stack([grid_x - start[0], grid_y - start[1]], axis=-1)
        t = np.clip((points @ segment) / length_sq, 0.0, 1.0)
        nearest_x = start[0] + t * segment[0]
        nearest_y = start[1] + t * segment[1]
        distance_sq = (grid_x - nearest_x) ** 2 + (grid_y - nearest_y) ** 2
        return np.exp(-distance_sq / (2.0 * (width * 0.65) ** 2))

    def _diffuse(self, density: Array, amount: float) -> Array:
        if amount <= 0.0:
            return density
        padded = np.pad(density, 1, mode="edge")
        neighbors = (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        ) / 4.0
        return (1.0 - amount) * density + amount * neighbors


def initial_topology_density(
    *,
    resolution: int,
    bounds: tuple[float, float],
    prior: SubjectiveTopologyPrior,
) -> Array:
    if resolution <= 1:
        raise ValueError("resolution must be greater than 1")
    if prior.mode == "flat" or prior.strength == 0.0:
        return np.zeros((resolution, resolution), dtype=float)

    lower, upper = bounds
    axis = np.linspace(lower, upper, resolution)
    grid_x, grid_y = np.meshgrid(axis, axis)

    if prior.mode == "basin":
        radius_sq = grid_x**2 + grid_y**2
        density = prior.strength * np.exp(-radius_sq / (2.0 * prior.width**2))
    elif prior.mode == "ridge":
        density = prior.strength * np.exp(-(grid_y**2) / (2.0 * prior.width**2))
        density *= 0.65 + 0.35 * np.cos(np.pi * grid_x) ** 2
    elif prior.mode == "random_wells":
        rng = np.random.default_rng(prior.seed)
        density = np.zeros((resolution, resolution), dtype=float)
        for _ in range(prior.well_count):
            center = rng.uniform(lower, upper, size=2)
            width = prior.width * float(rng.uniform(0.65, 1.35))
            weight = prior.strength * float(rng.uniform(0.35, 1.0))
            density += weight * np.exp(
                -((grid_x - center[0]) ** 2 + (grid_y - center[1]) ** 2)
                / (2.0 * width**2)
            )
    else:
        raise ValueError(f"unsupported subjective topology prior: {prior.mode}")

    return np.clip(density, 0.0, 1.0)
