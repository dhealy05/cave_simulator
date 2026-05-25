from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from cave.observation.experience import (
    FeatureVector,
    Presentation,
    feature_axis_label,
    presentation_for_object,
)
from cave.observation.structural import EpisodeFrame
from cave.observation.views import presentation_for_episode_input


@dataclass(frozen=True)
class Annotation:
    name: str
    value: Any


@dataclass(frozen=True)
class RenderTransform:
    x: float
    y: float
    scale: float = 1.0
    opacity: float = 1.0
    rotation: float = 0.0
    z_index: int = 0


@dataclass(frozen=True)
class RenderedObject:
    source_id: str | None
    presentation: Presentation | None
    transform: RenderTransform
    role: str
    style: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ViewState:
    name: str
    rendered_objects: list[RenderedObject]
    annotations: list[Annotation] = field(default_factory=list)


class ViewProjection(Protocol):
    def project(self, frame: EpisodeFrame) -> ViewState:
        ...


@dataclass(frozen=True)
class WallViewParams:
    travel_direction: str = "left_to_right"
    focal_x: float = 0.5
    y_position: float = 0.5


@dataclass(frozen=True)
class TimelineViewParams:
    time_min: float = 0.0
    time_max: float = 1.0
    show_event_markers: bool = True


@dataclass(frozen=True)
class LookbackViewParams:
    tau: float
    max_age: float
    min_strength: float = 0.05
    stack_mode: str = "horizontal"


@dataclass(frozen=True)
class SubjectiveTopologyViewParams:
    lower_bound: float = -1.0
    upper_bound: float = 1.0


class WallPOVProjection:
    def __init__(self, params: WallViewParams | None = None) -> None:
        self.params = params or WallViewParams()

    def project(self, frame: EpisodeFrame) -> ViewState:
        rendered = []
        inputs_by_id = frame.episode.input_by_id()
        for input_id in frame.observation.active_inputs:
            item = inputs_by_id.get(input_id)
            if item is None:
                continue
            phase = self.phase_within_extent(item.start, item.end, frame.observation.t)
            x = self.left_to_right_x(phase)
            if self.params.travel_direction == "right_to_left":
                x = 1.0 - x
            rendered.append(
                RenderedObject(
                    source_id=item.id,
                    presentation=presentation_for_episode_input(
                        item,
                        frame.episode.vocabulary,
                    ),
                    transform=RenderTransform(
                        x=x,
                        y=self.params.y_position,
                        scale=1.0,
                        opacity=1.0,
                    ),
                    role="current_presentation",
                    style={"kind": item.kind},
                )
            )
        return ViewState(name="wall_pov", rendered_objects=rendered)

    def phase_within_extent(self, start: float, end: float, t: float) -> float:
        return min(1.0, max(0.0, (t - start) / (end - start)))

    def left_to_right_x(self, phase: float) -> float:
        return phase


class TimelineProjection:
    def __init__(
        self,
        params: TimelineViewParams | None = None,
    ) -> None:
        self.params = params or TimelineViewParams()

    def project(self, frame: EpisodeFrame) -> ViewState:
        rendered: list[RenderedObject] = []
        if self.params.show_event_markers:
            for item in frame.episode.inputs:
                start_x = self.time_to_x(item.start)
                end_x = self.time_to_x(item.end)
                rendered.append(
                    RenderedObject(
                        source_id=item.id,
                        presentation=presentation_for_episode_input(
                            item,
                            frame.episode.vocabulary,
                        ),
                        transform=RenderTransform(
                            x=self.time_to_x(item.center),
                            y=0.32,
                            scale=max(0.03, end_x - start_x),
                            opacity=1.0,
                        ),
                        role="event_interval",
                        style={
                            "kind": item.kind,
                            "start_x": start_x,
                            "end_x": end_x,
                        },
                    )
                )

        rendered.append(
            RenderedObject(
                source_id=None,
                presentation=None,
                transform=RenderTransform(
                    x=self.time_to_x(frame.observation.t),
                    y=frame.observation.attention,
                    scale=1.0,
                    opacity=1.0,
                    z_index=10,
                ),
                role="timeline_pointer",
            )
        )
        attention_curve = frame.episode.metadata.get("attention_curve")
        curve = (
            [
                {
                    "t": float(point["t"]),
                    "x": self.time_to_x(float(point["t"])),
                    "attention": float(point["value"]),
                }
                for point in attention_curve
            ]
            if attention_curve is not None
            else [
                {
                    "t": obs.t,
                    "x": self.time_to_x(obs.t),
                    "attention": obs.attention,
                }
                for obs in frame.episode.observations
            ]
        )
        annotations = [
            Annotation(
                "attention_curve",
                curve,
            )
        ]
        return ViewState(name="timeline", rendered_objects=rendered, annotations=annotations)

    def time_to_x(self, t: float) -> float:
        span = self.params.time_max - self.params.time_min
        if span <= 0.0:
            return 0.0
        return min(1.0, max(0.0, (t - self.params.time_min) / span))


class LookbackProjection:
    def __init__(self, params: LookbackViewParams) -> None:
        self.params = params

    def project(self, frame: EpisodeFrame) -> ViewState:
        if frame.episode.metadata.get("lookback_mode") in {
            "attention_context",
            "conversation_mock_memory",
        }:
            return self._project_attention_context(frame)
        rendered = []
        for index, item in enumerate(frame.topology_frame.memory_items):
            age = item.age(frame.observation.t)
            strength = float(item.strength)
            if age > self.params.max_age or strength < self.params.min_strength:
                continue
            x, y = self.position_for_item(age, index)
            rendered.append(
                RenderedObject(
                    source_id=item.source.id,
                    presentation=presentation_for_episode_input(
                        item.source,
                        frame.episode.vocabulary,
                    ),
                    transform=RenderTransform(
                        x=x,
                        y=y,
                        scale=1.0,
                        opacity=strength,
                    ),
                    role="memory_trace",
                    style={
                        "kind": item.source.kind,
                        "age": age,
                        "strength": strength,
                    },
                )
            )
        return ViewState(name="lookback", rendered_objects=rendered)

    def _project_attention_context(self, frame: EpisodeFrame) -> ViewState:
        inputs_by_id = frame.episode.input_by_id()
        weighted_ids = [
            (input_id, float(weight))
            for input_id, weight in frame.observation.attention_weights.items()
            if weight >= self.params.min_strength and input_id in inputs_by_id
        ]
        weighted_ids.sort(key=lambda item: inputs_by_id[item[0]].order_index)
        current_t = frame.observation.t
        rendered = []
        for input_id, strength in weighted_ids:
            item = inputs_by_id[input_id]
            x = 0.18 + 0.64 * (item.start / max(1.0, current_t))
            y = 0.36 + 0.34 * (1.0 - strength)
            rendered.append(
                RenderedObject(
                    source_id=item.id,
                    presentation=presentation_for_episode_input(
                        item,
                        frame.episode.vocabulary,
                    ),
                    transform=RenderTransform(
                        x=x,
                        y=y,
                        scale=0.45 + 0.85 * strength,
                        opacity=min(1.0, max(0.12, strength)),
                    ),
                    role="attention_context",
                    style={
                        "kind": item.kind,
                        "distance": max(0.0, current_t - item.start),
                        "strength": strength,
                    },
                )
            )
        return ViewState(name="lookback", rendered_objects=rendered)

    def position_for_item(self, age: float, index: int) -> tuple[float, float]:
        if self.params.stack_mode == "vertical":
            return 0.5, min(0.95, 0.15 + index * 0.16)
        x = 1.0 - min(1.0, age / self.params.max_age)
        return x, 0.5


class SubjectiveTopologyProjection:
    def __init__(self, params: SubjectiveTopologyViewParams | None = None) -> None:
        self.params = params or SubjectiveTopologyViewParams()

    def project(self, frame: EpisodeFrame) -> ViewState:
        rendered = []
        topology = frame.topology_frame.topology
        for well in topology.wells:
            rendered.append(
                RenderedObject(
                    source_id=well.source.id,
                    presentation=presentation_for_object(well.source),
                    transform=RenderTransform(
                        x=self.coord_to_unit(float(well.center[0])),
                        y=self.coord_to_unit(float(well.center[1])),
                        scale=max(0.02, float(well.weight)),
                        opacity=float(well.weight),
                    ),
                    role="subjective_topology_well",
                    style={
                        "kind": well.source.kind,
                        "width": well.width,
                        "weight": well.weight,
                    },
                )
            )
        return ViewState(
            name="subjective_topology",
            rendered_objects=rendered,
            annotations=[
                Annotation("feature_x", feature_axis_label(topology.feature_x)),
                Annotation("feature_y", feature_axis_label(topology.feature_y)),
                Annotation("bounds", topology.bounds),
                Annotation("density", topology.density.copy()),
            ],
        )

    def coord_to_unit(self, value: float) -> float:
        span = self.params.upper_bound - self.params.lower_bound
        if span <= 0.0:
            return 0.0
        return min(1.0, max(0.0, (value - self.params.lower_bound) / span))


def project_all(frame: EpisodeFrame) -> dict[str, ViewState]:
    max_age = float(frame.episode.metadata.get("memory_max_age", 6.0))
    return {
        "wall_pov": WallPOVProjection().project(frame),
        "lookback": LookbackProjection(
            LookbackViewParams(
                tau=float(frame.episode.metadata.get("memory_decay_tau", 2.0)),
                max_age=max_age,
            )
        ).project(frame),
        "timeline": TimelineProjection(
            params=TimelineViewParams(
                time_min=0.0,
                time_max=max(1.0, frame.episode.duration),
            )
        ).project(frame),
        "subjective_topology": SubjectiveTopologyProjection(
            SubjectiveTopologyViewParams(
                lower_bound=frame.topology_frame.topology.bounds[0],
                upper_bound=frame.topology_frame.topology.bounds[1],
            )
        ).project(frame),
    }



def view_to_dict(view: ViewState) -> dict[str, Any]:
    return {
        "name": view.name,
        "rendered_objects": [rendered_object_to_dict(obj) for obj in view.rendered_objects],
        "annotations": [
            {"name": annotation.name, "value": encode_value(annotation.value)}
            for annotation in view.annotations
        ],
    }


def rendered_object_to_dict(obj: RenderedObject) -> dict[str, Any]:
    return {
        "source_id": obj.source_id,
        "presentation": presentation_to_dict(obj.presentation),
        "transform": {
            "x": obj.transform.x,
            "y": obj.transform.y,
            "scale": obj.transform.scale,
            "opacity": obj.transform.opacity,
            "rotation": obj.transform.rotation,
            "z_index": obj.transform.z_index,
        },
        "role": obj.role,
        "style": encode_value(obj.style),
    }


def presentation_to_dict(presentation: Presentation | None) -> dict[str, Any] | None:
    if presentation is None:
        return None
    data = {
        "type": presentation.__class__.__name__,
        "modality": presentation.modality,
        "style": encode_value(presentation.style),
    }
    for key, value in vars(presentation).items():
        if key not in data:
            data[key] = encode_value(value)
    return data


def feature_vector_to_dict(features: FeatureVector) -> dict[str, float]:
    return dict(features.values)


def encode_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: encode_value(getattr(value, key))
            for key in value.__dataclass_fields__
        }
    if isinstance(value, list):
        return [encode_value(item) for item in value]
    if isinstance(value, tuple):
        return [encode_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): encode_value(item) for key, item in value.items()}
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
