from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cave.commitments.attention import AttentionState
from cave.observation.experience import Array, ExperienceObject
from cave.observation.sensing.sensor import Sensor, SensorResponse, visual_feature_sensor


@dataclass(frozen=True)
class Sensorium:
    sensors: tuple[Sensor, ...] = field(default_factory=lambda: (visual_feature_sensor(),))

    def channel_responses(
        self,
        objects: list[ExperienceObject],
        vocabulary: list[str],
    ) -> dict[str, SensorResponse]:
        responses: dict[str, Array] = {}
        for sensor in self.sensors:
            vector = sensor.transduce(objects, vocabulary)
            if sensor.channel not in responses:
                responses[sensor.channel] = np.zeros(len(vocabulary), dtype=float)
            responses[sensor.channel] += vector
        return {
            channel: SensorResponse(channel=channel, vector=vector)
            for channel, vector in responses.items()
        }

    def attended_input(
        self,
        responses: dict[str, SensorResponse],
        attention: AttentionState,
        vocabulary: list[str],
    ) -> Array:
        attended = np.zeros(len(vocabulary), dtype=float)
        impact = attention.impact()
        for channel, response in responses.items():
            attended += impact * attention.channel_weight(channel) * response.vector
        return attended


def default_sensorium() -> Sensorium:
    return Sensorium()
