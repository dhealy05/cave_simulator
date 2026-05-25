from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from cave.commitments.agency import apply_action_exposure
from cave.commitments.affect import ValenceState
from cave.commitments.attention import AttentionState, INTERNAL_EXPECTATION_CHANNEL
from cave.substrates.cavenet.blocks import (
    attention_gate,
    error_surprise_block,
    expectation_readout,
    learning_importance,
    learning_rate_block,
    state_input_from_workspace,
    workspace_block,
)
from cave.substrates.cavenet.config import CaveNetAdaptationPolicy, CaveNetConfig
from cave.substrates.cavenet.controller import CaveNetController, CaveNetControllerObservation
from cave.substrates.cavenet.state import CaveNetReadout, CaveNetState
from cave.observation.episodes import Episode, episode_from_cave_states
from cave.observation.experience import InputSequence
from cave.observation.sensing import Sensorium, default_sensorium
from cave.demonstrations.simulation import ModelParams
from cave.demonstrations.simulation.state import SceneState
from cave.demonstrations.state import SubjectState


BLOCK_NAMES = (
    "attention_gate",
    "workspace_block",
    "expectation_readout",
    "error_surprise_block",
    "value_objective_readout",
    "memory_topology_cell",
)


@dataclass
class CaveNet:
    sequence: InputSequence
    state: CaveNetState
    params: ModelParams
    vocabulary: list[str]
    sensorium: Sensorium = field(default_factory=default_sensorium)
    config: CaveNetConfig = field(default_factory=CaveNetConfig)
    adaptation_policy: CaveNetAdaptationPolicy = field(
        default_factory=CaveNetAdaptationPolicy
    )
    controller: CaveNetController | None = None
    config_history: list[dict[str, object]] = field(default_factory=list)
    last_valence: ValenceState | None = None
    readout: CaveNetReadout = field(
        default_factory=lambda: CaveNetReadout(block_names=BLOCK_NAMES)
    )

    @classmethod
    def from_subject_state(
        cls,
        *,
        sequence: InputSequence,
        subject_state: SubjectState,
        params: ModelParams,
        vocabulary: list[str],
        sensorium: Sensorium | None = None,
        config: CaveNetConfig | None = None,
        adaptation_policy: CaveNetAdaptationPolicy | None = None,
        controller: CaveNetController | None = None,
    ) -> "CaveNet":
        return cls(
            sequence=sequence,
            state=CaveNetState(subject_state=subject_state),
            params=params,
            vocabulary=list(vocabulary),
            sensorium=default_sensorium() if sensorium is None else sensorium,
            config=CaveNetConfig() if config is None else config,
            adaptation_policy=(
                CaveNetAdaptationPolicy()
                if adaptation_policy is None
                else adaptation_policy
            ),
            controller=controller,
        )

    def step(self, t: float) -> SceneState:
        current_objects = self.sequence.active_at(t)
        action = self.params.action_policy.choose_action(
            current_objects=current_objects,
            vocabulary=self.vocabulary,
        )
        state_objects = apply_action_exposure(current_objects, action)
        attention_state = self._attention_state_at(t)
        sensor_responses = self.sensorium.channel_responses(
            state_objects,
            self.vocabulary,
        )
        attended_input = attention_gate(
            sensorium=self.sensorium,
            sensor_responses=sensor_responses,
            attention=attention_state,
            vocabulary=self.vocabulary,
            gain=self.config.attention_gain,
        )
        workspace = workspace_block(
            compressor=self.params.workspace_compressor,
            attended_input=attended_input,
            vocabulary=self.vocabulary,
        )
        state_input = state_input_from_workspace(
            attended_input=attended_input,
            workspace=workspace,
            mode=self.params.workspace_input_mode,
            gain=self.config.state_input_gain,
        )
        expected_input = expectation_readout(
            self.state.subject_state.memory,
            self.vocabulary,
            attention=attention_state,
            gain=self.config.expectation_gain,
        )
        prediction = error_surprise_block(
            expected_input=expected_input,
            actual_input=state_input,
            surprise_gain=self.config.surprise_gain,
        )
        valence = self.params.valence_evaluator.evaluate(
            current_objects=state_objects,
            attention=attention_state,
            prediction=prediction,
            previous=self.last_valence,
        )
        objective = self.params.objective_evaluator.evaluate(
            prediction=prediction,
            valence=valence,
            attention=attention_state,
            compression_cost=workspace.compression_cost,
        )
        config_before_update = self.config
        importance = learning_importance(state_objects)
        learning_rate = learning_rate_block(
            learning_rule=self.params.learning_rule,
            base_rate=1.0 - self.state.subject_state.memory.retention,
            attention=attention_state,
            importance=importance,
            surprise=prediction.surprise,
            gain=self.config.learning_rate_gain,
        )
        topology_params = replace(
            self.params.topology,
            deposit_strength=(
                self.params.topology.deposit_strength
                * self.config.topology_deposit_gain
            ),
            expectation_deposit_strength=(
                self.params.topology.expectation_deposit_strength
                * self.config.topology_deposit_gain
            ),
            transition_strength=(
                self.params.topology.transition_strength
                * self.config.topology_transition_gain
            ),
        )
        self.state.subject_state.update(
            t,
            state_input,
            state_objects,
            attention_state,
            topology_params,
            vocabulary=self.vocabulary,
            prediction=prediction,
            learning_rate=learning_rate,
        )
        subject_snapshot = self.state.subject_state.snapshot()
        updated_attention = self.params.attention_policy.next_channel_weights(
            current_attention=attention_state,
            sensor_responses=sensor_responses,
            prediction=prediction,
            valence=valence,
            objective=objective,
        )
        if updated_attention is None:
            self.state.adaptive_channel_weights = None
            next_attention_channel_weights = dict(attention_state.channel_weights)
        else:
            self.state.adaptive_channel_weights = dict(updated_attention)
            next_attention_channel_weights = dict(updated_attention)
        self.last_valence = valence
        attention_impact = attention_state.impact()
        external_attention = attention_impact * sum(
            weight
            for channel, weight in attention_state.channel_weights.items()
            if channel != INTERNAL_EXPECTATION_CHANNEL
        )
        controller_observation = CaveNetControllerObservation(
            surprise=prediction.surprise,
            utility=objective.utility,
            compression_cost=workspace.compression_cost,
            memory_norm=_norm(subject_snapshot.memory.vector),
            attention=attention_state.scalar,
            external_attention=external_attention,
            internal_expectation_attention=(
                attention_state.internal_expectation_impact()
            ),
            topology_mass=float(np.sum(subject_snapshot.topology.density)),
        )
        if self.controller is None:
            self.config = self.adaptation_policy.adapt(
                self.config,
                surprise=prediction.surprise,
                utility=objective.utility,
                compression_cost=workspace.compression_cost,
            )
            controller_metadata = None
        else:
            self.config = self.controller.step(self.config, controller_observation)
            controller_metadata = self.controller.to_metadata()
        self.config_history.append(
            {
                "t": float(t),
                "before": config_before_update.to_dict(),
                "after": self.config.to_dict(),
                "surprise": float(prediction.surprise),
                "utility": float(objective.utility),
                "compression_cost": float(workspace.compression_cost),
                "controller_observation": controller_observation.to_dict(),
                "controller": controller_metadata,
            }
        )

        return SceneState(
            t=t,
            sequence=self.sequence,
            vocabulary=list(self.vocabulary),
            current_objects=list(state_objects),
            action=action,
            attention=attention_state.scalar,
            attention_state=attention_state,
            next_attention_channel_weights=next_attention_channel_weights,
            attention_profile=self.params.attention,
            sensor_responses=sensor_responses,
            attended_input_vector=attended_input.copy(),
            input_vector=state_input.copy(),
            workspace=workspace,
            workspace_input_mode=self.params.workspace_input_mode,
            prediction=prediction,
            valence=valence,
            objective=objective,
            learning_rate=learning_rate,
            learning_importance=importance,
            subject_state=subject_snapshot,
        )

    def run(self, *, start: float = 0.0, end: float | None = None, dt: float = 0.1) -> list[SceneState]:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        run_end = self.sequence.duration if end is None else end
        states = []
        t = start
        while t <= run_end + 1e-9:
            states.append(self.step(t))
            t += dt
        return states

    def episode(
        self,
        *,
        source_name: str = "cavenet",
        start: float = 0.0,
        end: float | None = None,
        dt: float = 0.1,
    ) -> Episode:
        states = self.run(start=start, end=end, dt=dt)
        final_topology_mass = (
            0.0
            if not states
            else float(np.sum(states[-1].subject_state.topology.density))
        )
        final_config = self.config.to_dict()
        return episode_from_cave_states(
            source_name,
            self.sequence,
            self.vocabulary,
            states,
            metadata={
                "source": "cave.substrates.cavenet",
                "adapter": "CaveNet",
                "cavenet": self.readout.to_metadata(),
                "cavenet_config": final_config,
                "cavenet_controller": (
                    None if self.controller is None else self.controller.to_metadata()
                ),
                "cavenet_initial_config": (
                    self.config_history[0]["before"]
                    if self.config_history
                    else final_config
                ),
                "cavenet_config_history": list(self.config_history),
                "cavenet_final_topology_mass": final_topology_mass,
                "memory_decay_tau": self.state.subject_state.memory.decay_tau,
                "memory_max_age": self.state.subject_state.memory.max_age,
                "memory_retention": self.state.subject_state.memory.retention,
                "topology_params": self.params.topology,
                "attention_curve": [
                    {
                        "t": float(t),
                        "value": self.params.attention.value_at(
                            float(t),
                            self.sequence.duration,
                        ),
                    }
                    for t in np.linspace(0.0, self.sequence.duration, 240)
                ],
            },
        )

    def _attention_state_at(self, t: float) -> AttentionState:
        base_state = self.params.attention.state_at(t, self.sequence.duration)
        if self.state.adaptive_channel_weights is None:
            return base_state
        return AttentionState(
            channel_weights=self.state.adaptive_channel_weights,
            capacity=base_state.capacity,
            high_gamma=base_state.high_gamma,
        )


@dataclass(frozen=True)
class CaveNetProducer:
    cavenet: CaveNet
    name: str = "cavenet"

    def run(
        self,
        *,
        start: float = 0.0,
        end: float | None = None,
        dt: float = 0.1,
    ) -> Episode:
        return self.cavenet.episode(
            source_name=self.name,
            start=start,
            end=end,
            dt=dt,
        )


def _norm(value) -> float:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.linalg.norm(array.ravel()) / np.sqrt(array.size))
