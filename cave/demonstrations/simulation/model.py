from __future__ import annotations

from dataclasses import dataclass, field

from cave.commitments.agency import apply_action_exposure
from cave.commitments.affect import ValenceState
from cave.commitments.attention import AttentionState
from cave.observation.experience import ExperienceObject, InputSequence
from cave.commitments.prediction import Predictor, default_predictor
from cave.observation.sensing import Sensorium, default_sensorium
from cave.demonstrations.simulation.params import ModelParams
from cave.demonstrations.simulation.state import SceneState
from cave.demonstrations.state import SubjectState


@dataclass
class ExperienceModel:
    sequence: InputSequence
    subject_state: SubjectState
    params: ModelParams
    vocabulary: list[str]
    sensorium: Sensorium = field(default_factory=default_sensorium)
    predictor: Predictor = field(default_factory=default_predictor)
    adaptive_channel_weights: dict[str, float] | None = None
    last_valence: ValenceState | None = None

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
        u_t = self.sensorium.attended_input(
            sensor_responses,
            attention_state,
            self.vocabulary,
        )
        workspace = self.params.workspace_compressor.compress(u_t, self.vocabulary)
        state_input = (
            workspace.reconstructed
            if self.params.workspace_input_mode == "workspace"
            else u_t
        )
        raw_expected_input = self.predictor.predict(
            self.subject_state.snapshot(),
            self.vocabulary,
        )
        expected_input = raw_expected_input * attention_state.internal_expectation_impact()
        prediction = self.predictor.evaluate(expected_input, state_input)
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
        learning_importance = self._learning_importance(state_objects)
        learning_rate = self.params.learning_rule.learning_rate(
            base_rate=1.0 - self.subject_state.memory.retention,
            attention=attention_state,
            importance=learning_importance,
            surprise=prediction.surprise,
        )

        topology_params = self.params.topology
        self.subject_state.update(
            t,
            state_input,
            state_objects,
            attention_state,
            topology_params,
            vocabulary=self.vocabulary,
            prediction=prediction,
            learning_rate=learning_rate,
        )
        subject_state_snapshot = self.subject_state.snapshot()
        updated_attention_channel_weights = self.params.attention_policy.next_channel_weights(
            current_attention=attention_state,
            sensor_responses=sensor_responses,
            prediction=prediction,
            valence=valence,
            objective=objective,
        )
        if updated_attention_channel_weights is None:
            self.adaptive_channel_weights = None
            next_attention_channel_weights = dict(attention_state.channel_weights)
        else:
            self.adaptive_channel_weights = dict(updated_attention_channel_weights)
            next_attention_channel_weights = dict(updated_attention_channel_weights)
        self.last_valence = valence

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
            attended_input_vector=u_t.copy(),
            input_vector=state_input.copy(),
            workspace=workspace,
            workspace_input_mode=self.params.workspace_input_mode,
            prediction=prediction,
            valence=valence,
            objective=objective,
            learning_rate=learning_rate,
            learning_importance=learning_importance,
            subject_state=subject_state_snapshot,
        )

    def run(self, start: float, end: float, dt: float) -> list[SceneState]:
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        states: list[SceneState] = []
        t = start
        while t <= end + 1e-9:
            states.append(self.step(t))
            t += dt
        return states

    def _learning_importance(self, current_objects: list[ExperienceObject]) -> float:
        if not current_objects:
            return 1.0
        total_salience = sum(max(0.0, obj.salience) for obj in current_objects)
        if total_salience <= 0.0:
            return 1.0
        return sum(
            max(0.0, obj.learning_weight) * max(0.0, obj.salience)
            for obj in current_objects
        ) / total_salience

    def _attention_state_at(self, t: float) -> AttentionState:
        base_state = self.params.attention.state_at(t, self.sequence.duration)
        if self.adaptive_channel_weights is None:
            return base_state
        return AttentionState(
            channel_weights=self.adaptive_channel_weights,
            capacity=base_state.capacity,
            high_gamma=base_state.high_gamma,
        )
