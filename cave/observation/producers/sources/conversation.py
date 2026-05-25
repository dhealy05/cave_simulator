from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from cave.observation.episodes import Episode, EpisodeInput, EpisodeObservation
from cave.observation.experience import TextPresentation
from cave.observation.producers.sources.gpt2 import (
    EpisodeProjection,
    _normalize_weights,
    _resolve_layer_index,
    _softmax,
    _top_predictions,
    attention_concentration,
    fit_episode_projection,
)
from cave.commitments.topology import SubjectiveTopologyParams, SubjectiveTopologyPrior


Array = np.ndarray


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    text: str
    id: str | None = None


@dataclass(frozen=True)
class ConversationSegment:
    id: str
    role: str
    text: str
    formatted_text: str
    start_token: int
    end_token: int
    order_index: int

    @property
    def token_count(self) -> int:
        return self.end_token - self.start_token


@dataclass(frozen=True)
class SelectedSegments:
    positions: list[int]
    weights: Array
    retained_mass: float


class ConversationProducer:
    name = "conversation"

    def __init__(
        self,
        model_path: str | Path = "lib/models/gpt2",
        *,
        name: str = "conversation",
        backend_name: str = "gpt2",
        feature_count: int = 8,
        context_selection: str = "attended_top_k",
        context_top_k: int = 8,
        attention_layer: int = -1,
        top_prediction_k: int = 8,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        if feature_count < 2:
            raise ValueError("feature_count must be at least 2")
        if context_top_k <= 0:
            raise ValueError("context_top_k must be positive")
        self.model_path = Path(model_path)
        self.name = name
        self.backend_name = backend_name
        self.feature_count = feature_count
        self.context_selection = context_selection
        self.context_top_k = context_top_k
        self.attention_layer = attention_layer
        self.top_prediction_k = top_prediction_k
        self._model = model
        self._tokenizer = tokenizer

    def run(
        self,
        turns: Sequence[ConversationTurn | tuple[str, str] | dict[str, object]],
        *,
        max_length: int | None = None,
    ) -> Episode:
        torch, model, tokenizer = self._load_runtime()
        turns = _coerce_turns(turns)
        full_text, segment_specs, spans = _format_turns_with_token_spans(
            turns,
            tokenizer,
        )
        untruncated = tokenizer(full_text, return_tensors="pt")
        token_ids = untruncated["input_ids"][0].detach().cpu().numpy().astype(int)
        if max_length is not None and token_ids.size > max_length:
            raise ValueError(
                "conversation token count exceeds max_length; truncation would "
                "invalidate segment spans"
            )
        if token_ids.size < 2:
            raise ValueError("conversation episode requires at least two tokens")

        model.eval()
        with torch.no_grad():
            outputs = model(
                **untruncated,
                output_attentions=True,
                output_hidden_states=True,
            )
        if outputs.attentions is None:
            raise ValueError(
                "conversation backend did not return attention tensors; load with "
                'attn_implementation="eager"'
            )

        embedding_matrix = (
            model.get_input_embeddings().weight.detach().cpu().numpy().astype(float)
        )
        logits = outputs.logits[0].detach().cpu().numpy().astype(float)
        hidden_states = outputs.hidden_states[-1][0].detach().cpu().numpy().astype(float)
        attentions = np.stack(
            [
                layer[0].detach().cpu().numpy().astype(float)
                for layer in outputs.attentions
            ],
            axis=0,
        )
        segments = [
            ConversationSegment(
                id=spec.id,
                role=spec.role,
                text=spec.text,
                formatted_text=spec.formatted_text,
                start_token=start,
                end_token=end,
                order_index=index,
            )
            for index, (spec, (start, end)) in enumerate(zip(segment_specs, spans))
        ]
        return build_conversation_episode(
            source_name=self.name,
            backend_name=self.backend_name,
            segments=segments,
            token_ids=token_ids,
            embedding_matrix=embedding_matrix,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
            feature_count=self.feature_count,
            context_selection=self.context_selection,
            context_top_k=self.context_top_k,
            attention_layer=self.attention_layer,
            top_prediction_k=self.top_prediction_k,
            model_name_or_path=str(self.model_path),
            tokenizer_name_or_path=str(self.model_path),
            decode_token=lambda token_id: tokenizer.decode(
                [int(token_id)],
                clean_up_tokenization_spaces=False,
            ),
        )

    def _load_runtime(self) -> tuple[Any, Any, Any]:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Conversation support needs the optional GPT-2 runtime. Install it "
                'with: python -m pip install -e ".[gpt2]"'
            ) from exc

        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                attn_implementation="eager",
            )
        return torch, model, tokenizer


ConversationEpisodeSource = ConversationProducer


def build_conversation_episode(
    *,
    source_name: str,
    segments: Sequence[ConversationSegment],
    token_ids: Sequence[int],
    embedding_matrix: Array,
    logits: Array,
    hidden_states: Array,
    attentions: Array,
    backend_name: str = "gpt2",
    feature_count: int = 8,
    context_selection: str = "attended_top_k",
    context_top_k: int = 8,
    attention_layer: int = -1,
    top_prediction_k: int = 8,
    model_name_or_path: str | None = None,
    tokenizer_name_or_path: str | None = None,
    decode_token: Callable[[int], str] | None = None,
) -> Episode:
    token_ids = np.asarray(token_ids, dtype=int)
    embedding_matrix = np.asarray(embedding_matrix, dtype=float)
    logits = np.asarray(logits, dtype=float)
    hidden_states = np.asarray(hidden_states, dtype=float)
    attentions = np.asarray(attentions, dtype=float)
    segments = tuple(segments)
    token_count = int(token_ids.size)

    if len(segments) < 2:
        raise ValueError("conversation episode requires at least two segments")
    if token_count < 2:
        raise ValueError("conversation episode requires at least two tokens")
    if logits.shape[0] < token_count or hidden_states.shape[0] < token_count:
        raise ValueError("logits and hidden_states must include every token")
    if attentions.ndim != 4:
        raise ValueError("attentions must have shape (layers, heads, tokens, tokens)")
    if feature_count < 2:
        raise ValueError("feature_count must be at least 2")

    _validate_segments(segments, token_count)
    token_embeddings = embedding_matrix[token_ids]
    raw_actual = np.vstack(
        [
            token_embeddings[segment.start_token : segment.end_token].mean(axis=0)
            for segment in segments
        ]
    )
    raw_memory = np.vstack(
        [hidden_states[segment.end_token - 1] for segment in segments]
    )
    raw_expected = _segment_expected_vectors(
        segments,
        logits,
        embedding_matrix,
    )
    projection = fit_episode_projection(
        np.vstack(
            [
                token_embeddings,
                hidden_states[:token_count],
                raw_actual,
                raw_memory,
                raw_expected[1:],
            ]
        ),
        feature_count=feature_count,
    )
    projected_segments = projection.project(raw_actual)
    projected_memory = projection.project(raw_memory)
    projected_expected = projection.project(raw_expected)
    feature_names = list(projection.feature_names)
    selected_layer = _resolve_layer_index(attention_layer, attentions.shape[0])

    inputs = [
        _episode_input_for_segment(segment, projected_segments[index])
        for index, segment in enumerate(segments)
    ]
    observations = []
    for index, segment in enumerate(segments[1:], start=1):
        prior_weights = _prior_segment_attention_weights(
            attentions,
            selected_layer,
            segments,
            index,
        )
        selected = select_active_segments(
            prior_weights,
            index,
            mode=context_selection,
            top_k=context_top_k,
        )
        active_inputs = [segments[position].id for position in selected.positions]
        attention_weights = {
            segments[position].id: float(weight)
            for position, weight in zip(selected.positions, selected.weights)
        }
        token_probabilities = _segment_token_probabilities(
            segment,
            token_ids,
            logits,
        )
        first_probabilities = _softmax(logits[segment.start_token - 1])
        observations.append(
            EpisodeObservation(
                t=float(index),
                t_normalized=float(index / (len(segments) - 1)),
                expected=projected_expected[index],
                actual=projected_segments[index],
                memory_state=projected_memory[index],
                surprise=float(
                    -np.mean(np.log(np.maximum(token_probabilities, 1e-300)))
                ),
                learning_rate=0.0,
                attention=attention_concentration(selected.weights),
                attention_weights=attention_weights,
                active_inputs=active_inputs,
                input_features={
                    segments[position].id: projected_segments[position]
                    for position in selected.positions
                },
                metadata={
                    "segment_id": segment.id,
                    "segment_index": index,
                    "segment_role": segment.role,
                    "segment_text": segment.text,
                    "formatted_text": segment.formatted_text,
                    "token_start": segment.start_token,
                    "token_end": segment.end_token,
                    "segment_token_count": segment.token_count,
                    "mean_token_probability": float(np.mean(token_probabilities)),
                    "top_predictions": _top_predictions(
                        first_probabilities,
                        top_prediction_k,
                        decode_token=decode_token,
                    ),
                    "attention_layer": selected_layer,
                    "context_selection": context_selection,
                    "retained_attention_mass": selected.retained_mass,
                    "selected_context_ids": active_inputs,
                    "memory_interpretation": "mock_prior_context",
                },
            )
        )

    return Episode(
        source_name=source_name,
        vocabulary=feature_names,
        inputs=inputs,
        observations=observations,
        duration=float(len(segments)),
        metadata={
            "source": "conversation.forward_pass",
            "adapter": "ConversationProducer",
            "backend": backend_name,
            "model_name_or_path": model_name_or_path,
            "tokenizer_name_or_path": tokenizer_name_or_path,
            "segment_mode": "turn",
            "segment_count": len(segments),
            "context_length": token_count,
            "feature_names": feature_names,
            "projection": {
                "method": "pca",
                "scope": "episode",
                "feature_count": feature_count,
            },
            "context_memory_mode": "prior_turns",
            "context_selection": context_selection,
            "context_top_k": context_top_k,
            "attention_layer": selected_layer,
            "attention_aggregation": "mean_heads_to_turns",
            "top_prediction_k": top_prediction_k,
            "surprise_log_base": "e",
            "memory_interpretation": "mock_prior_context",
            "mock_memory_note": (
                "Prior turns are supplied in context; they are not "
                "transformer-stored memories."
            ),
            "presentation_mode": "current_conversation_segment",
            "lookback_mode": "conversation_mock_memory",
            "topology_params": SubjectiveTopologyParams(
                feature_x=feature_names[0],
                feature_y=feature_names[1],
                prior=SubjectiveTopologyPrior(),
            ),
        },
    )


def select_active_segments(
    weights: Array,
    current_index: int,
    *,
    mode: str,
    top_k: int,
) -> SelectedSegments:
    weights = np.asarray(weights, dtype=float)
    if current_index <= 0:
        return SelectedSegments([], np.array([], dtype=float), 0.0)
    if weights.size != current_index:
        raise ValueError("weights must contain one value for each prior segment")
    if mode == "full_context":
        positions = list(range(current_index))
    elif mode == "recent_k":
        positions = list(range(max(0, current_index - top_k), current_index))
    elif mode == "attended_top_k":
        order = np.argsort(-weights, kind="stable")[: min(top_k, weights.size)]
        positions = sorted(int(index) for index in order)
    else:
        raise ValueError(
            "context_selection must be one of: "
            "attended_top_k, full_context, recent_k"
        )
    selected_weights = np.array([weights[position] for position in positions], dtype=float)
    retained_mass = float(np.sum(selected_weights))
    selected_weights = _normalize_weights(selected_weights)
    return SelectedSegments(
        positions=positions,
        weights=selected_weights,
        retained_mass=retained_mass,
    )


def conversation_text_from_turns(
    turns: Sequence[ConversationTurn | tuple[str, str] | dict[str, object]],
) -> str:
    turns = _coerce_turns(turns)
    return "".join(_format_turn(turn) for turn in turns)


@dataclass(frozen=True)
class _FormattedSegment:
    id: str
    role: str
    text: str
    formatted_text: str


def _episode_input_for_segment(
    segment: ConversationSegment,
    features: Array,
) -> EpisodeInput:
    label = _segment_label(segment.role, segment.text)
    return EpisodeInput(
        id=segment.id,
        kind=label,
        start=float(segment.order_index),
        end=float(segment.order_index + 1),
        order_index=segment.order_index,
        features=features,
        modality="conversation",
        presentation=TextPresentation(
            text=label,
            modality="conversation",
            style={
                "fill": "#ffffff",
                "stroke": "#1f2933",
                "text_color": "#111827",
                "wrap_width": 34,
            },
        ),
        metadata={
            "role": segment.role,
            "text": segment.text,
            "formatted_text": segment.formatted_text,
            "token_start": segment.start_token,
            "token_end": segment.end_token,
            "token_count": segment.token_count,
        },
    )


def _format_turns_with_token_spans(
    turns: Sequence[ConversationTurn],
    tokenizer: Any,
) -> tuple[str, list[_FormattedSegment], list[tuple[int, int]]]:
    specs = [
        _FormattedSegment(
            id=turn.id or f"turn:{index}",
            role=_normalize_role(turn.role),
            text=str(turn.text),
            formatted_text=_format_turn(turn),
        )
        for index, turn in enumerate(turns)
    ]
    spans: list[tuple[int, int]] = []
    prefix = ""
    for spec in specs:
        start = _token_count(tokenizer, prefix)
        prefix += spec.formatted_text
        end = _token_count(tokenizer, prefix)
        spans.append((start, end))
    return prefix, specs, spans


def _token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(text, return_tensors="pt")
    return int(encoded["input_ids"][0].numel())


def _coerce_turns(
    turns: Sequence[ConversationTurn | tuple[str, str] | dict[str, object]],
) -> tuple[ConversationTurn, ...]:
    coerced = []
    for index, item in enumerate(turns):
        if isinstance(item, ConversationTurn):
            coerced.append(item)
        elif isinstance(item, dict):
            role = str(item.get("role", "user"))
            text = str(item.get("text", ""))
            turn_id = item.get("id")
            coerced.append(
                ConversationTurn(
                    role=role,
                    text=text,
                    id=None if turn_id is None else str(turn_id),
                )
            )
        else:
            role, text = item
            coerced.append(ConversationTurn(role=str(role), text=str(text)))
    if len(coerced) < 2:
        raise ValueError("conversation episode requires at least two turns")
    return tuple(coerced)


def _format_turn(turn: ConversationTurn) -> str:
    role = _normalize_role(turn.role).capitalize()
    return f"{role}: {turn.text}\n"


def _normalize_role(role: str) -> str:
    normalized = str(role).strip().lower()
    return normalized or "user"


def _segment_label(role: str, text: str) -> str:
    clean = " ".join(str(text).split())
    if len(clean) > 96:
        clean = clean[:93].rstrip() + "..."
    return f"{role}: {clean}"


def _validate_segments(
    segments: Sequence[ConversationSegment],
    token_count: int,
) -> None:
    previous_end = 0
    seen_ids = set()
    for segment in segments:
        if segment.id in seen_ids:
            raise ValueError(f"duplicate conversation segment id: {segment.id}")
        seen_ids.add(segment.id)
        if segment.start_token < 0 or segment.end_token > token_count:
            raise ValueError("conversation segment token span is out of range")
        if segment.end_token <= segment.start_token:
            raise ValueError("conversation segment token span must be non-empty")
        if segment.start_token < previous_end:
            raise ValueError("conversation segment token spans must be ordered")
        previous_end = segment.end_token
    if segments[0].start_token != 0:
        raise ValueError("first conversation segment must start at token 0")


def _segment_expected_vectors(
    segments: Sequence[ConversationSegment],
    logits: Array,
    embedding_matrix: Array,
) -> Array:
    vectors = []
    for index, segment in enumerate(segments):
        if index == 0:
            vectors.append(
                embedding_matrix.mean(axis=0)
            )
            continue
        expected = []
        for token_position in range(segment.start_token, segment.end_token):
            probabilities = _softmax(logits[token_position - 1])
            expected.append(probabilities @ embedding_matrix)
        vectors.append(np.mean(np.vstack(expected), axis=0))
    return np.vstack(vectors)


def _segment_token_probabilities(
    segment: ConversationSegment,
    token_ids: Array,
    logits: Array,
) -> Array:
    probabilities = []
    for token_position in range(segment.start_token, segment.end_token):
        predicted = _softmax(logits[token_position - 1])
        probabilities.append(float(predicted[int(token_ids[token_position])]))
    return np.asarray(probabilities, dtype=float)


def _prior_segment_attention_weights(
    attentions: Array,
    selected_layer: int,
    segments: Sequence[ConversationSegment],
    current_index: int,
) -> Array:
    segment = segments[current_index]
    weights = np.zeros(current_index, dtype=float)
    samples = 0
    for query_position in range(segment.start_token, segment.end_token):
        token_weights = attentions[
            selected_layer,
            :,
            query_position,
            : query_position + 1,
        ].mean(axis=0)
        token_weights = _normalize_weights(token_weights)
        for prior_index, prior_segment in enumerate(segments[:current_index]):
            end = min(prior_segment.end_token, token_weights.size)
            if prior_segment.start_token >= end:
                continue
            weights[prior_index] += float(
                np.sum(token_weights[prior_segment.start_token:end])
            )
        samples += 1
    if samples > 0:
        weights /= samples
    return weights
