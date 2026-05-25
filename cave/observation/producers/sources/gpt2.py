from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from cave.observation.episodes import Episode, EpisodeInput, EpisodeObservation
from cave.observation.experience import TextPresentation
from cave.commitments.topology import SubjectiveTopologyParams, SubjectiveTopologyPrior


Array = np.ndarray


@dataclass(frozen=True)
class EpisodeProjection:
    feature_names: list[str]
    mean: Array
    components: Array
    low: Array
    high: Array

    def project(self, vectors: Array) -> Array:
        vectors = np.asarray(vectors, dtype=float)
        original_ndim = vectors.ndim
        if original_ndim == 1:
            vectors = vectors.reshape(1, -1)
        projected = (vectors - self.mean) @ self.components.T
        span = self.high - self.low
        normalized = np.empty_like(projected, dtype=float)
        constant = span <= 1e-12
        normalized[:, constant] = 0.5
        normalized[:, ~constant] = (
            (projected[:, ~constant] - self.low[~constant])
            / span[~constant]
        )
        normalized = np.clip(normalized, 0.0, 1.0)
        if original_ndim == 1:
            return normalized[0]
        return normalized


class GPT2Producer:
    name = "gpt2"

    def __init__(
        self,
        model_path: str | Path = "lib/models/gpt2",
        *,
        name: str = "gpt2",
        feature_count: int = 8,
        active_input_mode: str = "attended_top_k",
        active_top_k: int = 8,
        attention_layer: int = -1,
        top_prediction_k: int = 8,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        if feature_count < 2:
            raise ValueError("feature_count must be at least 2")
        if active_top_k <= 0:
            raise ValueError("active_top_k must be positive")
        self.model_path = Path(model_path)
        self.name = name
        self.feature_count = feature_count
        self.active_input_mode = active_input_mode
        self.active_top_k = active_top_k
        self.attention_layer = attention_layer
        self.top_prediction_k = top_prediction_k
        self._model = model
        self._tokenizer = tokenizer

    def run(self, text: str, *, max_length: int | None = None) -> Episode:
        torch, model, tokenizer = self._load_runtime()
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=max_length is not None,
            max_length=max_length,
        )
        token_ids = encoded["input_ids"][0].detach().cpu().numpy().astype(int)
        if token_ids.size < 2:
            raise ValueError("GPT-2 episode requires at least two tokens")

        model.eval()
        with torch.no_grad():
            outputs = model(
                **encoded,
                output_attentions=True,
                output_hidden_states=True,
            )
        if outputs.attentions is None:
            raise ValueError(
                "GPT-2 did not return attention tensors; load with "
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
        token_texts = [
            tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
            for token_id in token_ids
        ]
        return build_gpt2_episode(
            source_name=self.name,
            token_ids=token_ids,
            token_texts=token_texts,
            embedding_matrix=embedding_matrix,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
            feature_count=self.feature_count,
            active_input_mode=self.active_input_mode,
            active_top_k=self.active_top_k,
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
                "GPT-2 support is optional. Install it with: "
                'python -m pip install -e ".[gpt2]"'
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


GPT2EpisodeSource = GPT2Producer


def build_gpt2_episode(
    *,
    source_name: str,
    token_ids: Sequence[int],
    token_texts: Sequence[str],
    embedding_matrix: Array,
    logits: Array,
    hidden_states: Array,
    attentions: Array,
    feature_count: int = 8,
    active_input_mode: str = "attended_top_k",
    active_top_k: int = 8,
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
    token_count = int(token_ids.size)

    if token_count < 2:
        raise ValueError("GPT-2 episode requires at least two tokens")
    if len(token_texts) != token_count:
        raise ValueError("token_texts must match token_ids length")
    if logits.shape[0] < token_count or hidden_states.shape[0] < token_count:
        raise ValueError("logits and hidden_states must include every token")
    if attentions.ndim != 4:
        raise ValueError("attentions must have shape (layers, heads, tokens, tokens)")
    if feature_count < 2:
        raise ValueError("feature_count must be at least 2")

    token_embeddings = embedding_matrix[token_ids]
    projection = fit_episode_projection(
        np.vstack([token_embeddings, hidden_states[:token_count]]),
        feature_count=feature_count,
    )
    projected_tokens = projection.project(token_embeddings)
    feature_names = list(projection.feature_names)
    inputs = [
        EpisodeInput(
            id=_token_input_id(index),
            kind=_token_kind(token_texts[index]),
            start=float(index),
            end=float(index + 1),
            order_index=index,
            features=projected_tokens[index],
            modality="text",
            presentation=TextPresentation(
                text=_token_display_text(token_texts[index]),
                modality="text",
                style={
                    "fill": "#ffffff",
                    "stroke": "#1f2933",
                    "text_color": "#111827",
                },
            ),
            metadata={
                "token_id": int(token_ids[index]),
                "token_text": token_texts[index],
                "position": index,
            },
        )
        for index in range(token_count)
    ]

    selected_layer = _resolve_layer_index(attention_layer, attentions.shape[0])
    observations = []
    for index in range(1, token_count):
        probabilities = _softmax(logits[index - 1])
        actual_probability = float(probabilities[int(token_ids[index])])
        expected_raw = probabilities @ embedding_matrix
        expected = projection.project(expected_raw)
        actual = projected_tokens[index]
        memory_state = projection.project(hidden_states[index])
        context_weights = attentions[selected_layer, :, index, : index + 1].mean(axis=0)
        context_weights = _normalize_weights(context_weights)
        selected = select_active_context(
            context_weights,
            index,
            mode=active_input_mode,
            top_k=active_top_k,
        )
        active_inputs = [_token_input_id(position) for position in selected.positions]
        attention_weights = {
            _token_input_id(position): float(weight)
            for position, weight in zip(selected.positions, selected.weights)
        }
        top_predictions = _top_predictions(
            probabilities,
            top_prediction_k,
            decode_token=decode_token,
        )
        observations.append(
            EpisodeObservation(
                t=float(index),
                t_normalized=float(index / (token_count - 1)),
                expected=expected,
                actual=actual,
                memory_state=memory_state,
                surprise=float(-np.log(max(actual_probability, 1e-300))),
                learning_rate=0.0,
                attention=attention_concentration(selected.weights),
                attention_weights=attention_weights,
                active_inputs=active_inputs,
                input_features={
                    _token_input_id(position): projected_tokens[position]
                    for position in selected.positions
                },
                metadata={
                    "position": index,
                    "predicted_from_position": index - 1,
                    "token_id": int(token_ids[index]),
                    "token_text": token_texts[index],
                    "actual_token_probability": actual_probability,
                    "top_predictions": top_predictions,
                    "attention_layer": selected_layer,
                    "active_input_mode": active_input_mode,
                    "retained_attention_mass": selected.retained_mass,
                    "selected_context_ids": active_inputs,
                },
            )
        )

    return Episode(
        source_name=source_name,
        vocabulary=feature_names,
        inputs=inputs,
        observations=observations,
        duration=float(token_count),
        metadata={
            "source": "gpt2.forward_pass",
            "adapter": "GPT2Producer",
            "model_name_or_path": model_name_or_path,
            "tokenizer_name_or_path": tokenizer_name_or_path,
            "context_length": token_count,
            "feature_names": feature_names,
            "projection": {
                "method": "pca",
                "scope": "episode",
                "feature_count": feature_count,
            },
            "active_input_mode": active_input_mode,
            "active_top_k": active_top_k,
            "attention_layer": selected_layer,
            "attention_aggregation": "mean_heads",
            "top_prediction_k": top_prediction_k,
            "surprise_log_base": "e",
            "presentation_mode": "current_text",
            "lookback_mode": "attention_context",
            "topology_params": SubjectiveTopologyParams(
                feature_x=feature_names[0],
                feature_y=feature_names[1],
                prior=SubjectiveTopologyPrior(),
            ),
        },
    )


@dataclass(frozen=True)
class SelectedContext:
    positions: list[int]
    weights: Array
    retained_mass: float


def fit_episode_projection(vectors: Array, *, feature_count: int) -> EpisodeProjection:
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2:
        raise ValueError("vectors must have shape (samples, dimensions)")
    if feature_count <= 0:
        raise ValueError("feature_count must be positive")
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    components = np.zeros((feature_count, vectors.shape[1]), dtype=float)
    count = min(feature_count, vt.shape[0])
    if count > 0:
        components[:count] = vt[:count]
    projected = centered @ components.T
    return EpisodeProjection(
        feature_names=[f"pc{index + 1}" for index in range(feature_count)],
        mean=mean,
        components=components,
        low=projected.min(axis=0),
        high=projected.max(axis=0),
    )


def attention_concentration(weights: Array) -> float:
    weights = _normalize_weights(weights)
    if weights.size <= 1:
        return 1.0
    positive = weights[weights > 0.0]
    entropy = -float(np.sum(positive * np.log(positive)))
    max_entropy = float(np.log(weights.size))
    if max_entropy <= 0.0:
        return 1.0
    return float(np.clip(1.0 - entropy / max_entropy, 0.0, 1.0))


def select_active_context(
    weights: Array,
    current_index: int,
    *,
    mode: str,
    top_k: int,
) -> SelectedContext:
    weights = _normalize_weights(weights)
    if mode == "full_context":
        positions = list(range(current_index + 1))
    elif mode == "current_token":
        positions = [current_index]
    elif mode == "attended_top_k":
        order = np.argsort(-weights, kind="stable")[: min(top_k, weights.size)]
        positions = sorted(int(index) for index in order)
    else:
        raise ValueError(
            "active_input_mode must be one of: "
            "attended_top_k, full_context, current_token"
        )
    selected_weights = np.array([weights[position] for position in positions], dtype=float)
    retained_mass = float(np.sum(selected_weights))
    selected_weights = _normalize_weights(selected_weights)
    return SelectedContext(
        positions=positions,
        weights=selected_weights,
        retained_mass=retained_mass,
    )


def _softmax(values: Array) -> Array:
    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _normalize_weights(weights: Array) -> Array:
    weights = np.asarray(weights, dtype=float)
    if weights.size == 0:
        return weights
    weights = np.clip(weights, 0.0, None)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(weights.shape, 1.0 / weights.size, dtype=float)
    return weights / total


def _top_predictions(
    probabilities: Array,
    count: int,
    *,
    decode_token: Callable[[int], str] | None,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    order = np.argsort(-probabilities, kind="stable")[:count]
    return [
        {
            "token_id": int(token_id),
            "token_text": (
                None if decode_token is None else decode_token(int(token_id))
            ),
            "probability": float(probabilities[token_id]),
        }
        for token_id in order
    ]


def _resolve_layer_index(index: int, layer_count: int) -> int:
    resolved = index if index >= 0 else layer_count + index
    if resolved < 0 or resolved >= layer_count:
        raise ValueError(f"attention_layer {index} is out of range")
    return resolved


def _token_input_id(index: int) -> str:
    return f"tok:{index}"


def _token_kind(text: str) -> str:
    if text == "":
        return "<empty>"
    if text.strip() == "":
        return repr(text)
    return text


def _token_display_text(text: str) -> str:
    if text == "":
        return "<empty>"
    if text == " ":
        return "space"
    if text.strip() == "":
        return repr(text)
    return text.strip()
