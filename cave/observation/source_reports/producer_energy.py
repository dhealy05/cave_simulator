from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from cave.demonstrations.reports.cave_reference import reference_cave_report_spec
from cave.observation.energy import summarize_episode_energy
from cave.observation.episodes import Episode
from cave.observation.producers.sources.conversation import (
    ConversationSegment,
    build_conversation_episode,
)
from cave.observation.producers.sources.gpt2 import build_gpt2_episode
from cave.observation.projections import encode_value
from cave.observation.views import default_views
from cave.presentation.renderers.matplotlib_renderer import available_styles
from cave.presentation.reports.generate import write_producer_report
from cave.presentation.reports.specs import (
    ProducerReportSpec,
    ReportExtraAsset,
    ReportSection,
)


EpisodeFactoryMap = Mapping[str, Callable[[], Episode]]


def canonical_producer_energy_report_spec(
    *,
    dt: float = 0.5,
    fps: int = 4,
    include_assets: bool = True,
    episode_factories: EpisodeFactoryMap | None = None,
    style: str = "default",
) -> ProducerReportSpec:
    del include_assets
    factories = dict(episode_factories or default_canonical_episode_factories(dt=dt))

    def build_episode() -> Episode:
        return factories["cave"]()

    return ProducerReportSpec(
        id="canonical-producer-energy",
        title="Canonical Producer Energy Report",
        episode_factory=build_episode,
        input_summary="Cave, GPT-2-shaped, and conversation-shaped canonical producer episodes",
        description=(
            "Scores canonical producer Episode outputs with the shared proxy "
            "energy ledger. The language-producer fixtures use deterministic "
            "backend tensors so this report is runnable without downloading a "
            "local GPT-2 model."
        ),
        views=default_views(),
        extra_assets=(
            ReportExtraAsset(
                id="producer_energy_metrics",
                title="Canonical Producer Energy Metrics JSON",
                filename="canonical_producer_energy_metrics.json",
                writer=lambda episode, output: write_canonical_producer_energy_metrics_json(
                    output,
                    episode_factories=factories,
                ),
            ),
        ),
        checks=(lambda episode: check_canonical_producer_energy(factories),),
        frame_time=2.0,
        dt=dt,
        fps=fps,
        columns=2,
        style=style,
        config={
            "producer": "canonical_producer_energy",
            "dt": dt,
            "fps": fps,
            "style": style,
            "producer_ids": sorted(factories),
        },
        sections=(
            ReportSection(
                title="Question",
                body=(
                    "What compression, loss, rails, and subject-energy proxy "
                    "signals are already visible in the canonical producer "
                    "Episode contract?"
                ),
                asset_ids=("producer_energy_metrics",),
            ),
            ReportSection(
                title="Boundary",
                body=(
                    "This report scores emitted episodes. It does not measure "
                    "real GPU/CPU joules, and fixture-backed GPT-2/conversation "
                    "episodes should be read as adapter-level energy proxies."
                ),
            ),
        ),
    )


def default_canonical_episode_factories(*, dt: float = 0.5) -> dict[str, Callable[[], Episode]]:
    return {
        "cave": reference_cave_report_spec(
            dt=dt,
            fps=4,
            include_readme_assets=False,
        ).episode_factory,
        "gpt2": _fixture_gpt2_episode,
        "conversation": _fixture_conversation_episode,
    }


def check_canonical_producer_energy(
    episode_factories: EpisodeFactoryMap | None = None,
) -> dict[str, object]:
    factories = dict(episode_factories or default_canonical_episode_factories())
    episodes = {name: factory() for name, factory in factories.items()}
    metrics = {name: summarize_episode_energy(episode) for name, episode in episodes.items()}
    compact = _compact_metrics(metrics)
    roles = _roles(compact)
    errors = []
    for producer_id, summary in compact.items():
        if summary["subject_total"] <= 0.0:
            errors.append(f"{producer_id} has no subject energy proxy")
        if summary["mean_loss_presence"] <= 0.0:
            errors.append(f"{producer_id} has no loss/surprise proxy")
    if compact["cave"]["rail_independence"] <= 0.0:
        errors.append("cave rail independence did not resolve")
    return {
        "id": "canonical_producer_energy",
        "ok": not errors,
        "errors": errors,
        "metrics": compact,
        "roles": roles,
    }


def write_canonical_producer_energy_metrics_json(
    output: Path,
    *,
    episode_factories: EpisodeFactoryMap | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    result = check_canonical_producer_energy(episode_factories)
    output.write_text(json.dumps(encode_value(result), indent=2) + "\n", encoding="utf-8")


def _compact_metrics(metrics: dict[str, dict[str, object]]) -> dict[str, dict[str, float]]:
    compact: dict[str, dict[str, float]] = {}
    for name, summary in metrics.items():
        compact[name] = {
            "compression_load": float(summary["compression"]["compression_load"]),  # type: ignore[index]
            "mean_loss_presence": float(summary["loss"]["mean_loss_presence"]),  # type: ignore[index]
            "rails_total": float(summary["energy"]["rails_total"]),  # type: ignore[index]
            "subject_total": float(summary["energy"]["subject_total"]),  # type: ignore[index]
            "adaptive_extra_total": float(summary["energy"]["adaptive_extra_total"]),  # type: ignore[index]
            "rail_independence": float(summary["energy"]["rail_independence"]),  # type: ignore[index]
            "dynamic_energy_coupling": float(summary["coupling"]["dynamic_energy_coupling"]),  # type: ignore[index]
            "hick_slope": float(summary["coupling"]["hick_slope"]),  # type: ignore[index]
            "instantiation_proxy": float(summary["summary"]["instantiation_proxy"]),  # type: ignore[index]
        }
    return compact


def _roles(metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        "compression_load": {
            name: values["compression_load"] for name, values in metrics.items()
        },
        "loss_presence": {
            name: values["mean_loss_presence"] for name, values in metrics.items()
        },
        "rail_independence": {
            name: values["rail_independence"] for name, values in metrics.items()
        },
        "dynamic_energy_coupling": {
            name: values["dynamic_energy_coupling"] for name, values in metrics.items()
        },
        "instantiation_proxy": {
            name: values["instantiation_proxy"] for name, values in metrics.items()
        },
    }


def _fixture_gpt2_episode() -> Episode:
    token_ids = np.array([0, 1, 2, 3, 4], dtype=int)
    token_texts = ["Hello", " Paul", " likes", " cave", "."]
    embedding_matrix = np.array(
        [
            [1.0, 0.0, 0.1, 0.0],
            [0.0, 1.0, 0.0, 0.2],
            [0.2, 0.0, 1.0, 0.0],
            [0.0, 0.2, 0.0, 1.0],
            [0.7, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.7],
        ],
        dtype=float,
    )
    logits = np.full((len(token_ids), embedding_matrix.shape[0]), -2.0, dtype=float)
    for index, token_id in enumerate(token_ids[1:], start=1):
        logits[index - 1, token_id] = 4.0
        logits[index - 1, (token_id + 1) % embedding_matrix.shape[0]] = 1.0
    hidden_states = np.array(
        [
            [0.2, 0.1, 0.0, 0.0],
            [0.4, 0.2, 0.1, 0.2],
            [0.1, 0.5, 0.3, 0.1],
            [0.0, 0.3, 0.8, 0.4],
            [0.3, 0.2, 0.5, 0.9],
        ],
        dtype=float,
    )
    attentions = _causal_attention(token_count=len(token_ids), heads=2)
    return build_gpt2_episode(
        source_name="gpt2",
        token_ids=token_ids,
        token_texts=token_texts,
        embedding_matrix=embedding_matrix,
        logits=logits,
        hidden_states=hidden_states,
        attentions=attentions,
        feature_count=3,
        active_top_k=2,
        top_prediction_k=2,
        decode_token=lambda token_id: f"tok{token_id}",
    )


def _fixture_conversation_episode() -> Episode:
    turns = [
        ("user", "Prior context?"),
        ("assistant", "Protocol memory."),
        ("user", "Expected actual?"),
        ("assistant", "Token signals, turn views."),
    ]
    token_count = len(turns) * 2
    vocab_size = token_count + 2
    token_ids = np.array([(index * 2 + len(turns[index // 2][0])) % vocab_size for index in range(token_count)], dtype=int)
    embedding_matrix = np.eye(vocab_size, 4, dtype=float)
    for index in range(vocab_size):
        embedding_matrix[index] += np.array(
            [0.03 * index, 0.02 * (index % 3), 0.01 * len(turns), 0.04 * (index % 2)],
            dtype=float,
        )
    logits = np.full((token_count, vocab_size), -2.0, dtype=float)
    for index, token_id in enumerate(token_ids[1:], start=1):
        logits[index - 1, token_id] = 4.0
        logits[index - 1, (token_id + 1) % vocab_size] = 1.0
    hidden_states = np.vstack(
        [
            np.array([0.1 * (index + 1), 0.2 * (index % 3), 0.05 * len(turns[index // 2][1]), 0.15 * (index // 2)], dtype=float)
            for index in range(token_count)
        ]
    )
    segments = [
        ConversationSegment(
            id=f"turn:{index}",
            role=role,
            text=text,
            formatted_text=f"{role.capitalize()}: {text}\n",
            start_token=index * 2,
            end_token=index * 2 + 2,
            order_index=index,
        )
        for index, (role, text) in enumerate(turns)
    ]
    return build_conversation_episode(
        source_name="conversation",
        backend_name="fixture-gpt2",
        segments=segments,
        token_ids=token_ids,
        embedding_matrix=embedding_matrix,
        logits=logits,
        hidden_states=hidden_states,
        attentions=_causal_attention(token_count=token_count, heads=2),
        feature_count=3,
        context_selection="attended_top_k",
        context_top_k=2,
        top_prediction_k=2,
        decode_token=lambda token_id: f"tok{token_id}",
    )


def _causal_attention(*, token_count: int, heads: int) -> np.ndarray:
    attentions = np.zeros((1, heads, token_count, token_count), dtype=float)
    for head in range(heads):
        for target in range(token_count):
            weights = np.arange(1, target + 2, dtype=float)
            if head % 2 == 1:
                weights = weights[::-1]
            attentions[0, head, target, : target + 1] = weights / np.sum(weights)
    return attentions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate canonical producer energy report.")
    parser.add_argument("--output", type=Path, default=Path("out/reports/producers/energy"))
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--style", default="default", choices=available_styles())
    args = parser.parse_args()
    spec = canonical_producer_energy_report_spec(
        dt=args.dt,
        fps=args.fps,
        style=args.style,
    )
    outputs = write_producer_report(spec, args.output)
    print(f"wrote {outputs.report_md}")


if __name__ == "__main__":
    main()
