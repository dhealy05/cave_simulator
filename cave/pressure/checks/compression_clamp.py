from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from cave.observation.compression import summarize_episode_compression
from cave.observation.episodes import Episode, EpisodeInput, EpisodeObservation
from cave.observation.experience import Presentation
from cave.observation.projections import encode_value
from cave.observation.views import default_views
from cave.presentation.reports.specs import (
    ProducerReportSpec,
    ReportExtraAsset,
    ReportSection,
)


ClampVariant = Literal[
    "active",
    "fixed-container",
    "random-compressor",
    "shuffled-loss",
    "no-update",
    "oracle-rails",
]

CLAMP_VARIANTS: tuple[ClampVariant, ...] = (
    "active",
    "fixed-container",
    "random-compressor",
    "shuffled-loss",
    "no-update",
    "oracle-rails",
)

FEATURE_NAMES = ("threat", "reward", "clue", "context", "detail", "noise")
RELEVANCE = np.asarray([1.0, 0.9, 1.0, 0.55, 0.0, 0.0], dtype=float)
IRRELEVANCE = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=float)
FIXED_PRIORITY = (4, 5, 3, 1, 0, 2)
SHUFFLED_LOSS_DRIVER = np.asarray(
    [0.80, 0.15, 0.65, 0.20, 0.90, 0.10, 0.45, 0.75, 0.18, 0.55, 0.25, 0.70],
    dtype=float,
)

STREAM = np.asarray(
    [
        [0.05, 0.20, 0.10, 0.25, 0.85, 0.35],
        [0.10, 0.25, 0.55, 0.35, 0.90, 0.30],
        [0.15, 0.70, 0.45, 0.45, 0.88, 0.45],
        [0.82, 0.25, 0.35, 0.60, 0.86, 0.70],
        [0.78, 0.30, 0.65, 0.55, 0.80, 0.85],
        [0.35, 0.82, 0.50, 0.50, 0.78, 0.65],
        [0.88, 0.22, 0.78, 0.62, 0.70, 0.90],
        [0.30, 0.75, 0.82, 0.70, 0.74, 0.80],
        [0.18, 0.68, 0.72, 0.65, 0.82, 0.55],
        [0.62, 0.28, 0.60, 0.58, 0.92, 0.60],
        [0.15, 0.72, 0.42, 0.45, 0.88, 0.38],
        [0.08, 0.35, 0.30, 0.35, 0.82, 0.25],
    ],
    dtype=float,
)

CAPACITY_SCHEDULE = (6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6)
PHASES = (
    "open",
    "open",
    "mild",
    "clamp",
    "hard",
    "overload",
    "overload",
    "release",
    "release",
    "recovery",
    "recovery",
    "open",
)


@dataclass(frozen=True)
class CompressionClampConfig:
    eta: float = 0.62
    decay: float = 0.04

    def __post_init__(self) -> None:
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError("eta must be in [0, 1]")
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")


def compression_clamp_report_spec(
    *,
    dt: float = 1.0,
    fps: int = 4,
    include_assets: bool = True,
) -> ProducerReportSpec:
    extra_assets = ()
    if include_assets:
        extra_assets = (
            ReportExtraAsset(
                id="compression_clamp_metrics",
                title="Compression Clamp Metrics JSON",
                filename="compression_clamp_metrics.json",
                writer=lambda episode, output: write_compression_clamp_metrics_json(
                    output
                ),
            ),
            ReportExtraAsset(
                id="compression_clamp_stream",
                title="Compression Clamp Stream",
                filename="compression_clamp_stream.png",
                writer=lambda episode, output: save_compression_clamp_stream(output),
            ),
            ReportExtraAsset(
                id="compression_clamp_governance_animation",
                title="Compression Clamp Governance Animation",
                filename="compression_clamp_governance.gif",
                writer=lambda episode, output: save_compression_clamp_governance_animation(
                    output,
                    fps=fps,
                ),
            ),
            ReportExtraAsset(
                id="compression_clamp_timeline",
                title="Compression Clamp Timeline",
                filename="compression_clamp_timeline.png",
                writer=lambda episode, output: save_compression_clamp_timeline(output),
            ),
            ReportExtraAsset(
                id="compression_clamp_selectivity",
                title="Compression Clamp Selectivity",
                filename="compression_clamp_selectivity.png",
                writer=lambda episode, output: save_compression_clamp_selectivity(output),
            ),
            ReportExtraAsset(
                id="compression_clamp_controls",
                title="Compression Clamp Controls",
                filename="compression_clamp_controls.png",
                writer=lambda episode, output: save_compression_clamp_controls(output),
            ),
            ReportExtraAsset(
                id="compression_clamp_overlay",
                title="Compression Clamp Overlay",
                filename="compression_clamp_overlay.png",
                writer=lambda episode, output: save_compression_clamp_overlay(output),
            ),
        )

    return ProducerReportSpec(
        id="compression-clamp",
        title="Compression Clamp",
        episode_factory=lambda: build_compression_clamp_episode("active"),
        input_summary=(
            "same feature stream under an open-to-overload-to-release capacity clamp"
        ),
        description=(
            "A compression-pressure experiment that treats capacity as the "
            "independent variable. The active subject is only interesting if the "
            "clamp changes attention, memory update work, and response quality, "
            "rather than merely increasing compression ratio."
        ),
        views=default_views(),
        extra_assets=extra_assets,
        checks=(lambda episode: check_compression_clamp(),),
        frame_time=6.5,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "compression_clamp",
            "scenario": "compression_clamp",
            "dt": dt,
            "fps": fps,
            "variants": list(CLAMP_VARIANTS),
            "features": list(FEATURE_NAMES),
            "capacity_schedule": list(CAPACITY_SCHEDULE),
        },
        sections=(
            ReportSection(
                title="Question",
                body=(
                    "When the same stream is forced through a tighter bottleneck, "
                    "does the subject merely drop more data, or does it govern "
                    "attention, memory, update work, and action to preserve useful "
                    "control?"
                ),
                asset_ids=(
                    "compression_clamp_metrics",
                    "compression_clamp_stream",
                    "compression_clamp_governance_animation",
                    "compression_clamp_timeline",
                    "compression_clamp_selectivity",
                    "compression_clamp_controls",
                    "compression_clamp_overlay",
                ),
            ),
            ReportSection(
                title="Boundary",
                body=(
                    "This is still a synthetic calibration. The feature stream is "
                    "hand-authored so the report can isolate compression pressure "
                    "from world generation, rendering, and learned policy search."
                ),
            ),
        ),
    )


def compression_clamp_episodes(
    *,
    config: CompressionClampConfig | None = None,
) -> dict[str, Episode]:
    cfg = config or CompressionClampConfig()
    return {
        variant: build_compression_clamp_episode(variant, config=cfg)
        for variant in CLAMP_VARIANTS
    }


def build_compression_clamp_episode(
    variant: ClampVariant,
    *,
    config: CompressionClampConfig | None = None,
) -> Episode:
    cfg = config or CompressionClampConfig()
    memory = np.zeros(len(FEATURE_NAMES), dtype=float)
    previous_mask = np.zeros(len(FEATURE_NAMES), dtype=float)
    previous_loss = 0.0
    inputs: list[EpisodeInput] = []
    observations: list[EpisodeObservation] = []
    rng = np.random.default_rng(41)

    for index, source in enumerate(STREAM):
        capacity = int(CAPACITY_SCHEDULE[index])
        loss_signal = (
            float(SHUFFLED_LOSS_DRIVER[index % SHUFFLED_LOSS_DRIVER.size])
            if variant == "shuffled-loss"
            else previous_loss
        )
        mask = _selection_mask(source, memory, capacity, variant, loss_signal, rng)
        retained = source * mask
        reconstructed = retained.copy()
        expected = (
            retained.copy()
            if variant in {"fixed-container", "oracle-rails"}
            else memory.copy()
        )
        distortion = _distortion(source, reconstructed)
        prediction_gap = _weighted_norm(source - expected, RELEVANCE)
        prediction_loss = prediction_gap + 0.45 * distortion
        action_success = _action_success(source, retained, memory)
        if variant == "no-update":
            next_memory = memory.copy()
            update_work = 0.0
            ownership = "subject"
        elif variant in {"fixed-container", "oracle-rails"}:
            next_memory = retained.copy()
            update_work = float(capacity)
            ownership = "rails"
        else:
            eta = cfg.eta
            if variant in {"active", "shuffled-loss"}:
                eta = min(1.0, cfg.eta * (0.72 + 0.75 * _positive(loss_signal)))
            next_memory = memory * (1.0 - cfg.decay)
            next_memory[mask > 0.5] = (
                memory[mask > 0.5]
                + eta * (retained[mask > 0.5] - memory[mask > 0.5])
            )
            update_work = _energy(next_memory - memory)
            if variant in {"active", "shuffled-loss"}:
                update_work += 0.35 * _positive(loss_signal)
            ownership = "subject"
        source_energy = _energy(source)
        retained_energy = _energy(retained)
        attention_shift = float(np.mean(np.abs(mask - previous_mask)))
        relevant_retention = _retention(source, retained, RELEVANCE)
        irrelevant_retention = _retention(source, retained, IRRELEVANCE)
        selectivity = relevant_retention - irrelevant_retention
        input_id = f"clamp_{index:03d}"
        inputs.append(
            EpisodeInput(
                id=input_id,
                kind="compression_clamp",
                start=float(index),
                end=float(index + 1),
                order_index=index,
                features=source.copy(),
                salience=float(np.max(source)),
                presentation=Presentation(
                    style={
                        "label": f"{PHASES[index]} {capacity}",
                        "color": _phase_color(PHASES[index]),
                        "glyph": "circle",
                    },
                ),
                metadata={
                    "phase": PHASES[index],
                    "capacity": capacity,
                    "source_vector": source.copy(),
                    "variant": variant,
                },
            )
        )
        observations.append(
            EpisodeObservation(
                t=float(index) + 0.5,
                t_normalized=float(index) / max(1.0, float(len(STREAM) - 1)),
                expected=expected.copy(),
                actual=retained.copy(),
                memory_state=next_memory.copy(),
                surprise=prediction_loss,
                learning_rate=(
                    0.0
                    if variant in {"fixed-container", "oracle-rails", "no-update"}
                    else cfg.eta
                ),
                attention=float(capacity / len(FEATURE_NAMES)),
                attention_weights={
                    FEATURE_NAMES[pos]: float(mask[pos])
                    for pos in range(len(FEATURE_NAMES))
                },
                active_inputs=[input_id],
                input_features={input_id: source.copy()},
                metadata={
                    "source_vector": source.copy(),
                    "memory_previous": memory.copy(),
                    "compression": {
                        "source_load": float(len(FEATURE_NAMES)),
                        "admitted_load": float(capacity),
                        "state_capacity": float(capacity),
                        "compression_ratio": float(len(FEATURE_NAMES) / capacity),
                        "retained_energy": retained_energy,
                        "dropped_energy": max(0.0, source_energy - retained_energy),
                        "distortion": distortion,
                        "prediction_loss": prediction_loss,
                        "predictive_info": relevant_retention,
                        "update_work": update_work,
                        "energy_cost": update_work,
                        "ownership": ownership,
                    },
                    "compression_clamp": {
                        "variant": variant,
                        "phase": PHASES[index],
                        "capacity": capacity,
                        "selected_features": [
                            name
                            for name, selected in zip(FEATURE_NAMES, mask, strict=True)
                            if selected > 0.5
                        ],
                        "relevant_retention": relevant_retention,
                        "irrelevant_retention": irrelevant_retention,
                        "selectivity": selectivity,
                        "attention_shift": attention_shift,
                        "action_success": action_success,
                        "loss_signal": loss_signal,
                    },
                },
            )
        )
        previous_mask = mask
        previous_loss = prediction_loss
        memory = next_memory

    return Episode(
        source_name=f"compression-clamp:{variant}",
        vocabulary=list(FEATURE_NAMES),
        inputs=inputs,
        observations=observations,
        duration=float(len(STREAM)),
        metadata={
            "source": "cave.pressure.checks.compression_clamp",
            "adapter": "CompressionClampProducer",
            "variant": variant,
            "config": {
                "eta": cfg.eta,
                "decay": cfg.decay,
                "features": list(FEATURE_NAMES),
                "capacity_schedule": list(CAPACITY_SCHEDULE),
            },
        },
    )


def check_compression_clamp() -> dict[str, object]:
    episodes = compression_clamp_episodes()
    summaries = {
        name: summarize_episode_compression(episode)
        for name, episode in episodes.items()
    }
    metrics = {
        name: _clamp_metrics(episode, summaries[name])
        for name, episode in episodes.items()
    }
    roles = _roles(metrics)
    active = metrics["active"]
    errors = []
    if active["mean_selectivity"] <= metrics["random-compressor"]["mean_selectivity"]:
        errors.append(
            "active subject did not retain relevant structure better than random control"
        )
    if (
        active["lag_loss_to_update_coupling"]
        <= metrics["shuffled-loss"]["lag_loss_to_update_coupling"]
    ):
        errors.append("active subject did not exceed shuffled-loss lagged loss-to-update coupling")
    if (
        active["adaptive_governance_proxy"]
        <= metrics["fixed-container"]["adaptive_governance_proxy"]
    ):
        errors.append("active governance proxy did not exceed fixed-container control")
    if active["adaptive_governance_proxy"] <= metrics["oracle-rails"]["adaptive_governance_proxy"]:
        errors.append("active governance proxy did not exclude rails-supplied oracle state")
    if active["mean_action_success"] <= metrics["no-update"]["mean_action_success"]:
        errors.append("active subject did not improve action success over no-update control")
    if active["recovery_after_release"] <= 0.0:
        errors.append("active subject did not recover after clamp release")
    return {
        "id": "compression_clamp",
        "ok": not errors,
        "errors": errors,
        "metrics": metrics,
        "roles": roles,
    }


def write_compression_clamp_metrics_json(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    result = check_compression_clamp()
    output.write_text(json.dumps(encode_value(result), indent=2) + "\n", encoding="utf-8")


def save_compression_clamp_stream(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    active = build_compression_clamp_episode("active")
    t = np.asarray([obs.t for obs in active.observations], dtype=float)
    selected = np.asarray(
        [
            [
                obs.attention_weights.get(name, 0.0)
                for name in FEATURE_NAMES
            ]
            for obs in active.observations
        ],
        dtype=float,
    )
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 7.0), sharex=True, constrained_layout=True)
    im = axes[0].imshow(STREAM.T, aspect="auto", cmap="magma", interpolation="nearest")
    axes[0].set_yticks(np.arange(len(FEATURE_NAMES)))
    axes[0].set_yticklabels(FEATURE_NAMES)
    axes[0].set_title("Incoming Stream")
    fig.colorbar(im, ax=axes[0], label="source intensity")
    axes[1].step(t, CAPACITY_SCHEDULE, where="mid", color="#E45756", lw=2.4)
    axes[1].set_ylabel("slots")
    axes[1].set_title("Capacity Clamp")
    axes[1].grid(axis="y", alpha=0.24)
    im2 = axes[2].imshow(
        selected.T,
        aspect="auto",
        cmap="Greens",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    axes[2].set_yticks(np.arange(len(FEATURE_NAMES)))
    axes[2].set_yticklabels(FEATURE_NAMES)
    axes[2].set_title("Active Subject Admitted Features")
    axes[2].set_xlabel("timestep")
    fig.colorbar(im2, ax=axes[2], label="admitted")
    fig.suptitle("Compression Clamp Stream", fontsize=14)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_compression_clamp_governance_animation(
    output: Path,
    *,
    fps: int = 4,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    episode = build_compression_clamp_episode("active")
    observations = episode.observations
    rows = [_clamp_row(obs) for obs in observations]
    t = np.asarray([obs.t for obs in observations], dtype=float)
    capacities = np.asarray(CAPACITY_SCHEDULE, dtype=float)
    losses = np.asarray([obs.surprise for obs in observations], dtype=float)
    work = np.asarray([row["update_work"] for row in rows], dtype=float)
    success = np.asarray([row["action_success"] for row in rows], dtype=float)
    selectivity = np.asarray([row["selectivity"] for row in rows], dtype=float)
    memory = np.asarray([obs.memory_state for obs in observations], dtype=float)
    retained = np.asarray([obs.actual for obs in observations], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), constrained_layout=True)
    pressure_ax, bottleneck_ax, state_ax, response_ax = axes.flat

    def draw(frame_index: int) -> None:
        for ax in axes.flat:
            ax.clear()
        _draw_pressure_panel(pressure_ax, t, capacities, frame_index)
        _draw_bottleneck_panel(
            bottleneck_ax,
            STREAM[frame_index],
            retained[frame_index],
        )
        _draw_state_panel(
            state_ax,
            memory[frame_index],
            losses[: frame_index + 1],
            work[: frame_index + 1],
            t[: frame_index + 1],
        )
        _draw_response_panel(
            response_ax,
            t[: frame_index + 1],
            success[: frame_index + 1],
            selectivity[: frame_index + 1],
        )
        phase = observations[frame_index].metadata["compression_clamp"]["phase"]
        fig.suptitle(
            f"Compression Clamp Governance - t={frame_index} / {phase}",
            fontsize=14,
        )

    anim = animation.FuncAnimation(
        fig,
        draw,
        frames=len(observations),
        interval=int(1000 / max(1, fps)),
        repeat=True,
    )
    try:
        anim.save(output, writer=animation.PillowWriter(fps=fps), dpi=120)
    finally:
        plt.close(fig)


def save_compression_clamp_timeline(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    episodes = compression_clamp_episodes()
    fig, axes = plt.subplots(
        len(CLAMP_VARIANTS),
        1,
        figsize=(11.0, 10.0),
        sharex=True,
        constrained_layout=True,
    )
    for ax, name in zip(axes, CLAMP_VARIANTS, strict=True):
        rows = [_clamp_row(obs) for obs in episodes[name].observations]
        t = np.asarray([obs.t for obs in episodes[name].observations], dtype=float)
        loss = np.asarray([obs.surprise for obs in episodes[name].observations], dtype=float)
        work = np.asarray([row["update_work"] for row in rows], dtype=float)
        success = np.asarray([row["action_success"] for row in rows], dtype=float)
        ax.bar(t, work, width=0.42, color="#54A24B", alpha=0.35, label="update work")
        ax.plot(t, loss, color="#4C78A8", marker="o", lw=1.8, label="loss")
        ax.plot(t, success, color="#F58518", marker="s", lw=1.8, label="action success")
        ax.set_ylabel(name.replace("-", "\n"), fontsize=8)
        ax.set_ylim(bottom=0.0)
        ax.grid(axis="y", alpha=0.24)
    axes[0].legend(loc="upper right", ncols=3, fontsize=8)
    axes[-1].set_xlabel("timestep")
    fig.suptitle("Clamp Timeline: Loss, Work, Response", fontsize=14)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_compression_clamp_selectivity(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    episodes = compression_clamp_episodes()
    names = list(CLAMP_VARIANTS)
    x = np.arange(len(names), dtype=float)
    metrics = {name: _clamp_metrics(episodes[name]) for name in names}
    relevant = np.asarray(
        [metrics[name]["mean_relevant_retention"] for name in names],
        dtype=float,
    )
    irrelevant = np.asarray(
        [metrics[name]["mean_irrelevant_retention"] for name in names],
        dtype=float,
    )
    selectivity = relevant - irrelevant
    labels = [name.replace("-", "\n") for name in names]
    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    ax.bar(x - 0.2, relevant, width=0.36, color="#4C78A8", label="relevant retention")
    ax.bar(x + 0.2, irrelevant, width=0.36, color="#F58518", label="irrelevant retention")
    ax.plot(x, selectivity, color="#E45756", marker="o", lw=2.0, label="selectivity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(-0.1, 1.05)
    ax.grid(axis="y", alpha=0.24)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Compression Selectivity Under Clamp")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_compression_clamp_controls(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    result = check_compression_clamp()
    metrics = result["metrics"]  # type: ignore[index]
    names = list(CLAMP_VARIANTS)
    x = np.arange(len(names), dtype=float)
    fields = (
        ("lag_loss_to_update_coupling", "loss -> next update"),
        ("mean_action_success", "action success"),
        ("adaptive_governance_proxy", "governance proxy"),
    )
    labels = [name.replace("-", "\n") for name in names]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.3), constrained_layout=True)
    for ax, (field, title) in zip(axes, fields, strict=True):
        values = [float(metrics[name][field]) for name in names]  # type: ignore[index]
        ax.bar(x, values, color="#4C78A8", width=0.68)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.grid(axis="y", alpha=0.24)
        ax.set_ylim(bottom=min(0.0, min(values) - 0.05), top=max(values) + 0.12)
    fig.suptitle("Matched Clamp Controls", fontsize=14)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_compression_clamp_overlay(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    episode = build_compression_clamp_episode("active")
    rows = [_clamp_row(obs) for obs in episode.observations]
    t = np.asarray([obs.t for obs in episode.observations], dtype=float)
    memory = np.asarray([obs.memory_state for obs in episode.observations], dtype=float)
    source_relevance = STREAM * RELEVANCE
    work = np.asarray([row["update_work"] for row in rows], dtype=float)
    selectivity = np.asarray([row["selectivity"] for row in rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
    axes[0].plot(t, source_relevance[:, 0], color="#E45756", lw=2.0, label="source threat")
    axes[0].plot(t, source_relevance[:, 1], color="#54A24B", lw=2.0, label="source reward")
    axes[0].plot(t, source_relevance[:, 2], color="#4C78A8", lw=2.0, label="source clue")
    axes[0].plot(t, memory[:, 0], color="#E45756", ls="--", lw=1.8, label="memory threat")
    axes[0].plot(t, memory[:, 2], color="#4C78A8", ls="--", lw=1.8, label="memory clue")
    axes[0].set_title("Relevant Source Versus Memory")
    axes[0].set_xlabel("timestep")
    axes[0].grid(axis="y", alpha=0.24)
    axes[0].legend(loc="upper right", fontsize=8, ncols=2)
    axes[1].bar(t, work, width=0.42, color="#54A24B", alpha=0.35, label="update work")
    axes[1].plot(t, selectivity, color="#E45756", marker="o", lw=2.0, label="selectivity")
    axes[1].step(
        t,
        np.asarray(CAPACITY_SCHEDULE, dtype=float) / len(FEATURE_NAMES),
        where="mid",
        color="#212529",
        lw=1.8,
        label="capacity fraction",
    )
    axes[1].set_title("Costs On The Same Episode")
    axes[1].set_xlabel("timestep")
    axes[1].grid(axis="y", alpha=0.24)
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Active Compression Clamp Overlay", fontsize=14)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _draw_pressure_panel(
    ax: plt.Axes,
    t: np.ndarray,
    capacities: np.ndarray,
    frame_index: int,
) -> None:
    ax.step(t, capacities, where="mid", color="#E45756", lw=2.4)
    ax.scatter(t[frame_index], capacities[frame_index], color="#212529", s=54, zorder=3)
    ax.fill_between(
        t,
        0.0,
        capacities,
        step="mid",
        color="#E45756",
        alpha=0.12,
    )
    ax.set_title("1. Pressure Clamp")
    ax.set_ylabel("available slots")
    ax.set_xlim(float(t[0]) - 0.4, float(t[-1]) + 0.4)
    ax.set_ylim(0.0, len(FEATURE_NAMES) + 0.5)
    ax.grid(axis="y", alpha=0.24)


def _draw_bottleneck_panel(
    ax: plt.Axes,
    source: np.ndarray,
    retained: np.ndarray,
) -> None:
    y = np.arange(len(FEATURE_NAMES), dtype=float)
    retained_mask = retained > 0.0
    dropped = np.where(retained_mask, 0.0, source)
    ax.barh(y, source, color="#D7DEE8", height=0.72, label="incoming")
    ax.barh(y, retained, color="#4C78A8", height=0.44, label="retained")
    ax.barh(y, dropped, left=retained, color="#C44E52", height=0.22, label="dropped")
    for index, name in enumerate(FEATURE_NAMES):
        weight = RELEVANCE[index]
        if weight > 0.0:
            ax.text(
                1.03,
                index,
                "relevant",
                va="center",
                fontsize=7,
                color="#2F6F4E",
            )
    ax.set_title("2. Bottleneck Selection")
    ax.set_yticks(y)
    ax.set_yticklabels(FEATURE_NAMES, fontsize=8)
    ax.set_xlim(0.0, 1.25)
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=7)


def _draw_state_panel(
    ax: plt.Axes,
    memory: np.ndarray,
    losses: np.ndarray,
    work: np.ndarray,
    t: np.ndarray,
) -> None:
    x = np.arange(len(FEATURE_NAMES), dtype=float)
    colors = [
        (
            "#E45756"
            if name == "threat"
            else "#54A24B"
            if name == "reward"
            else "#4C78A8"
            if name == "clue"
            else "#8A8F98"
        )
        for name in FEATURE_NAMES
    ]
    ax.bar(x, memory, color=colors, alpha=0.78)
    pulse = float(work[-1]) if work.size else 0.0
    loss = float(losses[-1]) if losses.size else 0.0
    ax.plot(
        np.interp(t, (float(t[0]), float(t[-1])), (0.0, len(FEATURE_NAMES) - 1.0)),
        np.clip(losses, 0.0, 1.0),
        color="#E45756",
        lw=1.8,
        alpha=0.85,
        label="loss trace",
    )
    ax.text(
        0.02,
        0.94,
        f"loss {loss:.2f}\nupdate pulse {pulse:.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#CED4DA", "alpha": 0.88},
    )
    ax.set_title("3. Subject State / Work")
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_NAMES, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.15)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper right", fontsize=7)


def _draw_response_panel(
    ax: plt.Axes,
    t: np.ndarray,
    success: np.ndarray,
    selectivity: np.ndarray,
) -> None:
    ax.plot(t, success, color="#54A24B", marker="o", lw=2.0, label="response quality")
    ax.plot(t, selectivity, color="#4C78A8", marker="s", lw=1.8, label="selectivity")
    ax.axhline(0.0, color="#212529", lw=0.8, alpha=0.35)
    ax.scatter(t[-1], success[-1], color="#212529", s=46, zorder=3)
    ax.set_title("4. Response / Control")
    ax.set_xlabel("timestep")
    ax.set_ylim(-0.2, 1.05)
    ax.set_xlim(float(t[0]) - 0.4, float(STREAM.shape[0]) - 0.1)
    ax.grid(axis="y", alpha=0.24)
    ax.legend(loc="lower right", fontsize=7)


def _selection_mask(
    source: np.ndarray,
    memory: np.ndarray,
    capacity: int,
    variant: ClampVariant,
    loss_signal: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if capacity >= source.size:
        return np.ones(source.size, dtype=float)
    if variant == "fixed-container":
        order = list(FIXED_PRIORITY)
    elif variant == "random-compressor":
        order = list(rng.permutation(source.size))
    elif variant == "oracle-rails":
        scores = source * RELEVANCE
        order = list(np.argsort(-scores))
    else:
        prediction_gap = np.abs(source - memory)
        scores = (
            1.15 * source * RELEVANCE
            + 0.45 * prediction_gap * RELEVANCE
            + 0.20 * loss_signal * RELEVANCE
            + 0.04 * source
        )
        if variant == "shuffled-loss":
            scores = scores + 0.55 * loss_signal * IRRELEVANCE
        order = list(np.argsort(-scores))
    mask = np.zeros(source.size, dtype=float)
    for index in order[:capacity]:
        mask[int(index)] = 1.0
    return mask


def _clamp_metrics(
    episode: Episode,
    summary: dict[str, object] | None = None,
) -> dict[str, float]:
    if summary is None:
        summary = summarize_episode_compression(episode)
    rows = [_clamp_row(obs) for obs in episode.observations]
    losses = np.asarray([obs.surprise for obs in episode.observations], dtype=float)
    works = np.asarray([row["update_work"] for row in rows], dtype=float)
    shifts = np.asarray([row["attention_shift"] for row in rows], dtype=float)
    success = np.asarray([row["action_success"] for row in rows], dtype=float)
    improvements = np.maximum(0.0, losses[:-1] - losses[1:])
    relevant = np.asarray([row["relevant_retention"] for row in rows], dtype=float)
    irrelevant = np.asarray([row["irrelevant_retention"] for row in rows], dtype=float)
    subject_fraction = float(summary["work"]["ownership_subject_fraction"])  # type: ignore[index]
    lag_loss_to_update = _correlation(losses[:-1], works[1:])
    lag_loss_to_shift = _correlation(losses[:-1], shifts[1:])
    work_to_future_improvement = _correlation(works[:-1], improvements)
    recovery_after_release = float(np.mean(success[-3:]) - np.min(success[5:8]))
    mean_selectivity = float(np.mean(relevant - irrelevant))
    future_loss_improvement = float(np.mean(improvements))
    return {
        "mean_pressure": float(
            summary["pressure"]["mean_compression_ratio"]  # type: ignore[index]
        ),
        "mean_distortion": float(summary["distortion"]["mean_distortion"]),  # type: ignore[index]
        "update_work": float(summary["work"]["update_work"]),  # type: ignore[index]
        "ownership_subject_fraction": subject_fraction,
        "mean_relevant_retention": float(np.mean(relevant)),
        "mean_irrelevant_retention": float(np.mean(irrelevant)),
        "mean_selectivity": mean_selectivity,
        "mean_action_success": float(np.mean(success)),
        "lag_loss_to_update_coupling": lag_loss_to_update,
        "lag_loss_to_attention_shift": lag_loss_to_shift,
        "work_to_future_loss_improvement": work_to_future_improvement,
        "future_loss_improvement": future_loss_improvement,
        "recovery_after_release": recovery_after_release,
        "adaptive_governance_proxy": (
            _positive(mean_selectivity)
            * _positive(lag_loss_to_update)
            * _positive(lag_loss_to_shift)
            * _positive(float(np.mean(success)))
            * _positive(subject_fraction)
        ),
    }


def _clamp_row(observation: EpisodeObservation) -> dict[str, float]:
    clamp = observation.metadata["compression_clamp"]
    compression = observation.metadata["compression"]
    return {
        "capacity": float(clamp["capacity"]),
        "relevant_retention": float(clamp["relevant_retention"]),
        "irrelevant_retention": float(clamp["irrelevant_retention"]),
        "selectivity": float(clamp["selectivity"]),
        "attention_shift": float(clamp["attention_shift"]),
        "action_success": float(clamp["action_success"]),
        "update_work": float(compression["update_work"]),
    }


def _roles(metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        "selectivity": {
            name: values["mean_selectivity"] for name, values in metrics.items()
        },
        "lag_loss_to_update_coupling": {
            name: values["lag_loss_to_update_coupling"]
            for name, values in metrics.items()
        },
        "action_success": {
            name: values["mean_action_success"] for name, values in metrics.items()
        },
        "adaptive_governance_proxy": {
            name: values["adaptive_governance_proxy"]
            for name, values in metrics.items()
        },
        "active_minus_controls": {
            "selectivity_vs_random": (
                metrics["active"]["mean_selectivity"]
                - metrics["random-compressor"]["mean_selectivity"]
            ),
            "coupling_vs_shuffled": (
                metrics["active"]["lag_loss_to_update_coupling"]
                - metrics["shuffled-loss"]["lag_loss_to_update_coupling"]
            ),
            "proxy_vs_oracle_rails": (
                metrics["active"]["adaptive_governance_proxy"]
                - metrics["oracle-rails"]["adaptive_governance_proxy"]
            ),
            "success_vs_no_update": (
                metrics["active"]["mean_action_success"]
                - metrics["no-update"]["mean_action_success"]
            ),
        },
    }


def _action_success(source: np.ndarray, retained: np.ndarray, memory: np.ndarray) -> float:
    target = source * RELEVANCE
    if _energy(target) <= 1e-12:
        return max(0.0, 1.0 - _retention(source, retained, IRRELEVANCE))
    available = (0.65 * retained + 0.35 * memory) * RELEVANCE
    error = _weighted_norm(target - available, RELEVANCE)
    scale = _weighted_norm(target, RELEVANCE)
    return float(np.clip(1.0 - _safe_ratio(error, scale), 0.0, 1.0))


def _retention(source: np.ndarray, retained: np.ndarray, weights: np.ndarray) -> float:
    source_energy = _energy(source * weights)
    if source_energy <= 1e-12:
        return 0.0
    return float(np.clip(_energy(retained * weights) / source_energy, 0.0, 1.0))


def _distortion(source: np.ndarray, reconstructed: np.ndarray) -> float:
    source_energy = _energy(source)
    if source_energy <= 1e-12:
        return 0.0
    return float(_energy(source - reconstructed) / source_energy)


def _weighted_norm(value: np.ndarray, weights: np.ndarray) -> float:
    weighted = np.asarray(value, dtype=float) * weights
    if weighted.size == 0:
        return 0.0
    return float(np.linalg.norm(weighted) / np.sqrt(weighted.size))


def _energy(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=float)
    return float(np.sum(array * array))


def _correlation(source: np.ndarray, target: np.ndarray) -> float:
    if source.size < 2 or target.size < 2:
        return 0.0
    if float(np.std(source)) <= 1e-12 or float(np.std(target)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(source, target)[0, 1])


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 1e-12:
        return 0.0
    return float(numerator / denominator)


def _positive(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _phase_color(phase: str) -> str:
    return {
        "open": "#4C78A8",
        "mild": "#72B7B2",
        "clamp": "#F58518",
        "hard": "#E45756",
        "overload": "#B279A2",
        "release": "#54A24B",
        "recovery": "#59A14F",
    }.get(phase, "#4C78A8")
