from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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


CompressionVariant = Literal[
    "ratio-1-active",
    "ratio-5-active",
    "ratio-5-container",
    "ratio-5-random",
]

COMPRESSION_VARIANTS: tuple[CompressionVariant, ...] = (
    "ratio-1-active",
    "ratio-5-active",
    "ratio-5-container",
    "ratio-5-random",
)

LATENT_SEQUENCE = np.asarray(
    [
        [0.20, 0.80],
        [0.22, 0.78],
        [0.18, 0.82],
        [0.86, 0.24],
        [0.82, 0.28],
        [0.24, 0.76],
        [0.20, 0.80],
        [0.18, 0.82],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class PrimitiveCompressionConfig:
    state_dim: int = 2
    ratio: int = 5
    eta: float = 0.45

    def __post_init__(self) -> None:
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if self.ratio <= 0:
            raise ValueError("ratio must be positive")
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError("eta must be in [0, 1]")


def primitive_compression_report_spec(
    *,
    dt: float = 1.0,
    fps: int = 4,
    include_assets: bool = True,
) -> ProducerReportSpec:
    extra_assets = ()
    if include_assets:
        extra_assets = (
            ReportExtraAsset(
                id="primitive_compression_metrics",
                title="Primitive Compression Metrics JSON",
                filename="primitive_compression_metrics.json",
                writer=lambda episode, output: write_primitive_compression_metrics_json(output),
            ),
            ReportExtraAsset(
                id="primitive_compression_scorecard",
                title="Compression Ownership Scorecard",
                filename="primitive_compression_scorecard.png",
                writer=lambda episode, output: save_primitive_compression_scorecard(output),
            ),
            ReportExtraAsset(
                id="primitive_compression_timeline",
                title="Primitive Compression Timeline",
                filename="primitive_compression_timeline.png",
                writer=lambda episode, output: save_primitive_compression_timeline(output),
            ),
            ReportExtraAsset(
                id="primitive_compression_coupling",
                title="Primitive Compression Coupling",
                filename="primitive_compression_coupling.png",
                writer=lambda episode, output: save_primitive_compression_coupling(output),
            ),
            ReportExtraAsset(
                id="primitive_compression_overlay",
                title="Primitive Compression Episode Overlay",
                filename="primitive_compression_overlay.png",
                writer=lambda episode, output: save_primitive_compression_overlay(output),
            ),
            ReportExtraAsset(
                id="primitive_compression_pressure_sweep",
                title="Primitive Compression Pressure Sweep",
                filename="primitive_compression_pressure_sweep.png",
                writer=lambda episode, output: save_primitive_compression_pressure_sweep(output),
            ),
        )

    return ProducerReportSpec(
        id="primitive-compression",
        title="Primitive Compression Pressure",
        episode_factory=lambda: build_primitive_compression_episode("ratio-5-active"),
        input_summary=(
            "primitive recurrence under 1:1 and 5:1 source/state pressure "
            "with active, rails-container, and random-projection controls"
        ),
        description=(
            "A calibration report for cost accounting. It forces primitive "
            "subjects through different source/state ratios and asks whether "
            "distortion, subject-paid update work, and future loss improvement "
            "move in the expected direction."
        ),
        views=default_views(),
        extra_assets=extra_assets,
        checks=(lambda episode: check_primitive_compression(),),
        frame_time=3.5,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "primitive_compression",
            "scenario": "primitive_compression",
            "dt": dt,
            "fps": fps,
            "variants": list(COMPRESSION_VARIANTS),
        },
        sections=(
            ReportSection(
                title="Question",
                body=(
                    "When source load exceeds subject state capacity, does the "
                    "system merely hold a compact state, or does it pay update "
                    "work that reduces future loss?"
                ),
                asset_ids=(
                    "primitive_compression_metrics",
                    "primitive_compression_scorecard",
                    "primitive_compression_timeline",
                    "primitive_compression_coupling",
                    "primitive_compression_overlay",
                    "primitive_compression_pressure_sweep",
                ),
            ),
            ReportSection(
                title="Boundary",
                body=(
                    "This is a primitive calibration probe. The high-dimensional "
                    "source vectors are synthetic expansions of a two-dimensional "
                    "latent sequence, so the report tests cost accounting rather "
                    "than a rich sensory encoder."
                ),
            ),
        ),
    )


def primitive_compression_episodes(
    *,
    config: PrimitiveCompressionConfig | None = None,
) -> dict[str, Episode]:
    cfg = config or PrimitiveCompressionConfig()
    return {
        variant: build_primitive_compression_episode(variant, config=cfg)
        for variant in COMPRESSION_VARIANTS
    }


def build_primitive_compression_episode(
    variant: CompressionVariant,
    *,
    config: PrimitiveCompressionConfig | None = None,
) -> Episode:
    cfg = config or PrimitiveCompressionConfig()
    source_dim = cfg.state_dim if variant == "ratio-1-active" else cfg.state_dim * cfg.ratio
    vocabulary = [f"state_{index}" for index in range(cfg.state_dim)]
    memory = np.zeros(cfg.state_dim, dtype=float)
    inputs: list[EpisodeInput] = []
    observations: list[EpisodeObservation] = []
    projection = _random_projection(source_dim, cfg.state_dim)
    ownership = "rails" if variant == "ratio-5-container" else "subject"

    for index, latent in enumerate(LATENT_SEQUENCE):
        source = _source_vector(latent, source_dim)
        compressed = _compress_source(source, cfg.state_dim, variant, projection)
        reconstructed = _reconstruct_source(compressed, source_dim)
        expected = memory.copy() if variant != "ratio-5-container" else compressed.copy()
        error = compressed - expected
        surprise = _normalized_norm(error) + _distortion(source, reconstructed)
        if variant == "ratio-5-container":
            next_memory = compressed.copy()
            update_work = float(source_dim)
        else:
            next_memory = memory + cfg.eta * error
            update_work = _energy(next_memory - memory)
        retained_energy = _energy(reconstructed)
        source_energy = _energy(source)
        dropped_energy = max(0.0, source_energy - retained_energy)
        distortion = _distortion(source, reconstructed)
        predictive_info = _predictive_info(index, compressed)
        input_id = f"compression_{index:03d}"
        inputs.append(
            EpisodeInput(
                id=input_id,
                kind="compression_probe",
                start=float(index),
                end=float(index + 1),
                order_index=index,
                features=compressed.copy(),
                salience=1.0,
                presentation=Presentation(
                    style={
                        "label": f"step {index}",
                        "color": "#4C78A8",
                        "glyph": "circle",
                    },
                ),
                metadata={
                    "source_vector": source.copy(),
                    "latent": latent.copy(),
                    "variant": variant,
                },
            )
        )
        observations.append(
            EpisodeObservation(
                t=float(index) + 0.5,
                t_normalized=float(index) / max(1.0, float(len(LATENT_SEQUENCE) - 1)),
                expected=expected.copy(),
                actual=compressed.copy(),
                memory_state=next_memory.copy(),
                surprise=surprise,
                learning_rate=0.0 if variant == "ratio-5-container" else cfg.eta,
                attention=1.0,
                attention_weights={input_id: 1.0},
                active_inputs=[input_id],
                input_features={input_id: source.copy()},
                metadata={
                    "source_vector": source.copy(),
                    "memory_previous": memory.copy(),
                    "compression": {
                        "source_load": float(source_dim),
                        "admitted_load": float(cfg.state_dim),
                        "state_capacity": float(cfg.state_dim),
                        "compression_ratio": float(source_dim / cfg.state_dim),
                        "retained_energy": retained_energy,
                        "dropped_energy": dropped_energy,
                        "distortion": distortion,
                        "prediction_loss": surprise,
                        "predictive_info": predictive_info,
                        "update_work": update_work,
                        "energy_cost": update_work,
                        "ownership": ownership,
                    },
                    "primitive_compression": {
                        "variant": variant,
                        "latent": latent.copy(),
                        "source_dim": source_dim,
                        "state_dim": cfg.state_dim,
                        "reconstructed_source": reconstructed.copy(),
                    },
                },
            )
        )
        memory = next_memory

    return Episode(
        source_name=f"primitive-compression:{variant}",
        vocabulary=vocabulary,
        inputs=inputs,
        observations=observations,
        duration=float(len(LATENT_SEQUENCE)),
        metadata={
            "source": "cave.pressure.checks.primitive_compression",
            "adapter": "PrimitiveCompressionProducer",
            "variant": variant,
            "config": {
                "state_dim": cfg.state_dim,
                "ratio": cfg.ratio,
                "eta": cfg.eta,
            },
        },
    )


def check_primitive_compression() -> dict[str, object]:
    episodes = primitive_compression_episodes()
    metrics = {
        name: summarize_episode_compression(episode)
        for name, episode in episodes.items()
    }
    compact = _compact_metrics(metrics)
    roles = _roles(compact)
    errors = []
    if compact["ratio-5-active"]["compression_ratio"] <= compact["ratio-1-active"]["compression_ratio"]:
        errors.append("5:1 active run did not increase compression pressure")
    if compact["ratio-5-active"]["update_work"] <= 0.0:
        errors.append("5:1 active run did not pay subject update work")
    if compact["ratio-5-container"]["ownership_subject_fraction"] >= 0.05:
        errors.append("container control was counted as subject-paid work")
    if compact["ratio-5-active"]["paid_compression_proxy"] <= compact["ratio-5-container"]["paid_compression_proxy"]:
        errors.append("active compressor did not exceed rails container proxy")
    if compact["ratio-5-active"]["future_loss_improvement"] <= 0.0:
        errors.append("active compressor did not improve future loss")
    if compact["ratio-5-random"]["mean_distortion"] <= compact["ratio-5-active"]["mean_distortion"]:
        errors.append("random projection was not more distorted than active compression")
    return {
        "id": "primitive_compression",
        "ok": not errors,
        "errors": errors,
        "metrics": compact,
        "roles": roles,
    }


def write_primitive_compression_metrics_json(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    result = check_primitive_compression()
    output.write_text(json.dumps(encode_value(result), indent=2) + "\n", encoding="utf-8")


# Scorecard rendering palette.
_SCORE_PASS = "#5BA85A"
_SCORE_PASS_FILL = "#DCEEDB"
_SCORE_FAIL = "#D6504F"
_SCORE_FAIL_FILL = "#F6DCDB"
_SCORE_NEUTRAL_FILL = "#E7ECF1"
_SCORE_NEUTRAL_TEXT = "#48586A"
_SCORE_HERO = "#C8A23B"
_SCORE_INK = "#2A2F36"
_SCORE_ZERO_FILL = "#F2F4F6"
_SCORE_ZERO_TEXT = "#9AA6B2"

# Per-variant row labels: (ratio, who, tag).
_SCORE_ROWS: dict[str, tuple[str, str, str]] = {
    "ratio-1-active": ("1:1", "active", "baseline"),
    "ratio-5-active": ("5:1", "active", "earns it"),
    "ratio-5-container": ("5:1", "rails container", ""),
    "ratio-5-random": ("5:1", "random proj.", ""),
}
_SCORE_HERO_ROW = "ratio-5-active"

# Pass/fail thresholds for the scorecard cells.
_SCORE_DISTORTION_MAX = 0.05
_SCORE_OWNERSHIP_MIN = 0.5
_SCORE_FUTURE_MIN = 0.01
_SCORE_PROXY_EMPTY = 0.05


def _scorecard_rows(metrics: dict[str, dict[str, float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in COMPRESSION_VARIANTS:
        m = metrics[key]
        ratio = float(m["compression_ratio"])
        distortion = float(m["mean_distortion"])
        owned = float(m["ownership_subject_fraction"])
        future = float(m["future_loss_improvement"])
        proxy = float(m["paid_compression_proxy"])
        is_baseline = key == "ratio-1-active"
        rows.append(
            {
                "key": key,
                "compact": {
                    "top": f"{ratio:.0f}:1",
                    "bot": "calibration" if is_baseline else "compact",
                    "state": "neutral",
                },
                "distortion": {
                    "top": f"{distortion:.3f}",
                    "bot": "low" if distortion < _SCORE_DISTORTION_MAX else "high",
                    "state": "pass" if distortion < _SCORE_DISTORTION_MAX else "fail",
                },
                "owned": {
                    "top": f"{owned:.2f}",
                    "bot": "subject" if owned >= _SCORE_OWNERSHIP_MIN else "rails",
                    "state": "pass" if owned >= _SCORE_OWNERSHIP_MIN else "fail",
                },
                "future": {
                    "top": f"+{future:.3f}",
                    "bot": "improves" if future > _SCORE_FUTURE_MIN else "flat",
                    "state": "pass" if future > _SCORE_FUTURE_MIN else "fail",
                },
                "proxy": {"top": f"{proxy:.3f}", "state": "proxy", "value": proxy},
            }
        )
    return rows


def save_primitive_compression_scorecard(output: Path) -> None:
    """Render the compression ownership scorecard.

    Replaces the old four-panel pressure board and the dual-axis ownership chart
    with a single variant x criteria matrix. Only the 5:1 active subject passes
    every criterion, so only it earns a high paid-compression proxy; each control
    fails a different test (the rails container is not subject-owned, the random
    projection wrecks distortion). The raw container ``update_work`` penalty is
    never plotted on a value axis -- ownership is shown as a fraction instead.
    """
    from matplotlib.patches import FancyBboxPatch

    output.parent.mkdir(parents=True, exist_ok=True)
    result = check_primitive_compression()
    metrics = result["metrics"]  # type: ignore[index]
    rows = _scorecard_rows(metrics)

    col_keys = ["compact", "distortion", "owned", "future", "proxy"]
    col_titles = [
        "compact?\n(not enough)",
        "low\ndistortion",
        "subject-\nowned",
        "future\ngain",
        "PAID-COMPRESSION\nPROXY",
    ]
    label_w, col_w, proxy_w, row_h, header_h = 2.4, 2.0, 2.6, 1.0, 1.25
    widths = [col_w, col_w, col_w, col_w, proxy_w]
    n_rows = len(rows)
    total_w = label_w + sum(widths)
    total_h = header_h + n_rows * row_h

    fig, ax = plt.subplots(figsize=(total_w * 1.18, (total_h + 0.9) * 0.86))
    ax.set_xlim(0, total_w)
    ax.set_ylim(-0.9, total_h)
    ax.axis("off")
    ax.invert_yaxis()

    def col_x(i: int) -> float:
        return label_w + sum(widths[:i])

    for i, title in enumerate(col_titles):
        cx = col_x(i) + widths[i] / 2
        is_proxy = col_keys[i] == "proxy"
        ax.text(
            cx,
            header_h / 2,
            title,
            ha="center",
            va="center",
            fontsize=11 if is_proxy else 10,
            color=_SCORE_INK,
            fontweight="bold" if is_proxy else "semibold",
            linespacing=1.15,
        )

    fills = {
        "pass": (_SCORE_PASS_FILL, _SCORE_PASS, _SCORE_INK),
        "fail": (_SCORE_FAIL_FILL, _SCORE_FAIL, _SCORE_INK),
        "neutral": (_SCORE_NEUTRAL_FILL, "#C2CCD6", _SCORE_NEUTRAL_TEXT),
    }
    glyph = {"pass": "✓", "fail": "✗"}
    proxy_max = max(float(r["proxy"]["value"]) for r in rows) or 1.0

    for ri, row in enumerate(rows):
        y0 = header_h + ri * row_h
        yc = y0 + row_h / 2
        is_hero = row["key"] == _SCORE_HERO_ROW
        ratio, who, tag = _SCORE_ROWS[row["key"]]

        ax.text(
            label_w - 0.18,
            yc - 0.14,
            f"{ratio} {who}",
            ha="right",
            va="center",
            fontsize=11,
            color=_SCORE_INK,
            fontweight="bold" if is_hero else "normal",
        )
        if tag:
            ax.text(
                label_w - 0.18,
                yc + 0.26,
                tag,
                ha="right",
                va="center",
                fontsize=8.5,
                color=_SCORE_HERO if is_hero else _SCORE_NEUTRAL_TEXT,
                fontstyle="italic",
            )

        for i, ckey in enumerate(col_keys):
            cell = row[ckey]
            cx0, cw, pad = col_x(i), widths[i], 0.10
            if ckey == "proxy":
                frac = float(cell["value"]) / proxy_max
                if frac < _SCORE_PROXY_EMPTY:
                    fill, txt_color = _SCORE_ZERO_FILL, _SCORE_ZERO_TEXT
                else:
                    fill, txt_color = plt.cm.Greens(0.20 + 0.60 * frac), _SCORE_INK
                edge = _SCORE_HERO if is_hero else "#C2CCD6"
            else:
                fill, edge, txt_color = fills[cell["state"]]
            ax.add_patch(
                FancyBboxPatch(
                    (cx0 + pad, y0 + pad),
                    cw - 2 * pad,
                    row_h - 2 * pad,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    linewidth=2.2 if is_hero else 1.0,
                    edgecolor=edge,
                    facecolor=fill,
                )
            )

            cxc = cx0 + cw / 2
            if ckey == "proxy":
                ax.text(cxc, yc, cell["top"], ha="center", va="center",
                        fontsize=15 if is_hero else 12, color=txt_color,
                        fontweight="bold" if is_hero else "normal")
            elif ckey == "compact":
                ax.text(cxc, yc - 0.13, cell["top"], ha="center", va="center",
                        fontsize=12, color=txt_color, fontweight="bold")
                ax.text(cxc, yc + 0.26, cell["bot"], ha="center", va="center",
                        fontsize=8, color=txt_color)
            else:
                gc = _SCORE_PASS if cell["state"] == "pass" else _SCORE_FAIL
                ax.text(cxc - 0.42, yc - 0.12, glyph[cell["state"]], ha="center",
                        va="center", fontsize=13, color=gc, fontweight="bold")
                ax.text(cxc + 0.18, yc - 0.12, cell["top"], ha="center",
                        va="center", fontsize=11, color=txt_color)
                ax.text(cxc, yc + 0.27, cell["bot"], ha="center", va="center",
                        fontsize=8, color=txt_color)

    ax.text(0.0, -0.62,
            "Who paid for the compact state?  —  compression ownership scorecard",
            ha="left", va="center", fontsize=13.5, color=_SCORE_INK, fontweight="bold")
    ax.text(
        0.0, total_h + 0.42,
        "All four states are equally compact. Only the 5:1 active subject passes "
        "every criterion — so only it earns a high paid-compression proxy.\n"
        "Each control fails a different test: the rails container is not "
        "subject-owned; the random projection wrecks distortion.",
        ha="left", va="center", fontsize=8.8, color=_SCORE_NEUTRAL_TEXT, linespacing=1.3,
    )

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_primitive_compression_timeline(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    summaries = _compression_summaries()
    fig, axes = plt.subplots(
        len(COMPRESSION_VARIANTS),
        1,
        figsize=(10.5, 8.5),
        sharex=True,
        constrained_layout=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    for ax, name in zip(axes, COMPRESSION_VARIANTS, strict=True):
        trace = summaries[name]["trace"]
        t = np.asarray([row["t"] for row in trace], dtype=float)
        loss = np.asarray([row["prediction_loss"] for row in trace], dtype=float)
        distortion = np.asarray([row["distortion"] for row in trace], dtype=float)
        work = np.asarray([row["update_work"] for row in trace], dtype=float)
        ax.bar(t, work, width=0.42, color="#54A24B", alpha=0.32, label="update work")
        ax.plot(t, loss, color="#4C78A8", lw=2.0, marker="o", label="loss")
        ax.plot(t, distortion, color="#F58518", lw=2.0, marker="s", label="distortion")
        ax.set_ylabel(name.replace("ratio-", "").replace("-", "\n"), fontsize=8)
        ax.grid(axis="y", alpha=0.24)
        ax.set_ylim(bottom=0.0)
    axes[0].legend(loc="upper right", ncols=3, fontsize=8)
    axes[-1].set_xlabel("timestep")
    fig.suptitle("Primitive Compression Timeline", fontsize=14)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_primitive_compression_coupling(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    summaries = _compression_summaries()
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.0), constrained_layout=True)
    for ax, name in zip(axes.flat, COMPRESSION_VARIANTS, strict=True):
        trace = summaries[name]["trace"]
        loss = np.asarray([row["prediction_loss"] for row in trace], dtype=float)
        work = np.asarray([row["update_work"] for row in trace], dtype=float)
        coupling = float(summaries[name]["effect"]["loss_to_update_coupling"])
        ax.scatter(loss, work, color="#4C78A8", s=42, alpha=0.85)
        if loss.size > 1 and float(np.var(loss)) > 1e-12:
            slope, intercept = np.polyfit(loss, work, 1)
            xs = np.linspace(float(loss.min()), float(loss.max()), 64)
            ax.plot(xs, slope * xs + intercept, color="#E45756", lw=1.8)
        ax.set_title(f"{name}\nr={coupling:.2f}", fontsize=10)
        ax.set_xlabel("prediction loss")
        ax.set_ylabel("update work")
        ax.grid(alpha=0.24)
        ax.set_ylim(bottom=0.0)
    fig.suptitle("Loss-To-Work Coupling", fontsize=14)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_primitive_compression_overlay(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    episode = build_primitive_compression_episode("ratio-5-active")
    summary = summarize_episode_compression(episode)
    trace = summary["trace"]
    actual = np.asarray([obs.actual for obs in episode.observations], dtype=float)
    memory = np.asarray([obs.memory_state for obs in episode.observations], dtype=float)
    expected = np.asarray([obs.expected for obs in episode.observations], dtype=float)
    work = np.asarray([row["update_work"] for row in trace], dtype=float)
    distortion = np.asarray([row["distortion"] for row in trace], dtype=float)
    t = np.asarray([obs.t for obs in episode.observations], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)

    scatter = axes[0].scatter(
        memory[:, 0],
        memory[:, 1],
        c=work,
        s=80.0 + 800.0 * distortion,
        cmap="viridis",
        edgecolor="#212529",
        linewidth=0.7,
        zorder=3,
    )
    axes[0].plot(memory[:, 0], memory[:, 1], color="#129B63", lw=1.7, alpha=0.75, label="memory")
    axes[0].scatter(actual[:, 0], actual[:, 1], color="#4C78A8", marker="x", label="actual")
    axes[0].scatter(expected[:, 0], expected[:, 1], color="#F58518", marker="+", label="expected")
    axes[0].set_title("Trajectory Colored By Update Work")
    axes[0].set_xlabel("state_0")
    axes[0].set_ylabel("state_1")
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(alpha=0.22)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.colorbar(scatter, ax=axes[0], label="update work")

    axes[1].plot(t, [row["prediction_loss"] for row in trace], color="#4C78A8", lw=2.0, label="loss")
    axes[1].bar(t, work, width=0.42, color="#54A24B", alpha=0.35, label="update work")
    axes[1].scatter(t, distortion, s=60.0 + 800.0 * distortion, color="#F58518", alpha=0.75, label="distortion")
    axes[1].set_title("Costs On The Episode Timeline")
    axes[1].set_xlabel("timestep")
    axes[1].grid(axis="y", alpha=0.24)
    axes[1].legend(loc="upper right", fontsize=8)

    fig.suptitle("Primitive Compression Episode Overlay", fontsize=14)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_primitive_compression_pressure_sweep(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    ratios = (1, 2, 5, 10)
    active: list[dict[str, float]] = []
    random: list[dict[str, float]] = []
    for ratio in ratios:
        cfg = PrimitiveCompressionConfig(ratio=ratio)
        active_episode = build_primitive_compression_episode(
            "ratio-1-active" if ratio == 1 else "ratio-5-active",
            config=cfg,
        )
        random_episode = build_primitive_compression_episode(
            "ratio-1-active" if ratio == 1 else "ratio-5-random",
            config=cfg,
        )
        active.append(_sweep_point(summarize_episode_compression(active_episode)))
        random.append(_sweep_point(summarize_episode_compression(random_episode)))

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), constrained_layout=True)
    fields = (
        ("mean_distortion", "distortion"),
        ("update_work", "update work"),
        ("future_loss_improvement", "future loss improvement"),
    )
    for ax, (field, title) in zip(axes, fields, strict=True):
        ax.plot(ratios, [point[field] for point in active], marker="o", lw=2.0, label="active")
        ax.plot(ratios, [point[field] for point in random], marker="s", lw=2.0, label="random")
        ax.set_title(title)
        ax.set_xlabel("source/state ratio")
        ax.grid(alpha=0.24)
    axes[0].set_ylabel("metric value")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Primitive Compression Pressure Sweep", fontsize=14)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _compact_metrics(metrics: dict[str, dict[str, object]]) -> dict[str, dict[str, float]]:
    compact: dict[str, dict[str, float]] = {}
    for name, summary in metrics.items():
        compact[name] = {
            "compression_ratio": float(summary["pressure"]["compression_ratio"]),  # type: ignore[index]
            "mean_compression_ratio": float(summary["pressure"]["mean_compression_ratio"]),  # type: ignore[index]
            "mean_distortion": float(summary["distortion"]["mean_distortion"]),  # type: ignore[index]
            "mean_prediction_loss": float(summary["distortion"]["mean_prediction_loss"]),  # type: ignore[index]
            "update_work": float(summary["work"]["update_work"]),  # type: ignore[index]
            "energy_cost": float(summary["work"]["energy_cost"]),  # type: ignore[index]
            "subject_work": float(summary["work"]["subject_work"]),  # type: ignore[index]
            "rails_work": float(summary["work"]["rails_work"]),  # type: ignore[index]
            "amortized_training_work": float(summary["work"]["amortized_training_work"]),  # type: ignore[index]
            "ownership_subject_fraction": float(summary["work"]["ownership_subject_fraction"]),  # type: ignore[index]
            "future_loss_improvement": float(summary["effect"]["future_loss_improvement"]),  # type: ignore[index]
            "loss_to_update_coupling": float(summary["effect"]["loss_to_update_coupling"]),  # type: ignore[index]
            "retained_predictive_info": float(summary["effect"]["retained_predictive_info"]),  # type: ignore[index]
            "paid_compression_proxy": float(summary["summary"]["paid_compression_proxy"]),  # type: ignore[index]
        }
    return compact


def _compression_summaries() -> dict[str, dict[str, object]]:
    return {
        name: summarize_episode_compression(episode)
        for name, episode in primitive_compression_episodes().items()
    }


def _sweep_point(summary: dict[str, object]) -> dict[str, float]:
    return {
        "mean_distortion": float(summary["distortion"]["mean_distortion"]),  # type: ignore[index]
        "update_work": float(summary["work"]["update_work"]),  # type: ignore[index]
        "future_loss_improvement": float(summary["effect"]["future_loss_improvement"]),  # type: ignore[index]
    }


def _roles(metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        "compression_pressure": {
            name: values["compression_ratio"] for name, values in metrics.items()
        },
        "distortion": {
            name: values["mean_distortion"] for name, values in metrics.items()
        },
        "work": {
            name: values["update_work"] for name, values in metrics.items()
        },
        "ownership_subject_fraction": {
            name: values["ownership_subject_fraction"] for name, values in metrics.items()
        },
        "future_loss_improvement": {
            name: values["future_loss_improvement"] for name, values in metrics.items()
        },
        "paid_compression_proxy": {
            name: values["paid_compression_proxy"] for name, values in metrics.items()
        },
        "active_minus_container": {
            "paid_compression_proxy": (
                metrics["ratio-5-active"]["paid_compression_proxy"]
                - metrics["ratio-5-container"]["paid_compression_proxy"]
            ),
            "subject_work": (
                metrics["ratio-5-active"]["subject_work"]
                - metrics["ratio-5-container"]["subject_work"]
            ),
            "rails_work": (
                metrics["ratio-5-active"]["rails_work"]
                - metrics["ratio-5-container"]["rails_work"]
            ),
        },
    }


def _source_vector(latent: np.ndarray, source_dim: int) -> np.ndarray:
    if source_dim == latent.size:
        return latent.copy()
    repeats = source_dim // latent.size
    values: list[float] = []
    offsets = np.linspace(-0.08, 0.08, repeats)
    for value in latent:
        values.extend(float(np.clip(value + offset, 0.0, 1.0)) for offset in offsets)
    return np.asarray(values, dtype=float)


def _compress_source(
    source: np.ndarray,
    state_dim: int,
    variant: CompressionVariant,
    projection: np.ndarray,
) -> np.ndarray:
    if source.size == state_dim:
        return source.copy()
    if variant == "ratio-5-random":
        projected = projection @ source
        return np.clip(projected, 0.0, 1.0)
    return np.asarray(
        [float(np.mean(chunk)) for chunk in np.array_split(source, state_dim)],
        dtype=float,
    )


def _reconstruct_source(compressed: np.ndarray, source_dim: int) -> np.ndarray:
    if compressed.size == source_dim:
        return compressed.copy()
    repeats = source_dim // compressed.size
    return np.repeat(compressed, repeats)


def _random_projection(source_dim: int, state_dim: int) -> np.ndarray:
    rng = np.random.default_rng(17)
    matrix = rng.normal(0.0, 1.0, size=(state_dim, source_dim))
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(row_norms, 1e-12)


def _predictive_info(index: int, compressed: np.ndarray) -> float:
    if index + 1 >= len(LATENT_SEQUENCE):
        return 0.0
    future = LATENT_SEQUENCE[index + 1]
    return max(0.0, 1.0 - _normalized_norm(future - compressed))


def _distortion(source: np.ndarray, reconstructed: np.ndarray) -> float:
    source_energy = _energy(source)
    if source_energy <= 1e-12:
        return 0.0
    return _energy(source - reconstructed) / source_energy


def _energy(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=float)
    return float(np.sum(array * array))


def _normalized_norm(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.linalg.norm(array) / np.sqrt(array.size))
