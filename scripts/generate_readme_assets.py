from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cave import (
    AttentionProfile,
    CaveProducer,
    CorrectionView,
    ExpectationActualView,
    MemoryLookbackView,
    PresentationView,
    SubjectiveTopologyView,
    TimelineView,
    demo_model,
)
from cave.commitments.attention import AttentionChannelCurve, INTERNAL_EXPECTATION_CHANNEL
from cave.presentation.renderers import LayoutSpec, MatplotlibRenderer
from cave.presentation.renderers.topology_surface_renderer import save_topology_state_surface


OUT = ROOT / "results" / "readme"


TUTORIAL_ASSETS = {
    "09_same_world_subject_dashboard.png": ROOT
    / "out/tutorials/02_comparing_experiences/01_same_world_subject_dashboard.png",
    "10_state_effect_subtraction.png": ROOT
    / "out/tutorials/02_comparing_experiences/06_state_effect_subtraction.png",
    "11_population_episode_dashboard.png": ROOT
    / "out/tutorials/02_comparing_experiences/08_population_episode_dashboard.png",
    "12_population_topology_dashboard.png": ROOT
    / "out/tutorials/02_comparing_experiences/09_population_topology_dashboard.png",
    "13_population_topology_atlas.png": ROOT
    / "out/tutorials/02_comparing_experiences/10_population_topology_atlas.png",
    "14_population_trajectory_geometry.gif": ROOT
    / "out/tutorials/02_comparing_experiences/11_population_trajectory_geometry.gif",
    "15_cavenet_pressure_dashboard.png": ROOT
    / "out/tutorials/03_pressures_cavenet_evolved_subjects/01_cavenet_pressure_dashboard.png",
    "16_cavenet_config_history.png": ROOT
    / "out/tutorials/03_pressures_cavenet_evolved_subjects/02_cavenet_config_history.png",
    "17_role_evidence_board.png": ROOT
    / "out/tutorials/03_pressures_cavenet_evolved_subjects/05_role_evidence_board.png",
    "18_evolved_exposure_metrics.png": ROOT
    / "out/tutorials/03_pressures_cavenet_evolved_subjects/06_evolved_exposure_metrics.png",
    "19_cross_substrate_dashboard.png": ROOT
    / "out/tutorials/03_pressures_cavenet_evolved_subjects/08_cross_substrate_dashboard.png",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    episode = CaveProducer(demo_model()).run(dt=0.25)

    single_renderer = MatplotlibRenderer(
        layout=LayoutSpec(columns=1, figsize_per_cell=(5.2, 4.6), dpi=100),
        style="default",
    )

    single_renderer.save_animation(
        episode,
        [PresentationView()],
        OUT / "02_presentation_wall.gif",
        fps=5,
    )
    single_renderer.save_animation(
        episode,
        [TimelineView()],
        OUT / "03_timeline_attention.gif",
        fps=5,
    )
    single_renderer.save_animation(
        episode,
        [MemoryLookbackView()],
        OUT / "04_memory_lookback.gif",
        fps=5,
    )
    single_renderer.save_animation(
        episode,
        [ExpectationActualView()],
        OUT / "05_expectation_actual.gif",
        fps=5,
    )
    single_renderer.save_animation(
        episode,
        [CorrectionView()],
        OUT / "06_correction.gif",
        fps=5,
    )
    single_renderer.save_animation(
        episode,
        [SubjectiveTopologyView(grid_resolution=40)],
        OUT / "07_subjective_topology.gif",
        fps=5,
    )
    save_topology_state_surface(
        episode,
        OUT / "08_topology_surface.png",
        resolution=34,
        dpi=130,
    )
    _save_attention_profiles(OUT / "09_attention_profiles.png")
    _save_episode_readout(episode, OUT / "10_episode_readout.png")
    _copy_tutorial_assets()


def _save_attention_profiles(output: Path) -> None:
    duration = 5.1
    times = np.linspace(0.0, duration, 180)
    profiles = {
        "balanced sine": AttentionProfile(mode="sine", level=0.5, amplitude=0.5),
        "steady high": AttentionProfile(mode="constant", level=0.85),
        "low capacity": AttentionProfile(mode="constant", level=0.25),
    }
    split_profile = AttentionProfile(
        mode="sine",
        level=0.72,
        amplitude=0.22,
        phase=0.15,
        channel_weights={
            "visual": 0.45,
            "audio": 0.12,
            INTERNAL_EXPECTATION_CHANNEL: 0.43,
        },
        channel_curves={
            "visual": AttentionChannelCurve(
                mode="sine",
                level=0.52,
                amplitude=0.34,
                phase=0.0,
                cycles=1.0,
            ),
            INTERNAL_EXPECTATION_CHANNEL: AttentionChannelCurve(
                mode="sine",
                level=0.48,
                amplitude=0.34,
                phase=np.pi,
                cycles=1.0,
            ),
            "audio": AttentionChannelCurve(
                mode="sine",
                level=0.18,
                amplitude=0.14,
                phase=np.pi / 2.0,
                cycles=2.0,
            ),
        },
    )
    figure, axes = plt.subplots(2, 1, figsize=(8.6, 6.2), dpi=140, sharex=True)
    capacity_axis, channel_axis = axes
    colors = ["#2563eb", "#059669", "#dc2626"]
    for color, (label, profile) in zip(colors, profiles.items()):
        values = [profile.value_at(float(t), duration) for t in times]
        capacity_axis.plot(times, values, label=label, color=color, linewidth=2.2)
    split_capacity = [split_profile.value_at(float(t), duration) for t in times]
    capacity_axis.plot(
        times,
        split_capacity,
        label="split channels",
        color="#7c3aed",
        linewidth=2.0,
        linestyle="--",
    )
    capacity_axis.set_title("Attention profiles", loc="left", fontweight="bold")
    capacity_axis.set_ylabel("capacity")
    capacity_axis.set_ylim(-0.03, 1.03)
    capacity_axis.grid(True, alpha=0.22)
    capacity_axis.legend(frameon=False, loc="lower right", ncol=2)

    channel_colors = {
        "visual": "#2563eb",
        "audio": "#ea580c",
        INTERNAL_EXPECTATION_CHANNEL: "#059669",
    }
    channel_labels = {
        "visual": "visual",
        "audio": "audio",
        INTERNAL_EXPECTATION_CHANNEL: "internal expectation",
    }
    for channel, color in channel_colors.items():
        values = [
            split_profile.state_at(float(t), duration).channel_weight(channel)
            for t in times
        ]
        channel_axis.plot(
            times,
            values,
            label=channel_labels[channel],
            color=color,
            linewidth=2.2,
        )
    channel_axis.set_title("Split channel allocation", loc="left", fontweight="bold")
    channel_axis.set_xlabel("time")
    channel_axis.set_ylabel("channel weight")
    channel_axis.set_ylim(-0.03, 1.03)
    channel_axis.grid(True, alpha=0.22)
    channel_axis.legend(frameon=False, loc="upper right", ncol=3)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _save_episode_readout(episode, output: Path) -> None:
    times = np.array([obs.t for obs in episode.observations], dtype=float)
    surprise = np.array([obs.surprise for obs in episode.observations], dtype=float)
    attention = np.array([obs.attention for obs in episode.observations], dtype=float)
    learning = np.array([obs.learning_rate for obs in episode.observations], dtype=float)
    expected = np.stack([obs.expected for obs in episode.observations], axis=0)
    actual = np.stack([obs.actual for obs in episode.observations], axis=0)
    memory = np.stack([obs.memory_state for obs in episode.observations], axis=0)

    figure = plt.figure(figsize=(11.0, 6.4), dpi=140, facecolor="#f8fafc")
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(1.0, 1.15),
        left=0.08,
        right=0.97,
        top=0.90,
        bottom=0.10,
        wspace=0.25,
        hspace=0.35,
    )
    curve_axis = figure.add_subplot(grid[0, :])
    curve_axis.plot(times, attention, color="#2563eb", label="attention", linewidth=2.0)
    curve_axis.plot(times, surprise, color="#dc2626", label="surprise", linewidth=2.0)
    curve_axis.plot(times, learning, color="#059669", label="learning rate", linewidth=2.0)
    curve_axis.set_title("Episode observations over time", loc="left", fontweight="bold")
    curve_axis.set_xlabel("time")
    curve_axis.set_ylabel("readout value")
    curve_axis.grid(True, alpha=0.18)
    curve_axis.legend(frameon=False, ncol=3, loc="upper right")

    heat_axis = figure.add_subplot(grid[1, 0])
    heat = np.concatenate([expected.T, actual.T, memory.T], axis=0)
    heat_axis.imshow(heat, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(1.0, float(heat.max())))
    heat_axis.set_title("Expected / actual / memory vectors", loc="left", fontweight="bold")
    heat_axis.set_xlabel("timestep")
    heat_axis.set_ylabel("stacked vector rows")
    heat_axis.set_yticks(
        [
            expected.shape[1] / 2,
            expected.shape[1] + actual.shape[1] / 2,
            expected.shape[1] + actual.shape[1] + memory.shape[1] / 2,
        ]
    )
    heat_axis.set_yticklabels(["expected", "actual", "memory"])

    text_axis = figure.add_subplot(grid[1, 1])
    text_axis.axis("off")
    rows = [
        ("inputs", f"{len(episode.inputs)} objects"),
        ("observations", f"{len(episode.observations)} timesteps"),
        ("duration", f"{episode.duration:.2f}"),
        ("features", f"{len(episode.vocabulary)} dimensions"),
        ("final surprise", f"{surprise[-1]:.3f}"),
        ("final memory norm", f"{np.linalg.norm(memory[-1]):.3f}"),
    ]
    text_axis.set_title("Episode contract", loc="left", fontweight="bold")
    y = 0.88
    for key, value in rows:
        text_axis.text(0.0, y, key, fontsize=10, color="#475467", ha="left", va="center")
        text_axis.text(0.48, y, value, fontsize=10, color="#111827", ha="left", va="center")
        y -= 0.13
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _copy_tutorial_assets() -> None:
    missing = []
    for name, source in TUTORIAL_ASSETS.items():
        if source.exists():
            shutil.copy2(source, OUT / name)
        else:
            missing.append(source)
    if missing:
        print("Skipped tutorial-derived README assets; missing sources:")
        for path in missing:
            print(f"  {path.relative_to(ROOT)}")
        print("Run the tutorial notebooks to regenerate those optional assets.")


if __name__ == "__main__":
    main()
