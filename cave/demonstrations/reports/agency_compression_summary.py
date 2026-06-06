from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


REPORT_ROOT = Path("out/reports/cave")
DEFAULT_OUTPUT = Path("artifacts/results/cave/agency_compression_summary.png")


def main() -> None:
    output = DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)

    valence = _scenario_metrics("valence-attractor-repulsor")
    attention = _scenario_metrics("objective-attention-shift")
    compression = _scenario_metrics("representational-compression")
    preference = _subject_metrics("preference-shaped-topology")

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), dpi=150)
    figure.suptitle(
        "Agency and Compression in Cave",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    draw_pipeline(axes[0, 0])
    draw_valence(axes[0, 1], valence)
    draw_attention_compression(axes[1, 0], attention, compression)
    draw_preference(axes[1, 1], preference)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output)
    plt.close(figure)
    print(f"wrote {output}")


def draw_pipeline(axis) -> None:
    prepare(axis, "Mechanism Stack")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    labels = [
        ("attention", "what gets in"),
        ("workspace", "what detail survives"),
        ("objective", "what matters"),
        ("agency", "how exposure changes"),
        ("topology", "what accumulates"),
    ]
    x = 0.08
    y = 0.66
    width = 0.16
    for index, (name, note) in enumerate(labels):
        color = ["#dbeafe", "#e0f2fe", "#dcfce7", "#fee2e2", "#fef3c7"][index]
        rect = Rectangle(
            (x, y),
            width,
            0.18,
            facecolor=color,
            edgecolor="#334155",
            linewidth=1.0,
        )
        axis.add_patch(rect)
        axis.text(x + width / 2, y + 0.115, name, ha="center", va="center", fontsize=10, fontweight="bold")
        axis.text(x + width / 2, y + 0.052, note, ha="center", va="center", fontsize=8, color="#334155")
        if index < len(labels) - 1:
            arrow = FancyArrowPatch(
                (x + width + 0.01, y + 0.09),
                (x + width + 0.065, y + 0.09),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.0,
                color="#334155",
            )
            axis.add_patch(arrow)
        x += width + 0.075

    axis.text(
        0.08,
        0.36,
        "Same external world + different preferences produces different exposure,\n"
        "memory, and topology state. The world is not mutated; coupling is.",
        fontsize=10,
        color="#111827",
        ha="left",
        va="top",
    )


def draw_valence(axis, metrics: dict) -> None:
    prepare(axis, "Valence Is Not Surprise")
    names = ["neutral", "pleasant", "painful"]
    x = range(len(names))
    pain = [metrics[name]["pain"] for name in names]
    pleasure = [metrics[name]["pleasure"] for name in names]
    surprise = [metrics[name]["surprise"] for name in names]
    axis.bar([i - 0.23 for i in x], pain, width=0.22, color="#b42318", label="pain")
    axis.bar([i for i in x], pleasure, width=0.22, color="#2f855a", label="pleasure")
    axis.bar([i + 0.23 for i in x], surprise, width=0.22, color="#c2410c", alpha=0.55, label="surprise")
    axis.set_xticks(list(x), names)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("value")
    axis.grid(True, axis="y", color="#e5e7eb")
    axis.legend(frameon=False, loc="upper right", fontsize=8)
    axis.text(
        0.02,
        0.80,
        "Pleasure rises for a surprising pleasant event;\npain rises for the painful event.",
        transform=axis.transAxes,
        fontsize=9,
        va="top",
        color="#374151",
    )


def draw_attention_compression(axis, attention: dict, compression: dict) -> None:
    prepare(axis, "Objective Attention + Workspace Compression")
    audio_before = attention["first_attention_channels"]["audio"]
    audio_after = attention["first_next_attention_channels"]["audio"]
    compression_cost = compression["compression_cost"]
    reconstruction_error = compression["reconstruction_error"]
    values = [audio_before, audio_after, compression_cost, reconstruction_error]
    labels = ["audio attention\nbefore", "audio attention\nafter", "compression\ncost", "reconstruction\nerror"]
    colors = ["#93c5fd", "#2563eb", "#f59e0b", "#f97316"]
    axis.bar(range(len(values)), values, color=colors, width=0.55)
    axis.set_xticks(range(len(labels)), labels, fontsize=8)
    axis.set_ylim(0, 0.85)
    axis.set_ylabel("value")
    axis.grid(True, axis="y", color="#e5e7eb")
    axis.text(
        0.02,
        0.93,
        "A painful audio channel shifts next attention from "
        f"{audio_before:.2f} to {audio_after:.2f}.\n"
        f"Top-k workspace keeps {compression['active_features'][0]!r} and drops detail.",
        transform=axis.transAxes,
        fontsize=9,
        va="top",
        color="#374151",
    )


def draw_preference(axis, metrics: dict) -> None:
    prepare(axis, "Preference-Shaped Topology")
    warm_memory = metrics["warm_final_memory"]
    avoid_memory = metrics["threat_avoid_final_memory"]
    labels = ["warmth", "threat"]
    x = range(2)
    axis.bar([i - 0.18 for i in x], warm_memory, width=0.34, color="#2f855a", label="warm-pref subject")
    axis.bar([i + 0.18 for i in x], avoid_memory, width=0.34, color="#b42318", label="threat-avoid subject")
    axis.set_xticks(list(x), labels)
    axis.set_ylabel("final memory")
    axis.grid(True, axis="y", color="#e5e7eb")
    axis.legend(frameon=False, loc="upper right", fontsize=8)
    axis.text(
        0.02,
        0.82,
        "Actions: warm subject approaches warm_event;\n"
        "threat-sensitive subject avoids threat_event.\n"
        f"State-effect distance: {metrics['effect_distance']:.3f}",
        transform=axis.transAxes,
        fontsize=9,
        va="top",
        color="#374151",
    )


def prepare(axis, title: str) -> None:
    axis.set_title(title, loc="left", fontsize=12, fontweight="bold")
    axis.set_facecolor("#f8fafc")
    for spine in axis.spines.values():
        spine.set_color("#cbd5e1")


def _scenario_metrics(name: str) -> dict:
    data = _read_json(REPORT_ROOT / "scenarios" / name / "checks.json")
    return data["extra"][0]["metrics"]


def _subject_metrics(name: str) -> dict:
    data = _read_json(REPORT_ROOT / "subjects" / name / "checks.json")
    return data["metrics"]


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; generate reports before building the summary figure"
        )
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
