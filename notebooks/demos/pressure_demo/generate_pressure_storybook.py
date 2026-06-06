"""Generate the static panels for the pressure / controls storybook.

This storybook is different in spirit: the others mostly teach mechanisms, while
this one teaches the *logic of an experiment*. An evolved recurrent creature
learns to regulate exposure in a delayed-value world; the hero of the story is
the set of **controls** that try to break it. If the behaviour collapses when you
remove memory, recurrence, or temporal order, the behaviour was real.

All numbers are read from the committed result ladder
(``artifacts/results/result_ladder/checks/evolved-exposure.json``) — the evidence snapshot,
not a fresh genetic search. Run from the repository root:

    python notebooks/demos/pressure_demo/generate_pressure_storybook.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from cave import CaveProducer, demo_model, episode_set, labeled_episode
from cave.presentation.renderers.episode_set_dashboard import save_episode_set_dashboard
from cave.presentation.renderers.matplotlib_renderer.role_evidence import save_role_evidence_frame
from cave.pressure.checks.cavenet_pressure import build_pressure_episode
from cave.pressure.checks.evolved_exposure import build_evolved_exposure_episode

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]
OUT = BASE / "storybook_assets"
CHECK = (
    REPO
    / "artifacts"
    / "results"
    / "result_ladder"
    / "checks"
    / "evolved-exposure.json"
)

EVOLVED = "#009E73"
CONTROL = "#9CA3AF"
GOOD = "#0072B2"
BAD = "#D55E00"
TEXT = "#111827"
MUTED = "#6B7280"
GRID = "#D1D5DB"

# Display order: the real creature first, then the controls, each breaking one thing.
ORDER = ["evolved-recurrent", "shuffled-temporal", "hidden-reset", "random-recurrent", "non-recurrent"]
SHORT = {
    "evolved-recurrent": "evolved\n(intact)",
    "shuffled-temporal": "shuffled\ntime",
    "hidden-reset": "memory\nreset each step",
    "random-recurrent": "random\ngenome",
    "non-recurrent": "no memory\n(non-recurrent)",
}


def load_metrics():
    data = json.loads(CHECK.read_text(encoding="utf-8"))
    return data["extra"][0]["metrics"]


def _style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=TEXT)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, axis="y", color=GRID, linewidth=0.7, alpha=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _bar_colors():
    return [EVOLVED] + [CONTROL] * (len(ORDER) - 1)


# --------------------------------------------------------------------------- #
def page1_world(output: Path):
    """A schematic of the delayed-value world and compact recurrent controller."""
    fig, ax = plt.subplots(figsize=(11.2, 5.2), dpi=140)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # time arrow
    ax.annotate("", xy=(9.7, 0.75), xytext=(0.3, 0.75),
                arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1.6))
    ax.text(9.7, 0.45, "time", ha="right", va="top", color=MUTED, fontsize=10)

    def station(x, title, sub, color):
        ax.add_patch(mpatches.FancyBboxPatch((x - 1.05, 1.8), 2.1, 1.25,
                     boxstyle="round,pad=0.06", facecolor="white",
                     edgecolor=color, linewidth=2.0))
        ax.text(x, 2.65, title, ha="center", va="center", fontsize=11, fontweight="bold", color=TEXT)
        ax.text(x, 2.15, sub, ha="center", va="center", fontsize=9, color=MUTED)
        ax.plot([x, x], [1.8, 0.75], color=color, linewidth=1.2, linestyle=":")

    station(1.6, "CUE", "cue_good\nor cue_bad", GOOD)
    station(5.0, "DELAY", "neutral\n(no value)", MUTED)
    station(8.4, "OUTCOME", "good: +1 if open\nbad: -1 if open", BAD)

    ax.text(5.0, 4.65, "A delayed-value world, solved by one small recurrent controller",
            ha="center", fontsize=14, fontweight="bold", color=TEXT)
    ax.text(0.55, 4.05, "obs_t in [cue_good, cue_bad, good, bad, neutral]",
            ha="left", va="center", fontsize=9.5, color=TEXT)
    ax.text(0.55, 3.75, "h_t = tanh(W_x obs_t + W_h h_{t-1} + b_h), hidden_dim = 5",
            ha="left", va="center", fontsize=9.5, color=TEXT)
    ax.text(0.55, 3.45, "exposure_t = sigmoid(W_a h_t + b_a)",
            ha="left", va="center", fontsize=9.5, color=TEXT)
    ax.text(5.0, 0.05,
            "To win, the subject must carry the cue across the neutral delay, "
            "then open before good outcomes and close before bad outcomes.",
            ha="center", va="bottom", fontsize=10, color=TEXT)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def page2_solves(m, output: Path):
    """The evolved creature seeks good and avoids bad; a random one can't tell them apart."""
    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=140)
    eu = m["evolved-recurrent"]["utility"]
    ru = m["random-recurrent"]["utility"]
    groups = [f"evolved (intact)\nutility {eu:.1f}", f"random genome\nutility {ru:.1f}"]
    good = [m["evolved-recurrent"]["good_exposure"], m["random-recurrent"]["good_exposure"]]
    bad = [m["evolved-recurrent"]["bad_exposure"], m["random-recurrent"]["bad_exposure"]]
    x = np.arange(len(groups))
    w = 0.34
    ax.bar(x - w / 2, good, w, color=GOOD, label="exposure to GOOD cues")
    ax.bar(x + w / 2, bad, w, color=BAD, label="exposure to BAD cues")
    for xi, g, b in zip(x, good, bad):
        ax.text(xi - w / 2, g + 0.02, f"{g:.2f}", ha="center", fontsize=9, color=TEXT)
        ax.text(xi + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=9, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylim(0, 1.18)
    _style(ax, "He learns to open up for good and shut out bad", "fraction of exposure")
    ax.legend(loc="upper center", frameon=False, fontsize=10, ncol=2)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def page3_collapse(m, output: Path):
    """The hero panel: utility across the intact creature and the controls."""
    fig, ax = plt.subplots(figsize=(10.0, 4.8), dpi=140)
    vals = [m[v]["utility"] for v in ORDER]
    x = np.arange(len(ORDER))
    bars = ax.bar(x, vals, color=_bar_colors(), edgecolor="white", linewidth=1.0)
    for xi, val in zip(x, vals):
        ax.text(xi, val + 0.2, f"{val:.1f}", ha="center", fontsize=10, color=TEXT, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[v] for v in ORDER], fontsize=9.5)
    ax.set_ylim(min(0, min(vals)) - 0.5, max(vals) * 1.16)
    _style(ax, "Remove any one capacity and the skill collapses", "utility earned")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.annotate("the evolved creature", (0, vals[0]), xytext=(0.6, vals[0] * 0.82),
                fontsize=10, color=EVOLVED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=EVOLVED))
    ax.text(2.5, max(vals) * 0.5, "every control =\nthe SAME creature,\none ability removed",
            ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def page4_controls(m, output: Path):
    """Two diagnostics across variants: exposure discrimination, and whether a probe
    can read the future outcome out of the creature's hidden state."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=140)
    x = np.arange(len(ORDER))

    contrast = [m[v]["exposure_contrast"] for v in ORDER]
    ax1.bar(x, contrast, color=_bar_colors(), edgecolor="white", linewidth=1.0)
    for xi, val in zip(x, contrast):
        ax1.text(xi, val + 0.01, f"{val:.2f}", ha="center", fontsize=9, color=TEXT)
    ax1.set_xticks(x); ax1.set_xticklabels([SHORT[v] for v in ORDER], fontsize=8.5)
    ax1.set_ylim(0, 1.08)
    _style(ax1, "Does it tell good from bad?\n(exposure contrast)", "good − bad exposure")

    probe = [m[v]["probe_accuracy"] for v in ORDER]
    ax2.bar(x, probe, color=_bar_colors(), edgecolor="white", linewidth=1.0)
    for xi, val in zip(x, probe):
        ax2.text(xi, val + 0.01, f"{val:.2f}", ha="center", fontsize=9, color=TEXT)
    ax2.axhline(0.5, color=BAD, linestyle="--", linewidth=1.2)
    ax2.text(len(ORDER) - 0.5, 0.52, "chance", ha="right", va="bottom", fontsize=9, color=BAD)
    ax2.set_xticks(x); ax2.set_xticklabels([SHORT[v] for v in ORDER], fontsize=8.5)
    ax2.set_ylim(0, 1.08)
    _style(ax2, "Can a probe read the future\noutcome from its hidden state?", "probe accuracy")

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CaveNet: the built-in roles, pushed by pressure
# --------------------------------------------------------------------------- #
GAIN_KEYS = [
    ("attention_gain", "#0072B2", "attention"),
    ("learning_rate_gain", "#009E73", "learning rate"),
    ("topology_deposit_gain", "#D55E00", "topology deposit"),
    ("topology_transition_gain", "#9CA3AF", "topology transition"),
]


def page_cavenet_history(output: Path):
    episode = build_pressure_episode("adaptive", dt=0.25)
    history = episode.metadata["cavenet_config_history"]
    t = np.array([h["t"] for h in history], dtype=float)
    fig, ax = plt.subplots(figsize=(9.8, 4.6), dpi=140)
    for key, color, label in GAIN_KEYS:
        y = np.array([h["after"][key] for h in history], dtype=float)
        ax.plot(t, y, color=color, linewidth=2.4, marker="o", markersize=3, label=label)
    _style(ax, "CaveNet under pressure: weak gains climb on their own", "gain")
    ax.set_xlabel("time (s)", fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, ncol=2)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def page_role_board(output: Path):
    save_role_evidence_frame(output, dt=1.0)


def page_cross_substrate(output: Path):
    cave_ep = CaveProducer(demo_model()).run(dt=0.1)
    cavenet_ep = build_pressure_episode("adaptive", dt=0.25)
    evolved_ep = build_evolved_exposure_episode("evolved-recurrent")
    es = episode_set(
        [
            labeled_episode(cave_ep, id="native-cave", label="native Cave", group="cave"),
            labeled_episode(cavenet_ep, id="cavenet", label="CaveNet", group="cavenet"),
            labeled_episode(evolved_ep, id="evolved", label="evolved subject", group="evolved"),
        ],
        id="cross_substrate",
        title="Three substrates, one Episode contract",
        comparison_axis="substrate",
    )
    save_episode_set_dashboard(es, output, title="Three substrates through one Episode contract")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = load_metrics()
    page1_world(OUT / "01_delayed_world.png")
    page2_solves(m, OUT / "02_solves_it.png")
    page3_collapse(m, OUT / "03_utility_collapse.png")
    page4_controls(m, OUT / "04_controls.png")
    page_cavenet_history(OUT / "05_cavenet_config_history.png")
    page_role_board(OUT / "06_role_evidence_board.png")
    page_cross_substrate(OUT / "07_cross_substrate.png")
    print("evolved utility =", round(m["evolved-recurrent"]["utility"], 2),
          "| non-recurrent =", round(m["non-recurrent"]["utility"], 2),
          "| shuffled =", round(m["shuffled-temporal"]["utility"], 2))
    print("pages written to", OUT)


if __name__ == "__main__":
    main()
