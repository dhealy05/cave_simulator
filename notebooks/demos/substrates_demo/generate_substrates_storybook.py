"""Generate the static panels for the Cave *substrates* storybook.

This storybook varies the **machine**: native Cave, CaveNet (the same update path
written as a network), a stripped-down minimal subject, and an evolved recurrent
black box -- plus, in a coda, GPT-2 and conversation producers. The thesis is the
``Episode`` contract: utterly different internals, one shared, comparable shape.

Every generated panel is reproducible here (native / CaveNet / minimal / evolved
all run from committed code). The text-substrate coda copies committed reference
assets from ``artifacts/results/`` (GPT-2 and conversation need optional model deps to
*re-run*, but their outputs are checked in).

Run from the repository root:

    python notebooks/demos/substrates_demo/generate_substrates_storybook.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Initialize the scenarios package first to avoid a circular import when we pull
# the minimal-subject builder out of cave.pressure.checks.
import cave.demonstrations.scenarios  # noqa: F401

from cave import CaveProducer, demo_model, episode_set, labeled_episode
from cave.presentation.renderers.episode_set_dashboard import save_episode_set_dashboard
from cave.substrates.cavenet import (
    CaveNet,
    CaveNetProducer,
    compare_cavenet_to_cave,
)
from cave.pressure.checks.cavenet_pressure import build_pressure_episode
from cave.pressure.checks.evolved_exposure import build_evolved_exposure_episode
from cave.pressure.checks.preference_emergence import build_preference_emergence_episode

REPO = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
OUT = BASE / "storybook_assets"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GREY = "#999999"

FIELDS = ["expected", "actual", "memory_state", "surprise", "learning_rate", "attention"]
NUMBERS: dict[str, object] = {}


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _series(ep, field):
    out = []
    for o in ep.observations:
        v = getattr(o, field, None)
        if v is None:
            out.append(0.0)
            continue
        a = np.asarray(v, dtype=float)
        out.append(float(np.linalg.norm(a)) if a.size > 1 else float(a))
    return np.array(out)


def _times(ep):
    return np.array([o.t for o in ep.observations])


def _field_max(ep, field):
    m = 0.0
    for o in ep.observations:
        v = getattr(o, field, None)
        if v is None:
            continue
        a = np.asarray(v, dtype=float)
        if a.size:
            m = max(m, float(np.max(np.abs(a))))
    return m


# --------------------------------------------------------------------------- #
def build_episodes():
    native = CaveProducer(demo_model(seed=1)).run(dt=0.2)
    m = demo_model(seed=1)
    cn = CaveNet.from_subject_state(
        sequence=m.sequence,
        subject_state=m.subject_state,
        params=m.params,
        vocabulary=m.vocabulary,
        sensorium=m.sensorium,
    )
    cavenet = CaveNetProducer(cn).run(dt=0.2)
    comparison = compare_cavenet_to_cave(native, cavenet)
    minimal = build_preference_emergence_episode("minimal-preference")
    evolved = build_evolved_exposure_episode("evolved-recurrent")
    return native, cavenet, comparison, minimal, evolved


# --------------------------------------------------------------------------- #
def page_contract_matrix(episodes):
    names = ["native\nCave", "CaveNet", "minimal\nsubject", "evolved\nsubject"]
    keys = ["native", "cavenet", "minimal", "evolved"]
    eps = dict(zip(keys, episodes))
    grid = np.array([[_field_max(eps[k], f) for k in keys] for f in FIELDS])
    NUMBERS["contract_matrix"] = {
        k: {f: round(_field_max(eps[k], f), 3) for f in FIELDS} for k in keys
    }
    filled = grid > 1e-9
    display = np.where(filled, 1.0, 0.0)
    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=140)
    ax.imshow(display, cmap=plt.cm.RdYlGn, vmin=0, vmax=1, aspect="auto", alpha=0.55)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_yticks(np.arange(len(FIELDS)))
    ax.set_yticklabels([f.replace("_", " ") for f in FIELDS], fontsize=10)
    for i, f in enumerate(FIELDS):
        for j, k in enumerate(keys):
            v = grid[i, j]
            label = f"{v:.2f}" if filled[i, j] else "—"
            ax.text(
                j, i, label, ha="center", va="center", fontsize=10,
                color="#114411" if filled[i, j] else "#777777",
                fontweight="bold" if filled[i, j] else "normal",
            )
    ax.set_title(
        "One Episode contract; each machine fills the slots it has",
        fontsize=13, fontweight="bold", loc="left", pad=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "01_contract_matrix.png", bbox_inches="tight")
    plt.close(fig)


def page_cavenet_equivalence(native, cavenet, comparison):
    NUMBERS["cavenet_equivalence"] = {
        "ok": bool(comparison.ok),
        "max_actual_distance": float(comparison.metrics.get("max_actual_distance", float("nan"))),
        "max_memory_distance": float(comparison.metrics.get("max_memory_distance", float("nan"))),
    }
    t = _times(native)
    s_native = np.array([o.surprise for o in native.observations])
    s_cavenet = np.array([o.surprise for o in cavenet.observations])
    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=140)
    ax.plot(t, s_native, "-", color=BLUE, linewidth=5.0, alpha=0.45, label="native Cave", zorder=2)
    ax.plot(t, s_cavenet, "--", color=ORANGE, linewidth=2.0, label="CaveNet (network form)", zorder=3)
    ax.set_ylim(0, max(s_native.max(), 0.01) * 1.25)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    _style(ax, "The same update path, written as a network, agrees exactly", "time (s)", "surprise")
    ax.annotate(
        f"max actual distance {comparison.metrics.get('max_actual_distance', 0):.0e}\n"
        f"max memory distance {comparison.metrics.get('max_memory_distance', 0):.0e}",
        (0.04, 0.04), xycoords="axes fraction", fontsize=10, color="#555",
    )
    fig.tight_layout()
    fig.savefig(OUT / "02_cavenet_equivalence.png", bbox_inches="tight")
    plt.close(fig)


def page_minimal(minimal):
    NUMBERS["minimal"] = {
        "duration": minimal.duration,
        "n_obs": len(minimal.observations),
        "adapter": minimal.metadata.get("adapter"),
        "surprise_max": round(_field_max(minimal, "surprise"), 3),
        "memory_max": round(_field_max(minimal, "memory_state"), 3),
    }
    t = _times(minimal)
    s = np.array([o.surprise for o in minimal.observations])
    mem = _series(minimal, "memory_state")
    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=140)
    ax.plot(t, s, "-o", color=BLUE, markersize=3, linewidth=2.0, label="surprise", zorder=3)
    ax.plot(t, mem, "-", color=GREEN, linewidth=2.2, label="memory magnitude", zorder=3)
    ax.legend(loc="center right", frameon=False, fontsize=10)
    _style(
        ax,
        "A minimal associative subject still emits a full Cave trajectory",
        "time (s)",
        "value",
    )
    fig.tight_layout()
    fig.savefig(OUT / "03_minimal_subject.png", bbox_inches="tight")
    plt.close(fig)


def page_evolved(evolved):
    NUMBERS["evolved"] = {
        "duration": evolved.duration,
        "n_obs": len(evolved.observations),
        "adapter": evolved.metadata.get("adapter"),
        "vocab": evolved.vocabulary,
        "fills": {f: round(_field_max(evolved, f), 3) for f in FIELDS},
    }
    t = _times(evolved)
    mem = _series(evolved, "memory_state")
    att = np.array([o.attention for o in evolved.observations])
    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=140)
    ax.plot(t, mem, "-", color=GREEN, linewidth=2.2, label="hidden memory magnitude", zorder=3)
    ax.plot(t, att, "-", color=BLUE, linewidth=2.0, label="attention", zorder=3)
    ax.set_ylim(0, 2.3)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    _style(
        ax,
        "An evolved black box: carries state and attention, but emits no surprise",
        "time (s)",
        "value",
    )
    ax.annotate(
        "expected / surprise / learning rate: not emitted\n"
        "(this network has no explicit prediction step)",
        (0.03, 0.94), xycoords="axes fraction", va="top", fontsize=9.5, color="#555",
    )
    fig.tight_layout()
    fig.savefig(OUT / "04_evolved_subject.png", bbox_inches="tight")
    plt.close(fig)


def page_cross_substrate(native, cavenet, minimal, evolved):
    # Use the adaptive CaveNet + full native run so the four trajectories are
    # comparable lengths in the shared embedding space.
    native_full = CaveProducer(demo_model()).run(dt=0.25)
    cavenet_adaptive = build_pressure_episode("adaptive", dt=0.25)
    es = episode_set(
        [
            labeled_episode(native_full, id="native", label="native Cave", group="cave"),
            labeled_episode(cavenet_adaptive, id="cavenet", label="CaveNet", group="cavenet"),
            labeled_episode(minimal, id="minimal", label="minimal subject", group="minimal"),
            labeled_episode(evolved, id="evolved", label="evolved subject", group="evolved"),
        ],
        id="substrates",
        title="Four substrates",
        comparison_axis="substrate",
    )
    save_episode_set_dashboard(
        es, OUT / "05_cross_substrate.png", title="Four substrates through one Episode contract"
    )


def page_text_substrates():
    """Copy committed text-substrate reference assets (reproducible copy)."""
    copies = {
        REPO
        / "artifacts"
        / "results"
        / "gpt2"
        / "reference"
        / "frame.png": "06_gpt2_frame.png",
        REPO
        / "artifacts"
        / "results"
        / "gpt2"
        / "matrices"
        / "text-config"
        / "population.png": "06_gpt2_population.png",
        REPO
        / "artifacts"
        / "results"
        / "conversation"
        / "matrices"
        / "text-config"
        / "population.png": "06_conversation_population.png",
    }
    copied = []
    for src, dst in copies.items():
        if src.exists():
            shutil.copyfile(src, OUT / dst)
            copied.append(dst)
    NUMBERS["text_substrates_copied"] = copied


# --------------------------------------------------------------------------- #
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    native, cavenet, comparison, minimal, evolved = build_episodes()
    page_contract_matrix((native, cavenet, minimal, evolved))
    page_cavenet_equivalence(native, cavenet, comparison)
    page_minimal(minimal)
    page_evolved(evolved)
    page_cross_substrate(native, cavenet, minimal, evolved)
    page_text_substrates()

    import json

    print("\n=== VERIFIED NUMBERS (for prose) ===")
    print(json.dumps(NUMBERS, default=str, indent=1))
    print("\npanels written to", OUT)


if __name__ == "__main__":
    main()
