"""Generate the static panels for the Cave *scenarios* storybook.

This storybook follows the **canonical causal probes** documented in
``docs/experiments/scenarios.md`` and implemented in
``cave/demonstrations/scenarios``. Each probe makes *one* distinction
inspectable, written as: hypothesis / control / expected / observed.

Every panel is produced either by a package renderer (``MatplotlibRenderer`` view
stills, the topology-atlas renderer) or by a small custom plot whose numbers are
harvested live from the probe's own ``check_*`` function -- so no number in the
prose is hand-typed; they all trace to a committed probe.

Run from the repository root:

    python notebooks/demos/scenarios_demo/generate_scenarios_storybook.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from cave.observation.structural import episode_frames, structural_state_for_episode
from cave.observation.views import (
    AffectView,
    ExpectationActualView,
    PresentationView,
    default_views,
)
from cave.presentation.renderers.matplotlib_renderer.renderer import (
    LayoutSpec,
    MatplotlibRenderer,
)

from cave.demonstrations.scenarios import (
    attention_bottleneck,
    expectation_violation,
    importance_weighted_event,
    objective_attention_shift,
    representational_compression,
    role_dependency_contrasts,
    topology_atlas,
    unseen_modality,
    valence_attractor_repulsor,
)
from cave.presentation.renderers.topology_atlas_renderer import save_topology_atlas
from cave.pressure.checks import preference_emergence

BASE = Path(__file__).resolve().parent
OUT = BASE / "storybook_assets"

# Colorblind-safe palette, shared with the compare/pressure books.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GREY = "#999999"
RED = "#CC3311"

NUMBERS: dict[str, object] = {}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _frames(episode):
    return list(episode_frames(episode, structural_state_for_episode(episode)))


def _frame_at(frames, t):
    return min(frames, key=lambda f: abs(f.observation.t - t))


def _views_by_name(extra=()):
    by_name = {type(v).__name__: v for v in default_views()}
    for v in extra:
        by_name[type(v).__name__] = v
    return by_name


def _save_frame(episode, view_objs, t, slug, *, columns=None):
    frames = _frames(episode)
    frame = _frame_at(frames, t)
    cols = columns or len(view_objs)
    renderer = MatplotlibRenderer(
        layout=LayoutSpec(columns=cols, figsize_per_cell=(5.1, 4.3))
    )
    renderer.save_frame(frame, view_objs, OUT / f"{slug}.png")
    return frame


# --------------------------------------------------------------------------- #
# Act I -- the gate: external presence vs entry into state
# --------------------------------------------------------------------------- #
def page_unseen_modality():
    spec = unseen_modality.unseen_modality_report_spec(include_assets=False)
    ep = spec.episode_factory()
    m = spec.checks[0](ep)["metrics"]
    NUMBERS["unseen_modality"] = m
    # Both objects are fully present in the external sequence; only the one with
    # a matching sensor crosses into actual state. The visible flash enters at
    # full strength on its channel; the audio tone never enters at all.
    visible_entered = float(np.max(np.abs(m["visible_actual"])))
    unheard_entered = float(np.max(np.abs(m["unheard_actual"])))
    events = ["visible flash\n(visual sensor)", "unheard tone\n(no audio sensor)"]
    external = [1.0, 1.0]
    entered = [visible_entered, unheard_entered]
    x = np.arange(len(events))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=140)
    ax.bar(x - w / 2, external, w, label="present in external sequence", color=GREY, zorder=3)
    ax.bar(x + w / 2, entered, w, label="entered actual state", color=BLUE, zorder=3)
    for xi, a in zip(x, external):
        ax.text(xi - w / 2, a + 0.02, f"{a:.1f}", ha="center", fontsize=10, color="#555")
    for xi, a in zip(x, entered):
        ax.text(xi + w / 2, a + 0.02, f"{a:.1f}", ha="center", fontsize=10, color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(events)
    ax.set_ylim(0, 1.2)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    _style(ax, "External presence is not entry into state", "", "value")
    ax.annotate(
        f"sensor channels: {', '.join(m['unheard_sensor_channels'])} only",
        (0.5, 0.55), fontsize=10, color="#555", ha="center",
    )
    fig.tight_layout()
    fig.savefig(OUT / "01_unseen_modality.png", bbox_inches="tight")
    plt.close(fig)


def page_attention_bottleneck():
    spec = attention_bottleneck.attention_bottleneck_report_spec(include_assets=False)
    ep = spec.episode_factory()
    NUMBERS["attention_bottleneck"] = spec.checks[0](ep)["metrics"]
    by = _views_by_name()
    _save_frame(
        ep,
        [by["PresentationView"], by["ExpectationActualView"]],
        0.5,
        "02_attention_bottleneck",
    )


def page_compression():
    spec = representational_compression.representational_compression_report_spec(
        include_assets=False
    )
    ep = spec.episode_factory()
    m = spec.checks[0](ep)["metrics"]
    NUMBERS["compression"] = m
    attended = np.array(m["attended_input"], dtype=float)
    actual = np.array(m["actual"], dtype=float)
    labels = ["dominant", "secondary", "detail"]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=140)
    ax.bar(x - w / 2, attended, w, label="attended (after attention)", color=GREY, zorder=3)
    ax.bar(x + w / 2, actual, w, label="actual state (after workspace)", color=BLUE, zorder=3)
    for xi, a in zip(x, attended):
        ax.text(xi - w / 2, a + 0.012, f"{a:.3f}", ha="center", fontsize=9, color="#555")
    for xi, a in zip(x, actual):
        ax.text(xi + w / 2, a + 0.012, f"{a:.3f}", ha="center", fontsize=9, color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.6)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    _style(
        ax,
        "A top-1 workspace keeps the dominant feature, drops the rest",
        "feature",
        "value",
    )
    ax.annotate(
        f"compression cost {m['compression_cost']:.2f}\n"
        f"reconstruction error {m['reconstruction_error']:.2f}",
        (1.5, 0.34),
        fontsize=10,
        color="#555",
    )
    fig.tight_layout()
    fig.savefig(OUT / "03_compression.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Act II -- prediction and learning
# --------------------------------------------------------------------------- #
def page_expectation_violation():
    spec = expectation_violation.expectation_violation_report_spec(include_assets=False)
    ep = spec.episode_factory()
    m = spec.checks[0](ep)["metrics"]
    NUMBERS["expectation_violation"] = m

    # 04a -- surprise + learning rate across the four events (custom).
    ids = ["repeat_1", "repeat_2", "repeat_3", "violation"]
    s = [m["surprise"][i] for i in ids]
    lr = [m["learning_rate"][i] for i in ids]
    x = np.arange(len(ids))
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=140)
    ax.plot(x, s, "-o", color=BLUE, linewidth=2.4, markersize=7, label="surprise", zorder=3)
    ax.plot(x, lr, "-s", color=ORANGE, linewidth=2.0, markersize=6, label="learning rate", zorder=3)
    for xi, v in zip(x, s):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=9, color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(["repeat 1", "repeat 2", "repeat 3", "VIOLATION"])
    ax.set_ylim(0, 0.55)
    ax.legend(loc="upper center", frameon=False, fontsize=10)
    _style(ax, "Surprise falls with repetition, then spikes at the violation", "", "value")
    fig.tight_layout()
    fig.savefig(OUT / "04_violation_surprise.png", bbox_inches="tight")
    plt.close(fig)

    # 04b -- the violation as correction geometry (package view still).
    by = _views_by_name()
    fv = max(_frames(ep), key=lambda f: f.observation.surprise)
    NUMBERS["expectation_violation_t"] = fv.observation.t
    MatplotlibRenderer(
        layout=LayoutSpec(columns=2, figsize_per_cell=(5.1, 4.3))
    ).save_frame(
        fv, [by["ExpectationActualView"], by["CorrectionView"]], OUT / "04_violation_correction.png"
    )


def page_importance_weighted():
    spec = importance_weighted_event.importance_weighted_event_report_spec(
        include_assets=False
    )
    ep = spec.episode_factory()
    m = spec.checks[0](ep)["metrics"]
    NUMBERS["importance_weighted"] = m
    groups = ["learning rate", "memory movement", "attention weight"]
    ordinary = [m["ordinary_learning_rate"], m["ordinary_memory_delta"], m["ordinary_attention_weight"]]
    important = [m["important_learning_rate"], m["important_memory_delta"], m["important_attention_weight"]]
    x = np.arange(len(groups))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=140)
    ax.bar(x - w / 2, ordinary, w, label="ordinary event (weight 1)", color=GREY, zorder=3)
    ax.bar(x + w / 2, important, w, label="important event (higher weight)", color=ORANGE, zorder=3)
    for xi, a in zip(x, ordinary):
        ax.text(xi - w / 2, a + 0.012, f"{a:.2f}", ha="center", fontsize=9, color="#555")
    for xi, a in zip(x, important):
        ax.text(xi + w / 2, a + 0.012, f"{a:.2f}", ha="center", fontsize=9, color=ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    _style(ax, "Same update rule, one knob: learning_weight", "", "value")
    fig.tight_layout()
    fig.savefig(OUT / "05_importance_weighted.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Act III -- value steering
# --------------------------------------------------------------------------- #
def page_valence():
    spec = valence_attractor_repulsor.valence_attractor_repulsor_report_spec(
        include_assets=False
    )
    ep = spec.episode_factory()
    m = spec.checks[0](ep)["metrics"]
    NUMBERS["valence"] = m

    # 06a -- affect view still at the painful event (t=1.7).
    affect = AffectView()
    by = _views_by_name(extra=[affect])
    _save_frame(ep, [by["AffectView"], by["ExpectationActualView"]], 1.7, "06_valence_affect")

    # 06b -- net valence and utility across the three events (custom).
    events = ["neutral", "pleasant", "painful"]
    net = [m[e]["net"] for e in events]
    util = [m[e]["utility"] for e in events]
    surprise = [m[e]["surprise"] for e in events]
    x = np.arange(len(events))
    w = 0.27
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=140)
    ax.axhline(0, color="#888", linewidth=1)
    ax.bar(x - w, net, w, label="net valence", color=GREEN, zorder=3)
    ax.bar(x, util, w, label="utility", color=BLUE, zorder=3)
    ax.bar(x + w, surprise, w, label="surprise", color=GREY, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(events)
    ax.legend(loc="lower left", frameon=False, fontsize=10)
    _style(ax, "Affect is evaluated state, separate from surprise", "", "value")
    fig.tight_layout()
    fig.savefig(OUT / "06_valence_bars.png", bbox_inches="tight")
    plt.close(fig)


def page_objective_shift():
    spec = objective_attention_shift.objective_attention_shift_report_spec(
        include_assets=False
    )
    ep = spec.episode_factory()
    m = spec.checks[0](ep)["metrics"]
    NUMBERS["objective_shift"] = m
    first = m["first_attention_channels"]
    nxt = m["first_next_attention_channels"]
    channels = ["visual", "audio"]
    before = [first[c] for c in channels]
    after = [nxt[c] for c in channels]
    x = np.arange(len(channels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=140)
    ax.bar(x - w / 2, before, w, label="this step's channel weights", color=GREY, zorder=3)
    ax.bar(x + w / 2, after, w, label="next step (after objective pressure)", color=RED, zorder=3)
    for xi, a in zip(x, before):
        ax.text(xi - w / 2, a + 0.015, f"{a:.2f}", ha="center", fontsize=9, color="#555")
    for xi, a in zip(x, after):
        ax.text(xi + w / 2, a + 0.015, f"{a:.2f}", ha="center", fontsize=9, color=RED)
    ax.set_xticks(x)
    ax.set_xticklabels(["visual (neutral)", "audio (painful)"])
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper center", frameon=False, fontsize=10)
    _style(ax, "A valued signal redistributes the next step's attention", "", "channel weight")
    fig.tight_layout()
    fig.savefig(OUT / "07_objective_shift.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Act IV -- emergence and controls (bridge to the pressure book)
# --------------------------------------------------------------------------- #
def page_preference_emergence():
    spec = preference_emergence.preference_emergence_report_spec(include_assets=False)
    ep = spec.episode_factory()
    m = spec.checks[0](ep)["metrics"]
    NUMBERS["preference_emergence"] = {
        k: {kk: vv for kk, vv in v.items()} for k, v in m.items()
    }
    base = m["minimal-preference"]
    groups = ["surprise", "skill (exposure separation)"]
    early = [base["early_surprise"], base["early_skill"]]
    late = [base["late_surprise"], base["late_skill"]]
    x = np.arange(len(groups))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=140)
    ax.bar(x - w / 2, early, w, label="early", color=GREY, zorder=3)
    ax.bar(x + w / 2, late, w, label="late", color=GREEN, zorder=3)
    for xi, a in zip(x, early):
        ax.text(xi - w / 2, a + 0.015, f"{a:.2f}", ha="center", fontsize=9, color="#555")
    for xi, a in zip(x, late):
        ax.text(xi + w / 2, a + 0.015, f"{a:.2f}", ha="center", fontsize=9, color=GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    _style(ax, "Under preference pressure, surprise falls and skill rises", "", "value")
    fig.tight_layout()
    fig.savefig(OUT / "08_preference_emergence.png", bbox_inches="tight")
    plt.close(fig)


def page_role_dependency():
    spec = role_dependency_contrasts.role_dependency_contrasts_report_spec(
        include_assets=False
    )
    # The positive-control check returns the full variant x role matrix.
    variants = role_dependency_contrasts.role_dependency_contrast_variants()
    chk = spec.checks[0](spec.episode_factory())
    role_pass = chk["metrics"]  # variant -> {roles, raw}
    NUMBERS["role_dependency"] = {
        vid: vm["roles"] for vid, vm in role_pass.items()
    }
    variant_ids = [v.id for v in variants]
    role_ids = list(role_pass["positive-control"]["roles"].keys())
    grid = np.array(
        [[1 if role_pass[vid]["roles"][r] else 0 for vid in variant_ids] for r in role_ids]
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=140)
    ax.imshow(grid, cmap=plt.cm.RdYlGn, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(variant_ids)))
    ax.set_xticklabels([v.replace("-", "\n") for v in variant_ids], fontsize=9)
    ax.set_yticks(np.arange(len(role_ids)))
    ax.set_yticklabels([r.replace("_", " ") for r in role_ids], fontsize=9)
    for i in range(len(role_ids)):
        for j in range(len(variant_ids)):
            ax.text(
                j, i, "PASS" if grid[i, j] else "fail",
                ha="center", va="center",
                color="#114411" if grid[i, j] else "#771111",
                fontsize=9, fontweight="bold",
            )
    ax.set_title(
        "Each control preserves only the relations it actually implements",
        fontsize=13, fontweight="bold", loc="left", pad=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "09_role_dependency.png", bbox_inches="tight")
    plt.close(fig)


def page_topology_atlas():
    entries = topology_atlas.topology_atlas_entries(dt=0.2, fps=8)
    params = topology_atlas.topology_atlas_params()
    save_topology_atlas(entries, OUT / "10_topology_atlas.png", params)
    NUMBERS["topology_atlas"] = topology_atlas.topology_atlas_report_spec(
        include_assets=False
    ).checks[0](None)["metrics"]


# --------------------------------------------------------------------------- #
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    page_unseen_modality()
    page_attention_bottleneck()
    page_compression()
    page_expectation_violation()
    page_importance_weighted()
    page_valence()
    page_objective_shift()
    page_preference_emergence()
    page_role_dependency()
    page_topology_atlas()

    print("\n=== VERIFIED NUMBERS (for prose) ===")
    import json

    def _coerce(o):
        if isinstance(o, (np.floating, np.integer)):
            return round(float(o), 4)
        if isinstance(o, np.ndarray):
            return np.round(o, 4).tolist()
        return str(o)

    print(json.dumps(NUMBERS, default=_coerce, indent=1)[:4000])
    print("\npanels written to", OUT)


if __name__ == "__main__":
    main()
