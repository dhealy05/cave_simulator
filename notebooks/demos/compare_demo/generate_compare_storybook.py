"""Generate the static panels for the "Two Jimmys" comparison storybook.

This storybook sits in the comparison part of the map:

  primitive_demo/  Jimmy and the snake          (the kernel loop)
  main_demo/       Jimmy opens his eyes          (one full trajectory, six views)
  compare_demo/    Two Jimmys                     (comparing trajectories)

The lesson of Cave's Tutorial 2: *comparison is over trajectories, not
screenshots*. Two subjects with different dials walk the same wall of shapes;
we watch their trajectories diverge, isolate what each walk changed with a
matched baseline, and finally collapse whole walks to points whose distance is
how differently they experienced the same world.

Every panel reuses package APIs (run_subject, the standard views, the exported
embeddings, classical_mds). Run from the repository root:

    python notebooks/demos/compare_demo/generate_compare_storybook.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cave import (
    AttentionProfile,
    ExperienceObject,
    FeatureVector,
    InputSequence,
    MemoryParams,
    MemoryTrace,
    SubjectState,
    TemporalExtent,
    classical_mds,
    default_model_params,
    default_views,
    demo_sequence,
    episode_set,
    make_subject,
    run_subject,
    subjective_trajectory_embedding,
)
from cave.demonstrations.examples import DEFAULT_VOCABULARY
from cave.demonstrations.subjects.embeddings import no_input_memory_baseline
from cave.observation.structural import (
    episode_frames,
    memory_trajectory,
    structural_state_for_episode,
)
from cave.presentation.renderers.episode_set_dashboard import save_episode_set_dashboard
from cave.presentation.renderers.matplotlib_renderer.renderer import (
    LayoutSpec,
    MatplotlibRenderer,
)
from cave.presentation.renderers.topology_population_renderer import (
    save_topology_population_dashboard,
)
from cave.presentation.renderers.topology_surface_renderer import save_topology_state_surface
from cave.pressure.checks.population_trajectory_geometry import controlled_population_records

BASE = Path(__file__).resolve().parent
OUT = BASE / "storybook_assets"
TMP = BASE / "storybook_assets" / "_tmp"

COL = {
    "wide": "#0072B2",     # wide-eyed Jimmy
    "sleepy": "#D55E00",   # sleepy Jimmy
    "observed": "#0072B2",
    "baseline": "#9CA3AF",
    "effect": "#009E73",
    "grid": "#D1D5DB",
    "text": "#111827",
    "muted": "#6B7280",
}

SEQ = demo_sequence()
DURATION = SEQ.duration


# --------------------------------------------------------------------------- #
# Subjects
# --------------------------------------------------------------------------- #
def _params(retention: float, level: float, amplitude: float, tau: float, max_age: float):
    return replace(
        default_model_params(),
        memory=MemoryParams(retention=retention, decay_tau=tau, max_age=max_age),
        attention=AttentionProfile(mode="sine", level=level, amplitude=amplitude),
    )


def _subject(name, retention, level, amplitude, tau, max_age, prior=None):
    params = _params(retention, level, amplitude, tau, max_age)
    initial_state = None
    if prior is not None:
        trace = MemoryTrace(
            vector=np.asarray(prior, dtype=float),
            retention=params.memory.retention,
            decay_tau=params.memory.decay_tau,
            max_age=params.memory.max_age,
        )
        initial_state = SubjectState.initial(trace, params.topology)
    return make_subject(name, params=params, initial_state=initial_state)


# The two protagonists (from scratch).
WIDE = _subject("wide-eyed", retention=0.70, level=0.62, amplitude=0.38, tau=2.2, max_age=5.0)
SLEEPY = _subject("sleepy", retention=0.92, level=0.34, amplitude=0.24, tau=0.9, max_age=2.2)

# A triangle-shaped prior for the state-effect page ("already half-expects triangles").
TRIANGLE_PRIOR = 0.45 * np.array(
    [0.0, 0.62, 0.13, 0.52, 0.79, 1.0, 0.0, 0.85, 0.55], dtype=float
)
WIDE_PRIMED = _subject(
    "wide-eyed (primed)", retention=0.70, level=0.62, amplitude=0.38, tau=2.2, max_age=5.0,
    prior=TRIANGLE_PRIOR,
)

# A small family for the distance scatter.
FAMILY = [
    WIDE,
    _subject("wide-eyed twin", retention=0.68, level=0.60, amplitude=0.40, tau=2.1, max_age=5.0),
    _subject("balanced", retention=0.82, level=0.50, amplitude=0.50, tau=1.6, max_age=4.0),
    SLEEPY,
    _subject("very sleepy", retention=0.95, level=0.24, amplitude=0.18, tau=0.7, max_age=1.8),
]
FAMILY_COLORS = ["#0072B2", "#4C9BD0", "#7A7A7A", "#D55E00", "#A03C00"]


def _run(subject):
    return run_subject(SEQ, subject, dt=0.1)


def _surprise_series(run):
    obs = run.episode.observations
    return np.array([o.t for o in obs]), np.array([o.surprise for o in obs])


# --------------------------------------------------------------------------- #
# Shared style helpers
# --------------------------------------------------------------------------- #
def _shape_bands(ax):
    """Shade the four object intervals so the reader sees where shapes were."""
    bands = [(0.0, 1.2, "triangle"), (1.4, 2.7, "circle"), (2.9, 4.1, "square"), (4.4, 5.1, "gap")]
    lo, hi = ax.get_ylim()
    for start, end, label in bands:
        ax.axvspan(start, end, color="#000000", alpha=0.035, zorder=0)
        ax.text((start + end) / 2, lo + (hi - lo) * 0.03, label, ha="center", va="bottom",
                fontsize=8, color=COL["muted"])


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=COL["text"])
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, color=COL["grid"], linewidth=0.7, alpha=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _hstack(panel_paths, labels, output, height=540):
    """Paste single-view PNGs side by side with a label strip above each."""
    imgs = [Image.open(p).convert("RGB") for p in panel_paths]
    scaled = []
    for img in imgs:
        w = int(img.width * height / img.height)
        scaled.append(img.resize((w, height), Image.LANCZOS))
    pad, top = 16, 34
    total_w = sum(im.width for im in scaled) + pad * (len(scaled) + 1)
    canvas = Image.new("RGB", (total_w, height + top + pad), "white")
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    x = pad
    for im, label in zip(scaled, labels):
        canvas.paste(im, (x, top))
        draw.text((x + im.width // 2, top // 2), label, anchor="mm", fill=(17, 24, 39), font=font)
        x += im.width + pad
    canvas.save(output)


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page1_same_wall(wide_run):
    """One Presentation panel: both Jimmys see this exact wall."""
    frames = list(episode_frames(wide_run.episode, structural_state_for_episode(wide_run.episode)))
    frame = min(frames, key=lambda f: abs(f.observation.t - 0.6))
    by = {type(v).__name__: v for v in default_views()}
    MatplotlibRenderer(layout=LayoutSpec(columns=1, figsize_per_cell=(5.6, 4.4))).save_frame(
        frame, [by["PresentationView"]], OUT / "01_same_wall.png"
    )


def page2_surprise(wide_run, sleepy_run):
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=140)
    for run, key, name in ((wide_run, "wide", "Wide-eyed Jimmy"), (sleepy_run, "sleepy", "Sleepy Jimmy")):
        t, s = _surprise_series(run)
        ax.plot(t, s, color=COL[key], linewidth=2.4, label=name)
    ax.set_ylim(0, max(_surprise_series(wide_run)[1].max(), 0.3) * 1.18)
    _style(ax, "Same wall, two minds: surprise over time", "time (s)", "surprise")
    _shape_bands(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "02_surprise_curves.png")
    plt.close(fig)


def page3_circle(wide_run, sleepy_run):
    TMP.mkdir(parents=True, exist_ok=True)
    by = {type(v).__name__: v for v in default_views()}
    paths = []
    for run, tag in ((wide_run, "wide"), (sleepy_run, "sleepy")):
        frames = list(episode_frames(run.episode, structural_state_for_episode(run.episode)))
        frame = min(frames, key=lambda f: abs(f.observation.t - 1.5))
        p = TMP / f"circle_{tag}.png"
        MatplotlibRenderer(layout=LayoutSpec(columns=1, figsize_per_cell=(5.4, 4.4))).save_frame(
            frame, [by["ExpectationActualView"]], p
        )
        paths.append(p)
    _hstack(paths, ["Wide-eyed Jimmy", "Sleepy Jimmy"], OUT / "03_circle_side_by_side.png")


def page4_state_effect(primed_run):
    obs_t = np.array([o.t for o in primed_run.episode.observations])
    observed = memory_trajectory(primed_run.episode)
    baseline = no_input_memory_baseline(primed_run)
    effect = observed - baseline
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=140)
    ax.plot(obs_t, np.linalg.norm(observed, axis=1), color=COL["observed"], linewidth=2.4,
            label="observed memory (raw)")
    ax.plot(obs_t, np.linalg.norm(baseline, axis=1), color=COL["baseline"], linewidth=2.2,
            linestyle="--", label="baseline: prior decaying on a blank day")
    ax.plot(obs_t, np.linalg.norm(effect, axis=1), color=COL["effect"], linewidth=2.8,
            label="state effect = observed − baseline")
    _style(ax, "What did THIS walk change? (state-effect subtraction)", "time (s)", "memory magnitude")
    _shape_bands(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "04_state_effect.png")
    plt.close(fig)


def page5_distance(family_runs):
    embs = np.array([subjective_trajectory_embedding(r).reshape(-1) for r in family_runs])
    # pairwise euclidean distance, then classical MDS to 2-D
    diff = embs[:, None, :] - embs[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    coords = classical_mds(dist)
    fig, ax = plt.subplots(figsize=(7.6, 6.2), dpi=140)
    for (x, y), run, color in zip(coords, family_runs, FAMILY_COLORS):
        ax.scatter([x], [y], s=180, color=color, edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(run.subject.id, (x, y), textcoords="offset points", xytext=(10, 6),
                    fontsize=10, color=COL["text"])
    ax.set_title("Each whole walk is a point — near means similar experience",
                 loc="left", fontsize=12.5, fontweight="bold", color=COL["text"])
    ax.set_xlabel("trajectory similarity axis 1", fontsize=10)
    ax.set_ylabel("trajectory similarity axis 2", fontsize=10)
    ax.margins(0.18)
    ax.grid(True, color=COL["grid"], linewidth=0.7, alpha=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "05_distance_scatter.png", bbox_inches="tight")
    plt.close(fig)


def page_surfaces(wide_run, sleepy_run):
    """Topology state surfaces: each subject's inner landscape over the same wall."""
    save_topology_state_surface(wide_run.episode, OUT / "06_surface_wide.png")
    save_topology_state_surface(sleepy_run.episode, OUT / "06_surface_sleepy.png")


def page_dashboard(family_runs):
    """The canonical comparison artifact: three embeddings, scatter + distance matrices."""
    runs = [r for r in family_runs if r.subject.id in ("wide-eyed", "balanced", "sleepy")]
    es = episode_set(
        [r.as_labeled_episode(label=r.subject.id) for r in runs],
        id="three_subjects",
        title="Same wall, three subjects",
        comparison_axis="subject configuration",
    )
    save_episode_set_dashboard(es, OUT / "07_episode_set_dashboard.png",
                               title="The comparison tool: three embeddings at once")


# ---- Act II: same subject, different worlds ----------------------------------
_OBJS = {o.id: o.features.values for o in demo_sequence().objects}


def _world(name, feature_dicts):
    objs = []
    t = 0.0
    for i, feats in enumerate(feature_dicts):
        objs.append(ExperienceObject(
            id=f"{name}_{i}",
            temporal_extent=TemporalExtent(start=t, end=t + 1.0, order_index=i),
            features=FeatureVector(dict(feats)),
            salience=0.85,
        ))
        t += 1.2
    return InputSequence(objects=objs)


# One fixed subject for the world comparison (constant attention so a held
# expectation produces a clean violation spike).
FIXED = make_subject(
    "fixed Jimmy",
    params=replace(default_model_params(),
                   memory=MemoryParams(retention=0.60, decay_tau=2.2, max_age=6.0),
                   attention=AttentionProfile(mode="constant", level=0.85, amplitude=0.0)),
)
WORLDS = {
    "repeat, then violate": [_OBJS["evt_triangle"]] * 4 + [_OBJS["evt_circle"]],
    "ever-changing": [_OBJS["evt_triangle"], _OBJS["evt_circle"], _OBJS["evt_square"],
                      _OBJS["evt_circle"], _OBJS["evt_triangle"]],
}
WORLD_COLORS = {"repeat, then violate": "#0072B2", "ever-changing": "#D55E00"}


def page_worlds(output: Path):
    fig, ax = plt.subplots(figsize=(9.8, 4.6), dpi=140)
    for name, feats in WORLDS.items():
        run = run_subject(_world(name, feats), FIXED, dt=0.1)
        obs = run.episode.observations
        t = np.array([o.t for o in obs]); s = np.array([o.surprise for o in obs])
        ax.plot(t, s, color=WORLD_COLORS[name], linewidth=2.4, label=name)
    # mark the five object slots
    for i in range(5):
        ax.axvspan(i * 1.2, i * 1.2 + 1.0, color="#000000", alpha=0.03, zorder=0)
    ax.set_ylim(0, 0.30)
    _style(ax, "Same fixed Jimmy, two different worlds", "time (s)", "surprise")
    ax.annotate("expectation held...\nthen violated", (4.9, 0.235), xytext=(3.0, 0.265),
                fontsize=9.5, color="#0072B2",
                arrowprops=dict(arrowstyle="->", color="#0072B2"))
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def page_population(output: Path):
    records = [r.to_population_record() for r in controlled_population_records(
        treatment_count=2, start_count=3, event_count=5, seed=11, dt=1.0, end=None)]
    save_topology_population_dashboard(
        records, output, default_model_params().topology,
        title="A labelled population: treatments x starts x control conditions")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    wide_run, sleepy_run = _run(WIDE), _run(SLEEPY)
    primed_run = _run(WIDE_PRIMED)
    family_runs = [_run(s) for s in FAMILY]

    page1_same_wall(wide_run)
    page2_surprise(wide_run, sleepy_run)
    page3_circle(wide_run, sleepy_run)
    page4_state_effect(primed_run)
    page5_distance(family_runs)
    page_surfaces(wide_run, sleepy_run)
    page_dashboard(family_runs)
    page_worlds(OUT / "08_worlds_surprise.png")
    page_population(OUT / "09_population_dashboard.png")

    # cleanup temp
    if TMP.exists():
        for p in TMP.iterdir():
            p.unlink()
        TMP.rmdir()

    # report numbers for the prose
    def s_at(run, t):
        return min(run.episode.observations, key=lambda o: abs(o.t - t)).surprise
    print("wide-eyed  surprise@circle =", round(s_at(wide_run, 1.5), 3),
          "peak =", round(max(o.surprise for o in wide_run.episode.observations), 3))
    print("sleepy     surprise@circle =", round(s_at(sleepy_run, 1.5), 3),
          "peak =", round(max(o.surprise for o in sleepy_run.episode.observations), 3))
    print("pages written to", OUT)


if __name__ == "__main__":
    main()
