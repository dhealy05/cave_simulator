"""Generate the static panels for the Two Inks topology storybook.

This is the sibling of ``notebooks/demos/main_demo`` (the full-system
storybook) and ``notebooks/demos/primitive_demo`` (the kernel loop), but it
refuses to unify: it explains exactly one object -- the subjective topology --
from first principles, one ingredient per page.

The walk is the interactive game's canonical route (tree, rock, tree, snake)
run through the real game machinery (``build_sequence`` + ``make_game_subject``
+ ``run_subject``), and the field is built by the real engine
(``SubjectiveTopologyState.update``). The projection is the game's feature
plane (novelty x angularity) with a flat prior, so every page matches what the
trajectory-space game shows and what the topology atlas measures.

Pages 2-4 drive the state by hand (one stamp at a time, transitions off) so a
single deposit can be watched aging. Pages 5-10 use the full episode, ending on
the atlas split: experienced field, generated ink, sensed ink, and the
world-minus-dream difference.

Run from the repository root:

    python notebooks/demos/topology_demo/generate_topology_storybook.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch

from cave.commitments.memory.trace import MemoryTrace
from cave.commitments.topology import (
    SubjectiveTopologyParams,
    SubjectiveTopologyPrior,
    SubjectiveTopologyState,
)
from cave.demonstrations.examples import DEFAULT_VOCABULARY, default_model_params
from cave.demonstrations.subjects.runs import run_subject
from cave.interactive.game.core.subjects import make_game_subject
from cave.interactive.game.core.world import GAME_OBJECTS, build_sequence
from cave.observation.structural import structural_state_for_episode
from cave.presentation.renderers.topology_atlas_renderer import _atlas_metrics

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "storybook_assets"

#: The game's canonical route: calm, dull, calm again, then the lurch.
WALK = ("tree", "rock", "tree", "snake")
#: Full flat attention -- every term in the update rule is wide open and
#: visible. The contrast subject shows what the gate does when it half-closes.
NARRATOR = "excited"
CONTRAST = "sleepy"

ACCENTS = {"tree": "#16885f", "rock": "#7c7f86", "snake": "#c93e6d"}
#: The trajectory-space game's terrain transformation, verbatim
#: (cave/interactive/game/tutorials/trajectory_space/static/trajectory_space_app.js).
GAME_GAMMA = 0.6
GAME_THRESHOLDS = (0.06, 0.18, 0.34, 0.55, 0.78)
GAME_BAND_COLORS = ("#142a20", "#1d4733", "#16885f", "#3fae7e", "#f2b84b")
FIG_BG = "#f7f4ef"
PANEL_BG = "#111827"
SPINE = "#2f3a4a"
CREAM = "#f4f0df"
INK = "#24303f"
THREAD = "#129b63"


def book_params() -> SubjectiveTopologyParams:
    """The engine's topology params on the game's plane, with a flat prior."""

    return replace(
        default_model_params().topology,
        feature_x="novelty",
        feature_y="angularity",
        prior=SubjectiveTopologyPrior(),
    )


def fresh_state(params: SubjectiveTopologyParams) -> SubjectiveTopologyState:
    return SubjectiveTopologyState.initial(
        feature_x=params.feature_x,
        feature_y=params.feature_y,
        bounds=params.bounds,
        resolution=params.resolution,
        prior=params.prior,
    )


def empty_memory() -> MemoryTrace:
    return MemoryTrace(vector=np.zeros(len(DEFAULT_VOCABULARY)))


def landmark_centers(params: SubjectiveTopologyParams) -> dict[str, np.ndarray]:
    state = fresh_state(params)
    sequence = build_sequence(tuple(GAME_OBJECTS))
    return {
        str(obj.metadata["symbol"]): state.center_for_object(
            obj, params.feature_x, params.feature_y
        )
        for obj in sequence.objects
    }


# --- Shared drawing ---------------------------------------------------------


def density_panel(
    axis,
    density: np.ndarray,
    params: SubjectiveTopologyParams,
    *,
    vmax: float,
    vmin: float = 0.0,
    cmap: str = "magma",
    title: str | None = None,
    landmarks: dict[str, np.ndarray] | None = None,
    peak: bool = True,
) -> None:
    lower, upper = params.bounds
    axis.imshow(
        density,
        origin="lower",
        extent=(lower, upper, lower, upper),
        cmap=cmap,
        vmin=vmin,
        vmax=max(vmax, 1e-9),
        interpolation="bilinear",
        aspect="equal",
    )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor(PANEL_BG)
    for spine in axis.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)
    if title:
        axis.set_title(title, fontsize=8.5, fontweight="bold", color=INK, pad=6)
    if landmarks:
        for symbol, center in landmarks.items():
            axis.scatter(
                [center[0]], [center[1]],
                s=26, color=ACCENTS[symbol],
                edgecolors=CREAM, linewidths=0.7, zorder=5,
            )
            axis.annotate(
                GAME_OBJECTS[symbol].label,
                (center[0], center[1]),
                xytext=(0, -11), textcoords="offset points",
                ha="center", fontsize=6.5, color=CREAM, alpha=0.85,
            )
    if peak:
        axis.text(
            0.04, 0.95, f"peak {float(np.max(density)):.2f}",
            transform=axis.transAxes, ha="left", va="top",
            fontsize=7, color=CREAM, alpha=0.9,
        )


def book_figure(ncols: int, *, nrows: int = 1, cell: float = 3.1, top_pad: float = 0.78):
    figure, axes = plt.subplots(
        nrows, ncols,
        figsize=(cell * ncols + 0.5, cell * nrows + top_pad),
        facecolor=FIG_BG,
    )
    return figure, np.atleast_2d(axes)


def caption(figure, text: str) -> None:
    figure.text(
        0.5, 0.012, text,
        ha="center", va="bottom", fontsize=8, color="#475467", style="italic",
    )


def save(figure, slug: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{slug}.png"
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor=FIG_BG, pad_inches=0.12)
    plt.close(figure)
    print(f"wrote {path.relative_to(BASE)}")


def axis_arrows(axis, params: SubjectiveTopologyParams) -> None:
    axis.set_xlabel("novelty →", fontsize=8, color=INK, loc="right")
    axis.set_ylabel("↑ angularity", fontsize=8, color=INK, loc="top", rotation=0, labelpad=2)


# --- Episode plumbing -------------------------------------------------------


def run_walk(subject_id: str):
    sequence = build_sequence(WALK)
    subject = make_game_subject(subject_id)
    run = run_subject(sequence, subject, run_id=f"{subject_id}:{'-'.join(WALK)}")
    return sequence, run.episode


def topology_frames(episode, params: SubjectiveTopologyParams):
    structural = structural_state_for_episode(episode, topology_params=params)
    return structural.topology_frames


def frame_nearest(frames, t: float):
    return min(frames, key=lambda frame: abs(frame.t - t))


def active_windows(sequence) -> list[tuple[str, float, float]]:
    return [
        (
            str(obj.metadata["symbol"]),
            obj.temporal_extent.start,
            obj.temporal_extent.end,
        )
        for obj in sequence.objects
    ]


def encounter_points(sequence, frames):
    """One pick per object: peak surprise in its window, and where memory
    stood when the encounter ended."""

    picks = []
    for symbol, start, end in active_windows(sequence):
        in_window = [
            frame for frame in frames
            if frame.correction is not None and start <= frame.t < end
        ]
        if not in_window:
            continue
        peak = max(in_window, key=lambda frame: frame.correction.surprise)
        last = max(in_window, key=lambda frame: frame.t)
        picks.append((symbol, peak.correction.surprise, last.correction.after_point))
    return picks


def split_fields(params: SubjectiveTopologyParams, frames) -> dict[str, np.ndarray]:
    """The engine's own ledger at the end of the walk, split by ink.

    The atlas builds counterfactual point-stream densities; the book instead
    reads the tagged fields the state already carries, so the split matches
    the sensed/generated inks shown on every earlier page.
    """

    final = frames[-1].topology
    initial = fresh_state(params)
    return {
        "experienced": np.clip(
            np.asarray(final.density) - np.asarray(initial.density), 0.0, None
        ),
        "generated": np.asarray(final.expected_density),
        "sensed": np.asarray(final.actual_density),
    }


# --- Pages ------------------------------------------------------------------


def page_01_the_plane(params: SubjectiveTopologyParams) -> None:
    landmarks = landmark_centers(params)
    figure = plt.figure(figsize=(6.8, 6.4), facecolor=FIG_BG)
    grid = figure.add_gridspec(
        2, 1, height_ratios=(0.22, 1.0), hspace=0.24,
        left=0.09, right=0.95, top=0.9, bottom=0.07,
    )

    timeline = figure.add_subplot(grid[0])
    timeline.set_xlim(0.4, 4.9)
    timeline.set_ylim(-1.0, 1.4)
    timeline.set_axis_off()
    timeline.annotate(
        "", xy=(4.85, 0.0), xytext=(0.5, 0.0),
        arrowprops={"arrowstyle": "-|>", "color": "#98a2b3", "linewidth": 1.1},
    )
    timeline.text(4.85, -0.7, "time", ha="right", fontsize=8, color="#667085")
    slots = []
    for index, symbol in enumerate(WALK):
        x = index + 1.0
        slots.append((x, symbol))
        timeline.scatter([x], [0.0], s=90, color=ACCENTS[symbol], zorder=5,
                         edgecolors="white", linewidths=1.0)
        timeline.text(x, 0.75, GAME_OBJECTS[symbol].label, ha="center",
                      fontsize=8.5, fontweight="bold", color=INK)
    timeline.set_title(
        "The walk, as the world hands it over: an order, nothing else",
        fontsize=10, fontweight="bold", color=INK, pad=10,
    )

    plane = figure.add_subplot(grid[1])
    empty = np.zeros((params.resolution, params.resolution))
    density_panel(
        plane, empty, params, vmax=1.0, peak=False,
        landmarks=landmarks,
        title="The same walk, re-plotted by what things are",
    )
    axis_arrows(plane, params)
    tree = landmarks["tree"]
    plane.annotate(
        "visits 1 and 3 land on the same spot",
        (tree[0], tree[1]), xytext=(14, 26), textcoords="offset points",
        fontsize=7, color=CREAM, alpha=0.9,
        arrowprops={"arrowstyle": "-", "color": CREAM, "alpha": 0.5, "linewidth": 0.7},
    )
    for x, symbol in slots:
        center = landmarks[symbol]
        figure.add_artist(
            ConnectionPatch(
                xyA=(x, -0.55), coordsA=timeline.transData,
                xyB=(center[0], center[1]), coordsB=plane.transData,
                color=ACCENTS[symbol], alpha=0.45, linewidth=0.9, linestyle=":",
            )
        )
    caption(figure, "In time the two trees are far apart. In here, places are kinds -- they are the same place.")
    save(figure, "01_the_plane")


def page_02_one_stamp(params: SubjectiveTopologyParams) -> tuple[SubjectiveTopologyState, float]:
    quiet_params = replace(params, transition_strength=0.0)
    landmarks = landmark_centers(params)
    tree = build_sequence(("tree",)).objects[0]
    before = fresh_state(quiet_params)
    after = before.update(empty_memory(), [tree], quiet_params, current_attention=1.0)
    vmax = float(np.max(after.density))

    figure, axes = book_figure(2)
    density_panel(
        axes[0][0], before.density, params, vmax=vmax,
        landmarks={"tree": landmarks["tree"]}, peak=False,
        title="Before",
    )
    density_panel(
        axes[0][1], after.density, params, vmax=vmax,
        landmarks={"tree": landmarks["tree"]},
        title="After one update",
    )
    figure.suptitle("One experience, one deposit", fontsize=11, fontweight="bold", color=INK)
    caption(figure, "deposit = attention x salience x 0.22, laid as a soft disc (width 0.18). The moment ends; the mark stays.")
    save(figure, "02_one_stamp")
    return after, vmax


def page_03_the_stamp_ages(
    params: SubjectiveTopologyParams,
    stamped: SubjectiveTopologyState,
    vmax: float,
) -> None:
    quiet_params = replace(params, transition_strength=0.0)
    landmarks = {"tree": landmark_centers(params)["tree"]}
    snapshots = {0: stamped}
    state = stamped
    for step in range(1, 11):
        state = state.update(empty_memory(), [], quiet_params, current_attention=1.0)
        if step in (4, 10):
            snapshots[step] = state

    figure, axes = book_figure(3)
    for axis, (steps, snapshot) in zip(axes[0], sorted(snapshots.items())):
        density_panel(
            axis, snapshot.density, params, vmax=vmax, landmarks=landmarks,
            title="just stamped" if steps == 0 else f"{steps} quiet steps later",
        )
    figure.suptitle("Nothing else happens, and still the field changes", fontsize=11, fontweight="bold", color=INK)
    caption(figure, "Each step the field decays (x0.94) and diffuses (0.18). Forgetting is not a deletion -- it is sag and blur.")
    save(figure, "03_the_stamp_ages")


def page_04_stamps_pile(params: SubjectiveTopologyParams) -> None:
    quiet_params = replace(params, transition_strength=0.0)
    landmarks = {"tree": landmark_centers(params)["tree"]}
    tree = build_sequence(("tree",)).objects[0]
    state = fresh_state(quiet_params)
    snapshots = {}
    for visit in range(1, 5):
        state = state.update(empty_memory(), [tree], quiet_params, current_attention=1.0)
        if visit in (1, 2, 4):
            snapshots[visit] = state
        for _ in range(2):
            state = state.update(empty_memory(), [], quiet_params, current_attention=1.0)
    vmax = max(float(np.max(s.density)) for s in snapshots.values())

    figure, axes = book_figure(3)
    for axis, (visit, snapshot) in zip(axes[0], sorted(snapshots.items())):
        density_panel(
            axis, snapshot.density, params, vmax=vmax, landmarks=landmarks,
            title=f"visit {visit}" if visit > 1 else "visit 1",
        )
    figure.suptitle("Like lands on like: a hill is a habit", fontsize=11, fontweight="bold", color=INK)
    caption(figure, "Repeats of the same kind stack in the same place. The field keeps the statistics of the walk, not its sequence.")
    save(figure, "04_stamps_pile")


def page_05_two_inks(params: SubjectiveTopologyParams, sequence, frames) -> None:
    landmarks = landmark_centers(params)
    second_tree = sequence.objects[2]
    frame = frame_nearest(frames, second_tree.temporal_extent.end)
    state = frame.topology

    figure, axes = book_figure(2)
    density_panel(
        axes[0][0], state.expected_density, params,
        vmax=float(np.max(state.expected_density)), landmarks=landmarks,
        title="Generated ink -- where the dream stood",
    )
    lower, _ = params.bounds
    axes[0][0].annotate(
        "the dream begins at\n'nothing yet'",
        (lower + 0.18, lower + 0.18), xytext=(26, 52), textcoords="offset points",
        fontsize=6.5, color=CREAM, alpha=0.9,
        arrowprops={"arrowstyle": "-", "color": CREAM, "alpha": 0.5, "linewidth": 0.7},
    )
    density_panel(
        axes[0][1], state.actual_density, params,
        vmax=float(np.max(state.actual_density)), landmarks=landmarks,
        title="Sensed ink -- where the world landed",
    )
    figure.suptitle("Two inks on one map (mid-walk, after the second tree)", fontsize=11, fontweight="bold", color=INK)
    caption(figure, "The dream stamps every step at 0.14 -- everywhere and thin, starting at the empty corner. The world stamps only what attention admits, at 0.22 x salience -- rare and heavy.")
    save(figure, "05_two_inks")


def page_06_roads(params: SubjectiveTopologyParams, episode) -> None:
    landmarks = landmark_centers(params)
    with_roads = topology_frames(episode, params)[-1].topology.actual_density
    no_roads = topology_frames(
        episode, replace(params, transition_strength=0.0)
    )[-1].topology.actual_density
    roads = np.clip(np.asarray(with_roads) - np.asarray(no_roads), 0.0, None)
    vmax = float(np.max(with_roads))

    figure, axes = book_figure(3)
    density_panel(
        axes[0][0], no_roads, params, vmax=vmax, landmarks=landmarks,
        title="Stamps only (transitions switched off)",
    )
    density_panel(
        axes[0][1], with_roads, params, vmax=vmax, landmarks=landmarks,
        title="The real sensed ink",
    )
    density_panel(
        axes[0][2], roads, params, vmax=float(np.max(roads)), landmarks=landmarks,
        title="The difference: roads",
    )
    figure.suptitle("Attention wears paths between the places it visits", fontsize=11, fontweight="bold", color=INK)
    caption(figure, "Each shift of focus lays a faint ridge from the last attended place to the next: desire paths, carved by the order of attention.")
    save(figure, "06_roads")


def page_07_the_gatekeeper(params: SubjectiveTopologyParams) -> None:
    landmarks = landmark_centers(params)
    fields = {}
    for subject_id in (CONTRAST, NARRATOR):
        _, episode = run_walk(subject_id)
        fields[subject_id] = topology_frames(episode, params)[-1].topology.actual_density
    vmax = max(float(np.max(field)) for field in fields.values())

    figure, axes = book_figure(2)
    density_panel(
        axes[0][0], fields[CONTRAST], params, vmax=vmax, landmarks=landmarks,
        title="Sleepy -- attention 0.3\nthe world barely lands",
    )
    density_panel(
        axes[0][1], fields[NARRATOR], params, vmax=vmax, landmarks=landmarks,
        title="Excited -- attention 1.0\neverything lands hard",
    )
    figure.suptitle("Same walk, same objects -- the gate differs", fontsize=11, fontweight="bold", color=INK)
    caption(figure, "Attention multiplies the world's ink. The sensed country is not what happened; it is what was admitted.")
    save(figure, "07_the_gatekeeper")


def page_08_field_and_thread(params: SubjectiveTopologyParams, sequence, frames) -> None:
    landmarks = landmark_centers(params)
    final = frames[-1].topology.density
    vmax = float(np.max(final))
    thread = [
        frame.correction.after_point
        for frame in frames
        if frame.correction is not None
    ]
    first_expected = next(
        frame.correction.expected_point
        for frame in frames
        if frame.correction is not None
    )
    thread = np.array([first_expected, *thread])
    picks = encounter_points(sequence, frames)

    figure, axes = book_figure(2, cell=3.4)
    density_panel(
        axes[0][0], final, params, vmax=vmax, landmarks=landmarks,
        title="The topology: a terrain",
    )
    density_panel(
        axes[0][1], final, params, vmax=vmax, landmarks=landmarks, peak=False,
        title="The trajectory: a thread over it",
    )
    axes[0][1].plot(thread[:, 0], thread[:, 1], color=THREAD, linewidth=1.6, alpha=0.95, zorder=6)
    offsets = ((9, -3), (9, -13), (-13, 5), (9, 5))
    for index, (symbol, _, point) in enumerate(picks):
        axes[0][1].scatter([point[0]], [point[1]], s=26, color=ACCENTS[symbol],
                           edgecolors=CREAM, linewidths=0.7, zorder=7)
        dx, dy = offsets[index % len(offsets)]
        axes[0][1].annotate(
            str(index + 1),
            (point[0], point[1]), xytext=(dx, dy), textcoords="offset points",
            ha="right" if dx < 0 else "left",
            fontsize=7, fontweight="bold", color=CREAM,
        )
    legend = "\n".join(
        f"{index + 1}  {GAME_OBJECTS[symbol].label:<6} s={surprise:.2f}"
        for index, (symbol, surprise, _) in enumerate(picks)
    )
    axes[0][1].text(
        0.04, 0.95, legend,
        transform=axes[0][1].transAxes, ha="left", va="top",
        fontsize=6.5, color=CREAM, family="monospace", linespacing=1.5,
    )
    figure.suptitle("One episode, two readings: where memory walked, what it carved", fontsize=11, fontweight="bold", color=INK)
    caption(figure, "The thread is the path of memory states; step length is surprise. Notice it never arrives at the things themselves -- the lived path stays in the in-between.")
    save(figure, "08_field_and_thread")


def page_09_the_split(params: SubjectiveTopologyParams, frames) -> None:
    landmarks = landmark_centers(params)
    fields = split_fields(params, frames)
    delta = fields["sensed"] - fields["generated"]
    vmax = float(np.max(fields["experienced"]))
    delta_max = float(np.max(np.abs(delta)))

    figure, axes = book_figure(4, cell=2.9)
    density_panel(
        axes[0][0], fields["experienced"], params, vmax=vmax, landmarks=landmarks,
        title="The experienced field",
    )
    density_panel(
        axes[0][1], fields["generated"], params, vmax=vmax, landmarks=landmarks,
        title="The generated ink alone",
    )
    density_panel(
        axes[0][2], fields["sensed"], params, vmax=vmax, landmarks=landmarks,
        title="The sensed ink alone",
    )
    density_panel(
        axes[0][3], delta, params, vmax=delta_max, vmin=-delta_max,
        cmap="RdBu_r", landmarks=landmarks, peak=False,
        title="World minus dream",
    )
    axes[0][3].text(
        0.04, 0.95, "red: surprise\nblue: disappointment",
        transform=axes[0][3].transAxes, ha="left", va="top",
        fontsize=6.5, color=INK,
    )
    figure.suptitle("The split: one field, peeled into its parents", fontsize=11, fontweight="bold", color=INK)
    caption(figure, "Red is world without dream -- the snake hill. Blue is dream without world -- expectations nothing arrived to confirm.")
    save(figure, "09_the_split")


def page_10_two_countries(params: SubjectiveTopologyParams) -> None:
    landmarks = landmark_centers(params)
    rows = []
    for subject_id, label in ((CONTRAST, "Sleepy"), (NARRATOR, "Excited")):
        _, episode = run_walk(subject_id)
        fields = split_fields(params, topology_frames(episode, params))
        metrics = _atlas_metrics(
            experienced_delta=fields["experienced"],
            expected_density=fields["generated"],
            actual_density=fields["sensed"],
            params=params,
        )
        rows.append((
            label,
            fields["experienced"],
            fields["sensed"] - fields["generated"],
            metrics,
        ))

    vmax = max(float(np.max(experienced)) for _, experienced, _, _ in rows)
    delta_max = max(float(np.max(np.abs(delta))) for _, _, delta, _ in rows)

    figure, axes = book_figure(3, nrows=2, cell=2.9)
    for row, (label, experienced, delta, metrics) in enumerate(rows):
        density_panel(
            axes[row][0], experienced, params, vmax=vmax, landmarks=landmarks,
            title=f"{label}: the experienced country" if row == 0 else None,
        )
        axes[row][0].set_ylabel(label, fontsize=10, fontweight="bold", color=INK)
        density_panel(
            axes[row][1], delta, params, vmax=delta_max, vmin=-delta_max,
            cmap="RdBu_r", landmarks=landmarks, peak=False,
            title="world minus dream" if row == 0 else None,
        )
        metric_axis = axes[row][2]
        metric_axis.set_axis_off()
        metric_axis.text(
            0.05, 0.5,
            (
                f"mass    {metrics['experienced_mass']:.2f}\n"
                f"peak    {metrics['experienced_peak']:.2f}\n"
                f"center  {metrics['experienced_centroid_x']:.2f}, "
                f"{metrics['experienced_centroid_y']:.2f}\n"
                f"spread  {metrics['experienced_spread']:.2f}\n"
                f"AE delta {metrics['actual_expected_l2']:.2f}"
            ),
            ha="left", va="center", fontsize=9, color=INK, family="monospace",
            linespacing=1.6,
        )
        if row == 0:
            metric_axis.set_title("measured", fontsize=9, fontweight="bold", color=INK, pad=6)
    figure.suptitle("Same walk, different countries -- and countries can be compared", fontsize=11, fontweight="bold", color=INK)
    caption(figure, "Two diaries cannot be subtracted. Two fields can: mass, peak, center, spread, and how far the world ran from the dream.")
    save(figure, "10_two_countries")
    for label, _, _, metrics in rows:
        print(
            f"  {label}: mass={metrics['experienced_mass']:.2f} "
            f"peak={metrics['experienced_peak']:.2f} "
            f"spread={metrics['experienced_spread']:.2f} "
            f"AE={metrics['actual_expected_l2']:.2f}"
        )


def band_grid(density: np.ndarray, global_max: float) -> np.ndarray:
    """The game's height rule: normalize, soften with gamma, cut into bands."""

    t = np.power(np.clip(density / max(global_max, 1e-9), 0.0, 1.0), GAME_GAMMA)
    bands = np.zeros(t.shape, dtype=int)
    for threshold in GAME_THRESHOLDS:
        bands += (t >= threshold).astype(int)
    return bands


def page_11_from_ink_to_terrain(params: SubjectiveTopologyParams, frames) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    landmarks = landmark_centers(params)
    field = split_fields(params, frames)["experienced"]
    global_max = float(np.max(field))
    bands = band_grid(field, global_max)

    figure = plt.figure(figsize=(10.6, 4.1), facecolor=FIG_BG)
    grid = figure.add_gridspec(
        1, 3, width_ratios=(1.0, 1.0, 1.25),
        left=0.03, right=0.99, top=0.82, bottom=0.06, wspace=0.12,
    )

    flat = figure.add_subplot(grid[0])
    density_panel(
        flat, field, params, vmax=global_max, landmarks=landmarks,
        title="The ink (the finished field)",
    )

    banded = figure.add_subplot(grid[1])
    lower, upper = params.bounds
    banded.imshow(
        bands,
        origin="lower",
        extent=(lower, upper, lower, upper),
        cmap=ListedColormap((PANEL_BG, *GAME_BAND_COLORS)),
        norm=BoundaryNorm(np.arange(-0.5, len(GAME_THRESHOLDS) + 1), len(GAME_THRESHOLDS) + 1),
        interpolation="nearest",
        aspect="equal",
    )
    banded.set_xticks([])
    banded.set_yticks([])
    for spine in banded.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)
    banded.set_title(
        "Normalize, gamma 0.6, cut at\n0.06 / 0.18 / 0.34 / 0.55 / 0.78",
        fontsize=8.5, fontweight="bold", color=INK, pad=6,
    )
    for symbol, center in landmarks.items():
        banded.scatter([center[0]], [center[1]], s=22, color=ACCENTS[symbol],
                       edgecolors=CREAM, linewidths=0.7, zorder=5)

    # The 3D panel repeats the game finale: terraces raised one step per band,
    # the memory thread draped on top.
    relief = figure.add_subplot(grid[2], projection="3d")
    # Draw in add-order so the thread and landmarks sit on the terrain, the
    # way the game's finale ribbon does.
    relief.computed_zorder = False
    half = field.shape[0] // 2
    coarse = field[: half * 2, : half * 2].reshape(half, 2, half, 2).mean(axis=(1, 3))
    coarse_bands = band_grid(coarse, global_max)
    n = coarse_bands.shape[0]
    cell = (upper - lower) / n
    unit = 0.12
    elev_deg, azim_deg = 30, -58
    view = np.array([np.cos(np.radians(azim_deg)), np.sin(np.radians(azim_deg))])
    columns = sorted(
        (
            (lower + gx * cell, lower + gy * cell, int(coarse_bands[gy, gx]))
            for gy in range(n)
            for gx in range(n)
            if coarse_bands[gy, gx] > 0
        ),
        # Painter order for the fixed camera: farthest along the view first.
        key=lambda column: column[0] * view[0] + column[1] * view[1],
    )
    for x, y, band in columns:
        relief.bar3d(
            x, y, 0.0, cell, cell, band * unit,
            color=GAME_BAND_COLORS[band - 1],
            edgecolor="none", shade=True,
        )

    def height_at(point: np.ndarray) -> float:
        gx = int(np.clip((point[0] - lower) / cell, 0, n - 1))
        gy = int(np.clip((point[1] - lower) / cell, 0, n - 1))
        return float(coarse_bands[gy, gx]) * unit

    thread = np.array([
        frame.correction.after_point
        for frame in frames
        if frame.correction is not None
    ])
    relief.plot3D(
        thread[:, 0], thread[:, 1],
        [height_at(point) + 0.05 for point in thread],
        color=CREAM, linewidth=2.2, alpha=0.95,
    )
    for symbol, center in landmarks.items():
        z = height_at(center)
        relief.scatter([center[0]], [center[1]], [z + 0.05], s=20,
                       color=ACCENTS[symbol], edgecolors=CREAM, linewidths=0.6)
        relief.text(center[0], center[1], z + 0.22, GAME_OBJECTS[symbol].label,
                    fontsize=6.5, color=INK, ha="center")
    border = np.array([
        (lower, lower), (upper, lower), (upper, upper), (lower, upper), (lower, lower)
    ])
    relief.plot3D(border[:, 0], border[:, 1], np.zeros(5),
                  color="#98a2b3", linewidth=0.8, alpha=0.7)
    relief.set_box_aspect((1, 1, 0.32))
    relief.view_init(elev=elev_deg, azim=azim_deg)
    relief.set_axis_off()
    relief.set_title(
        "Raise one step per band:\nthe game's terrain",
        fontsize=8.5, fontweight="bold", color=INK, pad=0,
    )

    figure.suptitle(
        "From ink to terrain: the third dimension is the ink, stood up",
        fontsize=11, fontweight="bold", color=INK,
    )
    caption(figure, "height = (density / max) ^ 0.6, quantized into five terraces -- the exact rule the trajectory-space game draws. No new information: only relief.")
    save(figure, "11_from_ink_to_terrain")


def main() -> None:
    params = book_params()
    sequence, episode = run_walk(NARRATOR)
    frames = topology_frames(episode, params)
    picks = encounter_points(sequence, frames)
    print("encounters (peak surprise per stop):")
    for symbol, surprise, _ in picks:
        print(f"  {symbol}: {surprise:.3f}")

    page_01_the_plane(params)
    stamped, vmax = page_02_one_stamp(params)
    page_03_the_stamp_ages(params, stamped, vmax)
    page_04_stamps_pile(params)
    page_05_two_inks(params, sequence, frames)
    page_06_roads(params, episode)
    page_07_the_gatekeeper(params)
    page_08_field_and_thread(params, sequence, frames)
    page_09_the_split(params, frames)
    page_10_two_countries(params)
    page_11_from_ink_to_terrain(params, frames)


if __name__ == "__main__":
    main()
