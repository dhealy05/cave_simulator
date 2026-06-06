from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, Rectangle

from cave.presentation.renderers.matplotlib_renderer.styles import (
    RendererStyle,
    apply_axis_style,
    apply_figure_style,
    resolve_style,
)
from cave.pressure.checks.role_recovery_matrix import check_role_recovery_matrix


ROLE_ORDER = (
    "expectation",
    "selection",
    "value_retention",
    "regulation",
    "topology",
)

ROLE_LABELS = {
    "expectation": "Expectation",
    "selection": "Selection",
    "value_retention": "Value retention",
    "regulation": "Regulation",
    "topology": "Topology geometry",
}

ROLE_COLORS = {
    "expectation": "#2563eb",
    "selection": "#2f855a",
    "value_retention": "#b45309",
    "regulation": "#7c3aed",
    "topology": "#c2410c",
}

ROLE_METRIC_LABELS = {
    "expectation": "surprise drop / future readout",
    "selection": "diagnostic preservation",
    "value_retention": "value-shaped hidden retention",
    "regulation": "future coupling control",
    "topology": "latent separation / topology mass",
}


@dataclass(frozen=True)
class RoleEvidenceStage:
    id: str
    title: str
    pressure: str
    focus_role: str | None
    note: str
    scores: dict[str, float]


def role_evidence_scores(matrix: dict[str, object]) -> dict[str, float]:
    """Normalize role-recovery margins onto a compact 0-10 evidence scale."""

    return {
        "expectation": _score_from_margin(matrix, "expectation", "minimal_value_surprise_drop", cap=0.30),
        "selection": _score_from_margin(matrix, "selection", "cavenet_selection_margin", cap=0.40),
        "value_retention": _score_from_margin(matrix, "value_retention", "valued_focus_margin", cap=0.85),
        "regulation": _score_from_margin(matrix, "regulation", "cavenet_audio_delta", cap=0.30),
        "topology": _score_from_margin(matrix, "topology", "cavenet_topology_mass_gain", cap=70.0),
    }


def build_role_evidence_stages(
    matrix_result: dict[str, object] | None = None,
    *,
    dt: float = 1.0,
) -> list[RoleEvidenceStage]:
    """Build a narrative sequence for pressure-to-function evidence."""

    if matrix_result is None:
        matrix_result = check_role_recovery_matrix(dt=dt)
    matrix = matrix_result["matrix"]
    final_scores = role_evidence_scores(matrix)
    zero = {role: 0.0 for role in ROLE_ORDER}
    stages: list[RoleEvidenceStage] = [
        RoleEvidenceStage(
            id="setup",
            title="Role evidence board",
            pressure="Start with no role evidence credited",
            focus_role=None,
            note="Rows fill only when a pressure/control contrast supports the role.",
            scores=zero,
        )
    ]

    stage_specs = (
        (
            "recurrence",
            "Temporal recurrence + memory access",
            "Expectation",
            "expectation",
            "Repetition lowers surprise; no-memory controls stay near zero.",
        ),
        (
            "bottleneck",
            "Workspace bottleneck + competing input",
            "Selection",
            "selection",
            "Diagnostic signal survives while distractor mass is dropped.",
        ),
        (
            "value",
            "Preference / value pressure",
            "Value-shaped retention",
            "value_retention",
            "Memory or hidden state keeps what matters, not just what repeats.",
        ),
        (
            "control",
            "Objective feedback + future coupling",
            "Regulation",
            "regulation",
            "Exposure or attention changes before later consequences arrive.",
        ),
        (
            "geometry",
            "Repeated trajectories + geometry",
            "Topology-like organization",
            "topology",
            "State space separates into stable role-relevant regions.",
        ),
    )
    current = dict(zero)
    for stage_id, pressure, title, role, note in stage_specs:
        current = dict(current)
        current[role] = final_scores[role]
        stages.append(
            RoleEvidenceStage(
                id=stage_id,
                title=title,
                pressure=pressure,
                focus_role=role,
                note=note,
                scores=current,
            )
        )
    stages.append(
        RoleEvidenceStage(
            id="summary",
            title="Recovered function profile",
            pressure="All current pressure/function contrasts",
            focus_role=None,
            note="The board is a compact evidence accumulator, not a subject-state view.",
            scores=final_scores,
        )
    )
    return stages


def save_role_evidence_animation(
    output: str | Path,
    *,
    matrix_result: dict[str, object] | None = None,
    dt: float = 1.0,
    fps: int = 6,
    frames_per_stage: int = 10,
    hold_frames: int = 8,
    style: str | RendererStyle | None = None,
) -> None:
    """Render the pressure-to-function role evidence board as a GIF."""

    stages = build_role_evidence_stages(matrix_result, dt=dt)
    sequence = _interpolated_stage_sequence(stages, frames_per_stage=frames_per_stage, hold_frames=hold_frames)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    resolved = resolve_style(style)

    with plt.rc_context(resolved.rc_params()):
        figure = plt.figure(figsize=(10.4, 6.0), dpi=120)
        apply_figure_style(figure, resolved)
        axis = figure.add_subplot(1, 1, 1)

        def update(index: int):
            axis.clear()
            draw_role_evidence_board(axis, sequence[index], resolved)
            return (axis,)

        anim = animation.FuncAnimation(
            figure,
            update,
            frames=len(sequence),
            interval=1000 / max(1, fps),
            blit=False,
        )
        anim.save(output, writer=animation.PillowWriter(fps=fps), dpi=120)
        plt.close(figure)


def save_role_evidence_frame(
    output: str | Path,
    *,
    matrix_result: dict[str, object] | None = None,
    dt: float = 1.0,
    style: str | RendererStyle | None = None,
) -> None:
    """Render the final role evidence board as a static PNG."""

    stages = build_role_evidence_stages(matrix_result, dt=dt)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    resolved = resolve_style(style)
    with plt.rc_context(resolved.rc_params()):
        figure = plt.figure(figsize=(10.4, 6.0), dpi=120)
        apply_figure_style(figure, resolved)
        axis = figure.add_subplot(1, 1, 1)
        draw_role_evidence_board(axis, stages[-1], resolved)
        figure.savefig(output, dpi=120)
        plt.close(figure)


def draw_role_evidence_board(
    axis: Axes,
    stage: RoleEvidenceStage,
    style: RendererStyle | str | None = None,
) -> None:
    resolved = resolve_style(style)
    apply_axis_style(axis, resolved, "Pressure-to-function evidence", grid=False)
    axis.set_xlim(0.0, 12.55)
    axis.set_ylim(-1.05, len(ROLE_ORDER) + 2.25)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

    axis.text(
        0.15,
        len(ROLE_ORDER) + 1.58,
        stage.title,
        fontsize=17,
        fontweight="bold",
        ha="left",
        va="center",
        color=resolved.text_color,
    )
    _draw_pressure_ribbon(axis, stage, resolved)
    _draw_scale(axis, resolved)

    for index, role in enumerate(ROLE_ORDER):
        y = len(ROLE_ORDER) - 1 - index
        _draw_role_row(axis, role, y, stage.scores.get(role, 0.0), stage.focus_role == role, resolved)

    axis.text(
        0.15,
        -0.65,
        stage.note,
        ha="left",
        va="center",
        fontsize=9.5,
        color=resolved.muted_text_color,
    )


def _draw_pressure_ribbon(axis: Axes, stage: RoleEvidenceStage, style: RendererStyle) -> None:
    y = len(ROLE_ORDER) + 0.78
    axis.add_patch(
        Rectangle(
            (0.15, y - 0.2),
            11.7,
            0.42,
            facecolor=style.box_facecolor,
            edgecolor=style.spine_color,
            linewidth=style.linewidth(0.9),
            alpha=0.95,
        )
    )
    axis.text(
        0.42,
        y,
        "pressure:",
        ha="left",
        va="center",
        fontsize=9,
        color=style.muted_text_color,
        fontweight="bold",
    )
    axis.text(
        1.72,
        y,
        stage.pressure,
        ha="left",
        va="center",
        fontsize=10.5,
        color=style.text_color,
    )
    if stage.focus_role is not None:
        x0, x1 = 8.9, 10.7
        arrow = FancyArrowPatch(
            (x0, y),
            (x1, y),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=style.linewidth(1.1),
            color=ROLE_COLORS[stage.focus_role],
            alpha=0.88,
        )
        axis.add_patch(arrow)
        axis.text(
            10.85,
            y,
            ROLE_LABELS[stage.focus_role],
            ha="left",
            va="center",
            fontsize=9.5,
            color=ROLE_COLORS[stage.focus_role],
            fontweight="bold",
        )


def _draw_scale(axis: Axes, style: RendererStyle) -> None:
    y = len(ROLE_ORDER) + 0.15
    axis.text(3.25, y, "0", ha="center", va="center", fontsize=8, color=style.muted_text_color)
    axis.text(8.25, y, "5", ha="center", va="center", fontsize=8, color=style.muted_text_color)
    axis.text(11.25, y, "10", ha="center", va="center", fontsize=8, color=style.muted_text_color)
    axis.plot([3.25, 11.25], [y - 0.2, y - 0.2], color=style.guide_color, linewidth=style.linewidth(0.8), alpha=0.8)


def _draw_role_row(axis: Axes, role: str, y: float, score: float, focused: bool, style: RendererStyle) -> None:
    label_x = 0.18
    bar_x = 3.25
    bar_width = 8.0
    bar_height = 0.44
    value = float(np.clip(score, 0.0, 10.0))
    color = ROLE_COLORS[role]
    alpha = 0.92 if value > 0.05 else 0.25
    if focused:
        axis.add_patch(
            Rectangle(
                (0.06, y - 0.42),
                12.0,
                0.86,
                facecolor=color,
                edgecolor="none",
                alpha=0.07,
                zorder=0,
            )
        )
    axis.text(
        label_x,
        y + 0.12,
        ROLE_LABELS[role],
        ha="left",
        va="center",
        fontsize=10.8,
        color=style.text_color,
        fontweight="bold" if focused else "normal",
    )
    axis.text(
        label_x,
        y - 0.18,
        ROLE_METRIC_LABELS[role],
        ha="left",
        va="center",
        fontsize=8.0,
        color=style.muted_text_color,
    )
    axis.add_patch(
        Rectangle(
            (bar_x, y - bar_height / 2),
            bar_width,
            bar_height,
            facecolor=style.grid_color,
            edgecolor=style.spine_color,
            linewidth=style.linewidth(0.7),
            alpha=0.72,
            zorder=1,
        )
    )
    axis.add_patch(
        Rectangle(
            (bar_x, y - bar_height / 2),
            bar_width * (value / 10.0),
            bar_height,
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            zorder=2,
        )
    )
    for tick in range(0, 11):
        tx = bar_x + bar_width * tick / 10.0
        axis.plot(
            [tx, tx],
            [y - bar_height / 2, y + bar_height / 2],
            color=style.axes_facecolor,
            linewidth=style.linewidth(0.5),
            alpha=0.7,
            zorder=3,
        )
    axis.text(
        bar_x + bar_width + 0.18,
        y,
        f"{value:0.1f}",
        ha="left",
        va="center",
        fontsize=11.0,
        color=color if value > 0.05 else style.muted_text_color,
        fontweight="bold",
    )


def _interpolated_stage_sequence(
    stages: list[RoleEvidenceStage],
    *,
    frames_per_stage: int,
    hold_frames: int,
) -> list[RoleEvidenceStage]:
    if len(stages) <= 1:
        return stages
    sequence: list[RoleEvidenceStage] = []
    previous = stages[0]
    sequence.extend([previous] * max(1, hold_frames))
    for stage in stages[1:]:
        for frame_index in range(1, max(2, frames_per_stage) + 1):
            mix = frame_index / max(1, frames_per_stage)
            scores = {
                role: (1.0 - mix) * previous.scores.get(role, 0.0) + mix * stage.scores.get(role, 0.0)
                for role in ROLE_ORDER
            }
            sequence.append(
                RoleEvidenceStage(
                    id=stage.id,
                    title=stage.title,
                    pressure=stage.pressure,
                    focus_role=stage.focus_role,
                    note=stage.note,
                    scores=scores,
                )
            )
        sequence.extend([stage] * max(0, hold_frames))
        previous = stage
    return sequence


def _score_from_margin(matrix: dict[str, object], role: str, metric: str, *, cap: float) -> float:
    role_entry = matrix[role]
    margins = role_entry["margins"]
    value = float(margins[metric])
    return float(np.clip(10.0 * value / cap, 0.0, 10.0))
