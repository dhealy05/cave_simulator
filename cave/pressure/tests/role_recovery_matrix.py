from __future__ import annotations

import json
from pathlib import Path

from cave.observation.episodes import Episode
from cave.observation.projections import encode_value
from cave.observation.views import default_views
from cave.presentation.reports.specs import ProducerReportSpec, ReportExtraAsset, ReportSection
from cave.pressure.tests.regulation_recovery import check_regulation_recovery, build_regulation_episode
from cave.pressure.tests.role_recovery import check_role_recovery
from cave.pressure.tests.selection_recovery import check_selection_recovery
from cave.pressure.tests.topology_recovery import check_topology_recovery
from cave.pressure.tests.value_retention_recovery import check_value_retention_recovery


ROLE_CHECKS = {
    "expectation": check_role_recovery,
    "selection": check_selection_recovery,
    "value_retention": check_value_retention_recovery,
    "regulation": check_regulation_recovery,
    "topology": check_topology_recovery,
}


def role_recovery_matrix_report_spec(
    *,
    dt: float = 1.0,
    fps: int = 4,
    include_assets: bool = True,
) -> ProducerReportSpec:
    def build_episode() -> Episode:
        return build_regulation_episode("cave-adaptive", dt=dt)

    extra_assets = ()
    if include_assets:
        extra_assets = (
            ReportExtraAsset(
                id="role_recovery_matrix",
                title="Role Recovery Matrix JSON",
                filename="role_recovery_matrix.json",
                writer=lambda episode, output: write_role_recovery_matrix_json(output, dt=dt),
            ),
            ReportExtraAsset(
                id="role_evidence_board",
                title="Role Evidence Board",
                filename="role_evidence_board.gif",
                writer=lambda episode, output: _write_role_evidence_animation(output, dt=dt, fps=fps),
            ),
        )

    return ProducerReportSpec(
        id="role-recovery-matrix",
        title="Role Recovery Matrix",
        episode_factory=build_episode,
        input_summary="aggregate pass/fail matrix across role recovery reports",
        description=(
            "Aggregates the individual role recovery reports. This report does not "
            "introduce new behavior; it summarizes whether each role's recovery "
            "check currently passes and records compact margins."
        ),
        views=default_views(),
        extra_assets=extra_assets,
        checks=(lambda episode: check_role_recovery_matrix(dt=dt),),
        frame_time=0.0,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "role_recovery_matrix",
            "scenario": "role_recovery_matrix",
            "roles": list(ROLE_CHECKS),
        },
        sections=(
            ReportSection(
                title="Question",
                body=(
                    "The matrix asks which Cave-like roles have a demonstrated "
                    "failure mode and recovery path in the current substrate suite."
                ),
                asset_ids=("role_evidence_board", "role_recovery_matrix"),
            ),
        ),
    )


def check_role_recovery_matrix(*, dt: float = 1.0) -> dict[str, object]:
    results = {
        "expectation": check_role_recovery(dt=dt),
        "selection": check_selection_recovery(dt=dt),
        "value_retention": check_value_retention_recovery(),
        "regulation": check_regulation_recovery(dt=dt),
        "topology": check_topology_recovery(dt=dt),
    }
    matrix = {role: _role_status(role, result) for role, result in results.items()}
    passed = sum(1 for status in matrix.values() if status["status"] == "pass")
    errors = [
        f"{role} recovery failed"
        for role, status in matrix.items()
        if status["status"] != "pass"
    ]
    return {
        "id": "role_recovery_matrix",
        "ok": not errors,
        "errors": errors,
        "summary": {
            "role_count": len(matrix),
            "passed_count": passed,
        },
        "matrix": matrix,
        "results": results,
    }


def write_role_recovery_matrix_json(output: Path, *, dt: float = 1.0) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(encode_value(check_role_recovery_matrix(dt=dt)), indent=2) + "\n",
        encoding="utf-8",
    )


def _write_role_evidence_animation(output: Path, *, dt: float = 1.0, fps: int = 4) -> None:
    from cave.presentation.renderers.matplotlib_renderer.role_evidence import save_role_evidence_animation

    save_role_evidence_animation(output, dt=dt, fps=fps)


def _role_status(role: str, result: dict[str, object]) -> dict[str, object]:
    roles = result.get("roles", {})
    margins: dict[str, float] = {}
    if role == "expectation":
        recovery = roles["cavenet_learning_recovery"]
        minimal = roles["minimal_memory_recovery"]
        margins = {
            "cavenet_surprise_drop_gain": recovery["surprise_drop_gain"],
            "minimal_value_surprise_drop": minimal["value_surprise_drop"],
        }
    elif role == "selection":
        cavenet = roles["cavenet_bottleneck"]
        minimal = roles["minimal_diagnostic_selection"]
        margins = {
            "cavenet_selection_margin": cavenet["selection_margin"],
            "minimal_selection_margin": minimal["selection_margin"],
        }
    elif role == "value_retention":
        value = roles["value_shaped_retention"]
        margins = {
            "valued_focus_margin": value["focus_margin"],
            "value_memory_strength": value["value_memory_strength"],
        }
    elif role == "regulation":
        regulation = roles["future_attention_regulation"]
        margins = {
            "cave_audio_delta": regulation["cave_adaptive_audio_delta"],
            "cavenet_audio_delta": regulation["cavenet_adaptive_audio_delta"],
        }
    elif role == "topology":
        topology = roles["cavenet_topology_recovery"]
        proxy = roles["minimal_geometry_proxy"]
        margins = {
            "cavenet_topology_mass_gain": topology["topology_mass_gain"],
            "minimal_value_memory_strength": proxy["value_memory_strength"],
        }
    return {
        "status": "pass" if bool(result.get("ok")) else "fail",
        "margins": margins,
    }
