from __future__ import annotations

from cave.presentation.renderers.matplotlib_renderer.role_evidence import (
    ROLE_ORDER,
    build_role_evidence_stages,
    role_evidence_scores,
)


def test_role_evidence_scores_normalize_matrix_margins() -> None:
    matrix = _fake_matrix()

    scores = role_evidence_scores(matrix)

    assert set(scores) == set(ROLE_ORDER)
    assert scores["expectation"] == 5.0
    assert scores["selection"] == 10.0
    assert scores["value_retention"] == 5.0
    assert scores["regulation"] == 5.0
    assert scores["topology"] == 5.0


def test_build_role_evidence_stages_accumulates_roles() -> None:
    stages = build_role_evidence_stages({"matrix": _fake_matrix()})

    assert stages[0].scores == {role: 0.0 for role in ROLE_ORDER}
    assert stages[1].focus_role == "expectation"
    assert stages[1].scores["expectation"] > 0.0
    assert stages[2].scores["expectation"] == stages[1].scores["expectation"]
    assert stages[2].scores["selection"] > 0.0
    assert stages[-1].focus_role is None
    assert all(stages[-1].scores[role] > 0.0 for role in ROLE_ORDER)

def _fake_matrix() -> dict[str, object]:
    return {
        "expectation": {
            "status": "pass",
            "margins": {
                "minimal_value_surprise_drop": 0.15,
            },
        },
        "selection": {
            "status": "pass",
            "margins": {
                "cavenet_selection_margin": 0.40,
            },
        },
        "value_retention": {
            "status": "pass",
            "margins": {
                "valued_focus_margin": 0.425,
            },
        },
        "regulation": {
            "status": "pass",
            "margins": {
                "cavenet_audio_delta": 0.15,
            },
        },
        "topology": {
            "status": "pass",
            "margins": {
                "cavenet_topology_mass_gain": 35.0,
            },
        },
    }
