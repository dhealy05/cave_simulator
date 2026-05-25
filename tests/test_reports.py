from __future__ import annotations

import json

import numpy as np

from cave.demonstrations.reports.cave_matrices import (
    initial_conditions_matrix_report_spec,
    population_clusters_matrix_report_spec,
    subject_ablation_matrix_report_spec,
)
from cave.demonstrations.reports.cave_reference import reference_cave_report_spec
from cave.demonstrations.reports.cave_subjects import (
    preference_shaped_topology_report_spec,
    same_world_different_subjects_report_spec,
)
from cave.demonstrations.scenarios import (
    attention_bottleneck_report_spec,
    expectation_violation_report_spec,
    importance_weighted_event_report_spec,
    objective_attention_shift_report_spec,
    representational_compression_report_spec,
    role_dependency_contrasts_report_spec,
    topology_atlas_report_spec,
    unseen_modality_report_spec,
    valence_attractor_repulsor_report_spec,
)
from cave.observation.source_reports.conversation_matrices import (
    ConversationMatrixFixture,
    conversation_text_config_matrix_report_spec,
)
from cave.observation.source_reports.conversation_reference import (
    conversation_reference_report_spec,
)
from cave.observation.source_reports.gpt2_matrices import (
    gpt2_text_config_matrix_report_spec,
)
from cave.observation.source_reports.gpt2_reference import gpt2_reference_report_spec
from cave.observation.structural import frame_for_time, structural_state_for_episode
from cave.observation.views import (
    ObserverView,
    ObserverViewState,
    SubjectSurfaceView,
    SubjectSurfaceViewState,
)
from cave.presentation.renderers.matplotlib_renderer import LayoutSpec, MatplotlibRenderer
from cave.presentation.renderers.topology_population_renderer import (
    save_topology_scatter_migration,
    save_topology_population_dashboard,
    topology_trajectories,
)
from cave.presentation.reports.generate import write_producer_report
from cave.presentation.reports.matrix import write_matrix_report
from cave.presentation.reports.subject_comparison import write_subject_comparison_report
from cave.presentation.reports.suites import (
    DEFAULT_SUITE_MANIFEST,
    load_suite_manifest,
    run_report_suite,
)
from cave.pressure.tests.cavenet_ablation import cavenet_ablation_report_spec
from cave.pressure.tests.cavenet_controller import (
    cavenet_controller_learning_report_spec,
    cavenet_controller_population_report_spec,
    cavenet_controller_report_spec,
)
from cave.pressure.tests.cavenet_pressure import (
    cavenet_pressure_population_report_spec,
    cavenet_pressure_report_spec,
)
from cave.pressure.tests.common_behaviors import common_behaviors_report_spec
from cave.pressure.tests.evolved_exposure import evolved_exposure_report_spec
from cave.pressure.tests.evolved_exposure import write_evolved_exposure_observer_controls
from cave.pressure.tests.evolved_exposure_sweep import evolved_exposure_sweep_report_spec
from cave.pressure.tests.evolved_roles import evolved_roles_report_spec
from cave.pressure.tests.evolved_roles_sweep import evolved_roles_sweep_report_spec
from cave.pressure.tests.preference_emergence import preference_emergence_report_spec
from cave.pressure.tests.population_trajectory_geometry import (
    population_trajectory_geometry_report_spec,
    population_trajectory_geometry_sweep_report_spec,
)
from cave.pressure.tests.regulation_recovery import regulation_recovery_report_spec
from cave.pressure.tests.role_recovery import role_recovery_report_spec
from cave.pressure.tests.role_recovery_matrix import role_recovery_matrix_report_spec
from cave.pressure.tests.selection_recovery import selection_recovery_report_spec
from cave.pressure.tests.topology_recovery import topology_recovery_report_spec
from cave.pressure.tests.value_retention_recovery import (
    value_retention_recovery_report_spec,
)
from cave.observation.producers.sources.conversation import (
    ConversationSegment,
    ConversationTurn,
    build_conversation_episode,
)
from cave.observation.producers.sources.gpt2 import build_gpt2_episode


def test_result_ladder_manifest_defines_evidence_tiers() -> None:
    manifest = load_suite_manifest(DEFAULT_SUITE_MANIFEST)

    assert manifest["id"] == "result-ladder"
    assert [tier["id"] for tier in manifest["tiers"]] == [
        "walkthrough",
        "mechanism",
        "emergence",
        "causal_probes",
    ]
    entry_ids = {entry["id"] for entry in manifest["entries"]}
    assert "reference-cave" in entry_ids
    assert "population-clusters" in entry_ids
    assert "population-trajectory-geometry" in entry_ids
    assert "population-trajectory-geometry-sweep" in entry_ids
    assert "cavenet-controller-learning" in entry_ids
    assert "role-dependency-contrasts" in entry_ids
    assert "topology-atlas" in entry_ids
    assert all(entry["claim"] for entry in manifest["entries"])
    assert all(entry["kind"] in {"producer", "subject", "matrix"} for entry in manifest["entries"])


def test_population_trajectory_geometry_report_scores_treatment_structure(tmp_path) -> None:
    spec = population_trajectory_geometry_report_spec(
        treatment_count=3,
        start_count=9,
        event_count=4,
        seed=701,
        dt=0.25,
        end=2.5,
        samples=32,
        permutations=49,
        min_treatment_accuracy_lift=0.2,
        max_permutation_p=0.08,
    )

    outputs = write_matrix_report(spec, tmp_path / "population-trajectory-geometry")
    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))

    assert checks["ok"]
    result = checks["extra"][0]
    roles = result["roles"]
    assert roles["treatment_decoding"]["accuracy"] > roles["treatment_decoding"]["chance"]
    assert roles["treatment_separability"]["subjective_treatment_margin"] > 0.0
    assert roles["treatment_separability"]["subjective_permutation_p"] <= 0.08
    controls = roles["control_contrasts"]
    assert controls["external-only-attention"]["actual_decoding_lift"] > 0.0
    assert controls["internal-only-attention"]["actual_decoding_accuracy"] <= roles[
        "treatment_decoding"
    ]["chance"]


def test_population_trajectory_geometry_sweep_scores_stability(tmp_path) -> None:
    spec = population_trajectory_geometry_sweep_report_spec(
        seed_start=701,
        seed_count=3,
        treatment_count=3,
        start_count=6,
        event_count=4,
        dt=0.25,
        fps=4,
        end=2.5,
        samples=24,
        permutations=19,
        min_pass_rate=1.0,
        max_permutation_p=0.1,
        include_assets=False,
    )

    outputs = write_producer_report(spec, tmp_path / "population-trajectory-sweep")
    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))

    assert checks["ok"]
    roles = checks["extra"][0]["roles"]
    assert roles["robust_treatment_recovery"]["strict_pass_rate"] == 1.0
    assert roles["robust_treatment_recovery"]["baseline_decoding_pass_count"] == 3
    assert roles["control_collapse"]["zero_attention_subjective_collapse_count"] == 3
    assert roles["control_collapse"]["external_only_actual_preserved_count"] == 3
    assert roles["control_collapse"]["internal_only_actual_collapse_count"] == 3
    assert roles["control_collapse"]["no_memory_observed_memory_collapse_count"] == 3


def test_report_suite_dry_run_writes_manifest_and_index(tmp_path) -> None:
    result = run_report_suite(
        DEFAULT_SUITE_MANIFEST,
        output_root=tmp_path / "suite-dry",
        dry_run=True,
    )

    assert result.suite_id == "result-ladder"
    assert result.manifest_json.exists()
    assert result.index_json.exists()
    assert result.index_md.exists()
    assert len(result.entries) >= 10
    index = json.loads(result.index_json.read_text(encoding="utf-8"))
    assert index["dry_run"]
    assert index["entries"][0]["key_metrics"]
    assert "Walkthrough" in result.index_md.read_text(encoding="utf-8")


def test_report_suite_runs_selected_entry_and_indexes_metrics(tmp_path) -> None:
    result = run_report_suite(
        DEFAULT_SUITE_MANIFEST,
        output_root=tmp_path / "suite",
        entries=("attention-bottleneck",),
        skip_assets=True,
    )

    assert len(result.entries) == 1
    entry = result.entries[0]
    assert entry.id == "attention-bottleneck"
    assert entry.ok
    assert entry.report_md.exists()
    assert entry.checks_json.exists()
    assert entry.key_metrics["extra.attention_bottleneck.metrics.actual"] == [
        0.25,
        0.75,
    ]
    index = json.loads(result.index_json.read_text(encoding="utf-8"))
    assert index["entries"][0]["report"].endswith("mechanism/attention-bottleneck/report.md")


def test_cave_reference_report_writes_contract_bundle(tmp_path) -> None:
    spec = reference_cave_report_spec(
        dt=0.5,
        fps=4,
        include_readme_assets=False,
    )

    outputs = write_producer_report(spec, tmp_path / "reference")

    assert outputs.report_md.exists()
    assert outputs.episode_json.exists()
    assert outputs.metadata_json.exists()
    assert outputs.checks_json.exists()
    assert outputs.frame_png.exists()
    assert outputs.frame_png.stat().st_size > 0
    assert outputs.animation_gif.exists()
    assert outputs.animation_gif.stat().st_size > 0

    episode = json.loads(outputs.episode_json.read_text(encoding="utf-8"))
    metadata = json.loads(outputs.metadata_json.read_text(encoding="utf-8"))
    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))

    assert episode["source_name"] == "cave"
    assert metadata["source_name"] == "cave"
    assert metadata["episode_metadata"]["adapter"] == "CaveProducer"
    assert metadata["config"]["producer"] == "cave"
    assert checks["ok"]
    assert checks["input_count"] == 4
    assert checks["observation_count"] == checks["frame_count"]
    assert [item["id"] for item in episode["inputs"]] == [
        "evt_triangle",
        "evt_circle",
        "evt_square",
        "evt_gap",
    ]

    first = episode["frames"][0]
    assert first["active_input_ids"] == ["evt_triangle"]
    actual = np.array(first["actual"], dtype=float)
    expected = np.array(first["expected"], dtype=float)
    prediction_error = np.array(first["prediction_error"], dtype=float)
    memory_state = np.array(first["memory_state"], dtype=float)

    np.testing.assert_allclose(prediction_error, actual - expected)
    assert memory_state.shape == actual.shape
    assert "views" in first
    assert "wall_pov" in first["views"]
    assert "subjective_topology" in first["views"]


def test_cave_reference_walkthrough_skips_subject_surface_and_observer_assets() -> None:
    spec = reference_cave_report_spec(dt=0.5, fps=4, include_readme_assets=True)

    assert [asset.filename for asset in spec.view_assets] == [
        "01_presentation_wall.gif",
        "02_memory_lookback.gif",
        "03_timeline_tape.gif",
        "04_expectation_actual.gif",
        "05_prediction_correction_over_time.gif",
        "06_subjective_topology.gif",
        "07_multi_view_state.gif",
    ]
    assert [asset.filename for asset in spec.extra_assets] == ["08_topology_state_surface.png"]


def test_observer_view_renders_default_cave_frame(tmp_path) -> None:
    spec = reference_cave_report_spec(dt=0.5, fps=4, include_readme_assets=True)
    observer_assets = [asset for asset in spec.view_assets if asset.id == "observer"]
    assert observer_assets == []

    episode = spec.episode_factory()
    structural = structural_state_for_episode(episode)
    frame = frame_for_time(episode, 1.0, structural)
    view = ObserverView()
    state = view.project(frame)
    output = tmp_path / "observer.png"

    assert isinstance(state, ObserverViewState)
    assert 0.0 <= state.openness <= 1.0
    assert state.gaze_label != ""
    assert state.trail_points

    renderer = MatplotlibRenderer(layout=LayoutSpec(columns=1))
    renderer.save_frame(frame, [view], output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_subject_surface_view_renders_default_cave_frame(tmp_path) -> None:
    spec = reference_cave_report_spec(dt=0.5, fps=4, include_readme_assets=True)
    surface_assets = [asset for asset in spec.view_assets if asset.id == "subject_surface"]
    assert surface_assets == []

    episode = spec.episode_factory()
    structural = structural_state_for_episode(episode)
    frame = frame_for_time(episode, 1.0, structural)
    view = SubjectSurfaceView()
    state = view.project(frame)
    output = tmp_path / "subject_surface.png"

    assert isinstance(state, SubjectSurfaceViewState)
    assert 0.0 <= state.aperture <= 1.0
    assert state.input_label != ""
    assert state.trail_points

    renderer = MatplotlibRenderer(layout=LayoutSpec(columns=1))
    renderer.save_frame(frame, [view], output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_gpt2_reference_report_accepts_episode_factory(tmp_path) -> None:
    spec = gpt2_reference_report_spec(
        text="Hello Paul !",
        include_assets=False,
        episode_factory=fake_gpt2_episode,
        fps=4,
    )

    outputs = write_producer_report(spec, tmp_path / "gpt2-reference")

    episode = json.loads(outputs.episode_json.read_text(encoding="utf-8"))
    metadata = json.loads(outputs.metadata_json.read_text(encoding="utf-8"))
    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    report = outputs.report_md.read_text(encoding="utf-8")

    assert checks["ok"]
    assert episode["source_name"] == "gpt2"
    assert metadata["source_name"] == "gpt2"
    assert metadata["episode_metadata"]["adapter"] == "GPT2Producer"
    assert metadata["config"]["producer"] == "gpt2"
    assert len(episode["inputs"]) == 4
    assert len(episode["frames"]) == 3
    assert episode["metadata"]["presentation_mode"] == "current_text"
    assert "GPT-2 Reference Report" in report
    assert "## Walkthrough" in report
    assert outputs.frame_png.exists()
    assert outputs.animation_gif.exists()


def test_conversation_reference_report_accepts_episode_factory(tmp_path) -> None:
    spec = conversation_reference_report_spec(
        include_assets=False,
        episode_factory=lambda: fake_conversation_episode_for_report(
            (
                ConversationTurn("user", "Prior context?"),
                ConversationTurn("assistant", "Protocol memory."),
                ConversationTurn("user", "Expected actual?"),
            ),
        ),
        fps=4,
    )

    outputs = write_producer_report(spec, tmp_path / "conversation-reference")

    episode = json.loads(outputs.episode_json.read_text(encoding="utf-8"))
    metadata = json.loads(outputs.metadata_json.read_text(encoding="utf-8"))
    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    report = outputs.report_md.read_text(encoding="utf-8")

    assert checks["ok"]
    assert episode["source_name"] == "conversation"
    assert metadata["source_name"] == "conversation"
    assert metadata["episode_metadata"]["adapter"] == "ConversationProducer"
    assert metadata["config"]["producer"] == "conversation"
    assert len(episode["inputs"]) == 3
    assert len(episode["frames"]) == 2
    assert episode["metadata"]["memory_interpretation"] == "mock_prior_context"
    assert "Conversation Reference Report" in report
    assert "transformer-stored memories" in report
    assert outputs.frame_png.exists()
    assert outputs.animation_gif.exists()


def test_expectation_violation_scenario_report_checks_claim(tmp_path) -> None:
    spec = expectation_violation_report_spec(
        dt=0.1,
        fps=4,
        include_assets=False,
    )

    outputs = write_producer_report(spec, tmp_path / "expectation-violation")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    episode = json.loads(outputs.episode_json.read_text(encoding="utf-8"))
    scenario_check = checks["extra"][0]
    surprise = scenario_check["metrics"]["surprise"]
    learning_rate = scenario_check["metrics"]["learning_rate"]

    assert checks["ok"]
    assert scenario_check["id"] == "expectation_violation"
    assert surprise["repeat_1"] > surprise["repeat_2"] > surprise["repeat_3"]
    assert surprise["violation"] > surprise["repeat_3"] * 2.0
    assert learning_rate["violation"] > learning_rate["repeat_3"]

    violation_frame = next(
        frame
        for frame in episode["frames"]
        if frame["active_input_ids"] == ["violation"]
    )
    np.testing.assert_allclose(violation_frame["actual"], np.array([0.0, 0.5]))


def test_unseen_modality_scenario_report_checks_claim(tmp_path) -> None:
    spec = unseen_modality_report_spec(
        dt=0.1,
        fps=4,
        include_assets=False,
    )

    outputs = write_producer_report(spec, tmp_path / "unseen-modality")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    episode = json.loads(outputs.episode_json.read_text(encoding="utf-8"))
    scenario_check = checks["extra"][0]

    assert checks["ok"]
    assert scenario_check["id"] == "unseen_modality"
    assert [item["id"] for item in episode["inputs"]] == [
        "visible_flash",
        "unheard_tone",
    ]
    unheard_frame = next(
        frame
        for frame in episode["frames"]
        if frame["active_input_ids"] == ["unheard_tone"]
    )
    np.testing.assert_allclose(unheard_frame["actual"], np.array([0.0, 0.0]))


def test_attention_bottleneck_scenario_report_checks_claim(tmp_path) -> None:
    spec = attention_bottleneck_report_spec(
        dt=0.1,
        fps=4,
        include_assets=False,
    )

    outputs = write_producer_report(spec, tmp_path / "attention-bottleneck")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    episode = json.loads(outputs.episode_json.read_text(encoding="utf-8"))
    scenario_check = checks["extra"][0]

    assert checks["ok"]
    assert scenario_check["id"] == "attention_bottleneck"
    first = episode["frames"][0]
    assert first["active_input_ids"] == ["visual_marker", "audio_marker"]
    np.testing.assert_allclose(first["actual"], np.array([0.25, 0.75]))
    assert first["attention_weights"] == {
        "audio_marker": 0.75,
        "visual_marker": 0.25,
    }


def test_importance_weighted_event_scenario_report_checks_claim(tmp_path) -> None:
    spec = importance_weighted_event_report_spec(
        dt=0.1,
        fps=4,
        include_assets=False,
    )

    outputs = write_producer_report(spec, tmp_path / "importance-weighted-event")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    scenario_check = checks["extra"][0]
    metrics = scenario_check["metrics"]

    assert checks["ok"]
    assert scenario_check["id"] == "importance_weighted_event"
    assert metrics["important_learning_rate"] > metrics["ordinary_learning_rate"]
    assert metrics["important_memory_delta"] > metrics["ordinary_memory_delta"]
    assert metrics["important_attention_weight"] > metrics["ordinary_attention_weight"]


def test_valence_attractor_repulsor_scenario_report_checks_claim(tmp_path) -> None:
    spec = valence_attractor_repulsor_report_spec(
        dt=0.1,
        fps=4,
        include_assets=False,
    )

    outputs = write_producer_report(spec, tmp_path / "valence-attractor-repulsor")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    episode = json.loads(outputs.episode_json.read_text(encoding="utf-8"))
    scenario_check = checks["extra"][0]
    metrics = scenario_check["metrics"]

    assert checks["ok"]
    assert scenario_check["id"] == "valence_attractor_repulsor"
    assert metrics["neutral"]["pain"] == metrics["neutral"]["pleasure"] == 0.0
    assert metrics["pleasant"]["pleasure"] > metrics["pleasant"]["pain"]
    assert metrics["painful"]["pain"] > metrics["painful"]["pleasure"]
    assert "valence" in episode["frames"][0]["metadata"]
    assert "objective" in episode["frames"][0]["metadata"]


def test_objective_attention_shift_scenario_report_checks_claim(tmp_path) -> None:
    spec = objective_attention_shift_report_spec(
        dt=0.1,
        fps=4,
        include_assets=False,
    )

    outputs = write_producer_report(spec, tmp_path / "objective-attention-shift")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    scenario_check = checks["extra"][0]
    metrics = scenario_check["metrics"]

    assert checks["ok"]
    assert scenario_check["id"] == "objective_attention_shift"
    assert metrics["first_pain"] > 0.0
    assert (
        metrics["first_next_attention_channels"]["audio"]
        > metrics["first_attention_channels"]["audio"]
    )
    assert (
        metrics["second_attention_channels"]["audio"]
        == metrics["first_next_attention_channels"]["audio"]
    )


def test_representational_compression_scenario_report_checks_claim(tmp_path) -> None:
    spec = representational_compression_report_spec(
        dt=0.1,
        fps=4,
        include_assets=False,
    )

    outputs = write_producer_report(spec, tmp_path / "representational-compression")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    scenario_check = checks["extra"][0]
    metrics = scenario_check["metrics"]

    assert checks["ok"]
    assert scenario_check["id"] == "representational_compression"
    np.testing.assert_allclose(metrics["attended_input"], np.array([0.5, 0.25, 0.125]))
    np.testing.assert_allclose(metrics["actual"], np.array([0.5, 0.0, 0.0]))
    assert metrics["active_features"] == ["dominant"]
    assert metrics["compression_cost"] > 0.0


def test_preference_emergence_scenario_report_checks_ablations(tmp_path) -> None:
    spec = preference_emergence_report_spec(
        dt=0.2,
        fps=4,
        include_assets=False,
    )

    outputs = write_producer_report(spec, tmp_path / "preference-emergence")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    episode = json.loads(outputs.episode_json.read_text(encoding="utf-8"))
    scenario_check = checks["extra"][0]
    metrics = scenario_check["metrics"]
    roles = scenario_check["roles"]

    assert checks["ok"]
    assert scenario_check["id"] == "preference_emergence"
    assert metrics["minimal-preference"]["skill_gain"] > 0.25
    assert metrics["minimal-preference"]["late_skill"] > metrics["no-memory"]["late_skill"]
    assert metrics["minimal-preference"]["late_memory_strength"] > metrics["no-preference"]["late_memory_strength"]
    assert metrics["minimal-preference"]["late_diagnostic_attention"] > metrics["no-bottleneck"]["late_diagnostic_attention"]
    assert metrics["frequency-memory"]["late_skill"] > metrics["no-memory"]["late_skill"]
    assert roles["workspace_pressure_attention_like_selection"]["selection_margin"] > 0.0
    assert (
        roles["workspace_pressure_attention_like_selection"]["claim_kind"]
        == "diagnostic_input_weight_concentration"
    )
    assert not roles["workspace_pressure_attention_like_selection"][
        "full_dynamic_attention_claimed"
    ]
    assert not roles["workspace_pressure_attention_like_selection"][
        "internal_expectation_channel_claimed"
    ]
    assert roles["preference_pressure_value_shaped_memory"]["value_separation_margin"] > 0.0
    assert roles["delayed_consequence_prediction_like_readout"]["structure_margin"] > 0.0
    assert episode["metadata"]["adapter"] == "MinimalSubject"
    assert "minimal_subject" in episode["frames"][0]["metadata"]


def test_common_behaviors_report_checks_shared_substrate_roles(tmp_path) -> None:
    spec = common_behaviors_report_spec(dt=1.0, fps=4, include_assets=False)

    outputs = write_producer_report(spec, tmp_path / "common-behaviors")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    scenario_check = checks["extra"][0]
    roles = scenario_check["roles"]
    metrics = scenario_check["metrics"]
    equivalence = scenario_check["equivalence"]

    assert checks["ok"]
    assert scenario_check["id"] == "common_behaviors"
    for substrate in ("cave", "cavenet", "minimal_subject"):
        assert roles["expectation_repetition"][substrate]["surprise_drop"] > 0.0
        assert roles["expectation_repetition"][substrate]["violation_margin"] > 0.0
        assert roles["workspace_selection"][substrate]["active_feature_count"] <= 2
        assert roles["workspace_selection"][substrate]["dropped_mass"] > 0.0
        assert roles["value_separation"][substrate]["utility_contrast"] > 0.0
    assert metrics["expectation_repetition"]["minimal_subject"]["adapter"] == "MinimalSubject"
    assert equivalence["max_actual_distance"] <= 1e-12
    assert equivalence["max_memory_distance"] <= 1e-12


def test_role_recovery_report_checks_expectation_recovery(tmp_path) -> None:
    spec = role_recovery_report_spec(dt=1.0, fps=4, include_assets=False)

    outputs = write_producer_report(spec, tmp_path / "role-recovery")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    scenario_check = checks["extra"][0]
    roles = scenario_check["roles"]

    assert checks["ok"]
    assert scenario_check["id"] == "role_recovery"
    assert roles["cavenet_reference"]["surprise_drop"] > 0.0
    assert roles["cavenet_no_expectation"]["surprise_drop"] == 0.0
    assert (
        roles["cavenet_learning_recovery"]["adaptive_surprise_drop"]
        > roles["cavenet_learning_recovery"]["weak_surprise_drop"]
    )
    assert (
        roles["cavenet_learning_recovery"]["final_learning_gain"]
        > roles["cavenet_learning_recovery"]["initial_learning_gain"]
    )
    assert roles["minimal_memory_recovery"]["no_memory_surprise_drop"] == 0.0
    assert (
        roles["minimal_memory_recovery"]["value_surprise_drop"]
        > roles["minimal_memory_recovery"]["no_memory_surprise_drop"]
    )
    assert roles["minimal_memory_recovery"]["value_memory_strength"] > 0.0


def test_selection_recovery_report_checks_bottleneck_selection(tmp_path) -> None:
    spec = selection_recovery_report_spec(dt=1.0, fps=4, include_assets=False)

    outputs = write_producer_report(spec, tmp_path / "selection-recovery")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    scenario_check = checks["extra"][0]
    roles = scenario_check["roles"]

    assert checks["ok"]
    assert scenario_check["id"] == "selection_recovery"
    assert roles["cavenet_bottleneck"]["selected_active_feature_count"] <= 2
    assert roles["cavenet_bottleneck"]["no_bottleneck_active_feature_count"] > 2
    assert roles["cavenet_bottleneck"]["selection_margin"] > 0.0
    assert (
        roles["minimal_diagnostic_selection"]["bottleneck_late_diagnostic_attention"]
        > roles["minimal_diagnostic_selection"]["no_bottleneck_late_diagnostic_attention"]
    )
    assert (
        roles["minimal_diagnostic_selection"]["bottleneck_memory_strength"]
        > roles["minimal_diagnostic_selection"]["no_memory_strength"]
    )


def test_value_retention_recovery_report_checks_value_shaped_memory(tmp_path) -> None:
    spec = value_retention_recovery_report_spec(dt=1.0, fps=4, include_assets=False)

    outputs = write_producer_report(spec, tmp_path / "value-retention-recovery")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    roles = checks["extra"][0]["roles"]["value_shaped_retention"]

    assert checks["ok"]
    assert roles["no_preference_memory_strength"] == 0.0
    assert roles["value_valued_focus"] > roles["frequency_valued_focus"]
    assert roles["focus_margin"] > 0.5
    assert roles["value_memory_strength"] > 0.0


def test_regulation_recovery_report_checks_future_attention_shift(tmp_path) -> None:
    spec = regulation_recovery_report_spec(dt=1.0, fps=4, include_assets=False)

    outputs = write_producer_report(spec, tmp_path / "regulation-recovery")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    roles = checks["extra"][0]["roles"]["future_attention_regulation"]

    assert checks["ok"]
    assert roles["cave_fixed_audio_delta"] == 0.0
    assert roles["cavenet_fixed_audio_delta"] == 0.0
    assert roles["cave_adaptive_audio_delta"] > 0.0
    assert roles["cavenet_adaptive_audio_delta"] > 0.0


def test_topology_recovery_report_checks_geometry_recovery(tmp_path) -> None:
    spec = topology_recovery_report_spec(dt=1.0, fps=4, include_assets=False)

    outputs = write_producer_report(spec, tmp_path / "topology-recovery")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    roles = checks["extra"][0]["roles"]

    assert checks["ok"]
    assert roles["cavenet_topology_recovery"]["reference_topology_mass"] > 0.0
    assert roles["cavenet_topology_recovery"]["no_topology_mass"] == 0.0
    assert roles["cavenet_topology_recovery"]["topology_mass_gain"] > 0.0
    assert (
        roles["minimal_geometry_proxy"]["value_memory_strength"]
        > roles["minimal_geometry_proxy"]["no_memory_strength"]
    )


def test_role_recovery_matrix_report_summarizes_recovery_reports(tmp_path) -> None:
    spec = role_recovery_matrix_report_spec(dt=1.0, fps=4, include_assets=False)

    outputs = write_producer_report(spec, tmp_path / "role-recovery-matrix")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    scenario_check = checks["extra"][0]

    assert checks["ok"]
    assert scenario_check["id"] == "role_recovery_matrix"
    assert scenario_check["summary"]["role_count"] == 5
    assert scenario_check["summary"]["passed_count"] == 5
    assert set(scenario_check["matrix"]) == {
        "expectation",
        "selection",
        "value_retention",
        "regulation",
        "topology",
    }


def test_evolved_exposure_report_checks_recurrent_emergence(tmp_path) -> None:
    spec = evolved_exposure_report_spec(
        generations=30,
        population_size=32,
        world_count=12,
        evaluation_cycles=20,
        include_assets=False,
    )

    outputs = write_producer_report(spec, tmp_path / "evolved-exposure")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    scenario_check = checks["extra"][0]
    roles = scenario_check["roles"]

    assert checks["ok"]
    assert scenario_check["id"] == "evolved_exposure"
    assert roles["exposure_regulation"]["utility_gain_over_random"] > 1.0
    assert roles["exposure_regulation"]["utility_gain_over_nonrecurrent"] > 1.0
    assert roles["exposure_regulation"]["evolved_exposure_contrast"] > 0.5
    assert roles["exposure_regulation"]["reset_exposure_contrast"] < 0.1
    assert roles["latent_expectation_probe"]["evolved_probe_accuracy"] >= 0.8


def test_evolved_exposure_sweep_report_checks_seed_robustness(tmp_path) -> None:
    spec = evolved_exposure_sweep_report_spec(
        seed_start=17,
        seed_count=3,
        generations=20,
        population_size=24,
        world_count=8,
        evaluation_cycles=16,
        min_pass_rate=1.0,
        include_assets=False,
    )

    outputs = write_producer_report(spec, tmp_path / "evolved-exposure-sweep")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    scenario_check = checks["extra"][0]
    roles = scenario_check["roles"]

    assert checks["ok"]
    assert scenario_check["id"] == "evolved_exposure_sweep"
    assert roles["robust_expectation_regulation"]["strict_pass_rate"] == 1.0
    assert (
        roles["robust_expectation_regulation"]["utility_gain_over_random_pass_count"]
        == 3
    )
    assert (
        roles["robust_expectation_regulation"]["utility_gain_over_nonrecurrent_pass_count"]
        == 3
    )
    assert roles["robust_expectation_regulation"]["exposure_contrast_pass_count"] == 3
    assert roles["robust_expectation_regulation"]["probe_accuracy_pass_count"] == 3
    assert roles["control_collapse"]["median_abs_reset_exposure_contrast"] < 0.1
    assert (
        roles["control_collapse"]["median_abs_shuffled_exposure_contrast"]
        < roles["robust_expectation_regulation"]["median_evolved_exposure_contrast"]
    )


def test_evolved_observer_control_grid_renders(tmp_path) -> None:
    spec = evolved_exposure_report_spec(
        generations=1,
        population_size=4,
        world_count=2,
        evaluation_cycles=3,
        include_assets=True,
    )
    assert any(asset.id == "observer" for asset in spec.view_assets)
    assert any(asset.id == "observer_controls" for asset in spec.extra_assets)

    output = tmp_path / "observer_controls.gif"
    write_evolved_exposure_observer_controls(
        output,
        generations=1,
        population_size=4,
        world_count=2,
        evaluation_cycles=3,
        fps=2,
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_evolved_roles_report_checks_role_emergence(tmp_path) -> None:
    spec = evolved_roles_report_spec(
        generations=30,
        population_size=32,
        world_count=12,
        evaluation_cycles=32,
        include_assets=False,
    )

    outputs = write_producer_report(spec, tmp_path / "evolved-roles")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    scenario_check = checks["extra"][0]
    roles = scenario_check["roles"]

    assert checks["ok"]
    assert scenario_check["id"] == "evolved_roles"
    assert roles["selection_under_bottleneck"]["evolved_cue_total_ratio"] > 0.5
    assert roles["selection_under_bottleneck"]["cue_total_gain_over_nonrecurrent"] > 0.1
    assert roles["selection_under_bottleneck"]["claim_kind"] == "cue_weight_concentration"
    assert not roles["attention_claim_boundary"]["dynamic_attention_claimed"]
    assert not roles["attention_claim_boundary"]["internal_expectation_channel_claimed"]
    assert roles["value_shaped_retention"]["evolved_probe_accuracy"] >= 0.8
    assert roles["value_shaped_retention"]["signal_gain_over_reset"] > 2.0
    assert roles["exposure_regulation"]["evolved_good_exposure"] > roles["exposure_regulation"]["evolved_neutral_exposure"]
    assert roles["exposure_regulation"]["evolved_neutral_exposure"] > roles["exposure_regulation"]["evolved_bad_exposure"]
    assert roles["latent_topology"]["evolved_latent_value_signal"] > 5.0
    assert roles["latent_topology"]["topology_signal_gain_over_shuffled"] > 2.0


def test_evolved_roles_sweep_report_separates_role_strength(tmp_path) -> None:
    spec = evolved_roles_sweep_report_spec(
        seed_start=17,
        seed_count=3,
        generations=20,
        population_size=24,
        world_count=8,
        evaluation_cycles=24,
        min_core_pass_rate=1.0,
        min_selection_pass_rate=0.6,
        include_assets=False,
    )

    outputs = write_producer_report(spec, tmp_path / "evolved-roles-sweep")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    scenario_check = checks["extra"][0]
    roles = scenario_check["roles"]

    assert checks["ok"]
    assert scenario_check["id"] == "evolved_roles_sweep"
    assert roles["selection_like_readout"]["cue_total_pass_count"] == 2
    assert roles["selection_like_readout"]["cue_neutral_pass_count"] == 2
    assert roles["selection_like_readout"]["claim_kind"] == "cue_weight_concentration"
    assert not roles["attention_claim_boundary"]["dynamic_attention_claimed"]
    assert not roles["attention_claim_boundary"]["internal_expectation_channel_claimed"]
    assert roles["value_retention"]["probe_accuracy_pass_count"] == 3
    assert roles["value_retention"]["value_signal_reset_pass_count"] == 3
    assert roles["exposure_regulation"]["exposure_order_pass_count"] == 3
    assert roles["exposure_regulation"]["exposure_contrast_pass_count"] == 3
    assert roles["latent_geometry"]["latent_value_signal_pass_count"] == 3
    assert roles["latent_geometry"]["topology_gain_shuffled_pass_count"] == 3
    assert scenario_check["aggregate"]["strict_pass_rate"] < 1.0


def test_cavenet_ablation_report_checks_parameter_roles(tmp_path) -> None:
    spec = cavenet_ablation_report_spec(dt=0.2, fps=4, include_assets=False)

    outputs = write_producer_report(spec, tmp_path / "cavenet-ablation")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    scenario_check = checks["extra"][0]
    roles = scenario_check["roles"]
    comparisons = scenario_check["comparisons"]

    assert checks["ok"]
    assert scenario_check["id"] == "cavenet_ablation"
    assert comparisons["fixed"]["ok"]
    assert roles["attention_gate"]["zero_attention_actual_mass"] == 0.0
    assert roles["external_input_gate"]["zero_external_attention_gain_actual_mass"] == 0.0
    assert roles["attention_allocation"]["zero_capacity_actual_mass"] == 0.0
    assert roles["attention_allocation"]["zero_capacity_expected_mass"] == 0.0
    assert roles["attention_allocation"]["external_only_actual_mass"] > 0.0
    assert roles["attention_allocation"]["external_only_expected_mass"] == 0.0
    assert roles["attention_allocation"]["internal_only_actual_mass"] == 0.0
    assert roles["attention_allocation"]["internal_only_expected_mass"] > 0.0
    assert (
        roles["expectation_readout"]["zero_expectation_expected_mass"]
        < roles["expectation_readout"]["fixed_expected_mass"]
    )
    assert (
        roles["memory_cell"]["zero_learning_final_memory_mass"]
        < roles["memory_cell"]["fixed_final_memory_mass"]
    )
    assert (
        roles["surprise_readout"]["high_surprise_total"]
        > roles["surprise_readout"]["fixed_surprise_total"]
    )
    assert roles["topology_layer"]["zero_topology_mass"] == 0.0


def test_role_dependency_contrasts_report_records_observed_contrasts(tmp_path) -> None:
    spec = role_dependency_contrasts_report_spec(dt=0.2, fps=4, include_assets=False)

    outputs = write_producer_report(spec, tmp_path / "role-dependency-contrasts")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    scenario_check = checks["extra"][0]
    roles = scenario_check["roles"]
    metrics = scenario_check["metrics"]
    contrasts = scenario_check["contrasts"]

    assert checks["ok"]
    assert scenario_check["id"] == "role_dependency_contrasts"
    relation_presence = roles["relation_presence"]
    observed_absences = roles["observed_absences"]
    assert all(relation_presence["positive-control"].values())
    assert observed_absences["positive-control"] == []
    assert set(observed_absences["passive-recorder"]) == set(
        relation_presence["positive-control"]
    )
    assert "prediction_temporal_dependency" in observed_absences["random-recurrent"]
    assert "prediction_temporal_dependency" in observed_absences["cosmetic-topology"]
    assert "value_future_attention" in observed_absences["cosmetic-topology"]
    assert roles["missing_expected_absences"]["cosmetic-topology"] == []
    assert roles["cosmetic_topology_exceeds_control"]
    assert roles["cosmetic_topology_does_not_supply_prediction_history"]
    assert metrics["passive-recorder"]["raw"]["unheard_actual_mass"] > 0.0
    assert metrics["cosmetic-topology"]["raw"]["cosmetic_topology_mass"] > 0.0
    assert set(contrasts["prediction_temporal_dependency"]) == {
        "hypothesis",
        "intervention_control",
        "expected_contrast",
        "observed_contrast",
        "interpretation",
    }
    assert not contrasts["prediction_temporal_dependency"]["observed_contrast"][
        "cosmetic-topology"
    ]
    assert not contrasts["value_future_attention"]["observed_contrast"][
        "cosmetic-topology"
    ]


def test_topology_atlas_report_writes_shared_projection_assets(tmp_path) -> None:
    spec = topology_atlas_report_spec(dt=0.2, fps=4, include_assets=True)

    outputs = write_producer_report(spec, tmp_path / "topology-atlas")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    atlas_check = checks["extra"][0]
    metrics_json = outputs.directory / "assets" / "topology_atlas_metrics.json"
    atlas_png = outputs.directory / "assets" / "topology_atlas.png"
    dashboard_png = outputs.directory / "assets" / "episode_set_dashboard.png"
    distances_json = outputs.directory / "assets" / "episode_set_distances.json"
    payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    distances_payload = json.loads(distances_json.read_text(encoding="utf-8"))

    assert checks["ok"]
    assert atlas_check["id"] == "topology_atlas"
    assert atlas_check["metrics"]["row_count"] == 11
    assert atlas_png.exists()
    assert atlas_png.stat().st_size > 0
    assert dashboard_png.exists()
    assert dashboard_png.stat().st_size > 0
    assert payload["projection"]["feature_x"] == "atlas-x"
    assert payload["projection"]["feature_y"] == "atlas-y"
    assert payload["episode_set"]["comparison_axis"] == "situation_or_control"
    assert distances_payload["episode_set"]["comparison_axis"] == "situation_or_control"
    assert "state_effect" in distances_payload["distances"]
    assert len(payload["entries"]) == 11
    assert (
        payload["entries"]["expectation-violation"]["metrics"]["experienced_mass"]
        > 0.0
    )
    assert (
        payload["entries"]["attention-bottleneck"]["metrics"]["actual_expected_l2"]
        > 0.0
    )


def test_cavenet_pressure_report_checks_adaptive_recovery(tmp_path) -> None:
    spec = cavenet_pressure_report_spec(dt=0.2, fps=4, include_assets=False)

    outputs = write_producer_report(spec, tmp_path / "cavenet-pressure")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    scenario_check = checks["extra"][0]
    roles = scenario_check["roles"]

    assert checks["ok"]
    assert scenario_check["id"] == "cavenet_pressure"
    assert (
        roles["parameter_development"]["final_learning_gain"]
        > roles["parameter_development"]["initial_learning_gain"]
    )
    assert (
        roles["functional_recovery"]["adaptive_memory_mass"]
        > roles["functional_recovery"]["fixed_weak_memory_mass"]
    )
    assert (
        roles["functional_recovery"]["adaptive_topology"]
        > roles["functional_recovery"]["fixed_weak_topology"]
    )
    assert roles["reference_closeness"]["distance_improvement"] > 0.0
    assert (
        roles["reference_closeness"]["adaptive_memory_distance"]
        < roles["reference_closeness"]["fixed_weak_memory_distance"]
    )


def test_cavenet_pressure_population_report_checks_generalized_recovery(
    tmp_path,
) -> None:
    spec = cavenet_pressure_population_report_spec(
        sequence_count=4,
        event_count=4,
        seed=41,
        dt=0.2,
        fps=4,
        include_assets=False,
    )

    outputs = write_producer_report(spec, tmp_path / "cavenet-pressure-population")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    scenario_check = checks["extra"][0]
    roles = scenario_check["roles"]

    assert checks["ok"]
    assert scenario_check["id"] == "cavenet_pressure_population"
    assert roles["population_recovery"]["mean_distance_improvement"] > 0.0
    assert roles["population_recovery"]["improved_sequence_count"] >= 3
    assert roles["population_recovery"]["mean_memory_distance_improvement"] > 0.0
    assert (
        roles["functional_recovery"]["mean_adaptive_memory_mass"]
        > roles["functional_recovery"]["mean_fixed_weak_memory_mass"]
    )
    assert (
        roles["functional_recovery"]["mean_adaptive_topology"]
        > roles["functional_recovery"]["mean_fixed_weak_topology"]
    )
    assert roles["pressure_response"]["mean_adaptive_config_delta"] > 0.0


def test_cavenet_controller_report_checks_latent_control_and_ablations(
    tmp_path,
) -> None:
    spec = cavenet_controller_report_spec(dt=0.2, fps=4, include_assets=False)

    outputs = write_producer_report(spec, tmp_path / "cavenet-controller")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    scenario_check = checks["extra"][0]
    roles = scenario_check["roles"]

    assert checks["ok"]
    assert scenario_check["id"] == "cavenet_controller"
    assert roles["controlled_recovery"]["distance_improvement"] > 0.0
    assert (
        roles["controlled_recovery"]["controller_memory_mass"]
        > roles["controlled_recovery"]["fixed_weak_memory_mass"]
    )
    assert (
        roles["controlled_recovery"]["controller_topology"]
        > roles["controlled_recovery"]["fixed_weak_topology"]
    )
    assert roles["controller_state"]["latent_norm"] > 0.0
    assert roles["controller_state"]["config_delta"] > 0.0
    assert roles["controller_state"]["mean_attention_capacity"] > 0.0
    assert roles["controller_state"]["mean_external_attention"] > 0.0
    assert roles["controller_state"]["mean_internal_expectation_attention"] > 0.0
    assert (
        roles["input_ablation_effects"]["full_external_attention_gain"]
        > roles["input_ablation_effects"][
            "no_attention_capacity_external_attention_gain"
        ]
    )
    assert (
        roles["input_ablation_effects"]["full_learning_gain"]
        > roles["input_ablation_effects"]["no_memory_learning_gain"]
    )
    assert (
        roles["input_ablation_effects"]["full_topology_gain"]
        > roles["input_ablation_effects"]["no_topology_topology_gain"]
    )


def test_cavenet_controller_population_report_checks_generalized_control(
    tmp_path,
) -> None:
    spec = cavenet_controller_population_report_spec(
        sequence_count=4,
        event_count=4,
        seed=61,
        dt=0.2,
        fps=4,
        include_assets=False,
    )

    outputs = write_producer_report(spec, tmp_path / "cavenet-controller-population")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    scenario_check = checks["extra"][0]
    roles = scenario_check["roles"]

    assert checks["ok"]
    assert scenario_check["id"] == "cavenet_controller_population"
    assert roles["population_recovery"]["mean_distance_improvement"] > 0.0
    assert roles["population_recovery"]["improved_sequence_count"] >= 3
    assert roles["population_recovery"]["mean_memory_distance_improvement"] > 0.0
    assert (
        roles["functional_recovery"]["mean_controller_memory_mass"]
        > roles["functional_recovery"]["mean_fixed_weak_memory_mass"]
    )
    assert (
        roles["functional_recovery"]["mean_controller_topology"]
        > roles["functional_recovery"]["mean_fixed_weak_topology"]
    )
    assert roles["controller_state"]["mean_controller_latent_norm"] > 0.0
    assert roles["controller_state"]["mean_attention_capacity"] > 0.0
    assert roles["controller_state"]["mean_external_attention"] > 0.0
    assert roles["controller_state"]["mean_internal_expectation_attention"] > 0.0
    assert (
        roles["controller_state"]["mean_controller_latent_norm"]
        > roles["controller_state"]["mean_pressureless_latent_norm"]
    )
    assert (
        roles["input_ablation_effects"][
            "mean_external_attention_gain_drop_without_attention_capacity"
        ]
        > 0.0
    )
    assert (
        roles["input_ablation_effects"][
            "mean_learning_gain_drop_without_memory"
        ]
        > 0.0
    )
    assert (
        roles["input_ablation_effects"][
            "mean_topology_gain_drop_without_topology"
        ]
        > 0.0
    )


def test_cavenet_controller_learning_report_checks_plastic_readout(
    tmp_path,
) -> None:
    spec = cavenet_controller_learning_report_spec(
        sequence_count=4,
        event_count=4,
        seed=91,
        dt=0.2,
        fps=4,
        include_assets=False,
    )

    outputs = write_producer_report(spec, tmp_path / "cavenet-controller-learning")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    scenario_check = checks["extra"][0]
    roles = scenario_check["roles"]

    assert checks["ok"]
    assert scenario_check["id"] == "cavenet_controller_learning"
    assert roles["learned_recovery"]["mean_distance_improvement_over_static"] > 0.0
    assert roles["learned_recovery"]["improved_over_static_count"] >= 3
    assert (
        roles["learned_recovery"]["mean_distance_improvement_over_fixed_weak"]
        > 0.0
    )
    assert (
        roles["functional_recovery"]["mean_learning_memory_mass"]
        > roles["functional_recovery"]["mean_static_memory_mass"]
    )
    assert (
        roles["functional_recovery"]["mean_learning_topology"]
        > roles["functional_recovery"]["mean_static_topology"]
    )
    assert (
        roles["readout_learning"]["mean_learning_config_delta"]
        > roles["readout_learning"]["mean_static_config_delta"]
    )
    assert roles["readout_learning"]["mean_learning_readout_delta_norm"] > 0.0
    assert roles["readout_learning"]["mean_learning_readout_updates"] > 0.0


def test_same_world_different_subjects_report_checks_claim(tmp_path) -> None:
    spec = same_world_different_subjects_report_spec(
        event_count=3,
        seed=202,
        dt=0.5,
        end=2.0,
        samples=12,
    )

    outputs = write_subject_comparison_report(
        spec,
        tmp_path / "same-world-different-subjects",
    )

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    metadata = json.loads(outputs.metadata_json.read_text(encoding="utf-8"))

    assert checks["ok"]
    assert checks["run_count"] == 5
    assert metadata["config"]["scenario"] == "same_world_different_subjects"
    assert outputs.dashboard_png.exists()
    assert outputs.dashboard_png.stat().st_size > 0
    assert outputs.topology_atlas_png.exists()
    assert outputs.topology_atlas_png.stat().st_size > 0
    assert outputs.topology_atlas_metrics_json.exists()
    assert outputs.episode_set_distances_json.exists()
    atlas_payload = json.loads(
        outputs.topology_atlas_metrics_json.read_text(encoding="utf-8")
    )
    distances_payload = json.loads(
        outputs.episode_set_distances_json.read_text(encoding="utf-8")
    )
    assert atlas_payload["episode_set"]["comparison_axis"] == "subject"
    assert distances_payload["episode_set"]["comparison_axis"] == "subject"
    assert len(outputs.run_episode_jsons) == 5
    assert all(path.exists() for path in outputs.run_episode_jsons)
    assert checks["metrics"]["zero_prior_effect_distance"] <= 1e-12
    assert checks["metrics"]["zero_prior_observed_distance"] > 0.0
    assert checks["metrics"]["zero_full_effect_distance"] > 0.0


def test_preference_shaped_topology_report_checks_claim(tmp_path) -> None:
    spec = preference_shaped_topology_report_spec(
        dt=0.5,
        end=1.0,
        samples=8,
    )

    outputs = write_subject_comparison_report(
        spec,
        tmp_path / "preference-shaped-topology",
    )

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    metadata = json.loads(outputs.metadata_json.read_text(encoding="utf-8"))

    assert checks["ok"]
    assert checks["run_count"] == 2
    assert metadata["config"]["scenario"] == "preference_shaped_topology"
    assert outputs.topology_atlas_png.exists()
    assert outputs.topology_atlas_png.stat().st_size > 0
    assert outputs.episode_set_distances_json.exists()
    atlas_payload = json.loads(
        outputs.topology_atlas_metrics_json.read_text(encoding="utf-8")
    )
    assert atlas_payload["episode_set"]["comparison_axis"] == "subject"
    metrics = checks["metrics"]
    assert metrics["warm_action"]["kind"] == "approach"
    assert metrics["warm_action"]["target_id"] == "warm_event"
    assert metrics["threat_avoid_action"]["kind"] == "avoid"
    assert metrics["threat_avoid_action"]["target_id"] == "threat_event"
    assert metrics["effect_distance"] > 0.0


def test_subject_ablation_matrix_report_checks_population_geometry(tmp_path) -> None:
    spec = subject_ablation_matrix_report_spec(
        event_count=3,
        seed=303,
        dt=0.5,
        end=2.0,
        samples=12,
    )

    outputs = write_matrix_report(spec, tmp_path / "subject-ablation")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    metadata = json.loads(outputs.metadata_json.read_text(encoding="utf-8"))
    matrix_check = checks["extra"][0]
    metrics = matrix_check["metrics"]

    assert checks["ok"]
    assert matrix_check["id"] == "subject_ablation_matrix"
    assert metadata["run_count"] == 10
    assert metadata["sequences"] == ["Q0"]
    assert metadata["subjects"] == ["flat", "prior"]
    assert len(metadata["variants"]) == 5
    assert metadata["factor_levels"]["treatment"][0]["id"] == "Q0"
    assert {item["id"] for item in metadata["factor_levels"]["start_condition"]} == {
        "flat",
        "prior",
    }
    condition_roles = {
        item["id"]: item["role"]
        for item in metadata["factor_levels"]["condition"]
    }
    assert condition_roles["baseline"] == "baseline"
    assert condition_roles["zero-attention"] == "negative_control"
    assert outputs.dashboard_png.exists()
    assert outputs.population_png.exists()
    assert outputs.population_png.stat().st_size > 0
    assert outputs.cluster_summary_json.exists()
    assert len(outputs.distance_jsons) == 4
    assert len(outputs.cluster_jsons) == 4
    assert len(outputs.run_episode_jsons) == 10
    assert metrics["zero_attention_effect_distance"] <= 1e-12
    assert metrics["baseline_zero_attention_effect_distance"] > 0.0
    assert metrics["baseline_prior_effect_distance"] <= 1e-12
    assert metrics["baseline_prior_observed_distance"] > 0.0
    assert metrics["baseline_zero_predictor_internal_distance"] > 0.0

    effect_clusters = json.loads(
        (outputs.directory / "clusters" / "state_effect.json").read_text(
            encoding="utf-8",
        )
    )
    assert effect_clusters["embedding"] == "state_effect"
    assert "factor_ids" in effect_clusters["clusters"][0]
    clustered_labels = [
        label
        for cluster in effect_clusters["clusters"]
        for label in cluster["labels"]
    ]
    assert sorted(clustered_labels) == sorted(metadata["labels"])
    summary = json.loads(outputs.cluster_summary_json.read_text(encoding="utf-8"))
    assert [item["embedding"] for item in summary["embeddings"]] == [
        "state_effect",
        "observed_memory",
        "subjective_trajectory",
        "active_context",
    ]
    assert "largest_cluster_condition_counts" in summary["embeddings"][0]

    first_run_metadata = json.loads(
        (outputs.directory / "runs" / "q0" / "flat" / "baseline" / "metadata.json").read_text(
            encoding="utf-8",
        )
    )
    assert first_run_metadata["comparison_role"] == "baseline"
    assert first_run_metadata["matched_set_id"] == "Q0"
    assert first_run_metadata["factors"]["start_condition"]["id"] == "flat"
    assert first_run_metadata["factors"]["condition"]["id"] == "baseline"


def test_topology_population_renderer_uses_population_factors(tmp_path) -> None:
    spec = subject_ablation_matrix_report_spec(
        event_count=2,
        seed=909,
        dt=0.5,
        end=1.0,
        samples=8,
    )
    records = list(spec.run_factory())
    population_records = tuple(record.to_population_record() for record in records)
    params = records[0].run.subject.params.topology
    output = tmp_path / "topology_population.png"
    scatter_output = tmp_path / "topology_scatter_migration.gif"
    subjective_output = tmp_path / "topology_subjective_trajectory.gif"

    trajectories = topology_trajectories(population_records, params)
    save_topology_population_dashboard(
        population_records,
        output,
        params,
        title="Same Treatment, Different Starts",
    )
    save_topology_scatter_migration(
        population_records,
        scatter_output,
        params,
        color_factor="start_condition",
        marker_factor="condition",
        point_kind="centroid",
        fps=2,
    )
    save_topology_scatter_migration(
        population_records,
        subjective_output,
        params,
        color_factor="start_condition",
        marker_factor="condition",
        point_kind="subjective",
        fps=2,
    )

    assert len(trajectories) == len(records)
    assert output.exists()
    assert output.stat().st_size > 0
    assert scatter_output.exists()
    assert scatter_output.stat().st_size > 0
    assert subjective_output.exists()
    assert subjective_output.stat().st_size > 0
    assert {trajectory.treatment_id for trajectory in trajectories} == {"Q0"}
    assert {trajectory.start_condition_id for trajectory in trajectories} == {
        "flat",
        "prior",
    }
    assert "baseline" in {trajectory.condition_id for trajectory in trajectories}
    for trajectory in trajectories:
        assert trajectory.density_deltas.shape[0] == trajectory.times.size
        assert trajectory.centroids.shape == (trajectory.times.size, 2)
        assert trajectory.mass.shape == trajectory.times.shape
        assert trajectory.correction_distance.shape == trajectory.times.shape


def test_initial_conditions_matrix_varies_only_start_state(tmp_path) -> None:
    spec = initial_conditions_matrix_report_spec(
        condition_count=9,
        event_count=2,
        seed=707,
        dt=0.5,
        end=1.0,
        samples=8,
    )
    outputs = write_matrix_report(spec, tmp_path / "initial-conditions")
    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    metadata = json.loads(outputs.metadata_json.read_text(encoding="utf-8"))
    matrix_check = checks["extra"][0]

    assert checks["ok"]
    assert matrix_check["id"] == "initial_conditions_matrix"
    assert metadata["run_count"] == 9
    assert metadata["sequences"] == ["Q0"]
    assert metadata["variants"] == ["baseline"]
    assert len(metadata["subjects"]) == 9
    assert {item["id"] for item in metadata["factor_levels"]["condition"]} == {
        "baseline",
    }
    assert len(metadata["factor_levels"]["start_condition"]) == 9
    assert matrix_check["metrics"]["mean_memory_trajectory_distance"] > 0.0


def test_initial_conditions_matrix_crosses_treatments_and_starts(tmp_path) -> None:
    spec = initial_conditions_matrix_report_spec(
        condition_count=4,
        treatment_count=3,
        event_count=2,
        seed=717,
        dt=0.5,
        end=1.0,
        samples=8,
    )
    outputs = write_matrix_report(spec, tmp_path / "initial-treatments")
    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    metadata = json.loads(outputs.metadata_json.read_text(encoding="utf-8"))
    matrix_check = checks["extra"][0]

    assert checks["ok"]
    assert metadata["run_count"] == 12
    assert metadata["sequences"] == ["Q000", "Q001", "Q002"]
    assert metadata["variants"] == ["baseline"]
    assert len(metadata["subjects"]) == 4
    assert len(metadata["factor_levels"]["treatment"]) == 3
    assert len(metadata["factor_levels"]["start_condition"]) == 4
    assert matrix_check["metrics"]["sequence_count"] == 3
    assert matrix_check["metrics"]["subject_count"] == 4
    assert matrix_check["metrics"]["mean_memory_trajectory_distance"] > 0.0
    assert matrix_check["metrics"]["mean_same_start_treatment_distance"] > 0.0


def test_conversation_matrix_report_checks_context_configs(tmp_path) -> None:
    fixture = ConversationMatrixFixture(
        "mock-memory",
        (
            ConversationTurn("user", "Prior context?"),
            ConversationTurn("assistant", "Protocol memory."),
            ConversationTurn("user", "Expected actual?"),
            ConversationTurn("assistant", "Whole turns, token metrics."),
        ),
    )
    spec = conversation_text_config_matrix_report_spec(
        fixtures=(fixture,),
        episode_factory=fake_conversation_episode_for_report,
        samples=12,
    )

    outputs = write_matrix_report(spec, tmp_path / "conversation-matrix")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    metadata = json.loads(outputs.metadata_json.read_text(encoding="utf-8"))
    matrix_check = checks["extra"][0]
    metrics = matrix_check["metrics"]

    assert checks["ok"]
    assert matrix_check["id"] == "conversation_text_config_matrix"
    assert metadata["run_count"] == 3
    assert metadata["sequences"] == ["mock-memory"]
    assert metadata["subjects"] == ["conversation"]
    assert metrics["attended_top_1_max_active"] == 1
    assert metrics["recent_2_max_active"] <= 2
    assert metrics["full_context_max_active"] == 3
    assert outputs.dashboard_png.exists()
    assert outputs.population_png.exists()
    assert outputs.cluster_summary_json.exists()
    assert len(outputs.run_episode_jsons) == 3
    assert metrics["same_fixture_context_config_distance"] > 0.0


def test_population_clusters_matrix_report_writes_cluster_artifacts(tmp_path) -> None:
    spec = population_clusters_matrix_report_spec(
        sequence_count=3,
        event_count=3,
        seed=404,
        dt=0.5,
        end=2.0,
        samples=12,
    )

    outputs = write_matrix_report(spec, tmp_path / "population-clusters")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    metadata = json.loads(outputs.metadata_json.read_text(encoding="utf-8"))
    matrix_check = checks["extra"][0]
    metrics = matrix_check["metrics"]

    assert checks["ok"]
    assert matrix_check["id"] == "population_clusters_matrix"
    assert metadata["run_count"] == 30
    assert metadata["sequences"] == ["Q000", "Q001", "Q002"]
    assert len(outputs.cluster_jsons) == 4
    assert len(outputs.distance_jsons) == 4
    assert outputs.population_png.exists()
    assert outputs.population_png.stat().st_size > 0
    assert outputs.cluster_summary_json.exists()
    assert len(outputs.run_episode_jsons) == 30
    assert metrics["max_zero_attention_effect_distance"] <= 1e-12
    assert metrics["mean_observed_subject_distance_baseline"] > 0.0
    assert metrics["mean_internal_baseline_zero_predictor_distance"] > 0.0

    effect_clusters = json.loads(
        (outputs.directory / "clusters" / "state_effect.json").read_text(
            encoding="utf-8",
        )
    )
    clustered_labels = [
        label
        for cluster in effect_clusters["clusters"]
        for label in cluster["labels"]
    ]
    assert sorted(clustered_labels) == sorted(metadata["labels"])
    zero_cluster = next(
        cluster
        for cluster in effect_clusters["clusters"]
        if set(cluster["variant_ids"]) == {"zero-attention"}
    )
    assert zero_cluster["size"] == 6
    summary = json.loads(outputs.cluster_summary_json.read_text(encoding="utf-8"))
    effect_summary = next(
        item
        for item in summary["embeddings"]
        if item["embedding"] == "state_effect"
    )
    assert effect_summary["largest_cluster_size"] >= 6


def test_gpt2_text_config_matrix_report_compares_fixed_texts(tmp_path) -> None:
    spec = gpt2_text_config_matrix_report_spec(
        texts=(("hello", "hello text"), ("pattern", "pattern text")),
        feature_count=2,
        samples=8,
        episode_factory=fake_gpt2_matrix_episode,
    )

    outputs = write_matrix_report(spec, tmp_path / "gpt2-text-config")

    checks = json.loads(outputs.checks_json.read_text(encoding="utf-8"))
    metadata = json.loads(outputs.metadata_json.read_text(encoding="utf-8"))
    matrix_check = checks["extra"][0]
    metrics = matrix_check["metrics"]

    assert checks["ok"]
    assert matrix_check["id"] == "gpt2_text_config_matrix"
    assert metadata["run_count"] == 8
    assert metadata["sequences"] == ["hello", "pattern"]
    assert metadata["subjects"] == ["gpt2"]
    assert metadata["variants"] == [
        "attended-top-1",
        "attended-top-3",
        "current-token",
        "full-context",
    ]
    assert outputs.dashboard_png.exists()
    assert outputs.population_png.exists()
    assert outputs.population_png.stat().st_size > 0
    assert outputs.cluster_summary_json.exists()
    assert len(outputs.distance_jsons) == 4
    assert len(outputs.cluster_jsons) == 4
    assert len(outputs.run_episode_jsons) == 8
    assert metrics["text_count"] == 2
    assert metrics["config_count"] == 4
    assert metrics["same_text_context_config_distance"] > 0.0
    assert metrics["current_token_max_active"] == 1
    active_context_clusters = json.loads(
        (outputs.directory / "clusters" / "active_context.json").read_text(
            encoding="utf-8",
        )
    )
    assert active_context_clusters["embedding"] == "active_context"
    assert any(
        cluster["size"] == 2 and len(set(cluster["variant_ids"])) == 1
        for cluster in active_context_clusters["clusters"]
    )
    assert metrics["full_context_max_active"] > metrics["current_token_max_active"]
    assert metrics["different_text_internal_distance"] > 0.0


def fake_gpt2_episode():
    token_ids = np.array([0, 1, 2, 3], dtype=int)
    token_texts = ["Hello", " Paul", " likes", "!"]
    embedding_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    logits = np.array(
        [
            [0.0, 3.0, 1.0, -1.0, 0.5],
            [0.0, -1.0, 4.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 3.5, 0.2],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    hidden_states = np.array(
        [
            [0.2, 0.1, 0.0],
            [0.0, 0.5, 0.2],
            [0.1, 0.0, 0.7],
            [0.4, 0.2, 0.1],
        ],
        dtype=float,
    )
    attentions = np.array(
        [
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.8, 0.2, 0.0, 0.0],
                    [0.2, 0.3, 0.5, 0.0],
                    [0.1, 0.2, 0.3, 0.4],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.6, 0.4, 0.0, 0.0],
                    [0.1, 0.7, 0.2, 0.0],
                    [0.4, 0.1, 0.2, 0.3],
                ],
            ]
        ],
        dtype=float,
    )
    return build_gpt2_episode(
        source_name="gpt2",
        token_ids=token_ids,
        token_texts=token_texts,
        embedding_matrix=embedding_matrix,
        logits=logits,
        hidden_states=hidden_states,
        attentions=attentions,
        feature_count=2,
        active_top_k=2,
        top_prediction_k=2,
        decode_token=lambda token_id: f"tok{token_id}",
    )


def fake_gpt2_matrix_episode(text, config):
    if "pattern" in text:
        token_ids = np.array([0, 2, 4, 2, 4], dtype=int)
        token_texts = ["red", " blue", " red", " blue", " red"]
        hidden_offset = np.array([0.0, 0.5, 1.0, 0.0], dtype=float)
    else:
        token_ids = np.array([0, 1, 2, 3, 4], dtype=int)
        token_texts = ["Hello", " Paul", " likes", " cave", "."]
        hidden_offset = np.array([1.0, 0.0, 0.0, 0.5], dtype=float)

    embedding_matrix = np.array(
        [
            [1.0, 0.0, 0.1, 0.0],
            [0.0, 1.0, 0.0, 0.2],
            [0.2, 0.0, 1.0, 0.0],
            [0.0, 0.2, 0.0, 1.0],
            [0.7, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.7],
        ],
        dtype=float,
    )
    logits = np.full((len(token_ids), embedding_matrix.shape[0]), -2.0, dtype=float)
    for index, token_id in enumerate(token_ids[1:], start=1):
        logits[index - 1, token_id] = 4.0
        logits[index - 1, (token_id + 1) % embedding_matrix.shape[0]] = 1.0

    base_hidden = np.array(
        [
            [0.2, 0.1, 0.0, 0.0],
            [0.4, 0.2, 0.1, 0.2],
            [0.1, 0.5, 0.3, 0.1],
            [0.0, 0.3, 0.8, 0.4],
            [0.3, 0.2, 0.5, 0.9],
        ],
        dtype=float,
    )
    hidden_states = base_hidden + hidden_offset

    token_count = len(token_ids)
    attentions = np.zeros((1, 2, token_count, token_count), dtype=float)
    for head in range(2):
        for target in range(token_count):
            weights = np.arange(1, target + 2, dtype=float)
            if head == 1:
                weights = weights[::-1]
            attentions[0, head, target, : target + 1] = weights / np.sum(weights)

    return build_gpt2_episode(
        source_name="gpt2",
        token_ids=token_ids,
        token_texts=token_texts,
        embedding_matrix=embedding_matrix,
        logits=logits,
        hidden_states=hidden_states,
        attentions=attentions,
        feature_count=config.feature_count,
        active_input_mode=config.active_input_mode,
        active_top_k=config.active_top_k,
        top_prediction_k=2,
        decode_token=lambda token_id: f"tok{token_id}",
    )


def fake_conversation_episode_for_report(turns, config=None):
    turns = tuple(turns)
    token_count = len(turns) * 2
    vocab_size = max(6, token_count + 1)
    token_ids = np.array(
        [(index * 2 + len(turns[index % len(turns)].role)) % vocab_size for index in range(token_count)],
        dtype=int,
    )
    embedding_matrix = np.eye(vocab_size, 4, dtype=float)
    for index in range(vocab_size):
        embedding_matrix[index] += np.array(
            [
                0.03 * index,
                0.02 * (index % 3),
                0.01 * len(turns),
                0.04 * (index % 2),
            ],
            dtype=float,
        )
    logits = np.full((token_count, vocab_size), -2.0, dtype=float)
    for index, token_id in enumerate(token_ids[1:], start=1):
        logits[index - 1, token_id] = 4.0
        logits[index - 1, (token_id + 1) % vocab_size] = 1.0
    hidden_states = np.vstack(
        [
            np.array(
                [
                    0.1 * (index + 1),
                    0.2 * (index % 3),
                    0.05 * len(turns[index // 2].text),
                    0.15 * (index // 2),
                ],
                dtype=float,
            )
            for index in range(token_count)
        ]
    )
    attentions = np.zeros((1, 2, token_count, token_count), dtype=float)
    for head in range(2):
        for target in range(token_count):
            weights = np.arange(1, target + 2, dtype=float)
            if head == 1:
                weights = weights[::-1]
            attentions[0, head, target, : target + 1] = weights / np.sum(weights)

    segments = [
        ConversationSegment(
            id=f"turn:{index}",
            role=turn.role,
            text=turn.text,
            formatted_text=f"{turn.role.capitalize()}: {turn.text}\n",
            start_token=index * 2,
            end_token=index * 2 + 2,
            order_index=index,
        )
        for index, turn in enumerate(turns)
    ]
    context_selection = getattr(config, "context_selection", "attended_top_k")
    context_top_k = getattr(config, "context_top_k", 2)
    feature_count = getattr(config, "feature_count", 3)
    return build_conversation_episode(
        source_name="conversation",
        backend_name="fake-gpt2",
        segments=segments,
        token_ids=token_ids,
        embedding_matrix=embedding_matrix,
        logits=logits,
        hidden_states=hidden_states,
        attentions=attentions,
        feature_count=feature_count,
        context_selection=context_selection,
        context_top_k=context_top_k,
        top_prediction_k=2,
        decode_token=lambda token_id: f"tok{token_id}",
    )
