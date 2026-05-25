from __future__ import annotations

from cave.commitments.topology import SubjectiveTopologyParams, SubjectiveTopologyPrior
from cave.demonstrations.scenarios.attention_bottleneck import (
    attention_bottleneck_report_spec,
)
from cave.demonstrations.scenarios.expectation_violation import (
    expectation_violation_report_spec,
)
from cave.demonstrations.scenarios.importance_weighted_event import (
    importance_weighted_event_report_spec,
)
from cave.demonstrations.scenarios.objective_attention_shift import (
    objective_attention_shift_report_spec,
)
from cave.demonstrations.scenarios.representational_compression import (
    representational_compression_report_spec,
)
from cave.demonstrations.scenarios.role_dependency_contrasts import (
    build_role_dependency_contrast_episode,
)
from cave.demonstrations.scenarios.unseen_modality import unseen_modality_report_spec
from cave.demonstrations.scenarios.valence_attractor_repulsor import (
    valence_attractor_repulsor_report_spec,
)
from cave.observation.episode_runs import EpisodeSet, LabeledEpisode
from cave.observation.experience import FeatureProjection
from cave.presentation.renderers.topology_atlas_renderer import (
    save_topology_atlas,
    save_topology_atlas_metrics,
    topology_atlas_metrics_payload,
)
from cave.presentation.renderers.episode_set_dashboard import (
    save_episode_set_dashboard,
    save_episode_set_distances_json,
)
from cave.presentation.reports.specs import (
    ProducerReportSpec,
    ReportExtraAsset,
    ReportSection,
)


def topology_atlas_report_spec(
    *,
    dt: float = 0.2,
    fps: int = 8,
    include_assets: bool = True,
) -> ProducerReportSpec:
    params = topology_atlas_params()

    def entries() -> EpisodeSet:
        return topology_atlas_entries(dt=dt, fps=fps)

    extra_assets = ()
    if include_assets:
        extra_assets = (
            ReportExtraAsset(
                id="topology_atlas_png",
                title="Topology Atlas",
                filename="topology_atlas.png",
                writer=lambda _episode, output: save_topology_atlas(
                    entries(),
                    output,
                    params,
                ),
            ),
            ReportExtraAsset(
                id="topology_atlas_metrics_json",
                title="Topology Atlas Metrics JSON",
                filename="topology_atlas_metrics.json",
                writer=lambda _episode, output: save_topology_atlas_metrics(
                    entries(),
                    output,
                    params,
                ),
            ),
            ReportExtraAsset(
                id="episode_set_dashboard_png",
                title="Episode Set Dashboard",
                filename="episode_set_dashboard.png",
                writer=lambda _episode, output: save_episode_set_dashboard(
                    entries(),
                    output,
                    samples=24,
                    title="Scenario And Control Episode Set",
                ),
            ),
            ReportExtraAsset(
                id="episode_set_distances_json",
                title="Episode Set Distances JSON",
                filename="episode_set_distances.json",
                writer=lambda _episode, output: save_episode_set_distances_json(
                    entries(),
                    output,
                    samples=24,
                ),
            ),
        )

    return ProducerReportSpec(
        id="topology-atlas",
        title="Causal Probe: Topology Atlas",
        episode_factory=lambda: expectation_violation_report_spec(
            dt=dt,
            fps=fps,
            include_assets=False,
        ).episode_factory(),
        input_summary=(
            "scenario and control episodes reconstructed from one flat topology "
            "prior under a shared atlas projection"
        ),
        description=(
            "Renders a small-multiple atlas of topology traces across scenarios "
            "and role-dependency controls. The atlas is a derived readout: it "
            "does not change the model update, but makes scenario impacts "
            "visually comparable from the same initial topology state."
        ),
        views=(),
        view_assets=(),
        extra_assets=extra_assets,
        frame_time=1.0,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "topology_atlas",
            "scenario": "topology_atlas",
            "dt": dt,
            "fps": fps,
        },
        checks=(lambda _episode: check_topology_atlas(dt=dt, fps=fps),),
        sections=(
            ReportSection(
                title="Atlas",
                body=(
                    "Each row starts from the same flat topology prior and uses "
                    "the same two-axis atlas projection. The columns show the "
                    "experienced topology delta, the expected-input "
                    "counterfactual path, the actual-input path, and the "
                    "actual-minus-expected difference."
                ),
                asset_ids=("topology_atlas_png",),
            ),
            ReportSection(
                title="Population Geometry",
                body=(
                    "The same labeled episode set can also be read as a "
                    "population: distance matrices and MDS plots show which "
                    "situations and controls induce similar trajectory "
                    "signatures."
                ),
                asset_ids=("episode_set_dashboard_png",),
            ),
            ReportSection(
                title="Metrics",
                body=(
                    "The metrics record mass, peak density, centroid, spread, "
                    "actual/expected topology divergence, and episode-set "
                    "distance matrices."
                ),
                asset_ids=("topology_atlas_metrics_json", "episode_set_distances_json"),
            ),
        ),
    )


def topology_atlas_params() -> SubjectiveTopologyParams:
    return SubjectiveTopologyParams(
        feature_x=FeatureProjection(
            name="atlas-x",
            weights={
                "visual_signal": 1.0,
                "energy": 1.0,
                "impact": 1.0,
                "comfort": 1.0,
                "dominant": 1.0,
                "visual": 1.0,
            },
        ),
        feature_y=FeatureProjection(
            name="atlas-y",
            weights={
                "audio_signal": 1.0,
                "warmth": 1.0,
                "context": 1.0,
                "threat": 1.0,
                "secondary": 1.0,
                "detail": 0.5,
                "audio": 1.0,
                "value": -1.0,
            },
        ),
        resolution=64,
        prior=SubjectiveTopologyPrior(),
    )


def topology_atlas_entries(
    *,
    dt: float = 0.2,
    fps: int = 8,
) -> EpisodeSet:
    scenario_specs = (
        (
            "expectation-violation",
            "Expectation violation",
            expectation_violation_report_spec,
        ),
        ("unseen-modality", "Unseen modality", unseen_modality_report_spec),
        (
            "attention-bottleneck",
            "Attention bottleneck",
            attention_bottleneck_report_spec,
        ),
        (
            "importance-weighted-event",
            "Importance weighted",
            importance_weighted_event_report_spec,
        ),
        (
            "valence-attractor-repulsor",
            "Valence attractor",
            valence_attractor_repulsor_report_spec,
        ),
        (
            "objective-attention-shift",
            "Objective attention",
            objective_attention_shift_report_spec,
        ),
        (
            "representational-compression",
            "Compression",
            representational_compression_report_spec,
        ),
    )
    entries = [
        LabeledEpisode(
            id=entry_id,
            label=label,
            episode=spec_factory(dt=dt, fps=fps, include_assets=False).episode_factory(),
            group="scenario",
            series="situation",
        )
        for entry_id, label, spec_factory in scenario_specs
    ]
    entries.extend(
        [
            LabeledEpisode(
                id="role-positive-control",
                label="Role positive",
                episode=build_role_dependency_contrast_episode("positive-control"),
                group="role_dependency_control",
                series="control",
            ),
            LabeledEpisode(
                id="role-passive-recorder",
                label="Role passive",
                episode=build_role_dependency_contrast_episode("passive-recorder"),
                group="role_dependency_control",
                series="control",
            ),
            LabeledEpisode(
                id="role-random-recurrent",
                label="Role recurrent",
                episode=build_role_dependency_contrast_episode("random-recurrent"),
                group="role_dependency_control",
                series="control",
            ),
            LabeledEpisode(
                id="role-cosmetic-topology",
                label="Role cosmetic",
                episode=build_role_dependency_contrast_episode("cosmetic-topology"),
                group="role_dependency_control",
                series="control",
            ),
        ]
    )
    return EpisodeSet(
        id="topology-atlas",
        title="Topology Atlas",
        comparison_axis="situation_or_control",
        episodes=tuple(entries),
    )


def check_topology_atlas(*, dt: float = 0.2, fps: int = 8) -> dict[str, object]:
    params = topology_atlas_params()
    payload = topology_atlas_metrics_payload(
        topology_atlas_entries(dt=dt, fps=fps),
        params,
    )
    entries = payload["entries"]
    errors = []
    if len(entries) < 8:
        errors.append("topology atlas has too few rows")
    max_divergence = max(
        entry["metrics"]["actual_expected_l2"]
        for entry in entries.values()
    )
    max_mass = max(
        entry["metrics"]["experienced_mass"]
        for entry in entries.values()
    )
    if not max_divergence > 0.0:
        errors.append("topology atlas did not record actual/expected divergence")
    if not max_mass > 0.0:
        errors.append("topology atlas did not record experienced topology mass")
    return {
        "id": "topology_atlas",
        "ok": not errors,
        "errors": errors,
        "metrics": {
            "row_count": len(entries),
            "max_actual_expected_l2": max_divergence,
            "max_experienced_mass": max_mass,
        },
    }
