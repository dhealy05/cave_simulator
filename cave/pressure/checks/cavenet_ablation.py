from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

import numpy as np

from cave.substrates.cavenet import CaveNet, CaveNetConfig, CaveNetProducer, compare_cavenet_to_cave
from cave.observation.episodes import CaveProducer, Episode
from cave.demonstrations.examples import demo_model, model_for_sequence
from cave.presentation.reports.specs import ProducerReportSpec, ReportSection
from cave.commitments.attention import (
    external_only_attention_profile,
    internal_only_attention_profile,
    zero_attention_profile,
)
from cave.commitments.topology import SubjectiveTopologyParams, SubjectiveTopologyPrior
from cave.observation.views import default_views


@dataclass(frozen=True)
class CaveNetVariant:
    id: str
    label: str
    config: CaveNetConfig
    attention_profile: object | None = None
    initial_memory: np.ndarray | None = None


def cavenet_ablation_report_spec(
    *,
    dt: float = 0.2,
    fps: int = 8,
    include_assets: bool = True,
) -> ProducerReportSpec:
    def build_episode() -> Episode:
        return build_cavenet_episode(CaveNetConfig(), dt=dt)

    return ProducerReportSpec(
        id="cavenet-ablation",
        title="CaveNet: Parameter Ablation",
        episode_factory=build_episode,
        input_summary=f"demo sequence through fixed CaveNet(dt={dt})",
        description=(
            "Runs the fixed CaveNet representation and perturbed CaveNet variants "
            "against symbolic Cave. The default configuration should match the "
            "symbolic core path exactly; perturbed gains show which architecture "
            "roles depend on which blocks."
        ),
        views=default_views(),
        view_assets=(),
        frame_time=2.0,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "cavenet",
            "scenario": "cavenet_ablation",
            "dt": dt,
            "fps": fps,
        },
        checks=(lambda episode: check_cavenet_ablation(dt=dt),),
        sections=(
            ReportSection(
                title="Representation Equivalence",
                body=(
                    "The default CaveNet configuration re-expresses symbolic Cave "
                    "as fixed network-form blocks. Its core readouts should match "
                    "symbolic Cave exactly."
                ),
            ),
            ReportSection(
                title="Parameterized Blocks",
                body=(
                    "Ablations perturb explicit gains for external input, "
                    "expectation, learning, surprise, and topology deposits. "
                    "Attention-allocation controls separately set zero capacity, "
                    "external-only attention, and internal-only expectation "
                    "attention."
                ),
            ),
        ),
    )


def cavenet_variants() -> tuple[CaveNetVariant, ...]:
    return (
        CaveNetVariant("fixed", "Fixed CaveNet", CaveNetConfig()),
        CaveNetVariant(
            "zero-attention",
            "Zero external attention gain",
            CaveNetConfig(attention_gain=0.0),
        ),
        CaveNetVariant(
            "zero-attention-capacity",
            "Zero attention capacity",
            CaveNetConfig(),
            attention_profile=zero_attention_profile(),
        ),
        CaveNetVariant(
            "external-only-attention",
            "External-only attention",
            CaveNetConfig(),
            attention_profile=external_only_attention_profile(),
        ),
        CaveNetVariant(
            "internal-only-attention",
            "Internal-only attention",
            CaveNetConfig(),
            attention_profile=internal_only_attention_profile(),
            initial_memory=np.linspace(0.1, 0.9, len(demo_model().vocabulary)),
        ),
        CaveNetVariant(
            "zero-expectation",
            "Zero expectation readout",
            CaveNetConfig(expectation_gain=0.0),
        ),
        CaveNetVariant(
            "zero-learning",
            "Zero learning-rate gain",
            CaveNetConfig(learning_rate_gain=0.0),
        ),
        CaveNetVariant(
            "high-surprise",
            "High surprise gain",
            CaveNetConfig(surprise_gain=2.0),
        ),
        CaveNetVariant(
            "zero-topology",
            "Zero topology deposit",
            CaveNetConfig(topology_deposit_gain=0.0, topology_transition_gain=0.0),
        ),
    )


def build_cavenet_episode(
    config: CaveNetConfig,
    *,
    dt: float,
    source_name: str = "cavenet",
    attention_profile=None,
    initial_memory: np.ndarray | None = None,
) -> Episode:
    model = _base_model()
    if initial_memory is not None:
        model.subject_state.memory.vector = np.asarray(
            initial_memory,
            dtype=float,
        ).copy()
    params = model.params
    if attention_profile is not None:
        params = replace(params, attention=attention_profile)
    cavenet = CaveNet.from_subject_state(
        sequence=model.sequence,
        subject_state=model.subject_state,
        params=params,
        vocabulary=model.vocabulary,
        sensorium=model.sensorium,
        config=config,
    )
    return CaveNetProducer(cavenet, name=source_name).run(dt=dt)


def check_cavenet_ablation(*, dt: float) -> dict[str, object]:
    cave_episode = CaveProducer(_base_model()).run(dt=dt)
    episodes = {
        variant.id: build_cavenet_episode(
            variant.config,
            dt=dt,
            source_name=f"cavenet:{variant.id}",
            attention_profile=variant.attention_profile,
            initial_memory=variant.initial_memory,
        )
        for variant in cavenet_variants()
    }
    comparisons = {
        variant_id: compare_cavenet_to_cave(cave_episode, episode).to_dict()
        for variant_id, episode in episodes.items()
    }
    metrics = {
        variant_id: _variant_metrics(cave_episode, episode)
        for variant_id, episode in episodes.items()
    }
    roles = {
        "external_input_gate": {
            "fixed_actual_mass": metrics["fixed"]["actual_mass"],
            "zero_external_attention_gain_actual_mass": metrics["zero-attention"][
                "actual_mass"
            ],
            "zero_attention_actual_mass": metrics["zero-attention"]["actual_mass"],
        },
        "attention_gate": {
            "fixed_actual_mass": metrics["fixed"]["actual_mass"],
            "zero_attention_actual_mass": metrics["zero-attention"]["actual_mass"],
        },
        "attention_allocation": {
            "fixed_actual_mass": metrics["fixed"]["actual_mass"],
            "fixed_expected_mass": metrics["fixed"]["expected_mass"],
            "zero_capacity_actual_mass": metrics["zero-attention-capacity"][
                "actual_mass"
            ],
            "zero_capacity_expected_mass": metrics["zero-attention-capacity"][
                "expected_mass"
            ],
            "external_only_actual_mass": metrics["external-only-attention"][
                "actual_mass"
            ],
            "external_only_expected_mass": metrics["external-only-attention"][
                "expected_mass"
            ],
            "internal_only_actual_mass": metrics["internal-only-attention"][
                "actual_mass"
            ],
            "internal_only_expected_mass": metrics["internal-only-attention"][
                "expected_mass"
            ],
        },
        "expectation_readout": {
            "fixed_expected_mass": metrics["fixed"]["expected_mass"],
            "zero_expectation_expected_mass": metrics["zero-expectation"]["expected_mass"],
            "zero_expectation_internal_distance": comparisons["zero-expectation"]["metrics"][
                "max_expected_distance"
            ],
        },
        "memory_cell": {
            "fixed_final_memory_mass": metrics["fixed"]["final_memory_mass"],
            "zero_learning_final_memory_mass": metrics["zero-learning"][
                "final_memory_mass"
            ],
        },
        "surprise_readout": {
            "fixed_surprise_total": metrics["fixed"]["surprise_total"],
            "high_surprise_total": metrics["high-surprise"]["surprise_total"],
        },
        "topology_layer": {
            "fixed_topology_mass": metrics["fixed"]["topology_mass"],
            "zero_topology_mass": metrics["zero-topology"]["topology_mass"],
        },
    }

    errors = []
    if not comparisons["fixed"]["ok"]:
        errors.append("fixed CaveNet did not match symbolic Cave")
    if not (
        roles["external_input_gate"]["zero_external_attention_gain_actual_mass"]
        < roles["external_input_gate"]["fixed_actual_mass"]
    ):
        errors.append("zero external attention gain did not reduce actual input mass")
    if not roles["attention_allocation"]["zero_capacity_actual_mass"] == 0.0:
        errors.append("zero attention capacity did not collapse actual input mass")
    if not roles["attention_allocation"]["zero_capacity_expected_mass"] == 0.0:
        errors.append("zero attention capacity did not collapse expected input mass")
    if not roles["attention_allocation"]["external_only_actual_mass"] > 0.0:
        errors.append("external-only attention did not preserve actual input mass")
    if not roles["attention_allocation"]["external_only_expected_mass"] == 0.0:
        errors.append("external-only attention did not collapse expected input mass")
    if not roles["attention_allocation"]["internal_only_actual_mass"] == 0.0:
        errors.append("internal-only attention did not collapse actual input mass")
    if not roles["attention_allocation"]["internal_only_expected_mass"] > 0.0:
        errors.append("internal-only attention did not preserve expected input mass")
    if not roles["expectation_readout"]["zero_expectation_expected_mass"] < roles["expectation_readout"]["fixed_expected_mass"]:
        errors.append("zero expectation gain did not reduce expected input mass")
    if not roles["memory_cell"]["zero_learning_final_memory_mass"] < roles["memory_cell"]["fixed_final_memory_mass"]:
        errors.append("zero learning gain did not reduce final memory mass")
    if not roles["surprise_readout"]["high_surprise_total"] > roles["surprise_readout"]["fixed_surprise_total"]:
        errors.append("high surprise gain did not increase surprise")
    if not roles["topology_layer"]["zero_topology_mass"] < roles["topology_layer"]["fixed_topology_mass"]:
        errors.append("zero topology gain did not reduce topology mass")

    return {
        "id": "cavenet_ablation",
        "ok": not errors,
        "errors": errors,
        "comparisons": comparisons,
        "metrics": metrics,
        "roles": roles,
    }


def _variant_metrics(cave_episode: Episode, episode: Episode) -> dict[str, float]:
    actual_mass = sum(_norm(obs.actual) for obs in episode.observations)
    expected_mass = sum(_norm(obs.expected) for obs in episode.observations)
    surprise_total = sum(obs.surprise for obs in episode.observations)
    final_memory_mass = _norm(episode.observations[-1].memory_state)
    topology_mass = float(episode.metadata.get("cavenet_final_topology_mass", 0.0))
    return {
        "actual_mass": actual_mass,
        "expected_mass": expected_mass,
        "surprise_total": surprise_total,
        "final_memory_mass": final_memory_mass,
        "topology_mass": topology_mass,
        "cave_actual_mass": sum(_norm(obs.actual) for obs in cave_episode.observations),
    }


def _base_model():
    model = demo_model(seed=1)
    params = replace(
        model.params,
        topology=SubjectiveTopologyParams(prior=SubjectiveTopologyPrior()),
    )
    return model_for_sequence(
        model.sequence,
        params=params,
        vocabulary=model.vocabulary,
    )


def _norm(value) -> float:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.linalg.norm(array.ravel()) / np.sqrt(array.size))
