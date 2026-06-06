from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from cave.observation.energy import EnergyLedger, summarize_episode_energy
from cave.observation.episodes import Episode, EpisodeObservation
from cave.observation.projections import encode_value
from cave.observation.views import default_views
from cave.presentation.reports.specs import (
    ProducerReportSpec,
    ReportExtraAsset,
    ReportSection,
)
from cave.pressure.checks.cavenet_controller import build_controller_episode
from cave.pressure.checks.common_behaviors import (
    PROBES,
    SUBSTRATES,
    build_common_behavior_episode,
)
from cave.pressure.checks.evolved_exposure import build_evolved_exposure_episode


SUBSTRATE_ENERGY_VARIANTS = (
    "cave",
    "cavenet",
    "minimal_subject",
    "cavenet_controller",
    "evolved_recurrent",
    "evolved_hidden_reset",
)


def substrate_energy_report_spec(
    *,
    dt: float = 1.0,
    fps: int = 4,
    include_assets: bool = True,
) -> ProducerReportSpec:
    def build_episode() -> Episode:
        return substrate_energy_episodes(dt=dt)["evolved_recurrent"]

    extra_assets = ()
    if include_assets:
        extra_assets = (
            ReportExtraAsset(
                id="substrate_energy_metrics",
                title="Substrate Energy Metrics JSON",
                filename="substrate_energy_metrics.json",
                writer=lambda episode, output: write_substrate_energy_metrics_json(
                    output,
                    dt=dt,
                ),
            ),
        )

    return ProducerReportSpec(
        id="substrate-energy",
        title="Substrate Energy Report",
        episode_factory=build_episode,
        input_summary=(
            "minimal subject, Cave, CaveNet, controlled CaveNet, and evolved "
            "recurrent subject scored with the energy estimator"
        ),
        description=(
            "Scores the project substrates directly. This is the substrate-level "
            "counterpart to the canonical producer energy report: it asks where "
            "compression, latent update work, controller work, and rails cost "
            "actually appear."
        ),
        views=default_views(),
        extra_assets=extra_assets,
        checks=(lambda episode: check_substrate_energy(dt=dt),),
        frame_time=3.0,
        dt=dt,
        fps=fps,
        columns=2,
        config={
            "producer": "substrate_energy",
            "scenario": "substrate_energy",
            "dt": dt,
            "fps": fps,
            "variants": list(SUBSTRATE_ENERGY_VARIANTS),
        },
        sections=(
            ReportSection(
                title="Question",
                body=(
                    "Where does the energy estimator light up when applied to "
                    "the actual project substrates rather than a calibration "
                    "probe?"
                ),
                asset_ids=("substrate_energy_metrics",),
            ),
            ReportSection(
                title="Boundary",
                body=(
                    "Minimal, Cave, and CaveNet are scored on shared common "
                    "behavior probes. Controlled CaveNet is scored on its "
                    "controller run. Evolved subject energy is a latent-state "
                    "proxy and offline evolution is counted as rails."
                ),
            ),
        ),
    )


def substrate_energy_episodes(*, dt: float = 1.0) -> dict[str, Episode]:
    episodes: dict[str, Episode] = {}
    for substrate in SUBSTRATES:
        episodes[substrate] = _merge_probe_episodes(
            substrate,
            [
                build_common_behavior_episode(substrate, probe, dt=dt)
                for probe in PROBES
            ],
        )
    episodes["cavenet_controller"] = build_controller_episode(
        "controller-full",
        dt=0.2,
    )
    episodes["evolved_recurrent"] = build_evolved_exposure_episode(
        "evolved-recurrent",
        generations=12,
        population_size=20,
        world_count=6,
        evaluation_cycles=12,
    )
    episodes["evolved_hidden_reset"] = build_evolved_exposure_episode(
        "hidden-reset",
        generations=12,
        population_size=20,
        world_count=6,
        evaluation_cycles=12,
    )
    return {
        name: annotate_substrate_energy(name, episode)
        for name, episode in episodes.items()
    }


def annotate_substrate_energy(name: str, episode: Episode) -> Episode:
    if name.startswith("evolved"):
        return _annotate_evolved_energy(name, episode)
    if name == "cavenet_controller":
        return _annotate_cavenet_controller_energy(episode)
    if name == "minimal_subject":
        return _annotate_minimal_energy(episode)
    if name in {"cave", "cavenet"}:
        return _annotate_cave_like_energy(name, episode)
    return episode


def check_substrate_energy(*, dt: float = 1.0) -> dict[str, object]:
    episodes = substrate_energy_episodes(dt=dt)
    metrics = {name: summarize_episode_energy(episode) for name, episode in episodes.items()}
    compact = _compact_metrics(metrics)
    for name, episode in episodes.items():
        compact[name]["latent_compression_load"] = _latent_compression_load(episode)
        compact[name]["latent_update_energy"] = _latent_update_energy(episode)
    roles = _roles(compact)
    errors = []
    for name, values in compact.items():
        if values["subject_total"] <= 0.0:
            errors.append(f"{name} has no subject energy proxy")
        if values["mean_loss_presence"] <= 0.0:
            errors.append(f"{name} has no loss/pressure proxy")
    if compact["minimal_subject"]["compression_load"] <= 0.0:
        errors.append("minimal_subject did not expose compression load")
    if compact["evolved_recurrent"]["rails_total"] <= compact["evolved_recurrent"]["subject_total"]:
        errors.append("evolved recurrent offline rails did not dominate subject run cost")
    if compact["evolved_recurrent"]["dynamic_energy_coupling"] <= compact["evolved_hidden_reset"]["dynamic_energy_coupling"]:
        errors.append("evolved recurrent did not exceed hidden-reset energy coupling")
    if compact["cavenet_controller"]["dynamic_energy_coupling"] <= compact["cavenet"]["dynamic_energy_coupling"]:
        errors.append("controller did not exceed plain CaveNet energy coupling")
    return {
        "id": "substrate_energy",
        "ok": not errors,
        "errors": errors,
        "metrics": compact,
        "roles": roles,
    }


def write_substrate_energy_metrics_json(output: Path, *, dt: float = 1.0) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    result = check_substrate_energy(dt=dt)
    output.write_text(json.dumps(encode_value(result), indent=2) + "\n", encoding="utf-8")


def _annotate_evolved_energy(name: str, episode: Episode) -> Episode:
    observations = []
    genome_size = _evolved_genome_size(episode)
    training_steps = _evolved_training_steps(episode)
    rails_per_step = training_steps / max(1, len(episode.observations))
    for obs in episode.observations:
        meta = obs.metadata.get("evolved_subject", {})
        hidden_before = np.asarray(meta.get("hidden_before", obs.memory_state), dtype=float)
        hidden_after = np.asarray(meta.get("hidden_after", obs.memory_state), dtype=float)
        hidden_delta = float(np.sum((hidden_after - hidden_before) ** 2))
        exposure = float(meta.get("exposure", obs.attention))
        next_exposure = float(meta.get("next_exposure", exposure))
        outcome_value = float(meta.get("outcome_value", 0.0))
        utility = float(meta.get("utility", 0.0))
        loss_proxy = (
            abs(outcome_value) * abs(1.0 - exposure)
            + max(0.0, -utility)
        )
        energy = EnergyLedger(
            rails_base=rails_per_step,
            subject_base=1.0,
            sensing=float(obs.actual.size),
            attention_update=abs(next_exposure - exposure) * 10.0,
            controller_update=hidden_delta,
            memory_update=float(hidden_after.size),
            adaptive_extra=hidden_delta,
        )
        observations.append(
            replace(
                obs,
                metadata={
                    **obs.metadata,
                    "energy": energy.to_metadata(),
                    "energy_loss_proxy": loss_proxy,
                    "energy_notes": {
                        "kind": name,
                        "genome_size": genome_size,
                        "training_steps": training_steps,
                        "hidden_delta_energy": hidden_delta,
                    },
                },
            )
        )
    return replace(episode, observations=observations)


def _annotate_cavenet_controller_energy(episode: Episode) -> Episode:
    history = episode.metadata.get("cavenet_config_history", [])
    observations = []
    for index, obs in enumerate(episode.observations):
        config_delta = 0.0
        controller_norm = 0.0
        if isinstance(history, list) and index < len(history):
            item = history[index]
            if isinstance(item, dict):
                before = item.get("before", {})
                after = item.get("after", {})
                config_delta = _dict_delta_norm(before, after)
                controller = item.get("controller")
                if isinstance(controller, dict):
                    controller_norm = float(controller.get("latent_norm", 0.0))
        loss_proxy = float(obs.surprise) + float(
            obs.metadata.get("objective", {}).get("compression_cost", 0.0)
        )
        energy = EnergyLedger(
            rails_base=1.0,
            scheduler=1.0,
            subject_base=1.0,
            sensing=float(obs.actual.size),
            attention_update=1.0 + abs(float(obs.attention)),
            compression_compute=float(obs.actual.size)
            * (1.0 + float(obs.metadata.get("workspace", {}).get("compression_cost", 0.0))),
            prediction_compute=float(obs.actual.size),
            loss_compute=1.0,
            memory_update=float(obs.memory_state.size),
            topology_update=1.0,
            controller_update=controller_norm + 10.0 * config_delta,
            adaptive_extra=controller_norm + 10.0 * config_delta,
        )
        observations.append(
            replace(
                obs,
                metadata={
                    **obs.metadata,
                    "energy": energy.to_metadata(),
                    "energy_loss_proxy": loss_proxy,
                    "energy_notes": {
                        "kind": "cavenet_controller",
                        "config_delta": config_delta,
                        "controller_norm": controller_norm,
                    },
                },
            )
        )
    return replace(episode, observations=observations)


def _annotate_minimal_energy(episode: Episode) -> Episode:
    observations = []
    for obs in episode.observations:
        source_energy = _input_energy(obs)
        retained_energy = _active_source_energy(obs)
        dropped_energy = max(0.0, source_energy - retained_energy)
        compression_cost = 0.0 if source_energy <= 1e-12 else dropped_energy / source_energy
        loss_proxy = float(obs.surprise) + compression_cost
        energy = EnergyLedger(
            rails_base=1.0,
            scheduler=1.0,
            subject_base=1.0,
            sensing=float(obs.actual.size),
            attention_update=float(len(obs.attention_weights)),
            compression_compute=float(obs.actual.size) * (1.0 + compression_cost),
            prediction_compute=float(obs.expected.size),
            loss_compute=1.0,
            memory_update=float(obs.memory_state.size),
        )
        observations.append(
            replace(
                obs,
                metadata={
                    **obs.metadata,
                    "energy": energy.to_metadata(),
                    "energy_loss_proxy": loss_proxy,
                    "workspace": {
                        "represented": obs.actual[np.abs(obs.actual) > 1e-12].copy(),
                        "reconstructed": obs.actual.copy(),
                        "retained_energy": retained_energy,
                        "dropped_energy": dropped_energy,
                        "compression_cost": compression_cost,
                        "reconstruction_error": 0.0,
                        "active_features": list(
                            obs.metadata.get("minimal_subject", {}).get("workspace_weights", {}).keys()
                        ),
                        "method": "minimal_subject_top_k",
                    },
                },
            )
        )
    return replace(episode, observations=observations)


def _annotate_cave_like_energy(name: str, episode: Episode) -> Episode:
    observations = []
    for obs in episode.observations:
        workspace = obs.metadata.get("workspace", {})
        compression_cost = float(workspace.get("compression_cost", 0.0)) if isinstance(workspace, dict) else 0.0
        loss_proxy = float(obs.surprise) + compression_cost
        energy = EnergyLedger(
            rails_base=1.0,
            scheduler=1.0,
            subject_base=1.0,
            sensing=float(obs.actual.size),
            attention_update=float(len(obs.attention_weights)),
            compression_compute=float(obs.actual.size) * (1.0 + compression_cost),
            prediction_compute=float(obs.expected.size),
            loss_compute=1.0,
            memory_update=float(obs.memory_state.size),
            topology_update=1.0,
        )
        observations.append(
            replace(
                obs,
                metadata={
                    **obs.metadata,
                    "energy": energy.to_metadata(),
                    "energy_loss_proxy": loss_proxy,
                    "energy_notes": {"kind": name},
                },
            )
        )
    return replace(episode, observations=observations)


def _merge_probe_episodes(name: str, episodes: list[Episode]) -> Episode:
    inputs = []
    observations = []
    t_offset = 0.0
    order_offset = 0
    for episode in episodes:
        probe = episode.source_name.split(":")[-1]
        id_map = {}
        for item in episode.inputs:
            new_id = f"{probe}:{item.id}"
            id_map[item.id] = new_id
            inputs.append(
                replace(
                    item,
                    id=new_id,
                    start=item.start + t_offset,
                    end=item.end + t_offset,
                    order_index=item.order_index + order_offset,
                    metadata={**item.metadata, "probe": probe},
                )
            )
        for obs in episode.observations:
            observations.append(
                replace(
                    obs,
                    t=obs.t + t_offset,
                    active_inputs=[id_map.get(input_id, input_id) for input_id in obs.active_inputs],
                    attention_weights={
                        id_map.get(input_id, input_id): weight
                        for input_id, weight in obs.attention_weights.items()
                    },
                    input_features={
                        id_map.get(input_id, input_id): value
                        for input_id, value in obs.input_features.items()
                    },
                    metadata={**obs.metadata, "probe": probe},
                )
            )
        t_offset += episode.duration + 1.0
        order_offset += len(episode.inputs)
    duration = max((item.end for item in inputs), default=0.0)
    return Episode(
        source_name=f"substrate-energy:{name}",
        vocabulary=list(episodes[0].vocabulary) if episodes else [],
        inputs=inputs,
        observations=observations,
        duration=duration,
        metadata={
            "source": "cave.pressure.checks.substrate_energy",
            "adapter": f"SubstrateEnergy:{name}",
            "substrate": name,
        },
    )


def _compact_metrics(metrics: dict[str, dict[str, object]]) -> dict[str, dict[str, float]]:
    compact: dict[str, dict[str, float]] = {}
    for name, summary in metrics.items():
        compact[name] = {
            "compression_load": float(summary["compression"]["compression_load"]),  # type: ignore[index]
            "mean_loss_presence": float(summary["loss"]["mean_loss_presence"]),  # type: ignore[index]
            "rails_total": float(summary["energy"]["rails_total"]),  # type: ignore[index]
            "subject_total": float(summary["energy"]["subject_total"]),  # type: ignore[index]
            "adaptive_extra_total": float(summary["energy"]["adaptive_extra_total"]),  # type: ignore[index]
            "rail_independence": float(summary["energy"]["rail_independence"]),  # type: ignore[index]
            "dynamic_energy_coupling": float(summary["coupling"]["dynamic_energy_coupling"]),  # type: ignore[index]
            "hick_slope": float(summary["coupling"]["hick_slope"]),  # type: ignore[index]
            "instantiation_proxy": float(summary["summary"]["instantiation_proxy"]),  # type: ignore[index]
        }
    return compact


def _roles(metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        "compression_load": {name: values["compression_load"] for name, values in metrics.items()},
        "loss_presence": {name: values["mean_loss_presence"] for name, values in metrics.items()},
        "rail_independence": {name: values["rail_independence"] for name, values in metrics.items()},
        "dynamic_energy_coupling": {name: values["dynamic_energy_coupling"] for name, values in metrics.items()},
        "instantiation_proxy": {name: values["instantiation_proxy"] for name, values in metrics.items()},
        "latent_compression_load": {name: values["latent_compression_load"] for name, values in metrics.items()},
        "latent_update_energy": {name: values["latent_update_energy"] for name, values in metrics.items()},
        "evolved_recurrent_minus_reset": {
            "dynamic_energy_coupling": (
                metrics["evolved_recurrent"]["dynamic_energy_coupling"]
                - metrics["evolved_hidden_reset"]["dynamic_energy_coupling"]
            ),
            "instantiation_proxy": (
                metrics["evolved_recurrent"]["instantiation_proxy"]
                - metrics["evolved_hidden_reset"]["instantiation_proxy"]
            ),
        },
    }


def _evolved_training_steps(episode: Episode) -> float:
    config = episode.metadata.get("evolution_config")
    subject_config = episode.metadata.get("subject_config")
    generations = float(getattr(config, "generations", 0.0))
    population = float(getattr(config, "population_size", 0.0))
    worlds = float(getattr(config, "world_count", 0.0))
    cycles = float(getattr(config, "cycles_per_world", 0.0))
    hidden = float(getattr(subject_config, "hidden_dim", 1.0))
    return max(0.0, generations * population * worlds * cycles * hidden)


def _evolved_genome_size(episode: Episode) -> float:
    subject_config = episode.metadata.get("subject_config")
    hidden = float(getattr(subject_config, "hidden_dim", 0.0))
    vocabulary = float(len(episode.vocabulary))
    if hidden <= 0.0:
        return 0.0
    return hidden * vocabulary + hidden * hidden + hidden + hidden + 1.0


def _dict_delta_norm(before: object, after: object) -> float:
    if not isinstance(before, dict) or not isinstance(after, dict):
        return 0.0
    keys = sorted(set(before) & set(after))
    if not keys:
        return 0.0
    delta = [
        float(after[key]) - float(before[key])
        for key in keys
        if isinstance(before.get(key), (int, float))
        and isinstance(after.get(key), (int, float))
    ]
    return float(np.linalg.norm(delta)) if delta else 0.0


def _input_energy(obs: EpisodeObservation) -> float:
    if not obs.input_features:
        return float(np.sum(obs.actual * obs.actual))
    return float(sum(np.sum(value * value) for value in obs.input_features.values()))


def _active_source_energy(obs: EpisodeObservation) -> float:
    if not obs.input_features:
        return float(np.sum(obs.actual * obs.actual))
    source = next(iter(obs.input_features.values()))
    active = np.abs(obs.actual) > 1e-12
    if source.shape != obs.actual.shape:
        return float(np.sum(obs.actual * obs.actual))
    return float(np.sum(source[active] * source[active]))


def _latent_compression_load(episode: Episode) -> float:
    if not episode.observations:
        return 0.0
    hidden_dim = max(1, int(episode.observations[0].memory_state.size))
    input_dim = max(1, len(episode.vocabulary))
    history_dim = input_dim * len(episode.observations)
    if history_dim <= hidden_dim:
        return 0.0
    return float(1.0 - hidden_dim / history_dim)


def _latent_update_energy(episode: Episode) -> float:
    total = 0.0
    for obs in episode.observations:
        notes = obs.metadata.get("energy_notes", {})
        if isinstance(notes, dict):
            total += float(notes.get("hidden_delta_energy", 0.0))
    return total
