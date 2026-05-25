from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cave.observation.pipeline import episode_payload
from cave.observation.episode_runs import episode_set
from cave.presentation.runs import slugify
from cave.presentation.renderers.topology_atlas_renderer import (
    save_topology_atlas,
    save_topology_atlas_metrics,
    shared_topology_params_for_episode_set,
)
from cave.presentation.renderers.episode_set_dashboard import (
    save_episode_set_distances_json,
)
from cave.demonstrations.subjects import (
    embedding_distance,
    state_effect_embedding,
    subjective_trajectory_embedding,
    memory_trajectory_embedding,
    save_subject_comparison_dashboard,
    threshold_clusters,
)
from cave.presentation.reports.specs import SubjectComparisonReportSpec


@dataclass(frozen=True)
class SubjectComparisonReportOutputs:
    directory: Path
    report_md: Path
    metadata_json: Path
    checks_json: Path
    dashboard_png: Path
    episode_set_distances_json: Path
    topology_atlas_png: Path
    topology_atlas_metrics_json: Path
    run_episode_jsons: tuple[Path, ...]


def write_subject_comparison_report(
    spec: SubjectComparisonReportSpec,
    output_dir: str | Path,
) -> SubjectComparisonReportOutputs:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    runs_dir = directory / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    runs, labels = spec.run_factory()
    runs = list(runs)
    labels = list(labels)
    if len(runs) == 0:
        raise ValueError("subject comparison report requires at least one run")
    if len(runs) != len(labels):
        raise ValueError("runs and labels must have the same length")

    effect = lambda run: state_effect_embedding(run, samples=spec.samples)
    observed = lambda run: memory_trajectory_embedding(run, samples=spec.samples)
    internal = lambda run: subjective_trajectory_embedding(run, samples=spec.samples)

    dashboard_png = directory / "dashboard.png"
    save_subject_comparison_dashboard(
        runs,
        labels,
        dashboard_png,
        effect_embedding=effect,
        observed_embedding=observed,
        internal_embedding=internal,
        title=spec.title,
    )
    comparison_set = episode_set(
        [run.as_labeled_episode(label) for run, label in zip(runs, labels)],
        id=spec.id,
        title=spec.title,
        comparison_axis="subject",
    )
    topology_params = shared_topology_params_for_episode_set(comparison_set)
    topology_atlas_png = directory / "topology_atlas.png"
    topology_atlas_metrics_json = directory / "topology_atlas_metrics.json"
    save_topology_atlas(comparison_set, topology_atlas_png, topology_params)
    save_topology_atlas_metrics(
        comparison_set,
        topology_atlas_metrics_json,
        topology_params,
    )
    episode_set_distances_json = directory / "episode_set_distances.json"
    save_episode_set_distances_json(
        comparison_set,
        episode_set_distances_json,
        samples=spec.samples,
    )

    run_episode_jsons = []
    for run, label in zip(runs, labels):
        run_dir = runs_dir / slugify(label)
        run_dir.mkdir(parents=True, exist_ok=True)
        episode_json = run_dir / "episode.json"
        episode_json.write_text(
            json.dumps(episode_payload(run.episode), indent=2) + "\n",
            encoding="utf-8",
        )
        metadata_json = run_dir / "metadata.json"
        metadata_json.write_text(
            json.dumps(
                {
                    "id": run.id,
                    "label": label,
                    "subject_id": run.subject.id,
                    "source_name": run.episode.source_name,
                    "sequence_length": len(run.sequence.objects),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        run_episode_jsons.append(episode_json)

    checks = subject_comparison_checks(
        runs,
        labels,
        samples=spec.samples,
        cluster_threshold=spec.cluster_threshold,
    )
    checks_json = directory / "checks.json"
    checks_json.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")

    metadata = {
        "id": spec.id,
        "title": spec.title,
        "run_count": len(runs),
        "labels": labels,
        "subjects": [run.subject.id for run in runs],
        "samples": spec.samples,
        "cluster_threshold": spec.cluster_threshold,
        "config": spec.config,
    }
    metadata_json = directory / "metadata.json"
    metadata_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    report_md = directory / "report.md"
    report_md.write_text(
        subject_comparison_report_markdown(spec, labels, checks),
        encoding="utf-8",
    )

    return SubjectComparisonReportOutputs(
        directory=directory,
        report_md=report_md,
        metadata_json=metadata_json,
        checks_json=checks_json,
        dashboard_png=dashboard_png,
        episode_set_distances_json=episode_set_distances_json,
        topology_atlas_png=topology_atlas_png,
        topology_atlas_metrics_json=topology_atlas_metrics_json,
        run_episode_jsons=tuple(run_episode_jsons),
    )


def subject_comparison_checks(
    runs,
    labels,
    *,
    samples: int,
    cluster_threshold: float,
) -> dict[str, object]:
    errors = []
    effect = lambda run: state_effect_embedding(run, samples=samples)
    observed = lambda run: memory_trajectory_embedding(run, samples=samples)
    internal = lambda run: subjective_trajectory_embedding(run, samples=samples)
    by_label = {label: run for label, run in zip(labels, runs)}
    sequence_signatures = {
        label: [obj.id for obj in run.sequence.objects]
        for label, run in by_label.items()
    }
    unique_sequences = {tuple(value) for value in sequence_signatures.values()}
    if len(unique_sequences) != 1:
        errors.append("runs do not share the same external sequence")

    if {"pref-warm", "pref-threat-avoid"}.issubset(by_label):
        return preference_shaped_topology_checks(
            by_label,
            labels,
            errors,
            effect=effect,
            observed=observed,
            cluster_threshold=cluster_threshold,
        )

    required = ["Q0-zero-flat", "Q0-full-flat", "Q0-zero-prior", "Q0-half-chan"]
    missing = [label for label in required if label not in by_label]
    if missing:
        errors.append(f"missing controlled runs: {', '.join(missing)}")

    metrics = {}
    if not missing:
        zero_flat = by_label["Q0-zero-flat"]
        full_flat = by_label["Q0-full-flat"]
        zero_prior = by_label["Q0-zero-prior"]
        half_chan = by_label["Q0-half-chan"]
        zero_prior_effect_distance = embedding_distance(
            effect(zero_flat),
            effect(zero_prior),
        )
        zero_prior_observed_distance = embedding_distance(
            observed(zero_flat),
            observed(zero_prior),
        )
        zero_full_effect_distance = embedding_distance(
            effect(zero_flat),
            effect(full_flat),
        )
        full_half_effect_distance = embedding_distance(
            effect(full_flat),
            effect(half_chan),
        )
        zero_full_internal_distance = embedding_distance(
            internal(zero_flat),
            internal(full_flat),
        )
        metrics.update(
            {
                "zero_prior_effect_distance": zero_prior_effect_distance,
                "zero_prior_observed_distance": zero_prior_observed_distance,
                "zero_full_effect_distance": zero_full_effect_distance,
                "full_half_effect_distance": full_half_effect_distance,
                "zero_full_internal_distance": zero_full_internal_distance,
            }
        )
        if zero_prior_effect_distance > cluster_threshold:
            errors.append("zero-attention prior changed state effect")
        if zero_prior_observed_distance <= cluster_threshold:
            errors.append("different prior did not change observed memory")
        if zero_full_effect_distance <= cluster_threshold:
            errors.append("zero and full attention did not diverge in state effect")
        if full_half_effect_distance <= cluster_threshold:
            errors.append("split channel attention did not diverge from full attention")
        if zero_full_internal_distance <= cluster_threshold:
            errors.append("zero and full attention did not diverge internally")

    clusters = threshold_clusters(runs, effect, threshold=cluster_threshold)
    return {
        "ok": not errors,
        "errors": errors,
        "run_count": len(runs),
        "clusters": [[labels[index] for index in cluster] for cluster in clusters],
        "metrics": metrics,
    }


def preference_shaped_topology_checks(
    by_label,
    labels,
    errors,
    *,
    effect,
    observed,
    cluster_threshold: float,
) -> dict[str, object]:
    warm = by_label["pref-warm"]
    threat_avoid = by_label["pref-threat-avoid"]
    warm_first = warm.episode.observations[0]
    threat_first = threat_avoid.episode.observations[0]
    warm_action = warm_first.metadata.get("action", {})
    threat_action = threat_first.metadata.get("action", {})
    effect_distance = embedding_distance(effect(warm), effect(threat_avoid))
    observed_distance = embedding_distance(observed(warm), observed(threat_avoid))
    warm_final = warm.episode.observations[-1].memory_state
    threat_final = threat_avoid.episode.observations[-1].memory_state
    metrics = {
        "warm_action": warm_action,
        "threat_avoid_action": threat_action,
        "effect_distance": effect_distance,
        "observed_distance": observed_distance,
        "warm_final_memory": warm_final.tolist(),
        "threat_avoid_final_memory": threat_final.tolist(),
    }
    if warm_action.get("kind") != "approach" or warm_action.get("target_id") != "warm_event":
        errors.append("warm-preference subject did not approach the warm event")
    if (
        threat_action.get("kind") != "avoid"
        or threat_action.get("target_id") != "threat_event"
    ):
        errors.append("threat-sensitive subject did not avoid the threat event")
    if effect_distance <= cluster_threshold:
        errors.append("preference-shaped actions did not change state effect")
    if observed_distance <= cluster_threshold:
        errors.append("preference-shaped actions did not change observed memory")
    if not warm_final[0] > threat_final[0]:
        errors.append("warm-preference subject did not retain more warmth")
    if not warm_final[1] > threat_final[1]:
        errors.append("threat avoidance did not reduce threat exposure relative to warm subject")

    clusters = threshold_clusters(by_label.values(), effect, threshold=cluster_threshold)
    label_list = list(by_label)
    return {
        "ok": not errors,
        "errors": errors,
        "run_count": len(by_label),
        "clusters": [[label_list[index] for index in cluster] for cluster in clusters],
        "metrics": metrics,
    }


def subject_comparison_report_markdown(
    spec: SubjectComparisonReportSpec,
    labels: list[str],
    checks: dict[str, object],
) -> str:
    lines = [
        f"# {spec.title}",
        "",
        spec.description,
        "",
        "## Run",
        "",
        f"- id: `{spec.id}`",
        f"- runs: {len(labels)}",
        f"- checks: {'pass' if checks['ok'] else 'fail'}",
        "",
        "## Outputs",
        "",
        "- [dashboard.png](dashboard.png)",
        "- [episode_set_distances.json](episode_set_distances.json)",
        "- [topology_atlas.png](topology_atlas.png)",
        "- [topology_atlas_metrics.json](topology_atlas_metrics.json)",
        "- [metadata.json](metadata.json)",
        "- [checks.json](checks.json)",
        "- [runs/](runs/)",
        "",
        "![Subject Comparison Dashboard](dashboard.png)",
        "",
        "![Topology Atlas](topology_atlas.png)",
        "",
        "## Labels",
        "",
    ]
    lines.extend(f"- `{label}`" for label in labels)
    if spec.sections:
        lines.extend(["", "## Walkthrough", ""])
        for section in spec.sections:
            lines.extend(["", f"### {section.title}", "", section.body.strip(), ""])
    lines.extend(["", "## Checks", ""])
    lines.append(f"- `same_world_different_subjects`: {'pass' if checks['ok'] else 'fail'}")
    metrics = checks.get("metrics", {})
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            lines.append(f"  - `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)
