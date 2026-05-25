from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from cave.observation.pipeline import episode_payload
from cave.observation.population import factor_levels_payload
from cave.presentation.runs import slugify
from cave.demonstrations.subjects import (
    active_context_embedding,
    state_effect_embedding,
    subjective_trajectory_embedding,
    memory_trajectory_embedding,
    pairwise_distance_matrix,
    save_subject_comparison_dashboard,
    threshold_clusters,
)
from cave.presentation.reports.specs import MatrixReportSpec, MatrixRunRecord
from cave.presentation.reports.matrix_viz import save_population_plot, write_cluster_summary


@dataclass(frozen=True)
class MatrixReportOutputs:
    directory: Path
    report_md: Path
    metadata_json: Path
    checks_json: Path
    dashboard_png: Path
    distance_jsons: tuple[Path, ...]
    cluster_jsons: tuple[Path, ...]
    cluster_summary_json: Path
    population_png: Path
    run_episode_jsons: tuple[Path, ...]


def write_matrix_report(
    spec: MatrixReportSpec,
    output_dir: str | Path,
) -> MatrixReportOutputs:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    runs_dir = directory / "runs"
    distances_dir = directory / "distances"
    clusters_dir = directory / "clusters"
    runs_dir.mkdir(parents=True, exist_ok=True)
    distances_dir.mkdir(parents=True, exist_ok=True)
    clusters_dir.mkdir(parents=True, exist_ok=True)

    records = list(spec.run_factory())
    if not records:
        raise ValueError("matrix report requires at least one run")
    labels = [record.label for record in records]
    runs = [record.run for record in records]

    dashboard_png = directory / "dashboard.png"
    save_subject_comparison_dashboard(
        runs,
        labels,
        dashboard_png,
        effect_embedding=lambda run: state_effect_embedding(run, samples=spec.samples),
        observed_embedding=lambda run: memory_trajectory_embedding(run, samples=spec.samples),
        internal_embedding=lambda run: subjective_trajectory_embedding(run, samples=spec.samples),
        title=spec.title,
    )

    run_episode_jsons = []
    for record in records:
        run_dir = (
            runs_dir
            / slugify(record.sequence_id)
            / slugify(record.subject_id)
            / slugify(record.variant_id)
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        episode_json = run_dir / "episode.json"
        episode_json.write_text(
            json.dumps(episode_payload(record.run.episode), indent=2) + "\n",
            encoding="utf-8",
        )
        metadata_json = run_dir / "metadata.json"
        metadata_json.write_text(
            json.dumps(
                {
                    "id": record.id,
                    "label": record.label,
                    "sequence_id": record.sequence_id,
                    "subject_id": record.subject_id,
                    "variant_id": record.variant_id,
                    "run_id": record.run.id,
                    "source_name": record.run.episode.source_name,
                    "comparison_role": record.comparison_role,
                    "matched_set_id": record.matched_set_id,
                    "replicate_id": record.replicate_id,
                    "group_id": record.group_id,
                    "factors": factor_levels_payload(record.factors),
                    "metadata": record.metadata,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        run_episode_jsons.append(episode_json)

    embeddings = {
        "state_effect": lambda run: state_effect_embedding(
            run,
            samples=spec.samples,
        ),
        "observed_memory": lambda run: memory_trajectory_embedding(
            run,
            samples=spec.samples,
        ),
        "subjective_trajectory": lambda run: subjective_trajectory_embedding(
            run,
            samples=spec.samples,
        ),
        "active_context": lambda run: active_context_embedding(
            run,
            samples=spec.samples,
        ),
    }
    distance_jsons = tuple(
        _write_distance_json(
            distances_dir / f"{name}.json",
            name,
            records,
            embedding,
        )
        for name, embedding in embeddings.items()
    )
    cluster_jsons = tuple(
        _write_cluster_json(
            clusters_dir / f"{name}.json",
            name,
            records,
            embedding,
            threshold=spec.cluster_thresholds.get(name, 1e-12),
        )
        for name, embedding in embeddings.items()
    )
    cluster_summary_json = write_cluster_summary(
        cluster_jsons,
        directory / "cluster_summary.json",
    )
    population_png = save_population_plot(
        records,
        distance_jsons,
        directory / "population.png",
    )

    checks = matrix_report_checks(records, spec.checks, cluster_jsons)
    checks_json = directory / "checks.json"
    checks_json.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")

    metadata = {
        "id": spec.id,
        "title": spec.title,
        "run_count": len(records),
        "sequences": sorted({record.sequence_id for record in records}),
        "subjects": sorted({record.subject_id for record in records}),
        "variants": sorted({record.variant_id for record in records}),
        "comparison_roles": sorted({record.comparison_role for record in records}),
        "factor_levels": _factor_level_summary(records),
        "labels": labels,
        "samples": spec.samples,
        "cluster_thresholds": spec.cluster_thresholds,
        "config": spec.config,
    }
    metadata_json = directory / "metadata.json"
    metadata_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    report_md = directory / "report.md"
    report_md.write_text(
        matrix_report_markdown(spec, records, checks, cluster_summary_json),
        encoding="utf-8",
    )

    return MatrixReportOutputs(
        directory=directory,
        report_md=report_md,
        metadata_json=metadata_json,
        checks_json=checks_json,
        dashboard_png=dashboard_png,
        distance_jsons=distance_jsons,
        cluster_jsons=cluster_jsons,
        cluster_summary_json=cluster_summary_json,
        population_png=population_png,
        run_episode_jsons=tuple(run_episode_jsons),
    )


def matrix_report_checks(records, checks, cluster_jsons=()) -> dict[str, object]:
    errors = []
    extra = []
    labels = [record.label for record in records]
    if len(labels) != len(set(labels)):
        errors.append("matrix labels must be unique")
    for check in checks:
        try:
            result = check(records)
        except Exception as exc:  # pragma: no cover - defensive report boundary
            result = {
                "id": getattr(check, "__name__", type(check).__name__),
                "ok": False,
                "errors": [str(exc)],
            }
        extra.append(result)
    return {
        "ok": not errors and all(bool(result.get("ok")) for result in extra),
        "errors": errors,
        "run_count": len(records),
        "cluster_files": [str(path) for path in cluster_jsons],
        "extra": extra,
    }


def matrix_report_markdown(
    spec: MatrixReportSpec,
    records: list[MatrixRunRecord],
    checks: dict[str, object],
    cluster_summary_json: Path | None = None,
) -> str:
    lines = [
        f"# {spec.title}",
        "",
        spec.description,
        "",
        "## Run Matrix",
        "",
        f"- id: `{spec.id}`",
        f"- runs: {len(records)}",
        f"- sequences: {len({record.sequence_id for record in records})}",
        f"- subjects: {len({record.subject_id for record in records})}",
        f"- variants: {len({record.variant_id for record in records})}",
        f"- checks: {'pass' if checks['ok'] else 'fail'}",
        "",
        "## Outputs",
        "",
        "- [dashboard.png](dashboard.png)",
        "- [metadata.json](metadata.json)",
        "- [checks.json](checks.json)",
        "- [distances/](distances/)",
        "- [clusters/](clusters/)",
        "- [cluster_summary.json](cluster_summary.json)",
        "- [population.png](population.png)",
        "- [runs/](runs/)",
        "",
        "![Population](population.png)",
        "",
        "![Matrix Dashboard](dashboard.png)",
    ]
    if cluster_summary_json is not None and cluster_summary_json.exists():
        lines.extend(["", "## Cluster Summary", ""])
        summary = json.loads(cluster_summary_json.read_text(encoding="utf-8"))
        lines.extend(
            [
                "| Embedding | Clusters | Largest | Largest Variant Counts |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for item in summary["embeddings"]:
            lines.append(
                "| "
                f"{item['embedding']} | "
                f"{item['cluster_count']} | "
                f"{item['largest_cluster_size']} | "
                f"`{item['largest_cluster_variant_counts']}` |"
            )
    if spec.sections:
        lines.extend(["", "## Walkthrough", ""])
        for section in spec.sections:
            lines.extend([f"### {section.title}", "", section.body.strip(), ""])

    lines.extend(["", "## Checks", ""])
    for check in checks.get("extra", []):
        status = "pass" if check.get("ok") else "fail"
        lines.append(f"- `{check.get('id', 'check')}`: {status}")
        metrics = check.get("metrics")
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                lines.append(f"  - `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def _write_distance_json(
    output: Path,
    embedding_name: str,
    records: list[MatrixRunRecord],
    embedding: Callable,
) -> Path:
    distances = pairwise_distance_matrix([record.run for record in records], embedding)
    output.write_text(
        json.dumps(
            {
                "embedding": embedding_name,
                "labels": [record.label for record in records],
                "sequence_ids": [record.sequence_id for record in records],
                "subject_ids": [record.subject_id for record in records],
                "variant_ids": [record.variant_id for record in records],
                "comparison_roles": [record.comparison_role for record in records],
                "matched_set_ids": [record.matched_set_id for record in records],
                "replicate_ids": [record.replicate_id for record in records],
                "factor_ids": _factor_ids_by_record(records),
                "distances": distances.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def _write_cluster_json(
    output: Path,
    embedding_name: str,
    records: list[MatrixRunRecord],
    embedding: Callable,
    *,
    threshold: float,
) -> Path:
    clusters = threshold_clusters(
        [record.run for record in records],
        embedding,
        threshold=threshold,
    )
    output.write_text(
        json.dumps(
            {
                "embedding": embedding_name,
                "threshold": threshold,
                "clusters": [
                    {
                        "size": len(cluster),
                        "indices": list(cluster),
                        "labels": [records[index].label for index in cluster],
                        "sequence_ids": [records[index].sequence_id for index in cluster],
                        "subject_ids": [records[index].subject_id for index in cluster],
                        "variant_ids": [records[index].variant_id for index in cluster],
                        "comparison_roles": [
                            records[index].comparison_role for index in cluster
                        ],
                        "matched_set_ids": [
                            records[index].matched_set_id for index in cluster
                        ],
                        "replicate_ids": [
                            records[index].replicate_id for index in cluster
                        ],
                        "factor_ids": _factor_ids_by_record(
                            [records[index] for index in cluster]
                        ),
                    }
                    for cluster in clusters
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def _factor_level_summary(records: list[MatrixRunRecord]) -> dict[str, list[dict[str, object]]]:
    summary: dict[str, dict[str, object]] = {}
    for record in records:
        for name, level in record.factors.items():
            key = f"{name}:{level.id}"
            summary[key] = {
                "factor": name,
                "id": level.id,
                "label": level.label,
                "role": level.role,
                "metadata": level.metadata,
            }
    grouped: dict[str, list[dict[str, object]]] = {}
    for item in summary.values():
        grouped.setdefault(str(item["factor"]), []).append(item)
    return {
        factor: sorted(items, key=lambda item: str(item["id"]))
        for factor, items in sorted(grouped.items())
    }


def _factor_ids_by_record(records: list[MatrixRunRecord]) -> dict[str, list[str | None]]:
    factor_names = sorted({name for record in records for name in record.factors})
    return {
        factor: [record.factor_id(factor) for record in records]
        for factor in factor_names
    }
