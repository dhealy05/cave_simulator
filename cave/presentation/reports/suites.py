from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from cave.demonstrations.reports.cave_matrices import (
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
from cave.presentation.reports.generate import write_producer_report
from cave.presentation.reports.matrix import write_matrix_report
from cave.presentation.reports.specs import (
    MatrixReportSpec,
    ProducerReportSpec,
    SubjectComparisonReportSpec,
)
from cave.presentation.reports.subject_comparison import write_subject_comparison_report
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
from cave.pressure.tests.evolved_dissociation import evolved_dissociation_report_spec
from cave.pressure.tests.evolved_exposure import evolved_exposure_report_spec
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


SpecFactory = Callable[..., ProducerReportSpec | SubjectComparisonReportSpec | MatrixReportSpec]


SPEC_FACTORIES: dict[str, SpecFactory] = {
    "role_dependency_contrasts": role_dependency_contrasts_report_spec,
    "topology_atlas": topology_atlas_report_spec,
    "attention_bottleneck": attention_bottleneck_report_spec,
    "cavenet_ablation": cavenet_ablation_report_spec,
    "cavenet_controller": cavenet_controller_report_spec,
    "cavenet_controller_learning": cavenet_controller_learning_report_spec,
    "cavenet_controller_population": cavenet_controller_population_report_spec,
    "cavenet_pressure": cavenet_pressure_report_spec,
    "cavenet_pressure_population": cavenet_pressure_population_report_spec,
    "common_behaviors": common_behaviors_report_spec,
    "evolved_dissociation": evolved_dissociation_report_spec,
    "evolved_exposure": evolved_exposure_report_spec,
    "evolved_exposure_sweep": evolved_exposure_sweep_report_spec,
    "evolved_roles": evolved_roles_report_spec,
    "evolved_roles_sweep": evolved_roles_sweep_report_spec,
    "expectation_violation": expectation_violation_report_spec,
    "importance_weighted_event": importance_weighted_event_report_spec,
    "objective_attention_shift": objective_attention_shift_report_spec,
    "population_clusters_matrix": population_clusters_matrix_report_spec,
    "population_trajectory_geometry": population_trajectory_geometry_report_spec,
    "population_trajectory_geometry_sweep": population_trajectory_geometry_sweep_report_spec,
    "preference_emergence": preference_emergence_report_spec,
    "preference_shaped_topology": preference_shaped_topology_report_spec,
    "reference_cave": reference_cave_report_spec,
    "representational_compression": representational_compression_report_spec,
    "regulation_recovery": regulation_recovery_report_spec,
    "role_recovery": role_recovery_report_spec,
    "role_recovery_matrix": role_recovery_matrix_report_spec,
    "same_world_different_subjects": same_world_different_subjects_report_spec,
    "selection_recovery": selection_recovery_report_spec,
    "subject_ablation_matrix": subject_ablation_matrix_report_spec,
    "topology_recovery": topology_recovery_report_spec,
    "unseen_modality": unseen_modality_report_spec,
    "valence_attractor_repulsor": valence_attractor_repulsor_report_spec,
    "value_retention_recovery": value_retention_recovery_report_spec,
}

REPORT_WRITERS = {
    "producer": write_producer_report,
    "subject": write_subject_comparison_report,
    "matrix": write_matrix_report,
}

DEFAULT_SUITE_MANIFEST = Path("fixtures/report_suites/result_ladder.json")


@dataclass(frozen=True)
class SuiteEntryResult:
    id: str
    tier: str
    kind: str
    spec: str
    title: str
    claim: str
    directory: Path
    report_md: Path
    checks_json: Path
    metadata_json: Path | None
    ok: bool
    key_metrics: dict[str, Any]
    curated_artifacts: tuple[str, ...]


@dataclass(frozen=True)
class SuiteRunResult:
    suite_id: str
    output_root: Path
    manifest_json: Path
    index_json: Path
    index_md: Path
    entries: tuple[SuiteEntryResult, ...]


def load_suite_manifest(path: str | Path = DEFAULT_SUITE_MANIFEST) -> dict[str, Any]:
    manifest_path = Path(path)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _git_provenance() -> dict[str, Any]:
    def _run(args: list[str]) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    status = _run(["status", "--porcelain"])
    return {
        "sha": _run(["rev-parse", "HEAD"]),
        "branch": _run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status) if status is not None else None,
    }


def _provenance(manifest: dict[str, Any]) -> dict[str, Any]:
    config_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    provenance: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest()[:16],
        "python": platform.python_version(),
        "git": _git_provenance(),
    }
    try:
        import numpy as _np

        provenance["numpy"] = _np.__version__
    except Exception:  # numpy should be present, but never fail provenance over it
        provenance["numpy"] = None
    return provenance


def run_report_suite(
    manifest_path: str | Path = DEFAULT_SUITE_MANIFEST,
    *,
    output_root: str | Path | None = None,
    tiers: Iterable[str] | None = None,
    entries: Iterable[str] | None = None,
    skip_assets: bool = False,
    dry_run: bool = False,
) -> SuiteRunResult:
    manifest_path = Path(manifest_path)
    manifest = load_suite_manifest(manifest_path)
    root = Path(output_root or manifest["output_root"])
    selected_tiers = set(tiers or ())
    selected_entries = set(entries or ())
    root.mkdir(parents=True, exist_ok=True)

    entry_results: list[SuiteEntryResult] = []
    for entry in manifest["entries"]:
        if selected_tiers and entry["tier"] not in selected_tiers:
            continue
        if selected_entries and entry["id"] not in selected_entries:
            continue
        if dry_run:
            entry_results.append(_dry_entry_result(entry, root))
            continue
        entry_results.append(_run_suite_entry(entry, root, skip_assets=skip_assets))

    manifest_json = root / "suite.json"
    manifest_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    index_payload = suite_index_payload(manifest, entry_results, dry_run=dry_run)
    index_json = root / "index.json"
    index_json.write_text(json.dumps(index_payload, indent=2) + "\n", encoding="utf-8")
    index_md = root / "index.md"
    index_md.write_text(suite_index_markdown(manifest, entry_results, dry_run=dry_run), encoding="utf-8")
    return SuiteRunResult(
        suite_id=manifest["id"],
        output_root=root,
        manifest_json=manifest_json,
        index_json=index_json,
        index_md=index_md,
        entries=tuple(entry_results),
    )


def suite_index_payload(
    manifest: dict[str, Any],
    entry_results: Iterable[SuiteEntryResult],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    kwargs_by_id = {
        entry["id"]: entry.get("kwargs", {}) for entry in manifest.get("entries", [])
    }
    return {
        "id": manifest["id"],
        "title": manifest["title"],
        "description": manifest.get("description", ""),
        "dry_run": dry_run,
        "provenance": _provenance(manifest),
        "tiers": manifest.get("tiers", []),
        "entries": [
            {
                "id": result.id,
                "tier": result.tier,
                "kind": result.kind,
                "spec": result.spec,
                "title": result.title,
                "claim": result.claim,
                "ok": result.ok,
                "kwargs": kwargs_by_id.get(result.id, {}),
                "directory": result.directory.as_posix(),
                "report": result.report_md.as_posix(),
                "checks": result.checks_json.as_posix(),
                "metadata": result.metadata_json.as_posix() if result.metadata_json else None,
                "key_metrics": result.key_metrics,
                "curated_artifacts": list(result.curated_artifacts),
            }
            for result in entry_results
        ],
    }


def suite_index_markdown(
    manifest: dict[str, Any],
    entry_results: Iterable[SuiteEntryResult],
    *,
    dry_run: bool = False,
) -> str:
    by_tier: dict[str, list[SuiteEntryResult]] = {}
    for result in entry_results:
        by_tier.setdefault(result.tier, []).append(result)

    lines = [
        f"# {manifest['title']}",
        "",
        manifest.get("description", ""),
        "",
        f"- dry run: `{'true' if dry_run else 'false'}`",
        f"- entries: `{sum(len(items) for items in by_tier.values())}`",
        "",
    ]
    tier_titles = {tier["id"]: tier["title"] for tier in manifest.get("tiers", [])}
    tier_descriptions = {
        tier["id"]: tier.get("description", "") for tier in manifest.get("tiers", [])
    }
    for tier_id, results in by_tier.items():
        lines.extend([f"## {tier_titles.get(tier_id, tier_id)}", ""])
        description = tier_descriptions.get(tier_id)
        if description:
            lines.extend([description, ""])
        lines.extend(
            [
                "| Entry | Kind | Checks | Claim | Report |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for result in results:
            status = "pass" if result.ok else "fail"
            report = _relative_link(result.report_md, result.directory.parent.parent)
            lines.append(
                f"| `{result.id}` | `{result.kind}` | {status} | "
                f"{result.claim} | [{result.report_md.name}]({report}) |"
            )
        lines.append("")
        for result in results:
            if not result.key_metrics:
                continue
            lines.extend([f"### {result.id}", ""])
            for key, value in result.key_metrics.items():
                lines.append(f"- `{key}`: `{_format_metric_value(value)}`")
            lines.append("")
    return "\n".join(lines)


def _run_suite_entry(
    entry: dict[str, Any],
    output_root: Path,
    *,
    skip_assets: bool,
) -> SuiteEntryResult:
    kind = entry["kind"]
    factory = SPEC_FACTORIES[entry["spec"]]
    kwargs = _entry_kwargs(factory, entry.get("kwargs", {}), skip_assets=skip_assets)
    spec = factory(**kwargs)
    directory = output_root / entry["output"]
    writer = REPORT_WRITERS[kind]
    outputs = writer(spec, directory)
    checks_json = outputs.checks_json
    checks = json.loads(checks_json.read_text(encoding="utf-8"))
    metadata_json = getattr(outputs, "metadata_json", None)
    metadata = _read_json_if_exists(metadata_json)
    cluster_summary = _read_json_if_exists(getattr(outputs, "cluster_summary_json", None))
    sources = {
        "checks": checks,
        "metadata": metadata,
        "cluster_summary": cluster_summary,
        **checks,
    }
    return SuiteEntryResult(
        id=entry["id"],
        tier=entry["tier"],
        kind=kind,
        spec=entry["spec"],
        title=getattr(spec, "title", entry["id"]),
        claim=entry.get("claim", ""),
        directory=directory,
        report_md=outputs.report_md,
        checks_json=checks_json,
        metadata_json=metadata_json,
        ok=bool(checks.get("ok")),
        key_metrics=_extract_key_metrics(sources, entry.get("key_metrics", ())),
        curated_artifacts=tuple(entry.get("curated_artifacts", ())),
    )


def _dry_entry_result(entry: dict[str, Any], output_root: Path) -> SuiteEntryResult:
    directory = output_root / entry["output"]
    return SuiteEntryResult(
        id=entry["id"],
        tier=entry["tier"],
        kind=entry["kind"],
        spec=entry["spec"],
        title=entry["id"],
        claim=entry.get("claim", ""),
        directory=directory,
        report_md=directory / "report.md",
        checks_json=directory / "checks.json",
        metadata_json=directory / "metadata.json",
        ok=True,
        key_metrics={key: None for key in entry.get("key_metrics", ())},
        curated_artifacts=tuple(entry.get("curated_artifacts", ())),
    )


def _entry_kwargs(factory: SpecFactory, kwargs: dict[str, Any], *, skip_assets: bool) -> dict[str, Any]:
    accepted = set(inspect.signature(factory).parameters)
    filtered = {key: value for key, value in kwargs.items() if key in accepted}
    if skip_assets:
        for key in ("include_assets", "include_readme_assets"):
            if key in accepted:
                filtered[key] = False
    return filtered


def _extract_key_metrics(sources: dict[str, Any], metric_paths: Iterable[str]) -> dict[str, Any]:
    metrics = {}
    for metric_path in metric_paths:
        try:
            metrics[metric_path] = _resolve_metric_path(sources, metric_path)
        except (KeyError, IndexError, TypeError):
            metrics[metric_path] = None
    return metrics


def _resolve_metric_path(value: Any, metric_path: str) -> Any:
    parts = metric_path.split(".")
    if len(parts) >= 2 and parts[0] == "extra":
        extra_id = parts[1]
        for item in value.get("extra", []):
            if item.get("id") == extra_id:
                return _resolve_parts(item, parts[2:])
        raise KeyError(extra_id)
    return _resolve_parts(value, parts)


def _resolve_parts(value: Any, parts: list[str]) -> Any:
    current = value
    for part in parts:
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            raise TypeError(part)
    return current


def _read_json_if_exists(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(", ", ": "))
    return str(value)


def _relative_link(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a predefined Cave report suite.")
    parser.add_argument(
        "suite",
        nargs="?",
        default=DEFAULT_SUITE_MANIFEST.as_posix(),
        help="Path to a suite manifest JSON file.",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--tier", action="append", default=None, help="Run one tier; may be repeated.")
    parser.add_argument("--entry", action="append", default=None, help="Run one entry id; may be repeated.")
    parser.add_argument(
        "--skip-assets",
        action="store_true",
        help="Disable optional per-view assets where a report spec supports it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write only the suite manifest and index without running reports.",
    )
    args = parser.parse_args()

    result = run_report_suite(
        args.suite,
        output_root=args.output_root,
        tiers=args.tier,
        entries=args.entry,
        skip_assets=args.skip_assets,
        dry_run=args.dry_run,
    )
    print(f"wrote {result.index_md}")


if __name__ == "__main__":
    main()
