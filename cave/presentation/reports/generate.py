from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cave.observation.pipeline import episode_payload
from cave.presentation.renderers.matplotlib_renderer import LayoutSpec, MatplotlibRenderer
from cave.presentation.runs import ExperienceRun
from cave.observation.structural import episode_frames, frame_for_time, structural_state_for_episode
from cave.observation.views import default_views

from cave.presentation.reports.specs import ProducerReportSpec, ReportCheck, ReportSection


@dataclass(frozen=True)
class ProducerReportOutputs:
    directory: Path
    report_md: Path
    episode_json: Path
    metadata_json: Path
    checks_json: Path
    frame_png: Path
    animation_gif: Path
    assets: tuple[Path, ...]


def write_producer_report(
    spec: ProducerReportSpec,
    output_dir: str | Path,
) -> ProducerReportOutputs:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    assets_dir = directory / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    episode = spec.episode_factory()
    run = ExperienceRun(
        id=spec.id,
        episode=episode,
        input_summary=spec.input_summary,
        config={**spec.config, "style": spec.style},
    )
    episode_json = run.write_json(directory / "episode.json")
    metadata_json = run.write_metadata(directory / "metadata.json")

    selected_views = list(spec.views or default_views())
    renderer = MatplotlibRenderer(
        layout=LayoutSpec(columns=spec.columns),
        style=spec.style,
    )
    structural = structural_state_for_episode(episode)
    frame_time = spec.frame_time
    if frame_time is None:
        frame_time = _default_frame_time(episode)
    frame = frame_for_time(episode, frame_time, structural)

    frame_png = directory / "frame.png"
    renderer.save_frame(frame, selected_views, frame_png)
    animation_gif = directory / "animation.gif"
    renderer.save_animation(
        episode,
        selected_views,
        animation_gif,
        dt=spec.dt,
        fps=spec.fps,
    )

    asset_paths: list[Path] = []
    asset_paths_by_id: dict[str, Path] = {}
    for asset in spec.view_assets:
        output = assets_dir / asset.filename
        asset_renderer = MatplotlibRenderer(
            layout=LayoutSpec(columns=asset.columns, figsize_per_cell=(6.0, 6.0)),
            style=asset.style or spec.style,
        )
        if asset.kind == "frame":
            asset_renderer.save_frame(frame, list(asset.views), output)
        else:
            asset_renderer.save_animation(
                episode,
                list(asset.views),
                output,
                dt=spec.dt,
                fps=spec.fps,
            )
        asset_paths.append(output)
        asset_paths_by_id[asset.id] = output

    for asset in spec.extra_assets:
        output = assets_dir / asset.filename
        asset.writer(episode, output)
        asset_paths.append(output)
        asset_paths_by_id[asset.id] = output

    checks = producer_report_checks(episode, spec.checks)
    checks_json = directory / "checks.json"
    checks_json.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")

    report_md = directory / "report.md"
    report_md.write_text(
        producer_report_markdown(spec, checks, asset_paths, asset_paths_by_id),
        encoding="utf-8",
    )
    return ProducerReportOutputs(
        directory=directory,
        report_md=report_md,
        episode_json=episode_json,
        metadata_json=metadata_json,
        checks_json=checks_json,
        frame_png=frame_png,
        animation_gif=animation_gif,
        assets=tuple(asset_paths),
    )


def producer_report_checks(
    episode,
    extra_checks: list[ReportCheck] | tuple[ReportCheck, ...] = (),
) -> dict[str, object]:
    structural = structural_state_for_episode(episode)
    frames = episode_frames(episode, structural)
    errors = []
    for index, observation in enumerate(episode.observations):
        expected_error = observation.actual - observation.expected
        if not np.allclose(observation.error, expected_error):
            errors.append(f"frame {index}: prediction error is not actual - expected")
    extra_results = []
    for check in extra_checks:
        try:
            result = check(episode)
        except Exception as exc:  # pragma: no cover - defensive report boundary
            result = {
                "id": getattr(check, "__name__", type(check).__name__),
                "ok": False,
                "errors": [str(exc)],
            }
        extra_results.append(result)

    extra_ok = all(bool(result.get("ok")) for result in extra_results)

    return {
        "ok": not errors and extra_ok,
        "source_name": episode.source_name,
        "adapter": episode.metadata.get("adapter"),
        "input_count": len(episode.inputs),
        "observation_count": len(episode.observations),
        "frame_count": len(frames),
        "duration": episode.duration,
        "errors": errors,
        "extra": extra_results,
    }


def producer_report_markdown(
    spec: ProducerReportSpec,
    checks: dict[str, object],
    assets: list[Path],
    assets_by_id: dict[str, Path] | None = None,
) -> str:
    assets_by_id = assets_by_id or {}
    lines = [
        f"# {spec.title}",
        "",
        spec.description,
        "",
        "## Run",
        "",
        f"- id: `{spec.id}`",
        f"- source: `{checks['source_name']}`",
        f"- adapter: `{checks['adapter']}`",
        f"- input summary: {spec.input_summary}",
        f"- inputs: {checks['input_count']}",
        f"- observations: {checks['observation_count']}",
        f"- checks: {'pass' if checks['ok'] else 'fail'}",
        "",
        "## Standard Outputs",
        "",
        "- [episode.json](episode.json)",
        "- [metadata.json](metadata.json)",
        "- [checks.json](checks.json)",
        "- [frame.png](frame.png)",
        "- [animation.gif](animation.gif)",
    ]
    if assets:
        lines.extend(["", "## Assets", ""])
        for asset in assets:
            lines.append(f"- [{asset.name}](assets/{asset.name})")
    if spec.sections:
        lines.extend(["", "## Walkthrough", ""])
        lines.extend(_render_sections(spec.sections, assets_by_id))
    extra_checks = checks.get("extra", [])
    if extra_checks:
        lines.extend(["", "## Checks", ""])
        for check in extra_checks:
            status = "pass" if check.get("ok") else "fail"
            lines.append(f"- `{check.get('id', 'check')}`: {status}")
            metrics = check.get("metrics")
            if isinstance(metrics, dict) and metrics:
                for key, value in metrics.items():
                    lines.append(f"  - `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def _render_sections(
    sections: list[ReportSection] | tuple[ReportSection, ...],
    assets_by_id: dict[str, Path],
) -> list[str]:
    lines: list[str] = []
    for section in sections:
        lines.extend([f"### {section.title}", "", section.body.strip(), ""])
        for asset_id in section.asset_ids:
            asset = assets_by_id.get(asset_id)
            if asset is None:
                lines.append(f"- missing asset `{asset_id}`")
            else:
                alt = asset.stem.replace("_", " ").replace("-", " ").title()
                lines.append(f"![{alt}](assets/{asset.name})")
            lines.append("")
    return lines


def _default_frame_time(episode) -> float:
    if episode.observations:
        return episode.observations[len(episode.observations) // 2].t
    return 0.0
