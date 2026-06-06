#!/usr/bin/env python3
"""Harvest the report-suite output into committed results data and refresh docs.

The report suite writes a rich `index.json` under `out/` (gitignored). This
script turns that ephemeral output into the committed *data of record*:

    artifacts/results/result_ladder/index.json   light, provenance-stamped dataset
    artifacts/results/result_ladder/checks/*.json per-entry full check detail
    artifacts/results/result_ladder/metrics.md    canonical human-readable tables

It also refreshes AUTOGEN regions inside docs so prose numbers are never
hand-copied. A region is any block delimited by matching markers:

    <!-- AUTOGEN:summary -->            ... <!-- /AUTOGEN:summary -->
    <!-- AUTOGEN:provenance -->         ... <!-- /AUTOGEN:provenance -->
    <!-- AUTOGEN:tier:emergence -->     ... <!-- /AUTOGEN:tier:emergence -->
    <!-- AUTOGEN:entry:evolved-exposure --> ... <!-- /AUTOGEN:entry:evolved-exposure -->

Usage:
    python scripts/build_results_index.py            # regenerate + inject
    python scripts/build_results_index.py --check     # verify in sync (CI); exit 1 if stale
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE_INDEX = ROOT / "out" / "report_suites" / "result-ladder" / "index.json"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "results" / "result_ladder"
DEFAULT_DOCS = (
    ROOT / "docs" / "reporting" / "results.md",
    ROOT / "docs" / "papers" / "paper_functional_role_emergence.md",
)

MARKER_RE = re.compile(
    r"(?P<open><!--\s*AUTOGEN:(?P<key>\S+)\s*-->)"
    r"(?P<body>.*?)"
    r"(?P<close><!--\s*/AUTOGEN:(?P=key)\s*-->)",
    re.DOTALL,
)


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def _fmt_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt_value(item) for item in value)
    return str(value)


def _short_metric(key: str, spec: str) -> str:
    for prefix in (f"extra.{spec}.", "extra."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _status(ok: Any) -> str:
    return "pass" if ok else "fail"


# --------------------------------------------------------------------------- #
# region builders
# --------------------------------------------------------------------------- #
def _provenance_line(index: dict[str, Any]) -> str:
    prov = index.get("provenance", {})
    git = prov.get("git", {}) or {}
    sha = (git.get("sha") or "unknown")[:10]
    dirty = " (dirty tree)" if git.get("dirty") else ""
    return (
        f"_Generated from `{index.get('id', 'suite')}` run `{sha}`{dirty} at "
        f"{prov.get('generated_at', '?')} — config `{prov.get('config_sha256', '?')}`, "
        f"python {prov.get('python', '?')}, numpy {prov.get('numpy', '?')}. "
        f"Regenerate with `python scripts/build_results_index.py`._"
    )


def _summary_table(entries: list[dict[str, Any]]) -> str:
    lines = ["| Tier | Entry | Status | Claim |", "| --- | --- | --- | --- |"]
    for entry in entries:
        lines.append(
            f"| {entry['tier']} | `{entry['id']}` | {_status(entry['ok'])} | "
            f"{entry.get('claim', '')} |"
        )
    return "\n".join(lines)


def _tier_table(entries: list[dict[str, Any]], tier: str) -> str:
    rows = [e for e in entries if e["tier"] == tier]
    lines = ["| Entry | Status | Claim |", "| --- | --- | --- |"]
    for entry in rows:
        lines.append(f"| `{entry['id']}` | {_status(entry['ok'])} | {entry.get('claim', '')} |")
    return "\n".join(lines)


def _entry_block(entry: dict[str, Any]) -> str:
    spec = entry.get("spec", "")
    head = f"**{entry.get('title', entry['id'])}** — {_status(entry['ok'])}."
    claim = entry.get("claim", "")
    parts = [head]
    if claim:
        parts.append(claim)
    kwargs = entry.get("kwargs") or {}
    if kwargs:
        rendered = ", ".join(f"`{k}={v}`" for k, v in kwargs.items())
        parts.append(f"Run parameters: {rendered}.")
    metrics = entry.get("key_metrics") or {}
    if metrics:
        table = ["", "| Metric | Value |", "| --- | --- |"]
        for key, value in metrics.items():
            table.append(f"| `{_short_metric(key, spec)}` | {_fmt_value(value)} |")
        parts.append("\n".join(table))
    return "\n\n".join(parts)


def build_regions(index: dict[str, Any]) -> dict[str, str]:
    entries = index.get("entries", [])
    regions: dict[str, str] = {
        "provenance": _provenance_line(index),
        "summary": _summary_table(entries),
    }
    for tier in dict.fromkeys(e["tier"] for e in entries):
        regions[f"tier:{tier}"] = _tier_table(entries, tier)
    for entry in entries:
        regions[f"entry:{entry['id']}"] = _entry_block(entry)
    return regions


# --------------------------------------------------------------------------- #
# committed dataset
# --------------------------------------------------------------------------- #
def build_dataset(index: dict[str, Any], suite_index_path: Path) -> dict[str, Any]:
    keep = ("id", "tier", "kind", "spec", "title", "claim", "ok", "kwargs",
            "key_metrics", "curated_artifacts")
    return {
        "generated_by": "scripts/build_results_index.py",
        "source_suite_index": suite_index_path.relative_to(ROOT).as_posix(),
        "suite_id": index.get("id"),
        "title": index.get("title"),
        "provenance": index.get("provenance", {}),
        "entries": [{k: entry.get(k) for k in keep} for entry in index.get("entries", [])],
    }


def metrics_markdown(index: dict[str, Any], regions: dict[str, str]) -> str:
    lines = [
        f"# {index.get('title', 'Result Ladder')} — Generated Metrics",
        "",
        regions["provenance"],
        "",
        "> This file is generated by `scripts/build_results_index.py`. Do not edit by hand.",
        "",
        "## Summary",
        "",
        regions["summary"],
        "",
    ]
    for entry in index.get("entries", []):
        lines.extend([f"## {entry['id']}", "", regions[f"entry:{entry['id']}"], ""])
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# injection
# --------------------------------------------------------------------------- #
def _navigate(node: Any, dotpath: str) -> Any:
    for part in dotpath.split("."):
        if isinstance(node, list):
            node = node[int(part)]
        elif isinstance(node, dict):
            node = node[part]
        else:
            raise KeyError(part)
    return node


def _titleize(key: str) -> str:
    return " ".join(word.capitalize() for word in str(key).split("_"))


def _render_node(node: Any, columns: list[str] | None) -> str:
    # dict-of-dicts -> a variant x metric table
    if isinstance(node, dict) and node and all(isinstance(v, dict) for v in node.values()):
        cols = columns or list(dict.fromkeys(k for row in node.values() for k in row))
        lines = [
            "| Variant | " + " | ".join(_titleize(c) for c in cols) + " |",
            "| --- | " + " | ".join("---:" for _ in cols) + " |",
        ]
        for variant, row in node.items():
            cells = " | ".join(_fmt_value(row.get(c)) for c in cols)
            lines.append(f"| `{variant}` | {cells} |")
        return "\n".join(lines)
    # dict-of-scalars -> metric/value table
    if isinstance(node, dict):
        lines = ["| Metric | Value |", "| --- | --- |"]
        for key, value in node.items():
            lines.append(f"| `{key}` | {_fmt_value(value)} |")
        return "\n".join(lines)
    return f"`{_fmt_value(node)}`"


def _checks_region(key: str, out_dir: Path) -> str | None:
    """Render a region from a committed checks.json: checks:<id>:<dotpath>[:col,col,...]."""
    _, entry_id, rest = key.split(":", 2)
    if ":" in rest:
        dotpath, colspec = rest.split(":", 1)
        columns = [c.strip() for c in colspec.split(",") if c.strip()]
    else:
        dotpath, columns = rest, None
    path = out_dir / "checks" / f"{entry_id}.json"
    if not path.exists():
        return None
    return _render_node(_navigate(json.loads(path.read_text(encoding="utf-8")), dotpath), columns)


def _resolve_region(key: str, regions: dict[str, str], out_dir: Path) -> str | None:
    if key in regions:
        return regions[key]
    if key.startswith("checks:"):
        try:
            return _checks_region(key, out_dir)
        except (KeyError, IndexError, ValueError, TypeError):
            return None
    return None


def inject(
    text: str, regions: dict[str, str], out_dir: Path, doc_label: str
) -> tuple[str, list[str]]:
    missing: list[str] = []

    def repl(match: re.Match[str]) -> str:
        key = match.group("key")
        content = _resolve_region(key, regions, out_dir)
        if content is None:
            missing.append(key)
            return match.group(0)
        return f"{match.group('open')}\n{content}\n{match.group('close')}"

    new_text = MARKER_RE.sub(repl, text)
    for key in missing:
        print(f"  warn: {doc_label}: no generated content for AUTOGEN:{key}", file=sys.stderr)
    return new_text, missing


def _write_if_changed(path: Path, content: str, *, check: bool) -> bool:
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if current == content:
        return False
    if check:
        print(f"STALE: {path.relative_to(ROOT)}")
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite-index", type=Path, default=DEFAULT_SUITE_INDEX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--docs", type=Path, nargs="*", default=list(DEFAULT_DOCS))
    parser.add_argument("--check", action="store_true", help="Verify outputs are in sync; write nothing; exit 1 if stale.")
    args = parser.parse_args()

    if not args.suite_index.exists():
        print(f"error: suite index not found: {args.suite_index}\n"
              f"Run `python -m cave.presentation.reports.suites` first.", file=sys.stderr)
        return 2

    index = json.loads(args.suite_index.read_text(encoding="utf-8"))
    regions = build_regions(index)
    dataset = build_dataset(index, args.suite_index)

    stale = False

    # committed dataset
    stale |= _write_if_changed(
        args.out_dir / "index.json",
        json.dumps(dataset, indent=2) + "\n",
        check=args.check,
    )
    # per-entry checks copies
    for entry in index.get("entries", []):
        checks_path = ROOT / entry["checks"] if entry.get("checks") else None
        if checks_path and checks_path.exists():
            # Canonicalize with sorted keys so committed checks are byte-stable
            # across suite runs (Python hash randomization can otherwise reorder
            # dict keys between processes and produce spurious diffs).
            canonical = (
                json.dumps(
                    json.loads(checks_path.read_text(encoding="utf-8")),
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            stale |= _write_if_changed(
                args.out_dir / "checks" / f"{entry['id']}.json",
                canonical,
                check=args.check,
            )
    # canonical generated metrics doc
    stale |= _write_if_changed(
        args.out_dir / "metrics.md",
        metrics_markdown(index, regions),
        check=args.check,
    )

    # inject into docs
    for doc in args.docs:
        if not doc.exists():
            print(f"  skip: {doc} (missing)", file=sys.stderr)
            continue
        original = doc.read_text(encoding="utf-8")
        if "AUTOGEN:" not in original:
            continue
        injected, _ = inject(original, regions, args.out_dir, doc.relative_to(ROOT).as_posix())
        stale |= _write_if_changed(doc, injected, check=args.check)

    if args.check and stale:
        print("\nResults data/docs are stale. Run: python scripts/build_results_index.py", file=sys.stderr)
        return 1
    if not args.check:
        print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
