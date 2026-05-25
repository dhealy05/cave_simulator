# Setup Scripts

This directory is only for external setup helpers that do not belong to the
package runtime.

New experiment, report, renderer, or asset logic should live in `cave/` next to
the module that owns it.

Canonical entrypoints:

- Reference assets: `python -m cave.demonstrations.reports.reference_assets`
- Cave reference report: `python -m cave.demonstrations.reports.cave_reference`
- Population topology comparison: `python -m cave.demonstrations.reports.topology_comparison`
- Agency/compression summary: `python -m cave.demonstrations.reports.agency_compression_summary`
- Generic view rendering: `python -m cave.presentation.renderers.matplotlib_renderer --views <names>`
- Topology state surface: `python -m cave.presentation.renderers.topology_surface_renderer`
- Report suites: `python -m cave.presentation.reports.suites`

The `scripts/gpt2/` files are setup utilities for local GPT-2 model assets.
