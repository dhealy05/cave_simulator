from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from cave.presentation.renderers.matplotlib_renderer import available_styles
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
from cave.pressure.checks.preference_emergence import (
    preference_emergence_report_spec,
)
from cave.demonstrations.scenarios.representational_compression import (
    representational_compression_report_spec,
)
from cave.demonstrations.scenarios.role_dependency_contrasts import (
    role_dependency_contrasts_report_spec,
)
from cave.demonstrations.scenarios.topology_atlas import topology_atlas_report_spec
from cave.demonstrations.scenarios.unseen_modality import unseen_modality_report_spec
from cave.demonstrations.scenarios.valence_attractor_repulsor import (
    valence_attractor_repulsor_report_spec,
)
from cave.presentation.reports.generate import write_producer_report


SCENARIO_SPECS = {
    "attention-bottleneck": attention_bottleneck_report_spec,
    "expectation-violation": expectation_violation_report_spec,
    "importance-weighted-event": importance_weighted_event_report_spec,
    "objective-attention-shift": objective_attention_shift_report_spec,
    "preference-emergence": preference_emergence_report_spec,
    "representational-compression": representational_compression_report_spec,
    "role-dependency-contrasts": role_dependency_contrasts_report_spec,
    "topology-atlas": topology_atlas_report_spec,
    "unseen-modality": unseen_modality_report_spec,
    "valence-attractor-repulsor": valence_attractor_repulsor_report_spec,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Cave causal probe reports.")
    parser.add_argument(
        "scenario",
        choices=sorted(SCENARIO_SPECS.keys()),
        help="Scenario report to generate.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--style", default="default", choices=available_styles())
    parser.add_argument(
        "--skip-assets",
        action="store_true",
        help="Only write the standard report frame and animation.",
    )
    args = parser.parse_args()

    spec = SCENARIO_SPECS[args.scenario](
        dt=args.dt,
        fps=args.fps,
        include_assets=not args.skip_assets,
    )
    spec = replace(
        spec,
        style=args.style,
        config={**spec.config, "style": args.style},
    )

    output = args.output or Path("out/reports/cave/scenarios") / spec.id
    outputs = write_producer_report(spec, output)
    print(f"wrote {outputs.report_md}")


if __name__ == "__main__":
    main()
