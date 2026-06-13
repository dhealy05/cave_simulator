# Topology demo: Two Inks

A focused storybook that explains one object — the **subjective topology** —
from first principles, one ingredient per page. Where the main and primitive
demos unify everything, this book deliberately does not: it is the reference
for "why a terrain, and what is the height?"

- [storybook.md](storybook.md) — the book: eleven pages, plane → stamp →
  decay → hills → two inks → roads → attention gate → field-vs-thread → the
  split → comparison → the 3D terrain (the game's exact gamma/band/extrude
  rule).
- `generate_topology_storybook.py` — regenerates every panel from the real
  engine (`SubjectiveTopologyState.update`) over the interactive game's
  canonical walk (tree, rock, tree, snake) on the game's feature plane
  (novelty × angularity, flat prior).

Regenerate from the repository root:

```bash
python notebooks/demos/topology_demo/generate_topology_storybook.py
```

The split pages read the engine's own tagged fields (`expected_density` /
`actual_density`) rather than the atlas's counterfactual point streams, so
every page is consistent with the sensed/generated inks shown earlier in the
book. The measurement row on page 10 uses the atlas metrics
(`topology_atlas_renderer._atlas_metrics`).
