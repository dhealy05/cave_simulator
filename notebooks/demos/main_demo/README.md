# Main Demo — full-system storybook

The illustrated explainer for the **full** Cave model: the canonical
triangle / circle / square / gap sequence, nine-dimensional feature vectors, and
the six standard views — the same dashboard as the README multi-view GIF, taught
one panel at a time.

Start here:

- [storybook.md](storybook.md): "Jimmy opens his eyes" — a page-by-page build-up
  from a single Presentation panel to the full six-panel dashboard, with the real
  vectors and numbers at each beat.

This is the deeper sibling of [`../primitive_demo`](../primitive_demo/README.md),
which strips the same loop down to a 2-D dot (Jimmy and the snake). Read the
primitive storybook first for the intuition, then this one for the real model.

Storybook map:

```text
primitive_demo/  Jimmy and the snake          the kernel loop, no math
main_demo/       Jimmy opens his eyes         one full trajectory, six views  (this folder)
compare_demo/    Two Jimmys                   comparing trajectories
scenarios_demo/  Ten causal probes            one distinction at a time
pressure_demo/   How would we know?           pressure and matched controls
substrates_demo/ Four machines, one contract  Episode interoperability
gpt2_demo/       What GPT-2 expects next      language-model adapter deep dive
```

Regenerate the panels (produced by the existing `MatplotlibRenderer`, frozen one
instant per page) from the repository root:

```bash
python notebooks/demos/main_demo/generate_main_storybook.py
```

Panels are written to `storybook_assets/` and are checked in so the markdown
renders without running anything.
