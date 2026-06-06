# Compare Demo — the "Two Jimmys" storybook

This storybook is about **comparing** subjective trajectories. Two subjects with
different dials walk the same wall of shapes; we watch their trajectories
diverge, isolate what each walk changed with a matched baseline, and collapse
whole walks to points whose distance measures how differently they experienced
one identical world.

Start here:

- [storybook.md](storybook.md): "Two Jimmys" — surprise curves, a side-by-side
  circle moment, state-effect subtraction, and a distance map of a family of
  Jimmys.

Storybook map:

```text
primitive_demo/  Jimmy and the snake          the kernel loop, no math
main_demo/       Jimmy opens his eyes         one full trajectory, six views
compare_demo/    Two Jimmys                   comparing trajectories  (this folder)
scenarios_demo/  Ten causal probes            one distinction at a time
pressure_demo/   How would we know?           pressure and matched controls
substrates_demo/ Four machines, one contract  Episode interoperability
gpt2_demo/       What GPT-2 expects next      language-model adapter deep dive
```

It is the gentle on-ramp to
[Tutorial 2: Comparing Experiences](../../tutorials/02_comparing_experiences.ipynb),
which has the full machinery (embeddings, distance JSON, population records,
topology population dashboards and atlases).

Regenerate the panels (all from package APIs) from the repository root:

```bash
python notebooks/demos/compare_demo/generate_compare_storybook.py
```

Panels are written to `storybook_assets/` and checked in so the markdown renders
without running anything.
