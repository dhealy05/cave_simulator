# Scenarios Demo — the "Ten causal probes" storybook

This storybook covers the **canonical causal probes**. Where some books follow a
subject, this one follows the *questions* — ten tiny authored worlds, each built
to make **one** distinction inspectable and to fail loudly if the model doesn't
implement it.

Every page has the same concrete shape:

```text
fixture or synthetic sequence
    + subject configuration
    + CaveProducer / pressure-test runner
    -> Episode
    -> check_* metrics
    -> storybook panel
```

Most probes use JSON fixtures in `artifacts/inputs/cave/scenarios/`; the later control
pages use synthetic episodes or the pressure-test runners directly. The important
thing to track on each page is not just the external event, but the one subject
knob being varied: sensorium, attention weights, workspace capacity, learning
weight, valence/objective evaluator, memory/preference capacity, or fake-control
variant.

Start here:

- [storybook.md](storybook.md): ten probes in four acts — first a probe anatomy
  and page map, then what gets *in* (sensing, attention, compression), what
  prediction *costs* (violation, importance), what *value* steers (valence,
  objective attention), and what survives *controls* (preference emergence,
  role-dependency, topology atlas).

Storybook map:

```text
primitive_demo/  Jimmy and the snake          the kernel loop, no math
main_demo/       Jimmy opens his eyes         one full trajectory, six views
compare_demo/    Two Jimmys                   comparing trajectories
scenarios_demo/  Ten causal probes            one distinction at a time  (this folder)
pressure_demo/   How would we know?           pressure and matched controls
substrates_demo/ Four machines, one contract  Episode interoperability
gpt2_demo/       What GPT-2 expects next      language-model adapter deep dive
```

Each page mirrors a probe in
[docs/experiments/scenarios.md](../../../docs/experiments/scenarios.md) and
[Tutorial 3](../../tutorials/03_pressures_cavenet_evolved_subjects.ipynb). Every
number in the prose is read live from that probe's committed `check_*` function —
if a value were wrong, the probe's own test would be failing.

Regenerate the panels from the repository root:

```bash
python notebooks/demos/scenarios_demo/generate_scenarios_storybook.py
```

Panels are written to `storybook_assets/` and checked in so the markdown renders
without running anything.
