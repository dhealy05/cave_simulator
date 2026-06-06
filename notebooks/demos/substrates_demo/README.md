# Substrates Demo — the "Four machines, one contract" storybook

This storybook varies the **machine**: native Cave, CaveNet (the same update path
written as a network), a stripped-down minimal subject, and an evolved recurrent
black box — with a coda on GPT-2 and conversation producers. The thesis is the
`Episode` contract: utterly different internals, one comparable shape.

The four runnable substrates in the main story are:

| Substrate | What it actually is | Storybook source | Episode slots |
| --- | --- | --- | --- |
| Native Cave | the reference update loop: sensing, attention, workspace, expectation, error, memory, topology | `CaveProducer(demo_model(...))` | fills all contract slots |
| CaveNet | the same update path rewritten as named network blocks and gains | `CaveNetProducer(CaveNet.from_subject_state(...))` | fills all slots; default run matches native Cave exactly |
| Minimal subject | small associative subject with workspace capacity, decaying traces, and preference-weighted memory | `build_preference_emergence_episode("minimal-preference")` | fills all slots at a simpler scale |
| Evolved subject | five-dimensional recurrent controller with sigmoid exposure output | `build_evolved_exposure_episode("evolved-recurrent")` | fills actual, hidden-state memory, and attention/exposure; leaves prediction slots blank |

Start here:

- [storybook.md](storybook.md): the contract and who fills it, CaveNet's
  bit-exact equivalence to native Cave, the minimal and evolved subjects, all four
  in one comparison space, and the contract reaching text models.

Storybook map:

```text
primitive_demo/  Jimmy and the snake          the kernel loop, no math
main_demo/       Jimmy opens his eyes         one full trajectory, six views
compare_demo/    Two Jimmys                   comparing trajectories
scenarios_demo/  Ten causal probes            one distinction at a time
pressure_demo/   How would we know?           pressure and matched controls
substrates_demo/ Four machines, one contract  Episode interoperability  (this folder)
gpt2_demo/       What GPT-2 expects next      language-model adapter deep dive
```

It is the on-ramp to [`cave/substrates/`](../../../cave/substrates/) and the
cross-substrate results in
[Tutorial 3](../../tutorials/03_pressures_cavenet_evolved_subjects.ipynb). Every
number in the prose is computed live from the running substrates.

Regenerate the panels from the repository root:

```bash
python notebooks/demos/substrates_demo/generate_substrates_storybook.py
```

The native/CaveNet/minimal/evolved panels run from committed code; the GPT-2 and
conversation coda panels are copied verbatim from the committed `artifacts/results/`
reference assets (those producers need optional model dependencies to *re-run*).
Panels are written to `storybook_assets/` and checked in so the markdown renders
without running anything.
