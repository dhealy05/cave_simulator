# Pressure Demo — "How would we know it's real?"

This storybook has a different job: it teaches the **logic of a pressure
experiment**, not a mechanism. An evolved recurrent creature learns to regulate
exposure in a delayed-value world; the hero is the set of **controls** that try
to break the skill. If the behaviour collapses when you remove memory,
recurrence, or temporal order, the behaviour was real.

The subject in Act I is intentionally small and explicit:

```text
obs_t = one of [cue_good, cue_bad, good, bad, neutral]
h_t = tanh(W_x obs_t + W_h h_{t-1} + b_h)
exposure_t = sigmoid(W_a h_t + b_a)
```

It has a five-dimensional recurrent hidden state and one scalar exposure output.
It is not given a Cave expectation variable, a prediction loss, an explicit memory
trace, or a topology layer. The evaluation world is repeated
`cue_good/cue_bad -> neutral delay -> good/bad outcome`; reward or harm is only
received at the outcome, so the cue has to be carried across the neutral delay.

Start here:

- [storybook.md](storybook.md): the delayed-value world, the creature solving it,
  the subject architecture, the actual cue grammar, the utility collapse under
  controls, and two sharper probes — with the honest
  bounds kept visible.

Storybook map:

```text
primitive_demo/  Jimmy and the snake          the kernel loop, no math
main_demo/       Jimmy opens his eyes         one full trajectory, six views
compare_demo/    Two Jimmys                   comparing trajectories
scenarios_demo/  Ten causal probes            one distinction at a time
pressure_demo/   How would we know?           pressure and matched controls  (this folder)
substrates_demo/ Four machines, one contract  Episode interoperability
gpt2_demo/       What GPT-2 expects next      language-model adapter deep dive
```

It is the gentle on-ramp to
[Tutorial 3: Pressure Results And Substrate Demos](../../tutorials/03_pressures_cavenet_evolved_subjects.ipynb),
and should be read alongside the [scope note](../../../docs/orientation/scope_note.md).

Regenerate the panels (all numbers read from the committed result ladder, not a
fresh genetic search) from the repository root:

```bash
python notebooks/demos/pressure_demo/generate_pressure_storybook.py
```

Panels are written to `storybook_assets/` and checked in so the markdown renders
without running anything.
