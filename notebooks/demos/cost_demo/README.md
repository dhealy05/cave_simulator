# Cost Demo — "Who paid for the thought?"

This storybook teaches the cost and compression layer that now sits beside the
ordinary Cave trajectory views. The earlier storybooks ask what trajectory
formed: expected input, actual input, error, memory, topology. This one asks
what had to be paid for that trajectory to exist, and who governed that work.

The central distinction is:

```text
compressed state != active compression
```

A compact state can be supplied by rails, projected randomly, amortized from
prior training, or actively earned by the subject during the episode. The cost
reports separate those cases by reading source load, state capacity, distortion,
update work, ownership, and future effect.

Start here:

- [storybook.md](storybook.md): a page-by-page explanation of primitive
  compression pressure, ownership, loss-to-work coupling, the capacity clamp
  controls, and how cost accounting changes the interpretation of CaveNet,
  evolved subjects, GPT-2, and conversation adapters.

Storybook map:

```text
primitive_demo/  Jimmy and the snake          the kernel loop, no math
main_demo/       Jimmy opens his eyes         one full trajectory, six views
compare_demo/    Two Jimmys                   comparing trajectories
scenarios_demo/  Ten causal probes            one distinction at a time
pressure_demo/   How would we know?           pressure and matched controls
cost_demo/       Who paid for the thought?    compression, work, ownership  (this folder)
substrates_demo/ Four machines, one contract  Episode interoperability
gpt2_demo/       What GPT-2 expects next      language-model adapter deep dive
```

Refresh the local storybook asset copies from the repository root:

```bash
python notebooks/demos/cost_demo/generate_cost_storybook.py
```

The source figures are the committed README cost artifacts in `artifacts/results/readme/`.
The generator only copies those figures into `storybook_assets/`; it does not
rerun the full report ladder.
