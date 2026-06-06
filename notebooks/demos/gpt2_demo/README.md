# GPT-2 Demo — a language model read as a Cave episode

An illustrated, honest account of pointing Cave's update-loop instrument at a
**GPT-2 forward pass**: how `GPT2Producer` maps a transformer's signals onto the
`Episode` contract, which roles map cleanly, and where the analogy is a stretch.

Storybook map:

```text
primitive_demo/  Jimmy and the snake          the kernel loop, no math
main_demo/       Jimmy opens his eyes         one full trajectory, six views
compare_demo/    Two Jimmys                   comparing trajectories
scenarios_demo/  Ten causal probes            one distinction at a time
pressure_demo/   How would we know?           pressure and matched controls
substrates_demo/ Four machines, one contract  Episode interoperability
gpt2_demo/       What GPT-2 expects next      language-model adapter deep dive  (this folder)
```

Start here:

- [storybook.md](storybook.md): "What GPT-2 expects next" — one forward pass over
  *"Hello, my name is Paul and I like to "*, read six panels at a time, with the
  slot-by-slot mapping, what genuinely looks Cave-like (surprise as real token
  surprisal, the expect-then-see step, the model's own attention), and where it
  falls short (no correction step, `learning_rate = 0.0`, a proxy "memory", a
  per-prompt projection, derived topology) — then the turn: across a repeating
  pattern the surprise slot *does* fall and spike, the model's in-context learning
  showing through.

Most figures are the **committed reference outputs** for the GPT-2 and conversation
producers (under `artifacts/results/`), copied into `storybook_assets/`. The one exception is
the in-context-learning panel (`05_in_context_learning.png`), produced by a real
GPT-2 run from [generate_icl_panel.py](generate_icl_panel.py):

```bash
pipenv run python notebooks/demos/gpt2_demo/generate_icl_panel.py
```

Running GPT-2 needs the optional model weights at `lib/models/gpt2` — see
[GPT-2 setup](../../../docs/producers/gpt2_setup.md) — but the storybook renders from
the checked-in assets without running anything.

For the full adapter contract, timing, and current limits, see the
[GPT-2 Producer](../../../docs/producers/gpt2.md) reference.
