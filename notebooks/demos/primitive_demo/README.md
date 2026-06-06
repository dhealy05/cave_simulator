# Primitive Demo

This folder contains the stripped-down Cave trajectory kernel and the reader
materials that explain it. The top level is intentionally only an index; docs,
assets, scripts, runnable demos, and notes live in separate folders.

Start here:

- [docs/storybook.md](docs/storybook.md): the gentlest entry point — "Jimmy and the
  Snake," a no-math picture-book walkthrough of the loop, one encounter per page,
  closing with a filmstrip of the whole walk as one object. Regenerate the pages
  and filmstrips with `python notebooks/demos/primitive_demo/scripts/generate_storybook.py`, or
  make your own random walks with
  `python notebooks/demos/primitive_demo/scripts/generate_storybook.py --random --seed N`.
- [docs/walkthrough.md](docs/walkthrough.md): the main explainer, with embedded
  GIFs and the filmstrip views.
- [docs/walkthrough.html](docs/walkthrough.html): local HTML rendering of the same
  walkthrough, with the Cave Walker animation promoted as the central visual.
- [docs/comprehensive_demo.md](docs/comprehensive_demo.md): design notes for a
  fuller primitive walkthrough that starts with an empty wasteland, then shows
  objects, vectors, observers, attention schedules, recurrence, and topology.
- `notes/`: older implementation notes kept for reference, outside the main
  reader path.

Storybook map:

```text
primitive_demo/  Jimmy and the snake          the kernel loop, no math  (this folder)
main_demo/       Jimmy opens his eyes         one full trajectory, six views
compare_demo/    Two Jimmys                   comparing trajectories
scenarios_demo/  Ten causal probes            one distinction at a time
pressure_demo/   How would we know?           pressure and matched controls
substrates_demo/ Four machines, one contract  Episode interoperability
gpt2_demo/       What GPT-2 expects next      language-model adapter deep dive
```

The implementation is deliberately smaller than the full `Episode`/view stack.
It focuses on the primitive recurrence:

```text
E_t = M_{t-1}
P_t = U_t - E_t
M_t = M_{t-1} + eta * P_t
```

Run the demos from the repository root:

```bash
python notebooks/demos/primitive_demo/primitive_cave/primitive_demo.py
python notebooks/demos/primitive_demo/cave_walker/cave_walker_demo.py
python notebooks/demos/primitive_demo/scripts/render_walkthrough_html.py
python notebooks/demos/primitive_demo/scripts/generate_comprehensive_demo.py
```

Generated artifacts are written to each demo's gitignored `out/` directory. The
checked-in media under `assets/` are stable copies used by the docs. Random
scenarios are written under gitignored `generated/` and can be regenerated with
`scripts/generate_storybook.py --random`.
The interactive tutorial built on top of this primitive source now lives under
`cave/interactive/game/`.

Folder layout:

```text
docs/
  Reader-facing storybooks, walkthrough markdown, and local HTML rendering.

assets/
  Checked-in images and GIFs embedded by the docs.

scripts/
  Generators and one-off renderers for docs and assets.

kernel/
  Small compatibility wrapper around the shared primitive recurrence.

primitive_cave/
  Direct scalar and 2D primitive recurrence demo.

cave_walker/
  Object-world side-scroller demo code.

notes/
  Historical implementation notes for the primitive loop and Cave Walker.
```

The docs under `docs/` are the canonical narrative entry points.
