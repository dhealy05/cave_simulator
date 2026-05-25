# Primitive Demo

This folder contains the stripped-down Cave trajectory kernel and two visual
walkthroughs of it.

Start here:

- [walkthrough.md](walkthrough.md): the main explainer, with embedded GIFs.
- [primitive_loop_2d.gif](primitive_loop_2d.gif): checked-in abstract recurrence
  animation.
- [cave_walker.gif](cave_walker.gif): checked-in object-world companion
  animation.

The implementation is deliberately smaller than the full `Episode`/view stack.
It focuses on the primitive recurrence:

```text
E_t = M_{t-1}
P_t = U_t - E_t
M_t = M_{t-1} + eta * P_t
```

Run the demos from the repository root:

```bash
python notebooks/primitive_demo/primitive_cave/primitive_demo.py
python notebooks/primitive_demo/cave_walker/cave_walker_demo.py
```

Generated artifacts are written to each demo's `out/` directory. The root GIFs
are stable copies for the walkthrough.

Folder layout:

```text
primitive_engine.py
  Shared primitive recurrence kernel.

primitive_cave/
  Direct scalar and 2D primitive recurrence demo.

cave_walker/
  Object-world side-scroller demo, docs, assets, and outputs.
```

Older markdown files under `primitive_cave/` and `cave_walker/docs/` are kept
as implementation notes and historical planning docs. The root walkthrough is
the canonical narrative entry point.
