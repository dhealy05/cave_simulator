# Cave Walker

A small side-scroller companion demo for the primitive Cave recurrence.

## Purpose

Cave Walker is the intuitive version of the primitive demo.

The primitive demo shows the math directly:

```text
E_t = M_{t-1}
P_t = U_t - E_t
M_t = M_{t-1} + ηP_t
```

Cave Walker wraps that same logic in a tiny world.

A little subject walks through an episode and encounters simple objects:

```text
tree → tree → tree → snake → rock → tree
```

At each step, the subject expects something, receives the actual object, registers error, and updates memory.

The subjective trajectory is the path of the subject's changing internal state through that episode.

## Core claim

Cave Walker is not a new model.

It is a UI and object-world layer over the primitive Cave recurrence.

```text
object sequence
→ object-to-feature mapping
→ primitive rollout
→ side-scroller view
→ trajectory/debug view
```

The primitive engine only knows vectors.

The walker UI gives those vectors names, sprites, and a world sequence.

## Minimal object model

Each object has:

```text
id
label
sprite
features
optional value
```

Example:

```python
OBJECTS = {
    "tree": {
        "label": "Tree",
        "sprite": "object_tree.png",
        "features": [0.20, 0.80],
        "value": 0.20,
    },
    "rock": {
        "label": "Rock",
        "sprite": "object_rock.png",
        "features": [0.50, 0.30],
        "value": 0.00,
    },
    "snake": {
        "label": "Snake",
        "sprite": "object_snake.png",
        "features": [0.90, 0.90],
        "value": -1.00,
    },
}
```

The feature axes can be interpreted however the demo needs.

For a first version:

```text
feature x = danger / salience
feature y = height / liveliness
```

The exact semantics are less important than the rule:

```text
world object → feature vector → primitive update
```

## Example episode

```python
EPISODE = ["tree", "tree", "tree", "snake", "rock", "tree"]
```

The subject begins with a tree-like prior:

```python
M_0 = [0.20, 0.80]
η = 0.45
```

At each timestep:

```python
E_t = M_{t-1}
U_t = OBJECTS[current_object]["features"]
P_t = U_t - E_t
M_t = M_{t-1} + ηP_t
surprise_t = norm(P_t)
```

At the snake step, the subject expected something tree-like but encountered a snake-like input.

That creates a large error and bends the internal trajectory.

## What the viewer sees

Cave Walker should have four synchronized views.

### 1. World strip

A side-scroller world:

```text
[tree] [tree] [tree] [snake] [rock] [tree]
```

The little subject walks one tile per timestep.

This is the external sequence.

### 2. Subject HUD

A small panel near the subject:

```text
expects: tree-like
actual: snake
error: high
memory: shifting
```

The HUD should make the current update readable.

### 3. Subjective trajectory map

A 2D feature plane showing:

```text
E_t = expected point
U_t = actual point
P_t = arrow from expected to actual
M_t = updated memory point
```

Over time, the memory points form the memory projection of the subjective trajectory.

### 4. Accumulated topology-like field

A density field over the same feature plane showing where expectation, actual
input, and memory have deposited mass so far:

```text
L_t(x, y) = λL_{t-1}(x, y) + D_t(x, y)
```

This panel is the primitive precursor to the full Cave topology-like state.

## Important distinction

The world path is not the subjective trajectory.

The world path is:

```text
tree → tree → tree → snake → rock → tree
```

The subjective trajectory is:

```text
S_1 → S_2 → S_3 → S_4 → S_5 → S_6
```

where:

```text
S_t = (E_t, U_t, P_t, M_t)
```

Memory alone is not the whole trajectory.

The green memory path is a useful projection of the trajectory.

The full primitive trajectory includes expectation, actual input, error, and memory at every timestep.

## Why this helps

The primitive demo is correct but abstract.

Cave Walker gives the same recurrence an immediate story:

```text
The little subject expected trees.
It encountered a snake.
Prediction error spiked.
Memory shifted.
The internal state path bent.
That path is the subjective trajectory.
```

This makes the core Cave idea easier to see before introducing the full system.

## Asset list

The current renderer uses the GBA-style starter asset pack in
`notebooks/primitive_demo/cave_walker/assets/gba/`.

```text
sprites/subject_idle.png
sprites/subject_walk_1.png
sprites/subject_walk_2.png
sprites/subject_thinking.png
sprites/subject_surprised.png
sprites/subject_update.png

sprites/object_tree.png
sprites/object_rock.png
sprites/object_snake_1.png
sprites/object_snake_2.png
sprites/object_snake_alert.png

tiles/grass_tile.png
tiles/dirt_tile.png
tiles/platform_edge.png

background/cloud.png
background/hills_far.png
background/hills_near.png

ui/marker_expectation.png
ui/marker_actual.png
ui/marker_error.png
ui/marker_memory.png
```

These are intentionally simple placeholder assets. They establish the native
`240 x 160` pixel-art rendering pipeline, not the final art direction.

## Implementation shape

Keep the primitive recurrence separate from the UI.

```text
primitive_engine.py
  shared rollout_vectors(inputs, eta, memory_initial)

cave_walker_objects.py
  object definitions
  episode definitions
  object_to_features(...)

cave_walker_demo.py
  side-scroller rendering
  subject HUD
  internal state map
```

Good separation:

```text
engine = vector recurrence
objects = world-to-vector adapter
UI = side-scroller visualization
```

Avoid making the primitive engine know what a tree, rock, or snake is.

Run from the repository root:

```bash
python notebooks/primitive_demo/cave_walker/cave_walker_demo.py
```

Generated artifacts are written to `cave_walker/out/`:

```text
cave_walker_frame.png
cave_walker.gif
cave_walker_rollout.json
```

## One-sentence summary

Cave Walker is a small side-scroller UI over the primitive Cave loop: a little subject encounters objects, predicts from memory, errs against actual input, updates, and leaves behind a visible subjective trajectory.
