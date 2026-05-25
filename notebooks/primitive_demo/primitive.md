# Primitive Cave

A short intuition note for the stripped-down version of Cave.

## The basic idea

Primitive Cave is the smallest useful version of the larger Cave system.

It does not try to model the whole subject profile, renderer, topology stack, value system, or comparison machinery. It isolates the basic loop that makes an input sequence become a subject-side trajectory.

The primitive loop is:

```text
input → expectation → error → update
```

Expanded slightly:

```text
admitted input
→ expectation from prior memory
→ actual-minus-expected error
→ memory update
→ changed next expectation
```

This is the kernel.

## Why this matters

The larger Cave system can sound complex because it includes many layers:

```text
objects, sensors, attention, expectation, error, memory, value,
workspace compression, exposure, topology, views, comparison
```

Primitive Cave asks:

> What is the smallest version that still produces a trajectory?

The answer is: a recurrent prediction/update loop.

If a system carries state forward, uses that state to form an expectation, compares the expectation with actual input, and updates its state from the mismatch, then the input sequence is no longer just an input sequence.

It has become a state history.

That state history is the primitive subjective trajectory.

## The minimal variables

At each timestep `t`, the primitive system only needs:

| Symbol | Meaning |
|---|---|
| `U_t` | actual admitted input |
| `E_t` | expectation |
| `P_t` | prediction error |
| `M_t` | memory after update |

The minimal state is:

```text
S_t = (U_t, E_t, P_t, M_t)
```

The trajectory is the ordered sequence:

```text
τ_min = (S_1, S_2, ..., S_T)
```

So the trajectory is not just memory.

Memory is one coordinate of the trajectory. The full primitive trajectory includes what was expected, what arrived, how wrong the expectation was, and how memory changed.

## The minimal equations

A very small version can be written as:

```text
E_t = M_{t-1}
P_t = U_t - E_t
M_t = M_{t-1} + ηP_t
```

Where:

```text
η = learning rate
```

This means:

1. the subject expects from prior memory
2. actual admitted input arrives
3. error measures the mismatch
4. memory moves toward the actual input
5. the next expectation changes because memory changed

That is enough to create a history-dependent path.

## Why animation helps

Primitive Cave is best understood as an animation because the important thing is not a static object.

The important thing is the transition:

```text
S_t → S_{t+1}
```

At each timestep, the animation should show:

```text
E_t appears first
U_t arrives
P_t is drawn as the mismatch
M_t updates
M_t becomes the basis for the next expectation
```

In a one-dimensional toy example, this can be shown as values over time.

In a two-feature plane, it becomes even clearer:

```text
E_t = expected point
U_t = actual point
P_t = arrow from expected to actual
M_t = updated memory point
```

The arrow is the error.

The green memory path is not the whole trajectory; it is the memory projection of the trajectory.

## Primitive topology-like state

Once the primitive trajectory exists, we can accumulate a simple field over it.

For example:

```text
L_{t+1}(x, y) = λL_t(x, y) + D_t(x, y)
```

where `D_t` deposits density around the expected, actual, and memory points.

This field is not the trajectory itself.

It is an accumulated trace of the trajectory.

So the distinction is:

```text
trajectory = ordered state history
field = accumulated deposit from that history
```

This gives a very simple precursor to the full Cave topology-like state.

## Why the full Cave system still matters

Primitive Cave is enough to show the kernel:

```text
prior state → expectation → error → update → changed prior state
```

But it starts after input has already been admitted.

The full Cave system explains how input becomes admitted in the first place:

```text
world object
→ exposure
→ sensing
→ attention
→ admitted input
```

It also adds value, workspace compression, future exposure regulation, object memory, topology, and view boundaries.

So the primitive version is not a replacement for the full system.

It is the smallest visible engine inside it.

## The clean relationship

```text
Primitive Cave = minimal trajectory generator
Full Cave = complete subject-side architecture
```

Primitive Cave shows that a trajectory can form.

Full Cave explains why different subjects form different trajectories from the same external episode.

## One-sentence summary

Primitive Cave is the stripped-down prediction/update loop that turns admitted input into a history-dependent subject-side trajectory.
