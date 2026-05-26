# Cave

Cave uses Plato's allegory as a design frame for computational functions often
invoked in consciousness research: sensing, attention, memory, expectation,
error, learning, value, action/exposure, and topology-like state.

From these functions we can glean an observable, transferrable model for the subject's experience over time: *the subjective trajectory*.

We start with a simulation of the cave, with panels as follows:

1. A fixed point of view observes a wall against which objects appear.
2. The objects recede into the observer's memory
3. They are filtered by the observer's attention
4. The observer develops internal expectations
5. The observer develops internal predictions
6. We measure the observer's subjective trajectory

![Synchronized projections of one Cave episode](results/readme/01_multi_view_state.gif)

This is the basis of the simulated cave. We have assumed only that the subject

1. has a memory (he thinks back)
2. has an expectation (he thinks forward) and
3. has an experience (he stares ahead).

We proceed below to compare trajectories on a population basis and to apply treatments to different subject substrates in an effort to see whether functional primitives arise independently. We find that matched pressure plus sufficient capacity reliably bends population trajectories toward four of the five reference roles.

To render your own, follow the tutorials:

1. [Build and render one subjective trajectory](notebooks/tutorials/01_intro_to_cave_subjective_trajectory.ipynb);
2. [Compare trajectories across subjects, experiences, populations, and substrates](notebooks/tutorials/02_comparing_experiences.ipynb);
3. [Run pressure experiments under delay, bottleneck, value, and exposure demands](notebooks/tutorials/03_pressures_cavenet_evolved_subjects.ipynb).

and for a deeper dive, read up on

1. [Subjective Trajectories](docs/subjective_trajectories.pdf): construction vocabulary for subjective trajectories.
2. [Functional Role Emergence Under Pressure](docs/convergence_under_pressure.pdf): pressure/capacity/function thesis and current evidence.

## Build One Trajectory

Tutorial 1 and Paper 1 start with the same object: a temporal input sequence
passed through a configured subject.

An `ExperienceObject` is an external event with a time interval, feature vector,
salience, modality, and optional presentation metadata. A subject-side model
then decides what is sensed, attended, expected, missed, learned, retained, and
carried forward.

```python
from pathlib import Path

from cave import CaveProducer, default_views, demo_model
from cave.presentation.renderers import LayoutSpec, MatplotlibRenderer

out = Path("out/readme")
out.mkdir(parents=True, exist_ok=True)

episode = CaveProducer(demo_model()).run(dt=0.1)

renderer = MatplotlibRenderer(
    layout=LayoutSpec(columns=2, figsize_per_cell=(5.2, 4.2)),
)
renderer.save_animation(
    episode,
    default_views(),
    out / "trajectory.gif",
    dt=0.1,
    fps=8,
)
```

The same path is available from the CLI:

```bash
cave-render --demo --output out/readme/trajectory.gif --views all --columns 2
cave-run --demo --output out/readme/episode.json
```

The core update is easiest to see in the expectation/actual view: each timestep
has an expected vector, an attended actual vector, a signed error, a learning
rate, and an after-update memory state.

![Expected input, actual input, error, and update](results/readme/05_expectation_actual.gif)

For the full API walkthrough, see
[Tutorial 1](notebooks/tutorials/01_intro_to_cave_subjective_trajectory.ipynb).
For the formal construction story, see
[Paper 1: Subjective trajectories](docs/papers/paper_subjective_trajectories.md).

## Configure A Subject

The native Cave subject is configured through `ModelParams`: attention, memory,
topology, learning, workspace compression, value/objective evaluation, and
optional action or exposure policy.

```python
from dataclasses import replace

from cave import AttentionProfile, MemoryParams, default_model_params

params = replace(
    default_model_params(),
    attention=AttentionProfile(mode="sine", level=0.55, amplitude=0.35),
    memory=MemoryParams(retention=0.86, decay_tau=2.0, max_age=5.0),
)
```

Attention changes the timing and strength of admission into the subject-side
update. Split-channel attention can also redistribute access across external
input, audio, and internal expectation.

![Attention profile examples](results/readme/09_attention_profiles.png)

## Inspect Episodes

`Episode` is the common contract. Native Cave, GPT-2 text runs, conversation
runs, CaveNet, and pressure-test substrates all adapt into this shape.

Each observation can contain expected input, actual input, memory state,
surprise, learning rate, attention, active inputs, and metadata. Views and
dashboards read that state; they do not mutate the run.

```python
print(episode.duration)
print(episode.vocabulary)
print(episode.observations[-1].memory_state)
print(episode.observations[-1].surprise)
```

![Episode observation readout](results/readme/10_episode_readout.png)

The visual layer includes presentation, timeline, memory lookback,
expectation/actual, correction, affect/action, and subjective topology views.
Topology is an accumulated density over a chosen feature plane, useful as an
inspection surface rather than a literal mental map.

![Whole-run topology state surface](results/readme/08_topology_surface.png)

For view implementation details and image generation notes, see
[README image construction](docs/reporting/readme_image_construction.md).

## Compare Trajectories

Tutorial 2 moves from one trajectory to many. Comparison tools operate on
episodes, not screenshots: same world across different subjects, same subject
across different experience sequences, or different substrates exported through
the same `Episode` contract.

```python
from cave import episode_set, labeled_episode
from cave.presentation.renderers import (
    save_episode_set_dashboard,
    save_episode_set_distances_json,
)

episodes = episode_set(
    [
        labeled_episode(episode, id="baseline", label="baseline"),
        # labeled_episode(other_episode, id="low-capacity", label="low capacity"),
    ],
    id="subject_comparison",
    title="Subject Comparison",
    comparison_axis="subject configuration",
)

save_episode_set_dashboard(episodes, out / "comparison.png")
save_episode_set_distances_json(episodes, out / "comparison_distances.json")
```

Built-in embeddings include observed memory, state effect, actual input, and a
broader subjective trajectory embedding.

```text
observed memory = what the episode directly retained
state effect    = observed memory minus a matched baseline
trajectory      = expected, actual, error, memory, attention, and adaptation
```

State-effect subtraction is the key comparison idea: it isolates what the
current episode changed rather than confusing that change with prior state.

![Baseline-subtracted state effect](results/readme/10_state_effect_subtraction.png)

Population tools add factor labels such as treatment, start condition, subject
profile, mechanism condition, or substrate. They let a report ask whether
families of trajectories converge, separate, collapse under controls, or
preserve structure.

![Population topology dashboard](results/readme/12_population_topology_dashboard.png)

For the full comparison workflow, see
[Tutorial 2](notebooks/tutorials/02_comparing_experiences.ipynb).

## Pressure Experiments

### Hypothesis

Why should certain trajectory-transforming functions appear at all? Well, consciousness evolved for a reason. We propose the following:

```text
capacity + pressure -> useful mathematical function
```

and to test it, we subject three substrates of increasing distance from the named Cave architecture to matched pressures:

1. the reference model
2. a weakened network (CaveNet) whose gains can adapt
3. a compact recurrent subject evolved only on delayed exposure utility

We then ask whether the predicted trajectory deformations appear and collapse in patterned ways when the enabling capacity is removed, scrambled, or reset.

### Results

The strongest current non-reference result is the evolved-dissociation world:
rare cues carry true delayed value, common cues carry the same surface sign
without consequence, and distractors fill the delay. A weak recurrent subject
evolves exposure control that beats a frequency-counter by ~4.8 utility,
decodes the rare future outcome from hidden state at 1.0 accuracy, and
collapses to chance under reset, non-recurrent, and shuffled controls.

![Evolved exposure control metrics](results/readme/18_evolved_exposure_metrics.png)

CaveNet sits one rung lower as a calibration result: pressure signals move a
weakened, role-compatible architecture in the predicted direction, more
strongly when a learned controller maps pressure summaries into gains than
when a hand-written rule does. The gain history shows the mechanism; the
result ladder records how far it actually moves.

![Adaptive CaveNet config history](results/readme/16_cavenet_config_history.png)

Across substrates, expectation, value retention, regulation, and latent
topology emerge with clean control collapse. Selection is reported more
conservatively as cue-weight concentration rather than full dynamic attention.
The role evidence board reads as bounded functional resemblance — not
coordinate identity with Cave variables, and not a consciousness claim.

![Role evidence board](results/readme/17_role_evidence_board.png)

Every claim above is recorded with metrics, controls, and pass/fail status in
[`results/result_ladder/metrics.md`](results/result_ladder/metrics.md).

For the pressure-result walkthrough, see
[Tutorial 3](notebooks/tutorials/03_pressures_cavenet_evolved_subjects.ipynb).

For the paper framing, see
[Paper 2: Functional convergence under pressure](docs/convergence_under_pressure.pdf).
