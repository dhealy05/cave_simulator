"""Generate the in-context-learning panel for the GPT-2 storybook.

Unlike the rest of this storybook (which reuses committed reference assets), this
panel runs a real GPT-2 forward pass over a *repeating* pattern and reads the
per-token surprise straight out of the emitted Cave episode. The point is the
opposite of the rest of the book: here a genuinely Cave-like dynamic — surprise
falling as a pattern locks in, then spiking on a violation — shows up, because the
surprise slot is conditioned on the growing context (in-context learning).

Requires the optional GPT-2 weights at ``lib/models/gpt2`` (see
``docs/producers/gpt2_setup.md``). Run from the repository root:

    pipenv run python notebooks/demos/gpt2_demo/generate_icl_panel.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from cave.observation.producers.sources.gpt2 import GPT2Producer

BASE = Path(__file__).resolve().parent
OUT = BASE / "storybook_assets"

# Colour each token marker by the literal word, so the alternation is legible.
WORD_COLOR = {"red": "#D55E00", "blue": "#0072B2", "green": "#009E73"}
GREY = "#999999"

TEXT = "red blue red blue red blue red blue red blue red green"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    producer = GPT2Producer(
        "lib/models/gpt2",
        feature_count=8,
        active_input_mode="attended_top_k",
        active_top_k=8,
    )
    episode = producer.run(TEXT)

    positions, tokens, surprise, probs, colors = [], [], [], [], []
    for o in episode.observations:
        word = o.metadata.get("token_text", "").strip()
        positions.append(int(o.t))
        tokens.append(word)
        surprise.append(float(o.surprise))
        probs.append(float(o.metadata.get("actual_token_probability", float("nan"))))
        colors.append(WORD_COLOR.get(word, GREY))

    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=140)
    ax.plot(positions, surprise, "-", color=GREY, linewidth=1.6, zorder=2)
    ax.scatter(positions, surprise, c=colors, s=90, zorder=3, edgecolor="white", linewidth=1.2)

    # Mark the violation (last token, "green").
    vi = len(positions) - 1
    ax.annotate(
        f"violation\n'{tokens[vi]}'  surprise {surprise[vi]:.2f}",
        (positions[vi], surprise[vi]),
        textcoords="offset points", xytext=(-10, 22), ha="right", fontsize=10,
        color="#114411", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#114411", lw=1.4),
    )
    ax.annotate(
        f"first guess\nsurprise {surprise[0]:.2f}",
        (positions[0], surprise[0]),
        textcoords="offset points", xytext=(8, -4), ha="left", fontsize=10, color="#555",
    )
    ax.annotate(
        "pattern locks in\n(surprise falls)",
        (positions[len(positions) // 2], surprise[len(positions) // 2]),
        textcoords="offset points", xytext=(6, 34), ha="left", fontsize=10, color="#555",
        arrowprops=dict(arrowstyle="->", color="#999", lw=1.2),
    )

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{t}\n{tok}" for t, tok in zip(positions, tokens)], fontsize=9)
    ax.set_ylim(0, max(surprise) * 1.18)
    ax.set_title(
        "In-context learning: surprise falls as 'red blue' locks in, then spikes on 'green'",
        fontsize=12.5, fontweight="bold", loc="left", pad=10,
    )
    ax.set_xlabel("token position", fontsize=11)
    ax.set_ylabel("surprise  (−log P, nats)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "05_in_context_learning.png", bbox_inches="tight")
    plt.close(fig)

    numbers = {
        "text": TEXT,
        "per_token": [
            {"pos": p, "token": tok, "surprise": round(s, 3), "p_actual": round(pr, 4)}
            for p, tok, s, pr in zip(positions, tokens, surprise, probs)
        ],
        "first_surprise": round(surprise[0], 3),
        "min_surprise": round(min(surprise[:-1]), 3),
        "violation_surprise": round(surprise[-1], 3),
    }
    print("\n=== VERIFIED NUMBERS (for prose) ===")
    print(json.dumps(numbers, indent=1))
    print("\npanel written to", OUT / "05_in_context_learning.png")


if __name__ == "__main__":
    main()
