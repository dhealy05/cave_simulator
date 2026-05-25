from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test a local GPT-2 snapshot for episode extraction.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("lib/models/gpt2"),
    )
    parser.add_argument(
        "--text",
        default="Cave maps model episodes",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation="eager",
    )
    model.eval()
    inputs = tokenizer(args.text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            output_hidden_states=True,
        )

    attention_layers = 0 if outputs.attentions is None else len(outputs.attentions)
    print(f"tokens: {inputs['input_ids'].shape[-1]}")
    print(f"logits: {tuple(outputs.logits.shape)}")
    print(f"hidden layers: {len(outputs.hidden_states)}")
    print(f"attention layers: {attention_layers}")
    if outputs.attentions:
        print(f"attention[0]: {tuple(outputs.attentions[0].shape)}")
    print(f"device: {next(model.parameters()).device}")


if __name__ == "__main__":
    main()
