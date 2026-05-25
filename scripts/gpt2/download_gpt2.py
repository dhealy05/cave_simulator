from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_ALLOW_PATTERNS = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a GPT-2 Hugging Face snapshot into lib/models/.",
    )
    parser.add_argument(
        "--model",
        default="gpt2",
        help="Hugging Face model id to download.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Model revision, branch, or commit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lib/models/gpt2"),
        help="Local output directory for the model snapshot.",
    )
    parser.add_argument(
        "--include-pytorch-bin",
        action="store_true",
        help="Also download pytorch_model.bin. GPT-2 normally loads from safetensors.",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. Run: "
            'python -m pip install -e ".[gpt2]"'
        ) from exc

    allow_patterns = list(DEFAULT_ALLOW_PATTERNS)
    if args.include_pytorch_bin:
        allow_patterns.append("pytorch_model.bin")

    args.output.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        local_dir=str(args.output),
        allow_patterns=allow_patterns,
    )
    print(f"downloaded {args.model}@{args.revision} to {path}")


if __name__ == "__main__":
    main()
