# Local Model Artifacts

This directory is for local, untracked artifacts used while developing optional
adapters.

Recommended layout:

```text
lib/
    models/
        gpt2/
    cache/
```

`lib/models/` and `lib/cache/` are ignored by git. Do not commit model weights,
tokenizer downloads, or Hugging Face cache files.

To download GPT-2 into the expected local path:

```bash
python scripts/gpt2/download_gpt2.py --output lib/models/gpt2
```

If GPT-2 already exists elsewhere on this machine, either pass that path to the
future adapter or create a local symlink:

```bash
mkdir -p lib/models
ln -s /path/to/existing/gpt2 lib/models/gpt2
```
