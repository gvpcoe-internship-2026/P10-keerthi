"""Publish a trained LoRA adapter to the HuggingFace Hub and create a Gradio Space for it."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Callable

SPACE_TEMPLATE_DIR = Path(__file__).parent / "space_template"
HARDWARE_CHOICES = {
    "CPU basic (free, slow)": None,
    "ZeroGPU (needs HF PRO)": "zero-a10g",
    "T4 small (paid)": "t4-small",
}


def _api(token: str | None):
    from huggingface_hub import HfApi
    token = token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("A HuggingFace token with WRITE access is needed. Create one at "
                         "https://huggingface.co/settings/tokens")
    return HfApi(token=token), token


def get_username(token: str | None) -> str:
    api, _ = _api(token)
    return api.whoami()["name"]


def full_repo_id(name: str, user: str) -> str:
    name = name.strip()
    if "/" in name:
        return name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    if not safe:
        raise ValueError("Repository name is empty.")
    return f"{user}/{safe}"


def _load_summary(run_dir: str) -> dict:
    return json.loads((Path(run_dir) / "training_summary.json").read_text(encoding="utf-8"))


def _gradio_version() -> str:
    try:
        import gradio
        return gradio.__version__
    except Exception:  # noqa: BLE001
        return "5.49.1"


def model_card(summary: dict, repo_id: str, space_id: str | None) -> str:
    m = summary.get("metrics", {})
    rows = []
    for k in ("exact_match", "contains", "token_f1", "rouge_l"):
        if f"ft_{k}" in m:
            base = m.get(f"base_{k}")
            rows.append(f"| {k} | {base:.3f} | {m[f'ft_{k}']:.3f} |" if base is not None
                        else f"| {k} | - | {m[f'ft_{k}']:.3f} |")
    table = ("| metric | base model | fine-tuned |\n|---|---|---|\n" + "\n".join(rows)) \
        if rows else "No validation set was used."
    cfg = summary.get("config", {})
    n_scored = min(summary.get("n_val", 0), cfg.get("eval_max_samples", 10**9))
    demo = f"\n**Live demo:** https://huggingface.co/spaces/{space_id}\n" if space_id else ""
    return f"""---
base_model: {summary['base_model']}
library_name: peft
pipeline_tag: text-generation
tags:
- qlora
- lora
- unsloth
- trl
- question-answering
---

# {repo_id.split('/')[-1]}

LoRA adapter for **{summary['base_model']}**, fine-tuned with QLoRA on
{summary.get('n_train', '?')} question-answer pairs.
{demo}
## Validation results ({n_scored} held-out questions)

{table}

## Training setup

- LoRA r = {cfg.get('lora_r')}, alpha = {cfg.get('lora_alpha')}, layers: {', '.join(summary.get('lora_targets', []))}
- learning rate {cfg.get('learning_rate')}, epochs {cfg.get('num_train_epochs')}, max steps {cfg.get('max_steps')}
- 4-bit base model: {cfg.get('load_in_4bit')}, max sequence length {cfg.get('max_seq_length')}
- backend: {summary.get('backend')}

## How to use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("{summary['base_model']}", torch_dtype="auto")
model = PeftModel.from_pretrained(base, "{repo_id}")
tok = AutoTokenizer.from_pretrained("{repo_id}")
msgs = [{{"role": "user", "content": "your question"}}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
print(tok.decode(model.generate(ids, max_new_tokens=200)[0][ids.shape[1]:],
                 skip_special_tokens=True))
```
"""


def push_adapter(run_dir: str, repo_id: str, token: str | None, private: bool = False,
                 space_id: str | None = None) -> str:
    api, _ = _api(token)
    summary = _load_summary(run_dir)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(Path(run_dir) / "adapter", tmp, dirs_exist_ok=True)
        Path(tmp, "README.md").write_text(model_card(summary, repo_id, space_id),
                                          encoding="utf-8")
        shutil.copy(Path(run_dir) / "training_summary.json", tmp)
        api.upload_folder(folder_path=tmp, repo_id=repo_id, repo_type="model",
                          commit_message=f"Upload LoRA adapter ({summary['run_name']})")
    return f"https://huggingface.co/{repo_id}"


def render_space(summary: dict, adapter_repo: str, title: str, out_dir: Path) -> None:
    """Write the Space files (app.py, requirements.txt, README.md, space_config.json)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(SPACE_TEMPLATE_DIR / "app.py", out_dir / "app.py")
    shutil.copy(SPACE_TEMPLATE_DIR / "requirements.txt", out_dir / "requirements.txt")
    (out_dir / "space_config.json").write_text(json.dumps({
        "title": title,
        "base_model": summary["base_model"],
        "adapter_repo": adapter_repo,
        "system_prompt": summary.get("system_prompt", ""),
        "max_new_tokens": 256,
        "examples": summary.get("examples", []),
    }, indent=2), encoding="utf-8")
    short_title = title[:60]
    (out_dir / "README.md").write_text(f"""---
title: {json.dumps(short_title)}
emoji: 💬
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: {_gradio_version()}
app_file: app.py
pinned: false
short_description: QA chatbot fine-tuned with QLoRA
models:
- {summary['base_model']}
- {adapter_repo}
---

# {title}

Chat with **{summary['base_model']}** plus the LoRA adapter
[{adapter_repo}](https://huggingface.co/{adapter_repo}), fine-tuned on question-answer pairs.
""", encoding="utf-8")


def create_space(run_dir: str, space_id: str, adapter_repo: str, token: str | None,
                 private: bool = False, add_token_secret: bool = False,
                 hardware: str | None = None, title: str | None = None) -> str:
    api, tok = _api(token)
    summary = _load_summary(run_dir)
    # Show a few validation questions as clickable examples in the Space
    val = Path(run_dir) / "val.csv"
    if val.exists():
        import pandas as pd
        summary["examples"] = list(pd.read_csv(val)["question"].head(4))
    api.create_repo(space_id, repo_type="space", space_sdk="gradio", private=private,
                    exist_ok=True)
    if add_token_secret:
        api.add_space_secret(space_id, "HF_TOKEN", tok)
    with tempfile.TemporaryDirectory() as tmp:
        render_space(summary, adapter_repo, title or space_id.split("/")[-1], Path(tmp))
        api.upload_folder(folder_path=tmp, repo_id=space_id, repo_type="space",
                          commit_message="Deploy QA chatbot")
    if hardware:
        api.request_space_hardware(space_id, hardware)
    return f"https://huggingface.co/spaces/{space_id}"


def deploy_run(run_dir: str, token: str | None, model_repo: str, space_name: str | None,
               private: bool = False, add_token_secret: bool = False,
               hardware: str | None = None,
               log_fn: Callable[[str], None] | None = None) -> dict:
    """Push the adapter, then (optionally) create the Space. Returns the URLs."""
    say = log_fn or print
    user = get_username(token)
    model_id = full_repo_id(model_repo, user)
    space_id = full_repo_id(space_name, user) if space_name else None
    say(f"Uploading adapter to {model_id} ...")
    out = {"model_url": push_adapter(run_dir, model_id, token, private, space_id)}
    if space_id:
        if private and not add_token_secret:
            say("Note: the model repo is private, so the Space needs your token as the "
                "HF_TOKEN secret to read it.")
        say(f"Creating Space {space_id} ...")
        out["space_url"] = create_space(run_dir, space_id, model_id, token, private,
                                        add_token_secret, hardware)
    say("Done.")
    return out
