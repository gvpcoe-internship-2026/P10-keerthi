"""Deployment helpers that need no network: repo naming, model card and Space files."""

import json

import pytest

from qa_finetune.deploy import full_repo_id, model_card, render_space

SUMMARY = {
    "run_name": "phi4-mini-test", "base_model": "unsloth/Phi-4-mini-instruct",
    "backend": "unsloth", "system_prompt": "You are the library assistant.",
    "lora_targets": ["q_proj", "v_proj"], "n_train": 47, "n_val": 5,
    "metrics": {"base_token_f1": 0.21, "ft_token_f1": 0.74},
    "config": {"lora_r": 16, "lora_alpha": 16, "eval_max_samples": 50},
    "examples": ["Is there Wi-Fi?"],
}


def test_full_repo_id():
    assert full_repo_id("my adapter!", "keerthi") == "keerthi/my-adapter"
    assert full_repo_id("org/name", "keerthi") == "org/name"
    with pytest.raises(ValueError):
        full_repo_id("  ", "keerthi")


def test_model_card_has_metrics_and_base():
    card = model_card(SUMMARY, "keerthi/qa-adapter", "keerthi/qa-chat")
    assert "base_model: unsloth/Phi-4-mini-instruct" in card
    assert "| token_f1 | 0.210 | 0.740 |" in card
    assert "spaces/keerthi/qa-chat" in card


def test_render_space(tmp_path):
    render_space(SUMMARY, "keerthi/qa-adapter", "Library Bot", tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {
        "app.py", "requirements.txt", "README.md", "space_config.json"}
    cfg = json.loads((tmp_path / "space_config.json").read_text())
    assert cfg["adapter_repo"] == "keerthi/qa-adapter"
    assert cfg["system_prompt"] == SUMMARY["system_prompt"]
    readme = (tmp_path / "README.md").read_text()
    assert "sdk: gradio" in readme and "app_file: app.py" in readme
