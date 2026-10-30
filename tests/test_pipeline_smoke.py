"""End-to-end smoke test on CPU: clean CSV -> LoRA training -> metrics -> MLflow -> reload."""

import json
from pathlib import Path

from conftest import SAMPLE_CSV
from qa_finetune.config import TrainConfig
from qa_finetune.data import clean_dataframe, read_csv
from qa_finetune.modeling import generate_answer, load_finetuned
from qa_finetune.train import list_runs, run_training


def test_train_eval_save_reload(tiny_model, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # mlflow.db and outputs/ land in the temp folder
    df, rep = clean_dataframe(read_csv(str(SAMPLE_CSV)))
    assert rep.ok

    cfg = TrainConfig(model_id=tiny_model, backend="hf", load_in_4bit=True,
                      max_seq_length=128, max_steps=4, per_device_train_batch_size=4,
                      gradient_accumulation_steps=1, learning_rate=5e-3,
                      eval_max_samples=3, eval_max_new_tokens=8,
                      system_prompt="You are a helpful assistant", run_name="smoke")
    logs = []
    res = run_training(df, cfg, log_fn=logs.append)

    run_dir = Path(res.run_dir)
    assert (run_dir / "adapter" / "adapter_config.json").exists()
    summary = json.loads((run_dir / "training_summary.json").read_text())
    assert summary["base_model"] == tiny_model
    assert "ft_token_f1" in res.metrics and "base_token_f1" in res.metrics
    assert len([h for h in res.loss_history if "loss" in h]) == 4
    assert (run_dir / "predictions.csv").exists()
    assert any("Training finished" in m for m in logs)
    assert list_runs("outputs") == [str(run_dir)]

    import mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    runs = mlflow.search_runs(experiment_names=["qa-qlora-finetune"])
    assert len(runs) == 1 and runs.iloc[0]["tags.status"] == "completed"
    assert "metrics.ft_token_f1" in runs.columns

    model, tok, summ = load_finetuned(str(run_dir), backend="hf")
    out = generate_answer(model, tok, "Is there Wi-Fi?", summ["system_prompt"],
                          max_new_tokens=5)
    assert isinstance(out, str)
