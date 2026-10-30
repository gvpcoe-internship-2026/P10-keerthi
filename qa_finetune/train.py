"""QLoRA fine-tuning of a chat model on question-answer pairs, tracked with MLflow."""

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import TrainConfig, short_name
from .data import split_dataframe, tokenize_dataframe
from .metrics import score_all
from .modeling import (bf16_supported, cuda_available, generate_answer,
                       load_base_for_training, resolve_backend, set_inference, set_training)

log = logging.getLogger(__name__)
LogFn = Callable[[str], None]


@dataclass
class TrainResult:
    run_dir: str
    adapter_dir: str
    base_model: str
    backend: str
    metrics: dict
    loss_history: list[dict]
    predictions: pd.DataFrame
    mlflow_run_id: str | None = None
    stopped_early: bool = False
    model: object = field(default=None, repr=False)
    tokenizer: object = field(default=None, repr=False)


# --------------------------------------------------------------------------------------
# MLflow helpers (training still works if MLflow is missing)
# --------------------------------------------------------------------------------------

class _NoMLflow:
    def __getattr__(self, _):
        return lambda *a, **k: None


def _get_mlflow(experiment: str, say: LogFn):
    try:
        import mlflow
    except ImportError:
        say("MLflow is not installed - experiment tracking is off.")
        return _NoMLflow()
    if not os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment)
    return mlflow


def _flat_params(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat_params(v, key + "."))
        else:
            s = str(v)
            out[key] = s if len(s) <= 500 else s[:497] + "..."
    return out


# --------------------------------------------------------------------------------------
# Trainer construction (works across TRL / Transformers versions)
# --------------------------------------------------------------------------------------

def _filter_kwargs(cls, wanted: dict) -> dict:
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()) and len(params) < 5:
        return wanted
    return {k: v for k, v in wanted.items() if k in params}


def _build_trainer(model, tokenizer, train_ds, eval_ds, cfg: TrainConfig, out_dir: str,
                   warmup_steps: int, callbacks: list, say: LogFn):
    from transformers import DataCollatorForSeq2Seq, Trainer

    use_cuda = cuda_available()
    bf16 = bf16_supported()
    try:
        import bitsandbytes  # noqa: F401
        optim = "adamw_8bit" if use_cuda else "adamw_torch"
    except ImportError:
        optim = "adamw_torch"

    has_eval = eval_ds is not None and len(eval_ds) > 0
    wanted = dict(
        output_dir=out_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps if cfg.max_steps and cfg.max_steps > 0 else -1,
        learning_rate=cfg.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="epoch" if has_eval else "no",
        evaluation_strategy="epoch" if has_eval else "no",
        report_to="none",
        seed=cfg.seed,
        bf16=bf16,
        fp16=use_cuda and not bf16,
        optim=optim,
        use_cpu=not use_cuda,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},   # data is already tokenized
        max_seq_length=cfg.max_seq_length,
        max_length=cfg.max_seq_length,
        packing=False,
    )
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100,
                                      pad_to_multiple_of=8 if use_cuda else None)

    def trainer_kwargs(cls, args):
        kw = dict(model=model, args=args, train_dataset=train_ds,
                  eval_dataset=eval_ds if has_eval else None, data_collator=collator,
                  callbacks=callbacks)
        params = inspect.signature(cls.__init__).parameters
        kw["processing_class" if "processing_class" in params else "tokenizer"] = tokenizer
        return kw

    try:
        from trl import SFTConfig, SFTTrainer
        args = SFTConfig(**_filter_kwargs(SFTConfig, wanted))
        trainer = SFTTrainer(**trainer_kwargs(SFTTrainer, args))
        say("Using TRL SFTTrainer.")
        return trainer
    except Exception as e:  # noqa: BLE001  - version mismatch safety net
        say(f"SFTTrainer could not be built ({type(e).__name__}: {e}); "
            "falling back to transformers.Trainer.")
        from transformers import TrainingArguments
        args = TrainingArguments(**_filter_kwargs(TrainingArguments, wanted))
        return Trainer(**trainer_kwargs(Trainer, args))


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------

def evaluate_qa(model, tokenizer, df: pd.DataFrame, cfg: TrainConfig, backend: str,
                say: LogFn, label: str, stop_event: threading.Event | None = None):
    df = df.head(cfg.eval_max_samples)
    if df.empty:
        return {}, []
    set_inference(model, backend)
    preds = []
    t0 = time.time()
    for i, q in enumerate(df["question"], 1):
        if stop_event is not None and stop_event.is_set():
            break
        preds.append(generate_answer(model, tokenizer, q, cfg.system_prompt,
                                     max_new_tokens=cfg.eval_max_new_tokens))
        if i % 10 == 0 or i == len(df):
            say(f"  {label}: answered {i}/{len(df)} validation questions")
    scores = score_all(preds, list(df["answer"][:len(preds)]))
    say(f"  {label} scores: " + ", ".join(f"{k}={v:.3f}" for k, v in scores.items())
        + f"  ({time.time() - t0:.0f}s)")
    return scores, preds


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------

def run_training(df: pd.DataFrame, cfg: TrainConfig, log_fn: LogFn | None = None,
                 stop_event: threading.Event | None = None,
                 hf_token: str | None = None,
                 history_out: list | None = None) -> TrainResult:
    """Fine-tune on a cleaned frame with columns 'question' and 'answer'.

    log_fn receives progress messages; history_out (if given) is filled live with one dict
    per logging step, so a UI can plot the loss while training runs.
    """
    from datasets import Dataset
    from transformers import TrainerCallback

    def say(msg: str) -> None:
        log.info(msg)
        if log_fn:
            log_fn(msg)

    backend = resolve_backend(cfg.backend)
    if backend == "unsloth":
        import unsloth  # noqa: F401  - must be imported before transformers/trl patches
    model_id = cfg.resolved_model_id
    run_name = cfg.run_name or f"{short_name(model_id)}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = Path(cfg.output_root) / run_name
    adapter_dir = run_dir / "adapter"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ------------------------------------------------------------------------
    train_df, val_df = split_dataframe(df, cfg.val_ratio, cfg.seed)
    train_df.to_csv(run_dir / "train.csv", index=False)
    val_df.to_csv(run_dir / "val.csv", index=False)
    say(f"Data: {len(train_df)} training rows, {len(val_df)} validation rows.")

    # ---- model -----------------------------------------------------------------------
    say(f"Loading {model_id} with backend '{backend}' "
        f"({'4-bit QLoRA' if cfg.load_in_4bit and cuda_available() else 'LoRA'}) ...")
    t0 = time.time()
    model, tokenizer, targets = load_base_for_training(
        model_id, backend, cfg.load_in_4bit, cfg.max_seq_length, cfg.lora_r, cfg.lora_alpha,
        cfg.lora_dropout, cfg.seed, hf_token)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    say(f"Model ready in {time.time() - t0:.0f}s. LoRA on {targets}; trainable params "
        f"{trainable:,} of {total:,} ({100 * trainable / max(total, 1):.2f}%).")

    train_ex, skipped_tr = tokenize_dataframe(tokenizer, train_df, cfg.system_prompt,
                                              cfg.max_seq_length)
    val_ex, skipped_va = tokenize_dataframe(tokenizer, val_df, cfg.system_prompt,
                                            cfg.max_seq_length)
    if skipped_tr or skipped_va:
        say(f"Skipped {skipped_tr + skipped_va} rows whose question is longer than "
            f"max_seq_length={cfg.max_seq_length}.")
    if not train_ex:
        raise ValueError("No training examples left after tokenization.")
    train_ds = Dataset.from_list(train_ex)
    eval_ds = Dataset.from_list(val_ex) if val_ex else None
    lens = [len(e["input_ids"]) for e in train_ex]
    say(f"Tokens per example: mean {sum(lens) / len(lens):.0f}, max {max(lens)}.")

    steps_per_epoch = math.ceil(len(train_ex) / (cfg.per_device_train_batch_size
                                                 * cfg.gradient_accumulation_steps))
    total_steps = cfg.max_steps if cfg.max_steps and cfg.max_steps > 0 else \
        math.ceil(steps_per_epoch * cfg.num_train_epochs)
    warmup_steps = int(round(total_steps * cfg.warmup_ratio))
    say(f"Plan: {total_steps} optimizer steps ({steps_per_epoch} per epoch), "
        f"warm-up {warmup_steps}.")

    # ---- tracking --------------------------------------------------------------------
    mlflow = _get_mlflow(cfg.mlflow_experiment, say)
    run = mlflow.start_run(run_name=run_name)
    run_id = getattr(getattr(run, "info", None), "run_id", None)
    mlflow.log_params(_flat_params({
        **cfg.to_dict(), "backend": backend, "lora_targets": ",".join(targets),
        "n_train": len(train_ex), "n_val": len(val_ex), "total_steps": total_steps,
        "gpu": _gpu_name()}))
    mlflow.log_artifact(str(run_dir / "train.csv"), "data")
    if len(val_df):
        mlflow.log_artifact(str(run_dir / "val.csv"), "data")

    history: list[dict] = history_out if history_out is not None else []
    stopped = {"flag": False}

    class Progress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            logs = logs or {}
            row = {"step": state.global_step, "epoch": round(state.epoch or 0, 3)}
            row.update({k: float(v) for k, v in logs.items()
                        if isinstance(v, (int, float)) and k != "epoch"})
            history.append(row)
            nums = {k: v for k, v in row.items() if k not in ("step", "epoch")}
            mlflow.log_metrics({k.replace("/", "_"): v for k, v in nums.items()},
                               step=state.global_step)
            if "loss" in logs:
                say(f"step {state.global_step}/{state.max_steps}  loss {logs['loss']:.4f}"
                    f"  lr {logs.get('learning_rate', 0):.2e}  epoch {state.epoch:.2f}")
            if "eval_loss" in logs:
                say(f"  validation loss {logs['eval_loss']:.4f}")

        def on_step_end(self, args, state, control, **kw):
            if stop_event is not None and stop_event.is_set():
                stopped["flag"] = True
                control.should_training_stop = True
                say("Stop requested - finishing this step and saving.")

    metrics: dict = {}
    preds_df = pd.DataFrame()
    try:
        # ---- baseline (fresh LoRA has B=0, so this is exactly the base model) --------
        base_scores, base_preds = {}, []
        if cfg.eval_base_model and len(val_df):
            say("Scoring the base model on the validation set (before training) ...")
            base_scores, base_preds = evaluate_qa(model, tokenizer, val_df, cfg, backend,
                                                  say, "base", stop_event)
            set_training(model, backend)

        # ---- train ---------------------------------------------------------------------
        trainer = _build_trainer(model, tokenizer, train_ds, eval_ds, cfg, str(run_dir),
                                 warmup_steps, [Progress()], say)
        say("Training started.")
        t_train = time.time()
        out = trainer.train()
        train_secs = time.time() - t_train
        say(f"Training finished in {train_secs / 60:.1f} min. "
            f"Final average loss {out.training_loss:.4f}.")
        metrics.update({"train_loss": float(out.training_loss),
                        "train_minutes": train_secs / 60})
        if cuda_available():
            import torch
            metrics["peak_gpu_mem_gb"] = torch.cuda.max_memory_reserved() / 1e9

        # ---- save adapter ----------------------------------------------------------------
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        say(f"LoRA adapter saved to {adapter_dir}")

        # ---- fine-tuned evaluation ---------------------------------------------------
        ft_scores, ft_preds = {}, []
        if len(val_df):
            say("Scoring the fine-tuned model on the validation set ...")
            ft_scores, ft_preds = evaluate_qa(model, tokenizer, val_df, cfg, backend, say,
                                              "fine-tuned")
        else:
            set_inference(model, backend)
        metrics.update({f"base_{k}": v for k, v in base_scores.items()})
        metrics.update({f"ft_{k}": v for k, v in ft_scores.items()})
        mlflow.log_metrics(metrics)

        if ft_preds:
            n = len(ft_preds)
            preds_df = val_df.head(n).copy()
            if base_preds:
                preds_df["base_model"] = (base_preds + [""] * n)[:n]
            preds_df["fine_tuned"] = ft_preds
            preds_df.to_csv(run_dir / "predictions.csv", index=False)
            mlflow.log_artifact(str(run_dir / "predictions.csv"), "eval")

        summary = {
            "run_name": run_name, "base_model": model_id, "backend": backend,
            "created": datetime.now().isoformat(timespec="seconds"),
            "system_prompt": cfg.system_prompt, "lora_targets": targets,
            "n_train": len(train_ex), "n_val": len(val_ex), "metrics": metrics,
            "mlflow_run_id": run_id, "stopped_early": stopped["flag"],
            "config": cfg.to_dict(),
        }
        (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2),
                                                       encoding="utf-8")
        pd.DataFrame(history).to_csv(run_dir / "loss_history.csv", index=False)
        mlflow.log_artifact(str(run_dir / "training_summary.json"))
        mlflow.log_artifact(str(run_dir / "loss_history.csv"))
        if cfg.log_adapter_to_mlflow:
            mlflow.log_artifacts(str(adapter_dir), "adapter")
        mlflow.set_tag("status", "stopped" if stopped["flag"] else "completed")
    except Exception:
        mlflow.set_tag("status", "failed")
        raise
    finally:
        mlflow.end_run()

    return TrainResult(run_dir=str(run_dir), adapter_dir=str(adapter_dir),
                       base_model=model_id, backend=backend, metrics=metrics,
                       loss_history=history, predictions=preds_df, mlflow_run_id=run_id,
                       stopped_early=stopped["flag"], model=model, tokenizer=tokenizer)


def _gpu_name() -> str:
    if not cuda_available():
        return "cpu"
    import torch
    return torch.cuda.get_device_name(0)


def list_runs(output_root: str = "outputs") -> list[str]:
    """Run folders that contain a saved adapter, newest first."""
    root = Path(output_root)
    if not root.exists():
        return []
    runs = [p for p in root.iterdir()
            if (p / "adapter").is_dir() and (p / "training_summary.json").exists()]
    return [str(p) for p in sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)]
