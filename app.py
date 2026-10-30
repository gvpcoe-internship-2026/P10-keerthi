"""QA Fine-Tuner: upload a CSV of question-answer pairs, fine-tune Phi-4-mini or Llama-3.2-3B
with QLoRA (Unsloth + PEFT + TRL), track runs in MLflow, and deploy to a HuggingFace Space.

Run:  python app.py            (local, http://127.0.0.1:7860)
      python app.py --share    (public link, e.g. on Google Colab)
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import traceback
from pathlib import Path

import gradio as gr
import pandas as pd

from qa_finetune.config import DEFAULT_MODEL, MODELS, TrainConfig
from qa_finetune.data import clean_dataframe, detect_columns, read_csv
from qa_finetune.deploy import HARDWARE_CHOICES, deploy_run
from qa_finetune.modeling import (cuda_available, free_memory, generate_answer,
                                  load_finetuned, resolve_backend, unsloth_available)
from qa_finetune.train import list_runs, run_training

ROOT = Path(__file__).resolve().parent
SAMPLE_CSV = ROOT / "data" / "sample_qa.csv"
OUTPUT_ROOT = os.getenv("QA_OUTPUT_ROOT", "outputs")

# One model in memory at a time (this app is meant for one user: local PC or Colab).
LOADED = {"model": None, "tokenizer": None, "run_dir": None, "system_prompt": "",
          "backend": None}
BUSY = threading.Lock()
STOP = threading.Event()


def environment_md() -> str:
    if cuda_available():
        import torch
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        hw = f"✅ GPU: **{gpu}** ({mem:.0f} GB)"
    else:
        hw = ("⚠️ **No GPU found.** QLoRA training of a 3-4B model needs an NVIDIA GPU "
              "(Colab T4 is enough). You can still prepare data, test saved runs slowly, "
              "and deploy.")
    us = "✅ Unsloth installed" if unsloth_available() else \
        "ℹ️ Unsloth not installed - the plain HuggingFace backend will be used"
    return f"{hw}  \n{us}"


def _unload():
    if LOADED["model"] is not None:
        m, t = LOADED["model"], LOADED["tokenizer"]
        LOADED.update(model=None, tokenizer=None, run_dir=None)
        free_memory(m, t)


# ------------------------------------------------------------------------------------
# Tab 1: data
# ------------------------------------------------------------------------------------

def on_upload(file_path):
    if not file_path:
        return None, gr.update(choices=[], value=None), gr.update(choices=[], value=None), \
            "Upload a CSV file to begin.", None, None
    try:
        raw = read_csv(file_path)
    except Exception as e:  # noqa: BLE001
        raise gr.Error(str(e))
    q, a = detect_columns(raw) if raw.shape[1] >= 2 else (None, None)
    cols = list(raw.columns)
    clean, report_md, preview = _clean(raw, q, a)
    return raw, gr.update(choices=cols, value=q), gr.update(choices=cols, value=a), \
        report_md, preview, clean


def _clean(raw, q, a):
    clean, rep = clean_dataframe(raw, q, a)
    return (clean if rep.ok else None), rep.to_markdown(), clean.head(20)


def on_columns(raw, q, a):
    if raw is None or not q or not a:
        return gr.update(), gr.update(), None
    clean, md, preview = _clean(raw, q, a)
    return md, preview, clean


# ------------------------------------------------------------------------------------
# Tab 2: training
# ------------------------------------------------------------------------------------

def _metrics_md(metrics: dict, stopped: bool) -> str:
    if not metrics:
        return ""
    lines = ["### Results" + (" (stopped early)" if stopped else ""),
             f"Training loss: **{metrics.get('train_loss', float('nan')):.4f}** "
             f"in {metrics.get('train_minutes', 0):.1f} min"]
    if "peak_gpu_mem_gb" in metrics:
        lines[-1] += f", peak GPU memory {metrics['peak_gpu_mem_gb']:.1f} GB"
    names = [k[3:] for k in metrics if k.startswith("ft_")]
    if names:
        lines += ["", "| metric | base model | fine-tuned | change |", "|---|---|---|---|"]
        for n in names:
            ft, base = metrics[f"ft_{n}"], metrics.get(f"base_{n}")
            if base is None:
                lines.append(f"| {n} | - | {ft:.3f} | |")
            else:
                lines.append(f"| {n} | {base:.3f} | {ft:.3f} | {ft - base:+.3f} |")
    return "\n".join(lines)


def _loss_frame(history: list[dict]) -> pd.DataFrame:
    rows = []
    for h in list(history):
        if "loss" in h:
            rows.append({"step": h["step"], "loss": h["loss"], "series": "train"})
        if "eval_loss" in h:
            rows.append({"step": h["step"], "loss": h["eval_loss"], "series": "validation"})
    # Fixed dtypes: an empty object-typed frame makes the chart treat loss as text
    return pd.DataFrame(rows, columns=["step", "loss", "series"]).astype(
        {"step": "int64", "loss": "float64", "series": "string"})


def start_training(clean_df, model_key, custom_model, backend, load_4bit, max_len, lora_r,
                   lora_alpha, lora_dropout, lr, epochs, max_steps, batch, grad_acc,
                   val_ratio, eval_samples, eval_base, system_prompt, hf_token):
    if clean_df is None or len(clean_df) == 0:
        raise gr.Error("Load a valid CSV in the Data tab first.")
    if not BUSY.acquire(blocking=False):
        raise gr.Error("A job is already running.")
    try:
        STOP.clear()
        _unload()
        cfg = TrainConfig(
            model_key=model_key, model_id=(custom_model or "").strip() or None,
            backend=backend, load_in_4bit=load_4bit, max_seq_length=int(max_len),
            lora_r=int(lora_r), lora_alpha=int(lora_alpha), lora_dropout=float(lora_dropout),
            learning_rate=float(lr), num_train_epochs=float(epochs),
            max_steps=int(max_steps), per_device_train_batch_size=int(batch),
            gradient_accumulation_steps=int(grad_acc), val_ratio=float(val_ratio),
            eval_max_samples=int(eval_samples), eval_base_model=bool(eval_base),
            system_prompt=(system_prompt or "").strip(), output_root=OUTPUT_ROOT)
        q: queue.Queue = queue.Queue()
        history: list[dict] = []
        box: dict = {}

        def worker():
            try:
                box["res"] = run_training(clean_df, cfg, log_fn=q.put, stop_event=STOP,
                                          hf_token=(hf_token or "").strip() or None,
                                          history_out=history)
            except Exception as e:  # noqa: BLE001
                box["err"] = e
                q.put(f"ERROR: {type(e).__name__}: {e}")
                q.put(traceback.format_exc(limit=3))
            finally:
                q.put(None)

        threading.Thread(target=worker, daemon=True).start()
        lines: list[str] = []
        no_change = gr.update()
        while True:
            try:
                msg = q.get(timeout=2.0)
            except queue.Empty:
                yield "\n".join(lines[-300:]), _loss_frame(history), no_change, no_change, \
                    no_change, no_change
                continue
            if msg is None:
                break
            lines.append(msg)
            yield "\n".join(lines[-300:]), _loss_frame(history), no_change, no_change, \
                no_change, no_change

        if "err" in box:
            raise gr.Error(str(box["err"]))
        res = box["res"]
        LOADED.update(model=res.model, tokenizer=res.tokenizer, run_dir=res.run_dir,
                      system_prompt=cfg.system_prompt, backend=res.backend)
        lines.append(f"\nRun saved in {res.run_dir}. The model is loaded in the Test tab.")
        runs = list_runs(OUTPUT_ROOT)
        yield ("\n".join(lines[-300:]), _loss_frame(history),
               _metrics_md(res.metrics, res.stopped_early),
               res.predictions if len(res.predictions) else None,
               gr.update(choices=runs, value=res.run_dir),
               gr.update(choices=runs, value=res.run_dir))
    finally:
        BUSY.release()


def stop_training():
    STOP.set()
    return "Stop requested - training will end after the current step."


# ------------------------------------------------------------------------------------
# Tab 3: test
# ------------------------------------------------------------------------------------

def load_run(run_dir, hf_token):
    if not run_dir:
        raise gr.Error("Choose a run.")
    if LOADED["run_dir"] == run_dir and LOADED["model"] is not None:
        return f"✅ `{run_dir}` is already loaded."
    if not BUSY.acquire(blocking=False):
        raise gr.Error("A job is running; wait for it to finish.")
    try:
        _unload()
        model, tok, summary = load_finetuned(run_dir, hf_token=(hf_token or "").strip() or None)
        LOADED.update(model=model, tokenizer=tok, run_dir=run_dir,
                      system_prompt=summary.get("system_prompt", ""),
                      backend=resolve_backend("auto"))
        return f"✅ Loaded `{run_dir}` (base model `{summary['base_model']}`)."
    finally:
        BUSY.release()


def ask(question, max_new_tokens, temperature):
    if LOADED["model"] is None:
        raise gr.Error("No model loaded. Train one, or pick a saved run and press Load.")
    if not (question or "").strip():
        raise gr.Error("Type a question.")
    return generate_answer(LOADED["model"], LOADED["tokenizer"], question.strip(),
                           LOADED["system_prompt"], int(max_new_tokens), float(temperature))


# ------------------------------------------------------------------------------------
# Tab 4: deploy
# ------------------------------------------------------------------------------------

def deploy(run_dir, token, model_repo, space_name, private, add_secret, hardware_label):
    if not run_dir:
        raise gr.Error("Choose a trained run.")
    if not (model_repo or "").strip():
        raise gr.Error("Enter a name for the model repository.")
    msgs: list[str] = []
    try:
        urls = deploy_run(run_dir, (token or "").strip() or None, model_repo,
                          (space_name or "").strip() or None, bool(private), bool(add_secret),
                          HARDWARE_CHOICES.get(hardware_label), log_fn=msgs.append)
    except Exception as e:  # noqa: BLE001
        raise gr.Error(f"Deploy failed: {e}")
    out = ["\n".join(f"- {m}" for m in msgs), "",
           f"**Model (LoRA adapter):** {urls['model_url']}"]
    if "space_url" in urls:
        out.append(f"**Space:** {urls['space_url']}  \nThe Space takes a few minutes to "
                   "build the first time. Copy this link into the Demo section of README.md.")
    return "\n".join(out)


# ------------------------------------------------------------------------------------
# Tab 5: experiments
# ------------------------------------------------------------------------------------

def mlflow_runs():
    try:
        import mlflow
        if not os.getenv("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
        df = mlflow.search_runs(experiment_names=["qa-qlora-finetune"])
    except Exception as e:  # noqa: BLE001
        return pd.DataFrame({"info": [f"No MLflow runs yet ({e})"]})
    if df.empty:
        return pd.DataFrame({"info": ["No MLflow runs yet."]})
    want = ["tags.mlflow.runName", "start_time", "tags.status", "params.resolved_model_id",
            "params.n_train", "params.lora_r", "params.learning_rate",
            "params.num_train_epochs", "metrics.train_loss", "metrics.base_token_f1",
            "metrics.ft_token_f1", "metrics.base_rouge_l", "metrics.ft_rouge_l",
            "metrics.train_minutes"]
    df = df[[c for c in want if c in df.columns]].copy()
    df.columns = [c.split(".", 1)[-1].replace("mlflow.", "") for c in df.columns]
    for c in df.select_dtypes("number").columns:
        df[c] = df[c].round(4)
    if "start_time" in df:
        df["start_time"] = df["start_time"].astype(str).str[:19]
    return df


# ------------------------------------------------------------------------------------
# UI
# ------------------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    runs = list_runs(OUTPUT_ROOT)
    with gr.Blocks(title="QA Fine-Tuner") as demo:
        gr.Markdown("# QA Fine-Tuner\nUpload question-answer pairs → fine-tune "
                    "Phi-4-mini or Llama-3.2-3B with QLoRA → test → deploy to a "
                    "HuggingFace Space.")
        gr.Markdown(environment_md())
        raw_state = gr.State(None)
        clean_state = gr.State(None)

        with gr.Tab("1. Data"):
            with gr.Row():
                csv_file = gr.File(label="CSV file (one question and one answer per row)",
                                   file_types=[".csv", ".tsv", ".txt"], type="filepath")
                with gr.Column():
                    sample_btn = gr.Button("Use the sample dataset (library help desk FAQ)")
                    q_col = gr.Dropdown(label="Question column", choices=[])
                    a_col = gr.Dropdown(label="Answer column", choices=[])
            report = gr.Markdown("Upload a CSV file to begin.")
            preview = gr.Dataframe(label="Preview (first 20 clean rows)", wrap=True)

        with gr.Tab("2. Train"):
            with gr.Row():
                model_key = gr.Radio(list(MODELS), value=DEFAULT_MODEL, label="Base model")
                epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                lr = gr.Number(value=2e-4, label="Learning rate")
            system_prompt = gr.Textbox(
                label="System prompt (optional, used in training and in the Space)",
                placeholder="e.g. You are the help desk assistant of Nimbus City Library.")
            with gr.Accordion("Advanced settings", open=False):
                with gr.Row():
                    lora_r = gr.Slider(4, 128, value=16, step=4, label="LoRA rank r")
                    lora_alpha = gr.Slider(4, 256, value=16, step=4, label="LoRA alpha")
                    lora_dropout = gr.Slider(0, 0.3, value=0.0, step=0.05,
                                             label="LoRA dropout")
                with gr.Row():
                    batch = gr.Slider(1, 16, value=2, step=1, label="Batch size per device")
                    grad_acc = gr.Slider(1, 32, value=4, step=1,
                                         label="Gradient accumulation steps")
                    max_steps = gr.Number(value=-1, precision=0,
                                          label="Max steps (-1 = use epochs)")
                with gr.Row():
                    max_len = gr.Slider(256, 4096, value=1024, step=128,
                                        label="Max sequence length (tokens)")
                    val_ratio = gr.Slider(0, 0.3, value=0.1, step=0.05,
                                          label="Validation share")
                    eval_samples = gr.Slider(0, 200, value=50, step=5,
                                             label="Max validation questions to score")
                with gr.Row():
                    eval_base = gr.Checkbox(value=True,
                                            label="Also score the base model (comparison)")
                    load_4bit = gr.Checkbox(value=True, label="Load base model in 4-bit (QLoRA)")
                    backend = gr.Dropdown(["auto", "unsloth", "hf"], value="auto",
                                          label="Backend")
                with gr.Row():
                    custom_model = gr.Textbox(
                        label="Custom base model id (optional, overrides the choice above)")
                    hf_token_train = gr.Textbox(
                        label="HF token (only for gated models)", type="password",
                        value=os.getenv("HF_TOKEN", ""))
            with gr.Row():
                train_btn = gr.Button("Start training", variant="primary")
                stop_btn = gr.Button("Stop", variant="stop")
            stop_msg = gr.Markdown()
            with gr.Row():
                train_log = gr.Textbox(label="Log", lines=18, max_lines=18, autoscroll=True)
                loss_plot = gr.LinePlot(x="step", y="loss", color="series",
                                        label="Loss", height=360)
            metrics_md = gr.Markdown()
            preds = gr.Dataframe(label="Validation answers: base vs fine-tuned", wrap=True)

        with gr.Tab("3. Test"):
            with gr.Row():
                test_run = gr.Dropdown(runs, value=runs[0] if runs else None,
                                       label="Saved run", allow_custom_value=True)
                load_btn = gr.Button("Load")
            load_status = gr.Markdown()
            question = gr.Textbox(label="Question", lines=2)
            with gr.Row():
                max_new = gr.Slider(16, 1024, value=256, step=16, label="Max new tokens")
                temperature = gr.Slider(0, 1.5, value=0.0, step=0.1,
                                        label="Temperature (0 = deterministic)")
            ask_btn = gr.Button("Ask", variant="primary")
            answer = gr.Textbox(label="Answer", lines=6)

        with gr.Tab("4. Deploy"):
            gr.Markdown(
                "Pushes the LoRA adapter to a **model repo** and creates a **Gradio Space** "
                "that loads base model + adapter. Needs a HuggingFace token with **write** "
                "access ([create one](https://huggingface.co/settings/tokens)).")
            deploy_run_dd = gr.Dropdown(runs, value=runs[0] if runs else None,
                                        label="Run to deploy", allow_custom_value=True)
            with gr.Row():
                token = gr.Textbox(label="HF write token", type="password",
                                   value=os.getenv("HF_TOKEN", ""))
                model_repo = gr.Textbox(label="Model repo name", value="qa-qlora-adapter")
                space_name = gr.Textbox(label="Space name (empty = no Space)",
                                        value="qa-qlora-chat")
            with gr.Row():
                private = gr.Checkbox(label="Private repos", value=False)
                add_secret = gr.Checkbox(
                    label="Save my token as the Space's HF_TOKEN secret (needed if private)",
                    value=False)
                hardware = gr.Dropdown(list(HARDWARE_CHOICES), value=list(HARDWARE_CHOICES)[0],
                                       label="Space hardware")
            deploy_btn = gr.Button("Deploy", variant="primary")
            deploy_out = gr.Markdown()

        with gr.Tab("5. Experiments"):
            gr.Markdown("All runs are logged to MLflow (`mlflow.db`). For the full MLflow "
                        "dashboard run: `mlflow ui --backend-store-uri sqlite:///mlflow.db`")
            refresh = gr.Button("Refresh")
            runs_table = gr.Dataframe(wrap=True)

        # events
        csv_file.change(on_upload, csv_file,
                        [raw_state, q_col, a_col, report, preview, clean_state])
        sample_btn.click(lambda: str(SAMPLE_CSV), None, csv_file)
        for dd in (q_col, a_col):
            dd.input(on_columns, [raw_state, q_col, a_col], [report, preview, clean_state])
        train_btn.click(
            start_training,
            [clean_state, model_key, custom_model, backend, load_4bit, max_len, lora_r,
             lora_alpha, lora_dropout, lr, epochs, max_steps, batch, grad_acc, val_ratio,
             eval_samples, eval_base, system_prompt, hf_token_train],
            [train_log, loss_plot, metrics_md, preds, test_run, deploy_run_dd])
        stop_btn.click(stop_training, None, stop_msg)
        load_btn.click(load_run, [test_run, hf_token_train], load_status)
        ask_btn.click(ask, [question, max_new, temperature], answer)
        question.submit(ask, [question, max_new, temperature], answer)
        deploy_btn.click(deploy, [deploy_run_dd, token, model_repo, space_name, private,
                                  add_secret, hardware], deploy_out)
        refresh.click(mlflow_runs, None, runs_table)
        demo.load(mlflow_runs, None, runs_table)
    return demo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--share", action="store_true", help="create a public gradio.live link")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    in_colab = "google.colab" in sys.modules
    build_ui().queue().launch(share=args.share or in_colab, server_name=args.host,
                              server_port=args.port)


if __name__ == "__main__":
    main()
