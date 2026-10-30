"""Command-line interface (same pipeline as the Gradio app).

Examples
  python -m qa_finetune.cli train --csv data/sample_qa.csv --model phi4 --epochs 3
  python -m qa_finetune.cli ask --run outputs/phi4-mini-20260601-101500 "Is there Wi-Fi?"
  python -m qa_finetune.cli deploy --run outputs/phi4-mini-20260601-101500 \
         --model-repo qa-qlora-adapter --space qa-qlora-chat
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import MODEL_ALIASES, MODELS, TrainConfig


def _train(a):
    from .data import clean_dataframe, read_csv
    from .train import run_training

    df, rep = clean_dataframe(read_csv(a.csv), a.question_col, a.answer_col)
    print(rep.to_markdown().replace("**", ""))
    if not rep.ok:
        sys.exit(1)
    cfg = TrainConfig(
        model_key=MODEL_ALIASES.get(a.model, a.model), model_id=a.model_id,
        backend=a.backend, load_in_4bit=not a.no_4bit, max_seq_length=a.max_len,
        lora_r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, learning_rate=a.lr,
        num_train_epochs=a.epochs, max_steps=a.max_steps, per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_acc, val_ratio=a.val_ratio,
        eval_max_samples=a.eval_samples, eval_base_model=not a.no_base_eval,
        system_prompt=a.system_prompt, output_root=a.output_root, run_name=a.run_name,
        seed=a.seed)
    res = run_training(df, cfg, log_fn=print)
    print(json.dumps(res.metrics, indent=2))
    print(f"Run folder: {res.run_dir}")


def _ask(a):
    from .modeling import generate_answer, load_finetuned
    model, tok, summary = load_finetuned(a.run, backend=a.backend)
    print(generate_answer(model, tok, a.question, summary.get("system_prompt", ""),
                          a.max_new_tokens, a.temperature))


def _deploy(a):
    from .deploy import deploy_run
    urls = deploy_run(a.run, a.token, a.model_repo, a.space, a.private, a.add_token_secret,
                      a.hardware)
    print(json.dumps(urls, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="qa_finetune")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="fine-tune on a CSV")
    t.add_argument("--csv", required=True)
    t.add_argument("--question-col")
    t.add_argument("--answer-col")
    t.add_argument("--model", default="phi4",
                   help=f"phi4 | llama | one of {list(MODELS)}")
    t.add_argument("--model-id", help="any HuggingFace id / local path (overrides --model)")
    t.add_argument("--backend", default="auto", choices=["auto", "unsloth", "hf"])
    t.add_argument("--no-4bit", action="store_true")
    t.add_argument("--max-len", type=int, default=1024)
    t.add_argument("--r", type=int, default=16)
    t.add_argument("--alpha", type=int, default=16)
    t.add_argument("--dropout", type=float, default=0.0)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--epochs", type=float, default=3)
    t.add_argument("--max-steps", type=int, default=-1)
    t.add_argument("--batch", type=int, default=2)
    t.add_argument("--grad-acc", type=int, default=4)
    t.add_argument("--val-ratio", type=float, default=0.1)
    t.add_argument("--eval-samples", type=int, default=50)
    t.add_argument("--no-base-eval", action="store_true")
    t.add_argument("--system-prompt", default="")
    t.add_argument("--output-root", default="outputs")
    t.add_argument("--run-name")
    t.add_argument("--seed", type=int, default=3407)
    t.set_defaults(func=_train)

    q = sub.add_parser("ask", help="ask a saved run a question")
    q.add_argument("--run", required=True)
    q.add_argument("--backend", default="auto")
    q.add_argument("--max-new-tokens", type=int, default=256)
    q.add_argument("--temperature", type=float, default=0.0)
    q.add_argument("question")
    q.set_defaults(func=_ask)

    d = sub.add_parser("deploy", help="push adapter + create HuggingFace Space")
    d.add_argument("--run", required=True)
    d.add_argument("--model-repo", required=True)
    d.add_argument("--space", help="Space name (omit to only push the adapter)")
    d.add_argument("--token", help="HF write token (default: HF_TOKEN env variable)")
    d.add_argument("--private", action="store_true")
    d.add_argument("--add-token-secret", action="store_true")
    d.add_argument("--hardware", help="e.g. zero-a10g or t4-small (default: free CPU)")
    d.set_defaults(func=_deploy)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
