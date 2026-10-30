# P10 — QA Fine-Tuner: QLoRA Fine-Tuning of Small LLMs from a CSV, Deployed to HuggingFace Spaces

**Student:** keerthi
**Mentor:** Dr. Kanthi Kiran Sirra
**Duration:** 25 May – 19 July 2026

## Problem
Build a Gradio app that takes a CSV of question-answer pairs and fine-tunes Phi-4-mini or Llama-3.2-3B
via QLoRA, then deploys the result to a HuggingFace Space.

## Approach

```
CSV ──► 1. Data check ──► 2. QLoRA training ──► 3. Evaluation ──► 4. Deploy
         columns,          Unsloth 4-bit base    base vs           adapter → HF model repo
         empties,          + LoRA (PEFT),        fine-tuned on     Space (Gradio chat) loads
         duplicates,       TRL SFTTrainer,       held-out          base + adapter
         train/val split   MLflow logging        questions
```

1. **Data.** The CSV is read with automatic encoding/separator detection. The question and answer
   columns are detected by name (`question/prompt/input…`, `answer/response/output…`) and can be
   changed in the app. Empty rows and duplicates are removed, and a seeded train/validation split
   (default 90/10) is made.
2. **Formatting.** Each pair is written with the **model's own chat template**
   (`<|user|>…<|assistant|>` for Phi-4-mini, `<|start_header_id|>…` for Llama 3.2), with an optional
   system prompt. The loss is computed **on the answer tokens only** (prompt tokens get label -100).
   Because the same template is used for chatting, training and inference formats match.
3. **QLoRA.** The base model is loaded in 4-bit NF4 with Unsloth; LoRA adapters (default r=16,
   alpha=16) are added to all attention and MLP projections; only about 1% of the weights are trained.
   TRL's `SFTTrainer` runs the training (8-bit AdamW, linear schedule, warm-up, fp16 on T4 / bf16 on
   Ampere+).
4. **Evaluation.** Before training (fresh LoRA = exact base model) and after training, the model
   answers the validation questions. Scores: exact match, "contains the gold answer", token F1 and
   ROUGE-L. The app shows a base-vs-fine-tuned table and every answer side by side.
5. **Tracking.** Every run is logged to MLflow: all settings, loss per step, validation loss, scores,
   the train/val data, predictions and the adapter.
6. **Deployment.** The adapter (about 50–120 MB, not the full 3–4B model) is pushed to a HuggingFace model repo
   with an auto-generated model card. A Gradio Space is created that downloads the base model and the
   adapter, merges them and serves a streaming chat. It is created on **ZeroGPU**, which free
   HuggingFace accounts can use for up to 2 Gradio Spaces. (Since 2026, Gradio Spaces on CPU need a
   paid PRO/Team plan.)

## Tech Stack
Unsloth, PEFT, TRL, MLflow, HuggingFace Hub, Gradio, Transformers, PyTorch

## Project structure

```
app.py                         Gradio app (Data → Train → Test → Deploy → Experiments tabs)
qa_finetune/
  config.py                    model choices and all hyper-parameters (TrainConfig)
  data.py                      CSV reading, checks, split, chat-template tokenization
  modeling.py                  model loading (Unsloth or plain HF), LoRA, generation
  train.py                     training loop, evaluation, MLflow logging, saving
  metrics.py                   exact match, contains, token F1, ROUGE-L
  deploy.py                    push adapter + model card, create the Space
  cli.py                       command-line version of the same pipeline
  space_template/              app.py + requirements.txt copied into the Space
data/sample_qa.csv             52 example QA pairs (fictional library help desk)
notebooks/
  Run_QA_FineTuner_on_Colab.ipynb    one-click Colab runner
  QLoRA_FineTuning_Phi4_Mini_Unsloth_10Step_Demo.ipynb   Week-1 demo
tests/                         16 tests, run on CPU with a tiny model
```

## How to run

### Google Colab (recommended — free T4 GPU)
Open `notebooks/Run_QA_FineTuner_on_Colab.ipynb` in Colab, select a **T4 GPU** runtime and run the
cells top to bottom. The app opens on a public `gradio.live` link.

### Your own machine with an NVIDIA GPU (Linux or WSL2)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py            # open http://127.0.0.1:7860
```

### Command line
```bash
python -m qa_finetune.cli train --csv data/sample_qa.csv --model phi4 --epochs 3 \
       --system-prompt "You are the help desk assistant of Nimbus City Library."
python -m qa_finetune.cli ask --run outputs/<run-folder> "Is there Wi-Fi?"
python -m qa_finetune.cli deploy --run outputs/<run-folder> --model-repo qa-qlora-adapter --space qa-qlora-chat
```
Use `--model llama` for Llama-3.2-3B. `python -m qa_finetune.cli train -h` lists every option.

### Tests (CPU, no download, ~10 s)
```bash
pip install -r requirements-cpu.txt
pytest
```

### MLflow dashboard
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000
```

## CSV format
One row per pair, with a header. Column names are detected automatically; other columns are ignored.
```csv
question,answer
What are the opening hours?,"Monday to Saturday, 8 AM to 8 PM."
```
At least 5 rows are required; 50+ is recommended.

## Hardware notes

| Where | Training | Chat speed |
|---|---|---|
| Colab T4 (16 GB) | Phi-4-mini / Llama-3.2-3B 4-bit: a few minutes for ~50 rows × 3 epochs | fast |
| HF Space, ZeroGPU (free accounts: up to 2 Spaces) | — | fast |
| HF Space, CPU basic (needs HF PRO) | — | slow (about 1 token/s) |
| HF Space, T4 small (paid per hour) | — | fast |

## Notes from the Week-1 demo notebook
* The demo trained with a `### Instruction / ### Response` text format but tested with the chat
  template, so training and testing saw different formats. The app now uses the chat template for both.
* The demo computed the loss on the question as well as the answer; now only answer tokens are trained.
* `SFTTrainer(tokenizer=…, dataset_text_field=…, max_seq_length=…)` changed in newer TRL versions; the
  app builds its arguments by checking which names the installed version accepts.

## Status
- [ ] Week 1: Foundation
- [ ] Week 2: Literature + baseline
- [ ] Week 3: Initial experiments
- [ ] Week 4: Working prototype (CIS evaluation)
- [ ] Weeks 5–7: Refinement
- [ ] Week 8: Final deliverable

## Demo
[Link here once deployed]
