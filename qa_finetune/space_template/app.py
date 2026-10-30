"""HuggingFace Space: chat with a base model + QLoRA adapter fine-tuned on QA pairs.

Settings come from space_config.json (written by qa_finetune.deploy). Works on:
  * free CPU Spaces (bfloat16 weights, slow but free),
  * ZeroGPU Spaces (needs HF PRO) and paid GPU Spaces (fast).
If the adapter repo is private, add your token as a Space secret named HF_TOKEN.
"""

import json
import os
from threading import Thread

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

try:
    import spaces  # available on HuggingFace Spaces
    ZERO_GPU = os.getenv("SPACES_ZERO_GPU", "").lower() in ("1", "true")
except ImportError:
    spaces, ZERO_GPU = None, False

with open("space_config.json", encoding="utf-8") as f:
    CFG = json.load(f)

TOKEN = os.getenv("HF_TOKEN")
DEVICE = "cuda" if (ZERO_GPU or torch.cuda.is_available()) else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cpu" or torch.cuda.is_bf16_supported() else torch.float16

print(f"Loading {CFG['base_model']} + {CFG['adapter_repo']} on {DEVICE} ({DTYPE}) ...")
tokenizer = AutoTokenizer.from_pretrained(CFG["adapter_repo"], token=TOKEN)
base = AutoModelForCausalLM.from_pretrained(CFG["base_model"], torch_dtype=DTYPE,
                                            token=TOKEN, low_cpu_mem_usage=True)
model = PeftModel.from_pretrained(base, CFG["adapter_repo"], token=TOKEN)
model = model.merge_and_unload()  # fold LoRA into the weights: faster generation
model.to(DEVICE).eval()
print("Model ready.")


def _messages(question: str) -> list[dict]:
    msgs = []
    if CFG.get("system_prompt"):
        msgs.append({"role": "system", "content": CFG["system_prompt"]})
    msgs.append({"role": "user", "content": question})
    return msgs


def _generate(question, max_new_tokens, temperature):
    inputs = tokenizer.apply_chat_template(_messages(question), add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True).to(DEVICE)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(**inputs, streamer=streamer, max_new_tokens=int(max_new_tokens),
                  pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.9)
    else:
        kwargs.update(do_sample=False)
    Thread(target=model.generate, kwargs=kwargs, daemon=True).start()
    text = ""
    for piece in streamer:
        text += piece
        yield text


if ZERO_GPU:
    _generate = spaces.GPU(duration=60)(_generate)


def chat(message, history, max_new_tokens, temperature):
    # The model was trained on single questions, so each turn is answered on its own.
    yield from _generate(message, max_new_tokens, temperature)


demo = gr.ChatInterface(
    fn=chat,
    title=CFG.get("title", "QA Chatbot"),
    description=(f"Base model **{CFG['base_model']}** + LoRA adapter "
                 f"[{CFG['adapter_repo']}](https://huggingface.co/{CFG['adapter_repo']}). "
                 + ("Running on CPU - answers may take a minute." if DEVICE == "cpu" else "")),
    additional_inputs=[
        gr.Slider(16, 1024, value=CFG.get("max_new_tokens", 256), step=16,
                  label="Max new tokens"),
        gr.Slider(0.0, 1.5, value=0.0, step=0.1, label="Temperature (0 = deterministic)"),
    ],
    examples=[[q] for q in CFG.get("examples", [])] or None,
    cache_examples=False,
)

if __name__ == "__main__":
    demo.queue().launch()
