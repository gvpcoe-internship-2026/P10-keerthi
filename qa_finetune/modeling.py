"""Loading models (Unsloth or plain HuggingFace), adding LoRA adapters, and generating answers."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .config import LORA_TARGET_MODULES
from .data import build_messages

log = logging.getLogger(__name__)

STANDARD_LORA = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def bf16_supported() -> bool:
    """True only on GPUs with native bfloat16 (Ampere or newer, e.g. A100/L4).
    torch.cuda.is_bf16_supported() can say True on a T4 through emulation, but Unsloth
    loads T4 models in float16, so we check the compute capability instead."""
    if not cuda_available():
        return False
    import torch
    return torch.cuda.get_device_capability(0)[0] >= 8


def unsloth_available() -> bool:
    try:
        import unsloth  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def resolve_backend(backend: str = "auto") -> str:
    """'unsloth' on a CUDA machine with Unsloth installed, otherwise plain 'hf'."""
    if backend == "auto":
        return "unsloth" if cuda_available() and unsloth_available() else "hf"
    if backend == "unsloth" and not cuda_available():
        raise RuntimeError("Unsloth needs an NVIDIA GPU (CUDA). Use Colab with a T4/L4/A100 "
                           "runtime, or choose backend 'hf'.")
    return backend


def _linear_names(model) -> set[str]:
    import torch
    return {name.split(".")[-1] for name, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and "lm_head" not in name}


def pick_target_modules(model, unsloth: bool = False) -> list[str]:
    """Every attention + MLP projection present in the model.

    Llama has separate q/k/v and gate/up layers; Phi-4-mini keeps them fused as
    qkv_proj and gate_up_proj, so both naming styles are included."""
    present = _linear_names(model)
    targets = [m for m in LORA_TARGET_MODULES if m in present]
    if not targets:
        raise RuntimeError(f"No LoRA target layers found. Linear layers: {sorted(present)}")
    return targets


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------

def load_base_for_training(model_id: str, backend: str, load_in_4bit: bool,
                           max_seq_length: int, lora_r: int, lora_alpha: int,
                           lora_dropout: float, seed: int, hf_token: str | None = None):
    """Load the base model (4-bit when possible) and attach fresh LoRA adapters."""
    token = hf_token or os.getenv("HF_TOKEN")
    if backend == "unsloth":
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id, max_seq_length=max_seq_length, dtype=None,
            load_in_4bit=load_in_4bit, token=token)
        targets = pick_target_modules(model)

        def add_lora(mods):
            return FastLanguageModel.get_peft_model(
                model, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=mods, bias="none", use_gradient_checkpointing="unsloth",
                random_state=seed)

        try:
            model = add_lora(targets)
        except Exception as e:  # noqa: BLE001 - older Unsloth may reject fused names
            standard = [m for m in targets if m in STANDARD_LORA]
            if not standard or standard == targets:
                raise
            log.warning("Unsloth rejected %s (%s); using %s", targets, e, standard)
            targets = standard
            model = add_lora(targets)
    else:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        kwargs: dict = {"token": token}
        use_4bit = load_in_4bit and cuda_available()
        if use_4bit:
            from transformers import BitsAndBytesConfig
            compute = torch.bfloat16 if bf16_supported() else torch.float16
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute)
            kwargs["device_map"] = {"": 0}
        elif cuda_available():
            kwargs["torch_dtype"] = torch.bfloat16 if bf16_supported() \
                else torch.float16
            kwargs["device_map"] = {"": 0}
        else:
            if load_in_4bit:
                log.warning("No GPU: 4-bit loading skipped, training plain LoRA in float32.")
            kwargs["torch_dtype"] = torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if use_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        targets = pick_target_modules(model)
        model = get_peft_model(model, LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
            target_modules=targets, task_type="CAUSAL_LM"))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer, targets


def load_finetuned(run_dir: str, backend: str = "auto", hf_token: str | None = None):
    """Load a saved run (outputs/<run>/) for chatting. Returns (model, tokenizer, summary)."""
    run = Path(run_dir)
    summary = json.loads((run / "training_summary.json").read_text(encoding="utf-8"))
    adapter_dir = str(run / "adapter")
    backend = resolve_backend(backend)
    base_id = summary["base_model"]
    max_len = summary["config"]["max_seq_length"]
    token = hf_token or os.getenv("HF_TOKEN")

    if backend == "unsloth":
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_dir, max_seq_length=max_len, dtype=None,
            load_in_4bit=summary["config"].get("load_in_4bit", True), token=token)
    else:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        kwargs: dict = {"token": token}
        if cuda_available():
            kwargs.update(torch_dtype=torch.bfloat16 if bf16_supported()
                          else torch.float16, device_map={"": 0})
        else:
            kwargs["torch_dtype"] = torch.float32
        base = AutoModelForCausalLM.from_pretrained(base_id, **kwargs)
        model = PeftModel.from_pretrained(base, adapter_dir)
    set_inference(model, backend)
    return model, tokenizer, summary


# --------------------------------------------------------------------------------------
# Mode switching and generation
# --------------------------------------------------------------------------------------

def set_inference(model, backend: str) -> None:
    if backend == "unsloth":
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    model.eval()


def set_training(model, backend: str) -> None:
    if backend == "unsloth":
        from unsloth import FastLanguageModel
        FastLanguageModel.for_training(model)
    model.train()


def generate_answer(model, tokenizer, question: str, system_prompt: str = "",
                    max_new_tokens: int = 256, temperature: float = 0.0,
                    top_p: float = 0.9) -> str:
    import torch

    inputs = tokenizer.apply_chat_template(
        build_messages(question, None, system_prompt), add_generation_prompt=True,
        return_tensors="pt", return_dict=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen_kwargs = dict(max_new_tokens=max_new_tokens, use_cache=True,
                      pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def free_memory(*objs) -> None:
    import gc
    for o in objs:
        del o
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
