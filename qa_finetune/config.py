"""Model choices and training hyper-parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Base models offered in the app. The Unsloth mirrors are ungated (no licence click needed)
# and hold the same weights as the official repos.
MODELS: dict[str, dict] = {
    "Phi-4-mini-instruct (3.8B)": {
        "id": "unsloth/Phi-4-mini-instruct",
        "short": "phi4-mini",
    },
    "Llama-3.2-3B-Instruct": {
        "id": "unsloth/Llama-3.2-3B-Instruct",
        "short": "llama3.2-3b",
    },
}
DEFAULT_MODEL = "Phi-4-mini-instruct (3.8B)"

# Short aliases for the command line.
MODEL_ALIASES = {"phi4": "Phi-4-mini-instruct (3.8B)", "llama": "Llama-3.2-3B-Instruct"}

# Linear layers that get LoRA adapters (same names in Phi-4-mini and Llama 3.2 after
# Unsloth loading; for plain HF loading of Phi-3/4 the fused names are added).
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "qkv_proj", "gate_up_proj",
]


def resolve_model_id(model_key: str | None, model_id: str | None = None) -> str:
    """Return the HuggingFace id to load. An explicit model_id always wins."""
    if model_id:
        return model_id
    key = MODEL_ALIASES.get(model_key or "", model_key) or DEFAULT_MODEL
    if key not in MODELS:
        raise ValueError(f"Unknown model '{model_key}'. Choose from {list(MODELS)}")
    return MODELS[key]["id"]


def short_name(model_id: str) -> str:
    for m in MODELS.values():
        if m["id"] == model_id:
            return m["short"]
    return model_id.rstrip("/").split("/")[-1].lower()[:30]


@dataclass
class TrainConfig:
    # model
    model_key: str = DEFAULT_MODEL
    model_id: str | None = None          # overrides model_key (any HF id or local path)
    backend: str = "auto"                # auto | unsloth | hf
    load_in_4bit: bool = True            # QLoRA = 4-bit base + LoRA adapters
    max_seq_length: int = 1024

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # optimisation
    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    max_steps: int = -1                  # > 0 overrides epochs
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407

    # data
    val_ratio: float = 0.1
    system_prompt: str = ""

    # evaluation
    eval_max_samples: int = 50
    eval_max_new_tokens: int = 128
    eval_base_model: bool = True         # also score the untrained model for comparison

    # output / tracking
    output_root: str = "outputs"
    run_name: str | None = None
    mlflow_experiment: str = "qa-qlora-finetune"
    log_adapter_to_mlflow: bool = True

    extra: dict = field(default_factory=dict)

    @property
    def resolved_model_id(self) -> str:
        return resolve_model_id(self.model_key, self.model_id)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["resolved_model_id"] = self.resolved_model_id
        return d
