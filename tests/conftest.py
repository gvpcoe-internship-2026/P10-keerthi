"""Shared test fixtures. A tiny random Llama model with a Phi-style chat template lets the
whole pipeline run on CPU in seconds, without downloading anything."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLE_CSV = ROOT / "data" / "sample_qa.csv"

PHI_STYLE_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|system|>{{ message['content'] }}<|end|>"
    "{% elif message['role'] == 'user' %}<|user|>{{ message['content'] }}<|end|>"
    "{% elif message['role'] == 'assistant' %}<|assistant|>{{ message['content'] }}<|end|>"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% else %}<|endoftext|>{% endif %}"
)
SPECIALS = ["<unk>", "<|pad|>", "<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>",
            "<|end|>"]


def build_tiny_model(path: Path) -> str:
    import pandas as pd
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    df = pd.read_csv(SAMPLE_CSV)
    words = set()
    for text in list(df["question"]) + list(df["answer"]) + ["you are a helpful assistant"]:
        words.update(re.findall(r"\w+|[^\w\s]+", str(text).lower()))
    vocab = {t: i for i, t in enumerate(SPECIALS + sorted(words))}

    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(SPECIALS)
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", pad_token="<|pad|>",
        eos_token="<|end|>", bos_token="<|endoftext|>")
    hf_tok.chat_template = PHI_STYLE_TEMPLATE

    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, pad_token_id=vocab["<|pad|>"],
                      eos_token_id=vocab["<|end|>"], bos_token_id=vocab["<|endoftext|>"])
    import torch
    torch.manual_seed(0)
    LlamaForCausalLM(cfg).save_pretrained(path)
    hf_tok.save_pretrained(path)
    return str(path)


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory) -> str:
    return build_tiny_model(tmp_path_factory.mktemp("tiny-model"))


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_model):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tiny_model)
