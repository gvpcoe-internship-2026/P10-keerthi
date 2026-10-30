import pandas as pd
import pytest

from conftest import SAMPLE_CSV
from qa_finetune.data import (clean_dataframe, detect_columns, read_csv, split_dataframe,
                              tokenize_example)
from qa_finetune.metrics import exact_match, rouge_l, score_all, token_f1


def test_read_sample_csv():
    df = read_csv(str(SAMPLE_CSV))
    assert list(df.columns) == ["question", "answer"]
    assert len(df) >= 50


def test_read_semicolon_and_bytes():
    raw = "Frage;Antwort\nWas?;Das.\n".encode("cp1252")
    df = read_csv(raw)
    assert df.shape == (1, 2)


@pytest.mark.parametrize("cols,expected", [
    (["Question", "Answer"], ("Question", "Answer")),
    (["id", "prompt", "completion"], ("prompt", "completion")),
    (["input", "output"], ("input", "output")),
    (["foo", "bar"], ("foo", "bar")),
])
def test_detect_columns(cols, expected):
    assert detect_columns(pd.DataFrame(columns=cols)) == expected


def test_clean_drops_empty_and_duplicates():
    df = pd.DataFrame({"q": ["a?", "a?", "", "b?", "c?", "d?", "e?", "f?"],
                       "a": ["1", "1", "x", "", "3", "4", "5", "6"]})
    out, rep = clean_dataframe(df, "q", "a")
    assert rep.empty_dropped == 2 and rep.duplicates_dropped == 1
    assert len(out) == 5 and rep.ok
    assert rep.warnings  # fewer than recommended rows


def test_clean_errors_on_too_few_rows():
    out, rep = clean_dataframe(pd.DataFrame({"q": ["a"], "a": ["b"]}))
    assert not rep.ok


def test_split_is_deterministic():
    df, _ = clean_dataframe(read_csv(str(SAMPLE_CSV)))
    tr1, va1 = split_dataframe(df, 0.1, 1)
    tr2, va2 = split_dataframe(df, 0.1, 1)
    assert tr1.equals(tr2) and va1.equals(va2)
    assert len(tr1) + len(va1) == len(df) and len(va1) >= 1


def test_labels_mask_the_prompt(tiny_tokenizer):
    ex = tokenize_example(tiny_tokenizer, "Is there Wi-Fi?", "Yes, free Wi-Fi.",
                          "You are a helpful assistant", 256)
    ids, labels = ex["input_ids"], ex["labels"]
    assert len(ids) == len(labels)
    n_masked = sum(1 for x in labels if x == -100)
    prompt = tiny_tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant"},
         {"role": "user", "content": "Is there Wi-Fi?"}],
        tokenize=False, add_generation_prompt=True)
    assert n_masked == len(tiny_tokenizer(prompt, add_special_tokens=False)["input_ids"])
    answer_part = tiny_tokenizer.decode([t for t in labels if t != -100])
    assert "yes" in answer_part and "<|end|>" in answer_part


def test_tokenize_skips_overlong_prompt(tiny_tokenizer):
    assert tokenize_example(tiny_tokenizer, "library " * 300, "x", "", 64) is None


def test_metrics():
    assert exact_match("The Paris.", "paris") == 1.0
    assert token_f1("open 8 AM to 8 PM", "8 AM to 8 PM") > 0.8
    assert abs(rouge_l("w x y z", "w y z") - 6 / 7) < 1e-9
    s = score_all(["paris", "x"], ["Paris", "y"])
    assert s["exact_match"] == 0.5


def test_lora_targets_cover_fused_phi_layers():
    """Phi-4-mini keeps q/k/v and gate/up fused; all of them must get LoRA."""
    from transformers import LlamaConfig, LlamaForCausalLM, Phi3Config, Phi3ForCausalLM

    from qa_finetune.modeling import pick_target_modules

    small = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                 num_attention_heads=4, num_key_value_heads=2)
    phi = Phi3ForCausalLM(Phi3Config(**small, pad_token_id=0))
    assert pick_target_modules(phi) == ["o_proj", "down_proj", "qkv_proj", "gate_up_proj"]
    llama = LlamaForCausalLM(LlamaConfig(**small))
    assert pick_target_modules(llama) == ["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"]
