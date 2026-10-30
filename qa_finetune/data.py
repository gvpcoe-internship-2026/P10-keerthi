"""Load, check and split a CSV of question-answer pairs, and turn it into training tokens."""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import pandas as pd

QUESTION_NAMES = ["question", "questions", "q", "prompt", "input", "instruction", "query", "user"]
ANSWER_NAMES = ["answer", "answers", "a", "response", "output", "completion", "target",
                "assistant", "reply"]

MIN_ROWS = 5
RECOMMENDED_ROWS = 50


@dataclass
class DataReport:
    rows_in_file: int = 0
    rows_kept: int = 0
    empty_dropped: int = 0
    duplicates_dropped: int = 0
    question_col: str = ""
    answer_col: str = ""
    avg_question_words: float = 0.0
    avg_answer_words: float = 0.0
    max_answer_words: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_markdown(self) -> str:
        lines = [
            f"**Rows in file:** {self.rows_in_file}  ",
            f"**Rows kept:** {self.rows_kept} "
            f"(empty dropped: {self.empty_dropped}, "
            f"duplicates dropped: {self.duplicates_dropped})  ",
            f"**Columns:** question = `{self.question_col}`, answer = `{self.answer_col}`  ",
            f"**Average length:** question {self.avg_question_words:.1f} words, "
            f"answer {self.avg_answer_words:.1f} words (longest answer {self.max_answer_words})",
        ]
        for e in self.errors:
            lines.append(f"\n❌ {e}")
        for w in self.warnings:
            lines.append(f"\n⚠️ {w}")
        if self.ok and not self.warnings:
            lines.append("\n✅ Data looks good.")
        return "\n".join(lines)


def read_csv(source) -> pd.DataFrame:
    """Read a CSV from a path, bytes or file-like object. Tries common encodings and separators."""
    raw: bytes
    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    elif hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
    else:
        with open(source, "rb") as f:
            raw = f.read()

    last_err: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
        try:
            # sep=None lets pandas sniff comma / semicolon / tab
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python", dtype=str,
                             keep_default_na=False)
            return df
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise ValueError(f"Could not read the CSV file: {last_err}")


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Guess which columns hold the question and the answer."""
    lower = {c.strip().lower(): c for c in df.columns}
    q = next((lower[n] for n in QUESTION_NAMES if n in lower), None)
    a = next((lower[n] for n in ANSWER_NAMES if n in lower and lower[n] != q), None)
    cols = list(df.columns)
    if q is None:
        q = next((c for c in cols if c != a), cols[0])
    if a is None:
        a = next((c for c in cols if c != q), cols[-1])
    return q, a


def clean_dataframe(df: pd.DataFrame, question_col: str | None = None,
                    answer_col: str | None = None) -> tuple[pd.DataFrame, DataReport]:
    """Return a clean two-column frame (question, answer) and a report of what was done."""
    rep = DataReport(rows_in_file=len(df))
    if df.shape[1] < 2:
        rep.errors.append("The CSV needs at least two columns (question and answer).")
        return pd.DataFrame(columns=["question", "answer"]), rep

    gq, ga = detect_columns(df)
    question_col = question_col or gq
    answer_col = answer_col or ga
    rep.question_col, rep.answer_col = question_col, answer_col
    for c in (question_col, answer_col):
        if c not in df.columns:
            rep.errors.append(f"Column `{c}` is not in the file.")
    if question_col == answer_col:
        rep.errors.append("Question and answer columns must be different.")
    if rep.errors:
        return pd.DataFrame(columns=["question", "answer"]), rep

    out = pd.DataFrame({
        "question": df[question_col].astype(str).str.strip(),
        "answer": df[answer_col].astype(str).str.strip(),
    })
    empty = (out["question"] == "") | (out["answer"] == "") | \
        out["question"].str.lower().isin(["nan", "none"]) | \
        out["answer"].str.lower().isin(["nan", "none"])
    rep.empty_dropped = int(empty.sum())
    out = out[~empty]
    before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    rep.duplicates_dropped = before - len(out)
    rep.rows_kept = len(out)

    if rep.rows_kept:
        qw = out["question"].str.split().str.len()
        aw = out["answer"].str.split().str.len()
        rep.avg_question_words = float(qw.mean())
        rep.avg_answer_words = float(aw.mean())
        rep.max_answer_words = int(aw.max())
        conflicting = out.groupby("question")["answer"].nunique()
        n_conf = int((conflicting > 1).sum())
        if n_conf:
            rep.warnings.append(f"{n_conf} question(s) have more than one different answer.")
        if rep.max_answer_words > 600:
            rep.warnings.append("Some answers are very long; they may be cut at the max "
                                "sequence length.")

    if rep.rows_kept < MIN_ROWS:
        rep.errors.append(f"Only {rep.rows_kept} usable rows; at least {MIN_ROWS} are needed.")
    elif rep.rows_kept < RECOMMENDED_ROWS:
        rep.warnings.append(f"{rep.rows_kept} rows is small; {RECOMMENDED_ROWS}+ rows give "
                            "more reliable results.")
    return out, rep


def split_dataframe(df: pd.DataFrame, val_ratio: float = 0.1,
                    seed: int = 3407) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle and split into train / validation. Small sets (< 10 rows) get no validation."""
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if val_ratio <= 0 or len(df) < 10:
        return df, df.iloc[0:0]
    n_val = max(1, int(round(len(df) * val_ratio)))
    n_val = min(n_val, len(df) - 1)
    return df.iloc[n_val:].reset_index(drop=True), df.iloc[:n_val].reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Chat-template formatting
# --------------------------------------------------------------------------------------

def build_messages(question: str, answer: str | None = None,
                   system_prompt: str = "") -> list[dict]:
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": question})
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


def tokenize_example(tokenizer, question: str, answer: str, system_prompt: str,
                     max_len: int) -> dict | None:
    """Tokenize one QA pair with the model's own chat template.

    The loss is computed on the answer tokens only: prompt tokens get label -100.
    This uses the same template that is used at inference time, so training and
    chatting see identical formatting.
    """
    prompt_text = tokenizer.apply_chat_template(
        build_messages(question, None, system_prompt), tokenize=False,
        add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(
        build_messages(question, answer, system_prompt), tokenize=False)

    if not full_text.startswith(prompt_text):
        # Rare: template renders the prompt differently when an answer follows.
        # Fall back to prompt + answer + eos.
        full_text = prompt_text + answer + (tokenizer.eos_token or "")

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    n_prompt = len(prompt_ids)
    if n_prompt >= max_len - 1:
        return None  # prompt alone does not fit
    full_ids = full_ids[:max_len]
    labels = [-100] * n_prompt + full_ids[n_prompt:]
    labels = labels[:len(full_ids)]
    if all(x == -100 for x in labels):
        return None
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def tokenize_dataframe(tokenizer, df: pd.DataFrame, system_prompt: str,
                       max_len: int) -> tuple[list[dict], int]:
    """Tokenize every row. Returns (examples, number_skipped)."""
    out, skipped = [], 0
    for q, a in zip(df["question"], df["answer"]):
        ex = tokenize_example(tokenizer, q, a, system_prompt, max_len)
        if ex is None:
            skipped += 1
        else:
            out.append(ex)
    return out, skipped
