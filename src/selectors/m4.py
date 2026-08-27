"""
M4 helpers: FOL→NL back-translation + BERT embedding (SemCo feature).

The M4 selector itself now lives in ``feature_select.run_from_features``
(``feature_key="backtrans_sim"``, ``method="4"``).  This module only provides
the shared building blocks used to *compute* that feature, imported by
``pw_metrics.py`` (step 6) and ``prepare_scorer_data.py`` (step 6):

  * ``_load_informalizer`` / ``_informalize`` — FOL→NL back-translation via the
    fine-tuned LLM + LoRA adapter (Qwen / Ministral).
  * ``_load_bert`` / ``_bert_embed`` / ``_cosine_similarity`` — BERT embeddings
    + cosine similarity.
  * ``_load_few_shot_examples`` / ``_build_few_shot_messages`` — few-shot
    informalization prompts.

Based on: SESC (Li et al., NeurIPS 2024) — Semantic Consistency (SemCo).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# -- project root ----------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent  # project/code/

# -- constants -------------------------------------------------------------
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0  # greedy for back-translation consistency

_FEW_SHOT_PATH = _ROOT / "data" / "few_shot_examples.json"

# Default few-shot examples for FOL → NL informalization (FOL, NL pairs)
# Used as fallback when few_shot_examples.json is unavailable.
_FEW_SHOT_FOL2NL = [
    # Generic logical patterns
    (
        "∀x (Dog(x) → Animal(x))",
        "All dogs are animals."
    ),
    (
        "∃x (Cat(x) ∧ Black(x))",
        "There exists a black cat."
    ),
    (
        "∀x (Student(x) → ∃y (Book(y) ∧ Read(x, y)))",
        "Every student reads at least one book."
    ),
    (
        "¬∃x (Person(x) ∧ ∀y (Task(y) → Complete(x, y)))",
        "No person completes every task."
    ),
    (
        "∀x ∀y (Parent(x, y) → Child(y, x))",
        "If someone is a parent of another, then the latter is a child of the former."
    ),
    (
        "∃x (Bird(x) ∧ ¬Fly(x))",
        "There is a bird that does not fly."
    ),
]

# Informalization system prompt
_INFORMALIZE_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Translate the given First-Order Logic "
    "(FOL) formula back into a single clear, concise natural language "
    "sentence. Output ONLY the natural language sentence, nothing else."
)

# -- lazy singletons -------------------------------------------------------
_informalizer_cache: Dict[str, tuple] = {}  # model_key → (tokenizer, model)
_bert_model = None
_bert_tokenizer = None


def _load_few_shot_examples(dataset_key: str) -> Optional[List[Tuple[str, str]]]:
    """Load few-shot (FOL, NL) examples for a dataset from few_shot_examples.json.

    Parameters
    ----------
    dataset_key : str
        One of "malls", "folio", "willow".

    Returns
    -------
    list of (FOL, NL) tuples, or None if the file is unavailable.
    """
    if not _FEW_SHOT_PATH.exists():
        return None
    try:
        with open(_FEW_SHOT_PATH, "r", encoding="utf-8") as f:
            all_examples = json.load(f)
        pairs = all_examples.get(dataset_key, [])
        if pairs:
            return [(fol, nl) for fol, nl in pairs]
        return None
    except (json.JSONDecodeError, KeyError, OSError):
        return None


# ======================================================================
# BERT Embedding Model (for cosine similarity — SESC §3.2)
# ======================================================================

_BERT_MODEL_NAME = "bert-base-uncased"


def _load_bert():
    """Load cached BERT model + tokenizer for embedding extraction.

    SESC uses BERT [Devlin et al.] to generate embeddings, then cosine
    similarity to evaluate semantic consistency.
    """
    global _bert_model, _bert_tokenizer
    if _bert_model is None:
        _bert_tokenizer = AutoTokenizer.from_pretrained(_BERT_MODEL_NAME)
        _bert_model = AutoModel.from_pretrained(_BERT_MODEL_NAME)
        _bert_model.eval()
        # Move to GPU if available
        if torch.cuda.is_available():
            _bert_model = _bert_model.to("cuda")
    return _bert_tokenizer, _bert_model


def _bert_embed(text: str) -> np.ndarray:
    """Compute BERT [CLS] embedding for a single text string.

    Returns a 1-D numpy array (768-d for bert-base-uncased).
    """
    tokenizer, model = _load_bert()
    device = next(model.parameters()).device
    enc = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=128, padding=True,
    ).to(device)
    with torch.no_grad():
        outputs = model(**enc)
        # [CLS] token embedding (SESC approach)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_emb[0].astype(np.float64)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D arrays."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


# ======================================================================
# Informalization LLM (FOL → NL back-translation)
# ======================================================================

_MODEL_REGISTRY: Dict[str, dict] = {
    "Qwen4b": {
        "base_model": "Qwen/Qwen3-4B",
        "adapter": _ROOT / "models" / "Qwen4b_finetuned",
    },
    "Qwen8b": {
        "base_model": "Qwen/Qwen3-8B",
        "adapter": _ROOT / "models" / "Qwen8b_finetuned",
    },
    "Ministral8b": {
        "base_model": "mistralai/Ministral-3-8B-Instruct-2512",
        "adapter": _ROOT / "models" / "Ministral8b_finetuned",
    },
}


def _resolve_model_cls(base_model: str):
    """Return the Auto class that loads ``base_model``.

    Mistral3 (``model_type == "mistral3"``) must be loaded with
    ``Mistral3ForConditionalGeneration``; everything else (Qwen3, ...) keeps
    ``AutoModelForCausalLM``.
    """
    from transformers import AutoConfig
    if AutoConfig.from_pretrained(base_model).model_type == "mistral3":
        from transformers import Mistral3ForConditionalGeneration
        return Mistral3ForConditionalGeneration
    return AutoModelForCausalLM


def _load_informalizer(model_key: str = "Qwen4b"):
    """Load the informalization LLM (fine-tuned model + LoRA adapter).

    Per SESC, the same LLM is used for informalization (FOL→NL) as for
    formalization (NL→FOL).  Since our formalization model is fine-tuned,
    we load the base instruct model with the same LoRA adapter.
    Falls back to base model only if the adapter directory is missing.
    """
    if model_key in _informalizer_cache:
        return _informalizer_cache[model_key]

    cfg = _MODEL_REGISTRY.get(model_key, _MODEL_REGISTRY["Qwen4b"])
    base_model_name = cfg["base_model"]
    adapter_path = cfg.get("adapter")

    # Mistral-only quirks: the tokenizer ships a broken regex (needs the
    # ``fix_mistral_regex`` flag) and its generation_config carries
    # ``max_length=262144`` which spams a warning on every generate() call.
    # Guard both behind the model tag so Qwen loaders stay untouched.
    is_mistral = model_key == "Ministral8b"

    tokenizer_kwargs = {"trust_remote_code": True}
    if is_mistral:
        tokenizer_kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = _resolve_model_cls(base_model_name).from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Null Mistral's default max_length so max_new_tokens takes effect cleanly
    # (mirrors generate_k10_ministral.py); Qwen is unaffected.
    if is_mistral:
        model.generation_config.max_length = None

    # Load LoRA adapter (same as formalization model)
    if adapter_path and adapter_path.exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()
        print(f"    + LoRA adapter: {adapter_path}", flush=True)
    else:
        print(f"    (no LoRA adapter at {adapter_path} — using base model)", flush=True)

    model.eval()

    _informalizer_cache[model_key] = (tokenizer, model)
    return tokenizer, model


def _build_few_shot_messages(
    fol: str,
    examples: Optional[List[Tuple[str, str]]] = None,
) -> List[dict]:
    """Build chat messages with few-shot FOL→NL examples + the target FOL.

    Parameters
    ----------
    fol : str
        The FOL formula to informalize.
    examples : list of (FOL, NL) tuples, or None
        Few-shot examples to use. None falls back to hardcoded defaults.
    """
    messages = [
        {"role": "system", "content": _INFORMALIZE_SYSTEM_PROMPT},
    ]
    shot_pairs = examples if examples is not None else _FEW_SHOT_FOL2NL
    for fol_ex, nl_ex in shot_pairs:
        messages.append({"role": "user", "content": f"FOL: {fol_ex}"})
        messages.append({"role": "assistant", "content": nl_ex})
    messages.append({"role": "user", "content": f"FOL: {fol}"})
    return messages


def _informalize(
    fol: str,
    model_key: str = "Qwen4b",
    sentence_id: int = 0,
    examples: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Back-translate a single FOL formula to natural language.

    Parameters
    ----------
    fol : str
        The FOL formula to informalize.
    model_key : str
        Model key for the base instruct model.
    sentence_id : int
        Sentence index for logging.
    examples : list of (FOL, NL) tuples, or None
        Few-shot examples. None falls back to hardcoded defaults.

    Returns
    -------
    str
        The back-translated natural language sentence (empty string on failure).
    """
    if not fol or not fol.strip():
        return ""

    try:
        tokenizer, model = _load_informalizer(model_key)
        device = next(model.parameters()).device

        messages = _build_few_shot_messages(fol, examples=examples)

        # Qwen3 defaults to a <think> reasoning block; disable it so we get the
        # direct NL answer (otherwise the first line is "<think>" and the
        # back-translation is both slow and wrong).
        chat_kwargs = {}
        if model_key.startswith("Qwen"):
            chat_kwargs["enable_thinking"] = False
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_kwargs,
        )

        enc = tokenizer(
            chat_text, return_tensors="pt", truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=(TEMPERATURE > 0),
                temperature=TEMPERATURE if TEMPERATURE > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        generated = outputs[0][enc["input_ids"].shape[1]:]
        nl = tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Clean up: remove trailing EOS artifacts, newlines
        nl = nl.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        # Take only the first line / sentence
        nl = nl.split("\n")[0].strip()

        return nl

    except Exception as e:
        # On any failure, return empty string — this candidate will be
        # deprioritized in the similarity ranking.
        return ""
