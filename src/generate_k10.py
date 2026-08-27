"""
K10 Multi-Candidate Generation
==============================

Generate k=10 stochastic FOL candidates per NL sentence at T=0.7.

Usage (Python)::

    from src.generate_k10 import run_k10
    run_k10("val", model="Qwen4b")

Usage (CLI)::

    python src/generate_k10.py --dataset val --model Qwen4b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent  # project/code/
DATA_DIR = ROOT / "data"
RESULTS_BASE = DATA_DIR / "results"

K = 10
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 256
TIMEOUT = 120

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant that translates Natural Language (NL) text in "
    "First-Order Logic (FOL) using only the given quantors and junctors:\n"
    "∀ (for all), ∃ (there exists), ¬ (not), ∧ (and), "
    "∨ (or), → (implies), ↔ (if and only if), ⊕ (xor).\n"
    "Start your answer with '\U0001D719=' followed by the FOL-formula. "
    "Do not include any other text."
)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: Dict[str, dict] = {
    "Qwen4b": {
        "base_model": "Qwen/Qwen3-4B",
        "adapter_path": "models/Qwen4b_finetuned",
        "results_subdir": "Qwen4b",
    },
    "Qwen8b": {
        "base_model": "Qwen/Qwen3-8B",
        "adapter_path": "models/Qwen8b_finetuned",
        "results_subdir": "Qwen8b",
    },
    "Ministral8b": {
        "base_model": "mistralai/Ministral-3-8B-Instruct-2512",
        "adapter_path": "models/Ministral8b_finetuned",
        "results_subdir": "Ministral8b",
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


def _resolve_model(model: str) -> dict:
    """Resolve model name → config dict. Supports registered names and custom paths."""
    if model in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[model]

    # Allow custom: "base_model|adapter_path|results_subdir"
    parts = model.split("|")
    if len(parts) == 3:
        return {
            "base_model": parts[0],
            "adapter_path": parts[1],
            "results_subdir": parts[2],
        }
    raise ValueError(
        f"Unknown model '{model}'. Registered: {list(_MODEL_REGISTRY)}. "
        f"Or pass 'base_model|adapter_path|results_subdir'."
    )


# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------

_tokenizer = None
_model = None
_current_model_key = None


def _load_model(model_cfg: dict):
    """Load base model + LoRA adapter (cached)."""
    global _tokenizer, _model, _current_model_key

    key = model_cfg["base_model"] + "|" + model_cfg["adapter_path"]
    if _model is not None and _current_model_key == key:
        return _tokenizer, _model

    adapter_full = ROOT / model_cfg["adapter_path"]

    _tokenizer = AutoTokenizer.from_pretrained(str(adapter_full), trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    # Suppress HF_TOKEN warning
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  VRAM: {vram_gb:.1f} GB")

    if vram_gb >= 22:
        print("  → bf16 full precision")
        base = _resolve_model_cls(model_cfg["base_model"]).from_pretrained(
            model_cfg["base_model"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        print("  → 4-bit NF4 quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        base = _resolve_model_cls(model_cfg["base_model"]).from_pretrained(
            model_cfg["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    # Patch HfApi.file_exists for local adapter paths
    from huggingface_hub import HfApi
    _orig = HfApi.file_exists

    def _patched(self, repo_id, *args, **kwargs):
        if os.path.isdir(str(repo_id)):
            return False
        return _orig(self, repo_id, *args, **kwargs)

    HfApi.file_exists = _patched
    try:
        _model = PeftModel.from_pretrained(base, str(adapter_full))
    finally:
        HfApi.file_exists = _orig

    _model.eval()

    trainable = sum(p.numel() for p in _model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in _model.parameters())
    print(f"  Base params:    {total / 1e9:.2f}B")
    print(f"  LoRA trainable: {trainable / 1e6:.1f}M ({100 * trainable / total:.2f}%)")

    _current_model_key = key
    return _tokenizer, _model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_fol(raw: str) -> str:
    """Extract clean FOL formula from model output."""
    import re
    s = raw.strip()
    if not s:
        return s

    s = re.sub(r'^[\U0001D719]\s*=\s*', "", s)

    # Fenced code block
    m = re.search(
        r"```(?:fol|logic|text)?\s*\n?(.*?)\n?```", s, re.DOTALL | re.IGNORECASE
    )
    if m:
        s = m.group(1).strip()

    for prefix in [
        "Output:", "FOL:", "Formula:", "Translation:",
        "output:", "fol:", "formula:", "translation:",
    ]:
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].strip()

    s = re.sub(r"[;]\s*$", "", s)
    s = re.sub(r",\s*\)", ")", s)
    s = re.sub(r"\.\.+", ".", s)
    s = s.replace("'", "").replace("'", "").replace("“", "").replace("”", "")

    lines = s.split("\n")
    for line in lines:
        line = line.strip()
        if line and any(c in line for c in ["∀", "∃", "→", "∧", "∨", "¬", "↔", "(", ")"]):
            if not line.lower().startswith(
                ("the", "this", "here", "i ", "we ", "note", "let")
            ):
                return line
    return s.strip()


class TimeoutError(Exception):
    pass


def _run_with_timeout(fn, sec, *args, **kwargs):
    result = [None]
    exc = [None]

    def target():
        try:
            result[0] = fn(*args, **kwargs)
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(sec)
    if t.is_alive():
        raise TimeoutError(f"Timeout({sec}s)")
    if exc[0]:
        raise exc[0]
    return result[0]


# ---------------------------------------------------------------------------
# Core: single-candidate generation
# ---------------------------------------------------------------------------


def _generate_one(nl_text: str, tokenizer, model, system_prompt: str) -> str:
    """Generate ONE stochastic FOL candidate at T=0.7."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": nl_text.strip()},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][input_len:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return _clean_fol(response)


# ---------------------------------------------------------------------------
# Core: k-candidate generation for one sentence
# ---------------------------------------------------------------------------


def generate_k_candidates(
    nl_text: str,
    tokenizer,
    model,
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    k: int = K,
) -> dict:
    """Generate k FOL candidates for one NL sentence.

    Returns dict with keys:
        nl, gt_fol (None here), candidates, n_success, n_empty, n_timeout, unique_count
    """
    txt = nl_text.strip().rstrip(".")
    candidates = []
    n_empty = 0
    n_timeout = 0

    for i in range(k):
        try:
            raw = _run_with_timeout(_generate_one, TIMEOUT, txt, tokenizer, model, system_prompt)
            fol = _clean_fol(raw)
            cand = {
                "candidate_id": i,
                "fol": fol if fol else None,
                "fol_hash": hashlib.md5(fol.encode()).hexdigest()[:16] if fol else None,
                "raw_output": raw,
                "timeout": False,
            }
            if not fol:
                n_empty += 1
            candidates.append(cand)
        except TimeoutError:
            candidates.append({
                "candidate_id": i, "fol": None, "fol_hash": None,
                "raw_output": None, "timeout": True,
            })
            n_timeout += 1
        except Exception as e:
            candidates.append({
                "candidate_id": i, "fol": None, "fol_hash": None,
                "raw_output": str(e)[:200], "timeout": False,
            })

    valid_hashes = [c["fol_hash"] for c in candidates if c["fol"]]
    return {
        "nl": txt,
        "gt_fol": None,
        "candidates": candidates,
        "n_success": len(valid_hashes),
        "n_empty": n_empty,
        "n_timeout": n_timeout,
        "unique_count": len(set(valid_hashes)),
    }


# ---------------------------------------------------------------------------
# Batch generation with checkpointing
# ---------------------------------------------------------------------------


def run_k10(
    dataset: str = "val",
    model: str = "Qwen4b",
    n_samples: Optional[int] = None,
    system_prompt: Optional[str] = None,
) -> List[dict]:
    """Generate k=10 candidates for each sentence in a dataset.

    Parameters
    ----------
    dataset : str
        Dataset name: "val", "test", "train", "test_folio", "test_willow".
        Can also pass a path to a JSON file.
    model : str
        Model key from registry ("Qwen4b", "Qwen8b", "Ministral8b")
        or custom "base_model|adapter|subdir".
    n_samples : int, optional
        Cap on number of sentences. Default: all.
    system_prompt : str, optional
        Custom system prompt.

    Returns
    -------
    List[dict] — one entry per sentence, each with `candidates` list.
    """
    model_cfg = _resolve_model(model)
    sys_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
    subdir = model_cfg["results_subdir"]

    # --- resolve dataset ---
    ds_map = {"test": "test", "val": "val", "train": "train",
              "test_folio": "test_folio", "test_willow": "test_willow"}
    ds_key = ds_map.get(dataset, dataset)
    data_path = DATA_DIR / f"{ds_key}.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    n_total = min(len(data), n_samples or len(data))
    data = data[:n_total]

    # --- output paths ---
    # k10 output goes to data/results/{subdir}/k10/{model}_k10_{ds_key}.json
    out_dir: Path = RESULTS_BASE / subdir / "k10"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = ds_key.replace("test_", "").replace("test", "")
    tag = f"{subdir}_k10_{suffix}" if suffix else f"{subdir}_k10"
    out_path = out_dir / f"{tag}.json"
    ckpt_path = out_dir / f"checkpoint_{tag}.json"

    # --- load model ---
    tokenizer, gen_model = _load_model(model_cfg)

    # --- resume from checkpoint ---
    completed_ids: Set[int] = set()
    all_results: List[dict] = []

    if ckpt_path.exists():
        with open(ckpt_path) as f:
            saved = json.load(f)
            all_results = saved.get("results", [])
            completed_ids = {r["sentence_id"] for r in all_results}
        print(f"  Resuming from checkpoint: {len(completed_ids)} already done\n")

    total_cand = len(completed_ids) * K
    total_ok = sum(r.get("n_success", 0) for r in all_results)
    total_unique = sum(r.get("unique_count", 0) for r in all_results)
    t0 = time.time()

    print("=" * 55)
    print(f"  K10 GENERATION")
    print(f"  Dataset: {ds_key}.json ({n_total} sentences)")
    print(f"  Model:   {model_cfg['base_model']} + LoRA")
    print(f"  K={K}  T={TEMPERATURE}  total: {n_total * K:,} calls")
    print(f"  Output:  {out_path}")
    print("=" * 55)
    print()

    for idx, item in enumerate(data):
        if idx in completed_ids:
            continue

        r = generate_k_candidates(item["NL"], tokenizer, gen_model, sys_prompt, k=K)
        r["sentence_id"] = idx
        r["gt_fol"] = item["FOL"]
        all_results.append(r)

        total_cand += K
        total_ok += r["n_success"]
        total_unique += r["unique_count"]

        done = len(all_results)
        if done % 10 == 0 or done == n_total:
            elapsed = time.time() - t0
            new_done = done - len(completed_ids)
            avg = elapsed / max(new_done, 1)
            rem = avg * (n_total - done)
            print(
                f"  [{done:>4}/{n_total}]  "
                f"success={100 * total_ok / max(total_cand, 1):.1f}%  "
                f"unique_ratio={total_unique / max(total_ok, 1):.2f}  "
                f"elapsed={elapsed / 60:.1f}m  ETA={rem / 60:.1f}m"
            )
            with open(ckpt_path, "w") as f:
                json.dump({
                    "results": all_results,
                    "total_candidates": total_cand,
                    "total_success": total_ok,
                    "total_unique": total_unique,
                    "total_done": done,
                    "last_update": datetime.now().isoformat(),
                }, f, indent=2, ensure_ascii=False)

        if done >= n_total:
            break

    elapsed = (time.time() - t0) / 60

    # --- save ---
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\n  === Done: {elapsed:.1f} min ===")
    print(f"  Candidates: {total_cand} | Success: {total_ok} | Unique: {total_unique}")
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")

    if len(all_results) >= n_total and ckpt_path.exists():
        os.remove(ckpt_path)
        print("  Removed checkpoint.")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate k=10 FOL candidates per NL sentence at T=0.7"
    )
    parser.add_argument("--dataset", default="val",
                        help="Dataset name: val, test, train, test_folio, test_willow")
    parser.add_argument("--model", default="Qwen4b",
                        help="Model key (Qwen4b, Qwen8b, Ministral8b) or custom")
    parser.add_argument("--n", type=int, default=None,
                        help="Max sentences (default: all)")
    args = parser.parse_args()

    run_k10(dataset=args.dataset, model=args.model, n_samples=args.n)


if __name__ == "__main__":
    main()
