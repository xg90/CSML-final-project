"""
Ministral8b — Standalone K10 Multi-Candidate Generation
========================================================

Self-contained k=10 stochastic FOL generation for the fine-tuned
Ministral-3-8B adapter (models/Ministral8b_finetuned).

Mirrors the Qwen k10 pipeline (src/run_inference.py -> run_inference_k10),
but is hardwired to Ministral so it can be run without touching the generic
pipeline.

Usage (Python)::

    from src.generate_k10_ministral import run_k10_ministral
    run_k10_ministral("test", n_samples=1000)

Usage (CLI)::

    python src/generate_k10_ministral.py --dataset test --n 1000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

import torch
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration
from peft import PeftModel

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent  # project/code/
DATA_DIR = ROOT / "data"
RESULTS_DIR = DATA_DIR / "results" / "Ministral8b" / "k10"

BASE_MODEL = "mistralai/Ministral-3-8B-Instruct-2512"
ADAPTER_PATH = ROOT / "models" / "Ministral8b_finetuned"

K = 10
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 256
TIMEOUT = 120

_SYSTEM_PROMPT = (
    "You are a helpful AI assistant that translates Natural Language (NL) text in "
    "First-Order Logic (FOL) using only the given quantors and junctors:\n"
    "∀ (for all), ∃ (there exists), ¬ (not), ∧ (and), "
    "∨ (or), → (implies), ↔ (if and only if), ⊕ (xor).\n"
    "Start your answer with '\U0001D719=' followed by the FOL-formula. "
    "Do not include any other text."
)

_DATASET_MAP = {
    "test": "test", "val": "val", "train": "train",
    "test_folio": "test_folio", "test_willow": "test_willow",
}

# ---------------------------------------------------------------------------
# Lazy model loading (cached)
# ---------------------------------------------------------------------------

_tokenizer = None
_model = None


def load_model():
    """Load Ministral base model + LoRA adapter once, then cache."""
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    if not ADAPTER_PATH.is_dir():
        raise FileNotFoundError(f"Adapter folder not found: {ADAPTER_PATH}")

    _tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_PATH), trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  VRAM: {vram_gb:.1f} GB")

    base = Mistral3ForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Ministral ships generation_config.max_length=262144; our per-call
    # max_new_tokens conflicts with it and transformers warns once per
    # generate() call (10k+ lines of noise for k=10 on 1000 samples).
    base.generation_config.max_length = None

    # Prevent peft from treating the local adapter dir as a Hub repo id.
    from huggingface_hub import HfApi
    _orig = HfApi.file_exists

    def _patched(self, repo_id, *args, **kwargs):
        if os.path.isdir(str(repo_id)):
            return False
        return _orig(self, repo_id, *args, **kwargs)

    HfApi.file_exists = _patched
    try:
        _model = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
    finally:
        HfApi.file_exists = _orig

    _model.eval()

    # Verify LoRA actually injected. The adapter targets the VLM's
    # language_model (34 layers) + vision_tower (24 layers) = 812 tensors.
    # A PEFT version older than the one that saved the adapter (0.20.0) can
    # silently inject zero layers and run the bare base model.
    n_lora = sum(1 for n, _ in _model.named_parameters() if "lora" in n)
    total = sum(p.numel() for p in _model.parameters())
    print(f"  Base params:    {total / 1e9:.2f}B")
    print(f"  LoRA layers:    {n_lora} (expected ~812)")
    if n_lora == 0:
        raise RuntimeError(
            "0 LoRA layers injected. The installed PEFT is likely older than the "
            "one that saved the adapter (needs >= 0.20.0). Run `pip install -U peft` "
            "and retry."
        )

    return _tokenizer, _model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_fol(raw: str) -> str:
    """Extract the clean FOL formula from raw model output."""
    s = raw.strip()
    if not s:
        return s

    s = re.sub(r'^[\U0001D719]\s*=\s*', "", s)

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
    s = s.replace("‘", "").replace("’", "").replace("“", "").replace("”", "")

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


def _generate_one(nl_text: str, tokenizer, model) -> str:
    """Generate ONE stochastic FOL candidate at T=0.7."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
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


def generate_k_candidates(nl_text: str, tokenizer, model) -> dict:
    """Generate k=K candidates for one NL sentence."""
    txt = nl_text.strip().rstrip(".")
    candidates = []
    n_empty = 0
    n_timeout = 0

    for i in range(K):
        try:
            raw = _run_with_timeout(_generate_one, TIMEOUT, txt, tokenizer, model)
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


def run_k10_ministral(dataset: str = "test", n_samples: Optional[int] = None) -> List[dict]:
    """Generate k=10 candidates for each sentence in a dataset."""
    ds_key = _DATASET_MAP.get(dataset, dataset)
    data_path = DATA_DIR / f"{ds_key}.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    n_total = min(len(data), n_samples or len(data))
    data = data[:n_total]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ds_key.replace("test_", "").replace("test", "")
    tag = f"Ministral8b_k10_{suffix}" if suffix else "Ministral8b_k10"
    out_path = RESULTS_DIR / f"{tag}.json"
    ckpt_path = RESULTS_DIR / f"checkpoint_{tag}.json"

    tokenizer, model = load_model()

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
    print(f"  K10 GENERATION (Ministral8b)")
    print(f"  Dataset: {ds_key}.json ({n_total} sentences)")
    print(f"  Model:   {BASE_MODEL} + LoRA")
    print(f"  K={K}  T={TEMPERATURE}  total: {n_total * K:,} calls")
    print(f"  Output:  {out_path}")
    print("=" * 55)
    print()

    for idx, item in enumerate(data):
        if idx in completed_ids:
            continue

        r = generate_k_candidates(item["NL"], tokenizer, model)
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
        description="Generate k=10 FOL candidates per NL sentence with Ministral8b"
    )
    parser.add_argument("--dataset", default="test",
                        help="Dataset name: test, val, train, test_folio, test_willow")
    parser.add_argument("--n", type=int, default=None,
                        help="Max sentences (default: all)")
    args = parser.parse_args()
    run_k10_ministral(dataset=args.dataset, n_samples=args.n)


if __name__ == "__main__":
    main()
