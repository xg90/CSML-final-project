"""
Ministral8b — Standalone T=0 (Greedy) Generation
=================================================

Self-contained single-candidate FOL generation (greedy decoding, T=0) for the
fine-tuned Ministral-3-8B adapter (models/Ministral8b_finetuned).

Mirrors the Qwen T=0 pipeline (src/run_inference.py -> run_inference), but is
hardwired to Ministral so it can be run without touching the generic pipeline.
Outputs the same flat result format as run_inference.py, so the results are
directly compatible with the existing evaluation scripts.

Usage (Python)::

    from src.generate_t0_ministral import run_t0_ministral
    run_t0_ministral("test", n_samples=1000)

Usage (CLI)::

    python src/generate_t0_ministral.py --dataset test --n 1000
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
RESULTS_DIR = DATA_DIR / "results" / "Ministral8b"

BASE_MODEL = "mistralai/Ministral-3-8B-Instruct-2512"
ADAPTER_PATH = ROOT / "models" / "Ministral8b_finetuned"

TEMPERATURE = 0.0
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
    # generate() call. Clear the default here.
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
# Core: single-candidate generation (T=0 greedy)
# ---------------------------------------------------------------------------


def _generate_one(nl_text: str, tokenizer, model) -> str:
    """Generate ONE deterministic FOL candidate with greedy decoding (T=0)."""
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
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][input_len:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return _clean_fol(response)


def translate_one(nl_text: str, tokenizer, model, sentence_id: int = 0) -> dict:
    """Generate one FOL formula for one NL sentence (T=0)."""
    result = {
        "sentence_id": sentence_id,
        "generation_method": "Ministral8b_T=0.0",
        "nl": nl_text.strip().rstrip("."),
        "fol": None,
        "fol_hash": None,
        "success": False,
        "error": None,
        "raw_output": None,
    }
    txt = nl_text.strip().rstrip(".")
    if not txt:
        result["error"] = "Empty sentence"
        return result
    try:
        raw = _run_with_timeout(_generate_one, TIMEOUT, txt, tokenizer, model)
        result["raw_output"] = raw
        fol = _clean_fol(raw)
        if fol:
            result["fol"] = fol
            result["fol_hash"] = hashlib.md5(fol.encode()).hexdigest()[:16]
            result["success"] = True
        else:
            result["error"] = "Empty FOL output"
        return result
    except TimeoutError:
        result["error"] = "Timeout"
        return result
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        return result


# ---------------------------------------------------------------------------
# Batch generation with checkpointing
# ---------------------------------------------------------------------------


def run_t0_ministral(dataset: str = "test", n_samples: Optional[int] = None) -> List[dict]:
    """Generate one greedy FOL candidate per sentence in a dataset (T=0)."""
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
    tag = f"Ministral8b_{suffix}" if suffix else "Ministral8b"
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

    n_success = sum(1 for r in all_results if r.get("success"))
    n_timeout = sum(1 for r in all_results if r.get("error") == "Timeout")
    n_error = sum(1 for r in all_results if r.get("error") and r.get("error") != "Timeout")
    t0 = time.time()

    print("=" * 55)
    print(f"  T0 GENERATION (Ministral8b)")
    print(f"  Dataset: {ds_key}.json ({n_total} sentences)")
    print(f"  Model:   {BASE_MODEL} + LoRA")
    print(f"  T={TEMPERATURE}  |  max_new_tokens={MAX_NEW_TOKENS}")
    print(f"  Output:  {out_path}")
    print("=" * 55)
    print()

    for idx, item in enumerate(data):
        if idx in completed_ids:
            continue

        r = translate_one(item["NL"], tokenizer, model, sentence_id=idx)
        r["gt_fol"] = item["FOL"]
        all_results.append(r)

        if r.get("success"):
            n_success += 1
        if r.get("error") == "Timeout":
            n_timeout += 1
        elif r.get("error"):
            n_error += 1

        done = len(all_results)
        if done % 100 == 0 or done == n_total:
            elapsed = time.time() - t0
            new_done = done - len(completed_ids)
            avg = elapsed / max(new_done, 1)
            rem = avg * (n_total - done)
            print(
                f"  [{done:>4}/{n_total}]  "
                f"success={100 * n_success / done:.1f}%({n_success}/{done})  "
                f"timeout={n_timeout}  error={n_error}  "
                f"elapsed={elapsed / 60:.1f}m  ETA={rem / 60:.1f}m"
            )
            with open(ckpt_path, "w") as f:
                json.dump({
                    "results": all_results,
                    "n_success": n_success,
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
    print(f"  Success: {n_success}/{n_total} ({100 * n_success / max(n_total, 1):.1f}%)")
    print(f"  Timeout: {n_timeout} | Error: {n_error}")
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
        description="Generate one greedy (T=0) FOL candidate per NL sentence with Ministral8b"
    )
    parser.add_argument("--dataset", default="test",
                        help="Dataset name: test, val, train, test_folio, test_willow")
    parser.add_argument("--n", type=int, default=None,
                        help="Max sentences (default: all)")
    args = parser.parse_args()
    run_t0_ministral(dataset=args.dataset, n_samples=args.n)


if __name__ == "__main__":
    main()
