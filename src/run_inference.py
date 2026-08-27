"""
NL-to-FOL Inference & Evaluation Pipeline
==========================================

Parameterized pipeline for running fine-tuned LLM inference followed by
multi-metric evaluation on any NL-FOL dataset (test.json, test_folio.json,
test_willow.json, etc.).

Supports different fine-tuned models via ModelConfig.

Usage (Python):
    from src.run_inference import run_inference
    run_inference("test_folio")

Usage with custom model:
    from src.run_inference import run_inference, ModelConfig
    config = ModelConfig(
        base_model="mistralai/Ministral-3-8B-Instruct-2512",
        adapter_path="models/Ministral8b_finetuned",
        results_subdir="Ministral8b",
    )
    run_inference("test", model_config=config)
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
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
# Constants (immutable, shared across all models)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent          # project/code/
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0
TIMEOUT = 120
CHUNK_SIZE = 50
Z3_TIMEOUT_MS = 10_000
MAX_ITEMS = 1000

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant that translates Natural Language (NL) text in "
    "First-Order Logic (FOL) using only the given quantors and junctors:\n"
    "∀ (for all), ∃ (there exists), ¬ (not), ∧ (and), "
    "∨ (or), → (implies), ↔ (if and only if), ⊕ (xor).\n"
    "Start your answer with '\U0001D719=' followed by the FOL-formula. "
    "Do not include any other text."
)

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Configuration for a fine-tuned NL-to-FOL model."""
    base_model: str = "Qwen/Qwen3-4B"
    adapter_path: str = "models/Qwen4b_finetuned"
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    results_subdir: str = "Qwen4b"

    @property
    def adapter_full_path(self) -> Path:
        return ROOT / self.adapter_path

    @property
    def results_dir(self) -> Path:
        d = DATA_DIR / "results" / self.results_subdir
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def results_k10_dir(self) -> Path:
        d = DATA_DIR / "results" / self.results_subdir / "k10"
        d.mkdir(parents=True, exist_ok=True)
        return d


# Pre-built configs
_MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "Qwen4b": ModelConfig(),
    "Qwen8b": ModelConfig(
        base_model="Qwen/Qwen3-8B",
        adapter_path="models/Qwen8b_finetuned",
        results_subdir="Qwen8b",
    ),
    "Ministral8b": ModelConfig(
        base_model="mistralai/Ministral-3-8B-Instruct-2512",
        adapter_path="models/Ministral8b_finetuned",
        results_subdir="Ministral8b",
    ),
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


# ---------------------------------------------------------------------------
# Globals (lazy-loaded, invalidated on config change)
# ---------------------------------------------------------------------------

_tokenizer = None
_model = None
_current_config = None


def _get_tokenizer(config: ModelConfig):
    global _tokenizer, _current_config
    if _tokenizer is None or _current_config is not config:
        _tokenizer = AutoTokenizer.from_pretrained(
            str(config.adapter_full_path), trust_remote_code=True
        )
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _current_config = config
    return _tokenizer


def _get_model(config: ModelConfig):
    global _model, _current_config
    if _model is None or _current_config is not config:
        os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')  # suppress HF_TOKEN warning
        tokenizer = _get_tokenizer(config)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {vram_gb:.1f} GB")

        if vram_gb >= 22:
            print("  -> using bf16 full precision")
            base = _resolve_model_cls(config.base_model).from_pretrained(
                config.base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            print("  -> using 4-bit NF4 quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            base = _resolve_model_cls(config.base_model).from_pretrained(
                config.base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

        # Monkey-patch HfApi.file_exists to prevent peft from treating local
        # paths as HuggingFace Hub repo IDs (returns 404 on newer peft/hub).
        from huggingface_hub import HfApi
        _orig_file_exists = HfApi.file_exists

        def _local_file_exists(self, repo_id, *args, **kwargs):
            if os.path.isdir(str(repo_id)):
                return False
            return _orig_file_exists(self, repo_id, *args, **kwargs)

        HfApi.file_exists = _local_file_exists
        try:
            _model = PeftModel.from_pretrained(base, str(config.adapter_full_path))
        finally:
            HfApi.file_exists = _orig_file_exists
        _model.eval()

        trainable = sum(p.numel() for p in _model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in _model.parameters())
        print(f"  Base params:    {total/1e9:.2f}B")
        print(f"  LoRA trainable: {trainable/1e6:.1f}M ({100*trainable/total:.2f}%)")
        _current_config = config
    return _model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def clean_fol_output(raw: str) -> str:
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


def _generate_fol(nl_text: str, config: ModelConfig) -> str:
    tokenizer = _get_tokenizer(config)
    model = _get_model(config)
    messages = [
        {"role": "system", "content": config.system_prompt},
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
            do_sample=(TEMPERATURE > 0),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][input_len:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return clean_fol_output(response)


def translate_one(nl_text: str, config: ModelConfig, sentence_id: int = 0) -> dict:
    result = {
        "sentence_id": sentence_id,
        "generation_method": f"{config.results_subdir}_T=0.0",
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
        raw = _run_with_timeout(_generate_fol, TIMEOUT, txt, config)
        result["raw_output"] = raw
        fol = clean_fol_output(raw)
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
# Inference loop
# ---------------------------------------------------------------------------


def _run_generation(dataset_name: str, n_samples: int, model_tag: str, config: ModelConfig) -> List[dict]:
    """Generate FOL for each entry in the dataset. Returns list of results."""
    data_path = DATA_DIR / f"{dataset_name}.json"
    out_path = config.results_dir / f"{model_tag}.json"
    ckpt_path = config.results_dir / f"checkpoint_{model_tag}.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    with open(data_path, encoding="utf-8") as f:
        test_data = json.load(f)

    n_total = min(len(test_data), n_samples)
    test_data = test_data[:n_total]

    completed_idx: Set[int] = set()
    all_results: List[dict] = []

    if ckpt_path.exists():
        with open(ckpt_path) as f:
            saved = json.load(f)
            all_results = saved.get("results", [])
            completed_idx = {r["sentence_id"] for r in all_results}
        print(f"  Resuming from checkpoint: {len(completed_idx)} already done")

    n_success = sum(1 for r in all_results if r.get("success"))
    n_timeout = sum(1 for r in all_results if r.get("error") == "Timeout")
    n_error = sum(
        1 for r in all_results
        if r.get("error") and r.get("error") != "Timeout"
    )
    t0 = time.time()

    print(f"  Model:    {config.base_model} + LoRA")
    print(f"  Data:     {data_path} ({n_total} entries)")
    print(f"  Output:   {out_path}")
    print(f"  T=0.0  |  max_tokens={MAX_NEW_TOKENS}")
    print()

    for idx, item in enumerate(test_data):
        if idx in completed_idx:
            continue
        r = translate_one(item["NL"], config, sentence_id=idx)
        r["gt_fol"] = item["FOL"]
        if r.get("fol"):
            r["fol_hash"] = hashlib.md5(r["fol"].encode()).hexdigest()[:16]
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
            new = done - len(completed_idx)
            avg = elapsed / max(new, 1)
            rem = avg * (n_total - done)
            sr = n_success / done if done > 0 else 0
            print(
                f"  [{done:>4}/{n_total}]  "
                f"success={sr:.3f}({n_success}/{done})  "
                f"timeout={n_timeout}  error={n_error}  "
                f"elapsed={elapsed/60:.1f}m  ETA={rem/60:.1f}m"
            )
            with open(ckpt_path, "w") as f:
                json.dump(
                    {
                        "results": all_results,
                        "n_success": n_success,
                        "total_done": done,
                        "last_update": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

    total_min = (time.time() - t0) / 60
    print(f"\n  === Generation Complete ===")
    print(f"  Total time: {total_min:.1f} min")
    print(f"  Success: {n_success}/{len(all_results)} "
          f"({100*n_success/max(len(all_results),1):.1f}%)")

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    file_size = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Saved {len(all_results)} results ({file_size:.1f} MB)")

    if len(all_results) >= n_total and ckpt_path.exists():
        os.remove(ckpt_path)
        print("  Removed checkpoint (all done)")

    return all_results


# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------


def _print_summary(all_results: List[dict], model_tag: str):
    print()
    print("=" * 60)
    print(f"  {model_tag} — RESULTS SUMMARY")
    print("=" * 60)

    n_total = len(all_results)
    n_success = sum(1 for r in all_results if r.get("success"))
    n_timeout = sum(1 for r in all_results if r.get("error") == "Timeout")
    n_empty = sum(1 for r in all_results if r.get("error") == "Empty FOL output")
    n_other = n_total - n_success - n_timeout - n_empty

    print(f"\n  Total sentences:      {n_total}")
    print(f"  Success:              {n_success} ({100*n_success/n_total:.1f}%)")
    print(f"  --- Errors ---")
    print(f"  Timeout:              {n_timeout}")
    print(f"  Empty FOL output:     {n_empty}")
    print(f"  Other:                {n_other}")

    all_hashes = [r.get("fol_hash") for r in all_results if r.get("fol_hash")]
    unique_hashes = len(set(all_hashes))
    print(f"\n  Unique FOL hashes:    {unique_hashes}/{n_total}")
    if n_total > 0:
        print(f"  Hash diversity:       {100*unique_hashes/n_total:.1f}%")

    fol_lens = [len(r.get("fol", "")) for r in all_results if r.get("fol")]
    if fol_lens:
        print(
            f"\n  FOL length — min: {min(fol_lens)}  max: {max(fol_lens)}  "
            f"mean: {sum(fol_lens)/len(fol_lens):.0f}  median: {sorted(fol_lens)[len(fol_lens)//2]}"
        )

    errs = [r.get("error") for r in all_results if r.get("error")]
    if errs:
        print(f"\n  Error types ({len(errs)}):")
        for e, c in Counter(errs).most_common(8):
            print(f"    {str(e)[:100]}: {c}")

    ok = [r for r in all_results if r.get("success")]
    n_show = min(3, len(ok))
    print(f"\n  === Sample Translations ({n_show} of {len(ok)} successful) ===")
    for r in ok[:n_show]:
        print(f"  [{r['sentence_id']}] NL:  {r['nl'][:80]}")
        print(f"       GEN: {r.get('fol', '')[:120]}")
        print(f"       GT:  {r.get('gt_fol', '')[:120]}")
        print()

    failed = [r for r in all_results if not r.get("success")]
    if failed:
        n_show_f = min(3, len(failed))
        print(f"  === Sample Failures ({n_show_f} of {len(failed)}) ===")
        for r in failed[:n_show_f]:
            print(f"  [{r['sentence_id']}] NL:  {r['nl'][:80]}")
            print(f"       Error: {r.get('error', 'unknown')}")
            print(f"       Raw:   {str(r.get('raw_output', ''))[:100]}")
            print()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _run_evaluation(dataset_name: str, model_tag: str, config: ModelConfig) -> List[dict]:
    """Run 7-metric evaluation on generated results."""
    gen_path = config.results_dir / f"{model_tag}.json"
    gt_path = DATA_DIR / f"{dataset_name}.json"
    eval_path = config.results_dir / f"{model_tag}_metrics.json"

    if not gen_path.exists():
        raise FileNotFoundError(f"Generated results not found: {gen_path}")

    sys.path.insert(0, str(ROOT))
    from src.eval.eval import compute_all_metrics

    with open(gen_path) as f:
        gen_data = json.load(f)
    with open(gt_path) as f:
        gt_raw = json.load(f)

    gt_map = {i: item["FOL"] for i, item in enumerate(gt_raw)}
    total = len(gen_data)

    already_done = 0
    if eval_path.exists():
        with open(eval_path) as f:
            prev = json.load(f)
            # only count entries that have ALL required metrics (handles schema upgrades)
            already_done = sum(
                1 for e in prev
                if "metrics" in e and all(k in e["metrics"] for k in METRIC_KEYS)
            )
        if already_done > 0:
            gen_data = prev
            print(f"  Resuming eval: {already_done}/{total} already done")
        else:
            print(f"  Old metrics file detected (missing keys) — recomputing from scratch")

    stop_at = min(total, already_done + MAX_ITEMS)

    print(f"  Input:   {gen_path} ({total} samples)")
    print(f"  Output:  {eval_path}")
    print(f"  Range:   {already_done} -> {stop_at} ({stop_at - already_done} items)")
    print(f"  Config:  chunk={CHUNK_SIZE}  timeout={Z3_TIMEOUT_MS}ms")
    print()

    if already_done >= stop_at:
        print("  Already complete - nothing to do.")
        return gen_data

    t0 = time.perf_counter()
    last_print = 0
    for i in range(already_done, stop_at):
        pred = gen_data[i].get("fol", "") or ""
        gt = gt_map[gen_data[i]["sentence_id"]]
        scores = compute_all_metrics(pred, gt, z3_timeout_ms=Z3_TIMEOUT_MS)
        gen_data[i]["metrics"] = scores
        gc.collect(2)  # fast collection after every entry

        # Full GC every 50 entries to reclaim NLTK leak (gen-2 objects)
        if (i + 1) % 50 == 0:
            gc.collect()

        done = i + 1 - already_done
        if done % 100 == 0 or i == stop_at - 1:
            with open(eval_path, "w") as f:
                json.dump(gen_data, f, indent=2, ensure_ascii=False)

            elapsed = time.perf_counter() - t0
            total_this = stop_at - already_done
            avg = elapsed / done
            rem = avg * (total_this - done)
            chunk_end = i + 1
            exec_ok = sum(
                1 for e in gen_data[:chunk_end]
                if e.get("metrics", {}).get("execution_rate") == 1.0
            )
            print(
                f"  [{chunk_end:>4}/{stop_at}]  "
                f"exec_rate={100*exec_ok/chunk_end:.1f}%  "
                f"elapsed={elapsed/60:.1f}m  ETA={rem/60:.1f}m"
            )
            last_print = chunk_end

    elapsed = (time.perf_counter() - t0) / 60
    print(f"\n  Done in {elapsed:.1f} min — saved to {eval_path}")
    if stop_at < total:
        print(f"  WARNING: {stop_at}/{total} done. Re-run to continue.")

    return gen_data


def _print_metrics(model_tag: str, config: ModelConfig):
    """Backward-compatible wrapper for internal callers."""
    suffix = model_tag.replace(config.results_subdir, "").lstrip("_")
    show_metrics(model=config.results_subdir, dataset_name=f"test_{suffix}" if suffix else "test")


# ---------------------------------------------------------------------------
# Public API — Metrics Display (callable from any notebook)
# ---------------------------------------------------------------------------


def show_metrics(
    *,
    model: str = "Qwen4b",
    dataset_name: str = "test",
) -> dict:
    """
    Read an existing _metrics.json file and print the aggregate results table.
    Use this from analysis/analysis_k10.ipynb or any notebook WITHOUT re-running evaluation.

    Usage:
        show_metrics()                              # Qwen4b on test
        show_metrics(model="Qwen8b")                # Qwen8b on test
        show_metrics(dataset_name="test_folio")      # Qwen4b on test_folio
        show_metrics(model="Qwen8b", dataset_name="test_willow")

    Parameters
    ----------
    model : str
        Model key in MODEL_REGISTRY ("Qwen4b", "Qwen8b", "Ministral8b").
    dataset_name : str
        Dataset file name without .json extension.

    Returns
    -------
    dict with keys: n_valid, n_total, metrics, execution_rate, predicate_gains
    """
    config = _MODEL_REGISTRY.get(model, ModelConfig())

    suffix = dataset_name.replace("test_", "").replace("test", "")
    model_tag = f"{config.results_subdir}_{suffix}" if suffix else config.results_subdir

    eval_path = config.results_dir / f"{model_tag}_metrics.json"
    if not eval_path.exists():
        print(f"  No metrics file found: {eval_path}")
        return {}

    with open(eval_path) as f:
        gen_data = json.load(f)

    metric_keys = [
        "execution_rate",
        "exact_match", "exact_match_pred_norm",
        "z3_le", "z3_le_pred_norm",
        "bleu", "bertscore",
    ]

    valid = [e for e in gen_data if "metrics" in e]
    print(f"\n  Valid entries: {len(valid)}/{len(gen_data)}")
    print()
    print(f"  {'Metric':<30s} {'Mean':>8s}")
    print("  " + "-" * 36)
    metric_summary = []
    for key in metric_keys:
        vals = [e["metrics"].get(key, 0.0) for e in valid]
        mean_v = float(np.mean(vals))
        metric_summary.append({"metric": key, "mean": mean_v})
        print(f"  {key:<28s} {mean_v:>8.4f}")

    n_gain_le = sum(
        1 for e in valid
        if e["metrics"].get("z3_le", 0.0) == 0.0 and e["metrics"].get("z3_le_pred_norm", 0.0) == 1.0
    )
    n_gain_em = sum(
        1 for e in valid
        if e["metrics"]["exact_match"] == 0.0
        and e["metrics"]["exact_match_pred_norm"] == 1.0
    )
    n_exec_ok = sum(1 for e in valid if e["metrics"].get("execution_rate", 0.0) == 1.0)
    exec_rate = 100 * n_exec_ok / len(valid) if valid else 0.0

    print(f"\n  Z3 Execution Rate: {n_exec_ok}/{len(valid)} "
          f"({exec_rate:.1f}%)")
    print(f"\n  Predicate-alignment gains (over {len(valid)}):")
    print(f"    LE z3:   +{n_gain_le} ({100*n_gain_le/len(valid):.1f}%)")
    print(f"    EM:      +{n_gain_em} ({100*n_gain_em/len(valid):.1f}%)")

    return {
        "n_valid": len(valid),
        "n_total": len(gen_data),
        "metrics": metric_summary,
        "execution_rate": exec_rate,
        "predicate_gains": {
            "le_z3": n_gain_le,
            "em": n_gain_em,
            "le_z3_pct": round(100 * n_gain_le / len(valid), 1) if valid else 0.0,
            "em_pct": round(100 * n_gain_em / len(valid), 1) if valid else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Public API — Inference & Evaluation
# ---------------------------------------------------------------------------


def run_inference(
    dataset_name: str = "test",
    model_tag: str | None = None,
    n_samples: int = 1000,
    skip_eval: bool = True,
    model: str = "Qwen4b",
) -> None:
    """
    Run inference (FOL generation) on a given dataset. Metrics NOT computed by default.
    Use run_eval_only() to compute metrics separately.

    Parameters
    ----------
    dataset_name : str
        Name of the JSON file in data/ (without .json extension).
    model_tag : str or None
        Tag for output file names. Defaults to "{model}" or "{model}_folio".
    n_samples : int
        Max number of entries to process.
    skip_eval : bool
        If True, skip the evaluation step.
    """
    config = _MODEL_REGISTRY.get(model, ModelConfig())

    if model_tag is None:
        suffix = dataset_name.replace("test_", "").replace("test", "")
        model_tag = f"{config.results_subdir}_{suffix}" if suffix else config.results_subdir

    print("=" * 55)
    print(f"  INFERENCE:  {dataset_name}.json")
    print(f"  MODEL TAG:  {model_tag}")
    print("=" * 55)

    # Step 1: Load model
    print("\n[1] Loading model ...")
    _get_model(config)
    print("  Model ready.")

    # Step 2: Generate
    print(f"\n[2] Running inference on {dataset_name}.json ...")
    all_results = _run_generation(dataset_name, n_samples, model_tag, config)

    # Step 3: Summary
    print(f"\n[3] Results summary ...")
    _print_summary(all_results, model_tag)

    # Step 4: Evaluate (off by default)
    if skip_eval:
        print(f"\n  [SKIP] Metrics not computed. Run eval notebook to compute.")
    else:
        print(f"\n[4] Running evaluation ...")
        _run_evaluation(dataset_name, model_tag, config)
        _print_metrics(model_tag, config)

    print(f"\n  Done. Output in {config.results_dir}/")


def run_eval_only(
    dataset_name: str = "test",
    model_tag: str | None = None,
    model: str = "Qwen4b",
) -> None:
    """Compute metrics for already-generated FOL results. Use after run_inference()."""
    config = _MODEL_REGISTRY.get(model, ModelConfig())

    if model_tag is None:
        suffix = dataset_name.replace("test_", "").replace("test", "")
        model_tag = f"{config.results_subdir}_{suffix}" if suffix else config.results_subdir

    gen_path = config.results_dir / f"{model_tag}.json"
    if not gen_path.exists():
        raise FileNotFoundError(
            f"{gen_path} not found — run inference notebook first"
        )

    print("=" * 55)
    print(f"  EVAL ONLY:  {model_tag}")
    print("=" * 55)

    print(f"\n[1] Running evaluation on {model_tag}.json ...")
    _run_evaluation(dataset_name, model_tag, config)
    _print_metrics(model_tag, config)
    print(f"\n  Done. Output in {config.results_dir}/")


# ---------------------------------------------------------------------------
# k10 Multi-Candidate Pipeline (T=0.7, K=10, pass@k evaluation)
# ---------------------------------------------------------------------------

K_CANDIDATES = 10
TEMP_K10 = 0.7


def _generate_single_k10(nl_text: str, config: ModelConfig) -> str:
    """Generate ONE stochastic FOL candidate at T=0.7."""
    tokenizer = _get_tokenizer(config)
    model = _get_model(config)
    messages = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": nl_text.strip()},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMP_K10,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][input_len:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return clean_fol_output(response)


def generate_k_candidates(nl_text: str, config: ModelConfig, sentence_id: int = 0) -> dict:
    """Generate k=K_CANDIDATES FOL candidates for one NL sentence."""
    result = {
        "sentence_id": sentence_id,
        "nl": nl_text.strip().rstrip("."),
        "gt_fol": None,
        "generation_method": f"{config.results_subdir}_k10_T={TEMP_K10}",
        "candidates": [],
        "n_success": 0,
        "n_empty": 0,
        "n_timeout": 0,
        "unique_count": 0,
        "error": None,
    }
    txt = nl_text.strip().rstrip(".")
    if not txt:
        result["error"] = "Empty sentence"
        return result

    candidates = []
    for i in range(K_CANDIDATES):
        try:
            raw = _run_with_timeout(_generate_single_k10, TIMEOUT, txt, config)
            fol = clean_fol_output(raw)
            cand = {
                "candidate_id": i,
                "fol": fol if fol else None,
                "fol_hash": hashlib.md5(fol.encode()).hexdigest()[:16] if fol else None,
                "raw_output": raw,
                "timeout": False,
            }
            if not fol:
                result["n_empty"] += 1
            candidates.append(cand)
        except TimeoutError:
            candidates.append({
                "candidate_id": i, "fol": None, "fol_hash": None,
                "raw_output": None, "timeout": True,
            })
            result["n_timeout"] += 1
        except Exception as e:
            candidates.append({
                "candidate_id": i, "fol": None, "fol_hash": None,
                "raw_output": str(e)[:200], "timeout": False,
            })

    result["candidates"] = candidates
    valid_fols = [c["fol_hash"] for c in candidates if c["fol"]]
    result["n_success"] = len(valid_fols)
    result["unique_count"] = len(set(valid_fols))
    return result


def _run_generation_k10(dataset_name: str, n_samples: int, model_tag: str, config: ModelConfig) -> List[dict]:
    """Generate k=K_CANDIDATES candidates for each sentence."""
    data_path = DATA_DIR / f"{dataset_name}.json"
    out_path = config.results_k10_dir / f"{model_tag}.json"
    ckpt_path = config.results_k10_dir / f"checkpoint_{model_tag}.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    with open(data_path, encoding="utf-8") as f:
        test_data = json.load(f)

    n_total = min(len(test_data), n_samples)
    test_data = test_data[:n_total]

    completed_idx: Set[int] = set()
    all_results: List[dict] = []

    if ckpt_path.exists():
        with open(ckpt_path) as f:
            saved = json.load(f)
            all_results = saved.get("results", [])
            completed_idx = {r["sentence_id"] for r in all_results}
        print(f"  Resuming from checkpoint: {len(completed_idx)} already done")

    total_candidates = len(completed_idx) * K_CANDIDATES
    total_success = sum(r.get("n_success", 0) for r in all_results)
    total_unique = sum(r.get("unique_count", 0) for r in all_results)
    t0 = time.time()

    print(f"  Model:    {config.base_model} + LoRA")
    print(f"  Data:     {data_path} ({n_total} entries)")
    print(f"  Output:   {out_path}")
    print(f"  T={TEMP_K10}  |  k={K_CANDIDATES}  |  total gens: {n_total * K_CANDIDATES:,}")
    print()

    for idx, item in enumerate(test_data):
        if idx in completed_idx:
            continue
        r = generate_k_candidates(item["NL"], config, sentence_id=idx)
        r["gt_fol"] = item["FOL"]
        all_results.append(r)

        total_candidates += K_CANDIDATES
        total_success += r["n_success"]
        total_unique += r["unique_count"]

        done = len(all_results)
        if done % 10 == 0 or done == n_total:
            elapsed = time.time() - t0
            new = done - len(completed_idx)
            avg = elapsed / max(new, 1)
            rem = avg * (n_total - done)
            print(
                f"  [{done:>4}/{n_total}]  "
                f"success={100*total_success/max(total_candidates,1):.1f}%  "
                f"unique_ratio={total_unique/max(total_success,1):.2f}  "
                f"elapsed={elapsed/60:.1f}m  ETA={rem/60:.1f}m"
            )
            with open(ckpt_path, "w") as f:
                json.dump(
                    {
                        "results": all_results,
                        "total_candidates": total_candidates,
                        "total_success": total_success,
                        "total_unique": total_unique,
                        "total_done": done,
                        "last_update": datetime.now().isoformat(),
                    },
                    f, indent=2, ensure_ascii=False,
                )

        if done >= n_total:
            break

    total_min = (time.time() - t0) / 60
    print(f"\n  === Generation Complete ===")
    print(f"  Total time: {total_min:.1f} min")
    print(f"  Candidates: {total_candidates} | Success: {total_success} | Unique: {total_unique}")

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    file_size = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Saved {len(all_results)} x up to {K_CANDIDATES} candidates ({file_size:.1f} MB)")

    if len(all_results) >= n_total and ckpt_path.exists():
        os.remove(ckpt_path)
        print("  Removed checkpoint (all done)")

    return all_results


def _print_summary_k10(all_results: List[dict], model_tag: str):
    """Print k10 generation summary."""
    print()
    print("=" * 60)
    print(f"  {model_tag} — k={K_CANDIDATES} MULTI-CANDIDATE SUMMARY")
    print("=" * 60)

    n_total = len(all_results)
    total_success = sum(r["n_success"] for r in all_results)
    total_empty = sum(r["n_empty"] for r in all_results)
    total_timeout = sum(r["n_timeout"] for r in all_results)
    total_gen = total_success + total_empty + total_timeout
    total_unique = sum(r["unique_count"] for r in all_results)

    print(f"\n  Sentences:            {n_total}")
    print(f"  Total generations:    {total_gen}")
    print(f"  Successful FOL:       {total_success} ({100*total_success/max(total_gen,1):.1f}%)")
    print(f"  Empty outputs:        {total_empty}")
    print(f"  Timeouts:             {total_timeout}")
    print(f"  Unique FOL hashes:    {total_unique}")
    print(f"  Diversity ratio:      {total_unique/max(total_success,1):.3f}")

    unique_per_sent = [r["unique_count"] for r in all_results]
    if unique_per_sent:
        print(f"\n  Per-sentence unique candidates:")
        print(f"    min: {min(unique_per_sent)}  max: {max(unique_per_sent)}  "
              f"mean: {np.mean(unique_per_sent):.1f}")
        for threshold in [1, 3, 5, 8, 10]:
            count = sum(1 for u in unique_per_sent if u >= threshold)
            print(f"    >={threshold} unique: {count}/{n_total} ({100*count/n_total:.1f}%)")

    # Sample showcase
    n_show = min(3, n_total)
    print(f"\n  === Sample Multi-Candidates ({n_show} of {n_total}) ===")
    for r in all_results[:n_show]:
        print(f"\n  [{r['sentence_id']}] NL: {r['nl'][:80]}")
        print(f"  GT: {r['gt_fol'][:120]}")
        print(f"  Generated {r['n_success']}/{K_CANDIDATES}  (unique: {r['unique_count']})")
        seen = set()
        for c in r["candidates"]:
            if c["fol"] and c["fol_hash"] not in seen:
                seen.add(c["fol_hash"])
                print(f"    [{c['candidate_id']}] {c['fol'][:100]}")


METRIC_KEYS = [
    "execution_rate", "exact_match", "exact_match_pred_norm",
    "z3_le", "z3_le_pred_norm",
    "bleu", "bleu_pred_norm",
    "bertscore", "bertscore_pred_norm",
]


def _run_eval_k10(dataset_name: str, model_tag: str, config: ModelConfig) -> List[dict]:
    """Compute core metric scores (BLEU/BERTScore disabled) for every k=10 candidate."""
    gen_path = config.results_k10_dir / f"{model_tag}.json"
    gt_path = DATA_DIR / f"{dataset_name}.json"
    eval_path = config.results_k10_dir / f"{model_tag}_metrics.json"

    sys.path.insert(0, str(ROOT))
    from src.eval.eval import compute_all_metrics

    with open(gen_path) as f:
        gen_data = json.load(f)
    with open(gt_path) as f:
        gt_raw = json.load(f)

    gt_map = {i: item["FOL"] for i, item in enumerate(gt_raw)}
    n_total = len(gen_data)

    # Resume from checkpoint
    already_done = 0
    k10_metrics_data: List[dict] = []
    if eval_path.exists():
        with open(eval_path) as f:
            prev = json.load(f)
            if isinstance(prev, list) and len(prev) > 0 and "candidates" in prev[0]:
                # only count entries where all candidates have complete metrics
                already_done = sum(
                    1 for e in prev
                    if all(
                        "metrics" in c and all(k in c["metrics"] for k in METRIC_KEYS)
                        for c in e.get("candidates", [])
                    )
                )
                if already_done > 0:
                    k10_metrics_data = prev
                    print(f"  Resuming metrics: {already_done}/{n_total} already done")
                else:
                    print(f"  Old metrics file (missing keys) — recomputing from scratch")

    total_candidates = n_total * K_CANDIDATES
    print(f"  Computing core metrics for up to {n_total} x {K_CANDIDATES} = {total_candidates:,} candidates...")
    print(f"  (Z3 only — BLEU/BERTScore disabled — a few minutes)")
    print()

    t0 = time.perf_counter()

    for sent_idx in range(already_done, n_total):
        r = gen_data[sent_idx]
        gt = gt_map[r["sentence_id"]]
        sent_entry = {
            "sentence_id": r["sentence_id"],
            "nl": r["nl"],
            "gt_fol": gt,
            "candidates": [],
        }
        for c in r["candidates"]:
            fol = c.get("fol", "") or ""
            if fol:
                scores = compute_all_metrics(
                    fol, gt, z3_timeout_ms=Z3_TIMEOUT_MS,
                    include_bleu=False, include_bertscore=False,
                )
            else:
                scores = {k: 0.0 for k in METRIC_KEYS}
            sent_entry["candidates"].append({
                "candidate_id": c["candidate_id"],
                "fol": fol,
                "metrics": scores,
            })

        k10_metrics_data.append(sent_entry)
        gc.collect(2)  # fast collection after every sentence (10 NLTK calls)
        if (sent_idx + 1) % 30 == 0:
            gc.collect()  # full GC to reclaim NLTK gen-2 leak

        if (sent_idx + 1) % 100 == 0:
            elapsed = (time.perf_counter() - t0) / 60
            done = sent_idx + 1
            avg = elapsed / max(done - already_done, 1)
            rem = avg * (n_total - done)
            print(f"  [{done:>4}/{n_total}]  elapsed={elapsed:.1f}m  ETA={rem:.1f}m")

            with open(eval_path, "w") as f:
                json.dump(k10_metrics_data, f, indent=2, ensure_ascii=False)

    elapsed = (time.perf_counter() - t0) / 60

    with open(eval_path, "w") as f:
        json.dump(k10_metrics_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Metrics saved to {eval_path} ({elapsed:.1f} min)")
    print(f"  Evaluated: {n_total} sentences x up to {K_CANDIDATES} candidates")

    return gen_data


def run_inference_k10(
    dataset_name: str = "test",
    model_tag: str | None = None,
    n_samples: int = 1000,
    skip_eval: bool = True,
    model: str = "Qwen4b",
) -> None:
    """
    Run k=10 multi-candidate inference (FOL generation only). Metrics NOT computed by default.
    Use run_eval_k10_only() to compute metrics separately.
    """
    config = _MODEL_REGISTRY.get(model, ModelConfig())

    if model_tag is None:
        suffix = dataset_name.replace("test_", "").replace("test", "")
        model_tag = f"{config.results_subdir}_k10_{suffix}" if suffix else f"{config.results_subdir}_k10"

    print("=" * 55)
    print(f"  K10 INFERENCE:  {dataset_name}.json")
    print(f"  MODEL TAG:      {model_tag}")
    print(f"  K={K_CANDIDATES}  T={TEMP_K10}")
    print("=" * 55)

    # Step 1: Load model
    print("\n[1] Loading model ...")
    _get_model(config)
    print("  Model ready.")

    # Step 2: Generate k=10 candidates
    print(f"\n[2] Running k={K_CANDIDATES} generation on {dataset_name}.json ...")
    all_results = _run_generation_k10(dataset_name, n_samples, model_tag, config)

    # Step 3: Summary
    print(f"\n[3] Results summary ...")
    _print_summary_k10(all_results, model_tag)

    # Step 4: Metric computation (off by default)
    if skip_eval:
        print(f"\n  [SKIP] Metrics not computed. Run eval notebook to compute.")
    else:
        print(f"\n[4] Computing per-candidate metrics ...")
        _run_eval_k10(dataset_name, model_tag, config)

    print(f"\n  Done. Output in {config.results_k10_dir}/")


def run_eval_k10_only(
    dataset_name: str = "test",
    model_tag: str | None = None,
    model: str = "Qwen4b",
) -> None:
    """Compute per-candidate metrics for already-generated k10 results. Use after run_inference_k10()."""
    config = _MODEL_REGISTRY.get(model, ModelConfig())

    if model_tag is None:
        suffix = dataset_name.replace("test_", "").replace("test", "")
        model_tag = f"{config.results_subdir}_k10_{suffix}" if suffix else f"{config.results_subdir}_k10"

    gen_path = config.results_k10_dir / f"{model_tag}.json"
    if not gen_path.exists():
        raise FileNotFoundError(
            f"{gen_path} not found — run k10 inference notebook first"
        )

    print("=" * 55)
    print(f"  K10 EVAL ONLY:  {model_tag}")
    print("=" * 55)

    print(f"\n[1] Computing per-candidate metrics for {model_tag}.json ...")
    _run_eval_k10(dataset_name, model_tag, config)
    print(f"\n  Done. Output in {config.results_k10_dir}/")
