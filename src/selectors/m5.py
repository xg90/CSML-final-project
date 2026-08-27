"""
M5: Qwen/Ministral + LoRA Selector
=============================

Loads the fine-tuned Qwen selector, scores each (NL, FOL) candidate,
and selects the candidate with the highest predicted LE confidence.

Usage::

    from src.selectors.m5 import run_m5
    run_m5(model="Qwen4b", dataset="test")
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# -- project root ----------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent  # project/code/
MODEL_DIR = _ROOT / "models"

# -- model registry — maps model tag → HuggingFace base model ----------------
_MODEL_REGISTRY: Dict[str, str] = {
    "Qwen4b":  "Qwen/Qwen3-4B",
    "Qwen8b":  "Qwen/Qwen3-8B",
    "Ministral8b": "mistralai/Ministral-3-8B-Instruct-2512",
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

# -- lazy cache ------------------------------------------------------------
_selector_cache: Dict[str, tuple] = {}


def _resolve_paths(model: str, dataset: str) -> Dict[str, Path]:
    _DS_MAP = {"test": "malls", "test_folio": "folio", "test_willow": "willow"}
    ds_folder = _DS_MAP.get(dataset, dataset)
    suffix = dataset.replace("test_", "").replace("test", "")
    tag = f"{model}_k10_{suffix}" if suffix else f"{model}_k10"
    k10_dir = _ROOT / "data" / "results" / model / "k10"
    k10_metrics = k10_dir / f"{tag}_metrics.json"
    gt_file = _ROOT / "data" / f"{dataset}.json"
    out_dir = k10_dir / ds_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    return {"k10_metrics": k10_metrics, "gt_file": gt_file, "out_dir": out_dir, "tag": tag}


def _load_selector(model_name: str):
    """Load cached M5 selector (Qwen/Ministral + LoRA + score_head)."""
    if model_name in _selector_cache:
        return _selector_cache[model_name]

    selector_path = MODEL_DIR / f"{model_name}_selector"
    if not selector_path.exists():
        raise FileNotFoundError(f"M5 selector not found: {selector_path}")

    base_model = _MODEL_REGISTRY.get(model_name)
    if base_model is None:
        raise ValueError(
            f"Unknown model '{model_name}' — add it to _MODEL_REGISTRY in m5.py."
        )
    is_mistral = model_name == "Ministral8b"

    # Ministral tokenizer ships a broken regex — needs the fix flag on load.
    tokenizer_kwargs = {"trust_remote_code": True}
    if is_mistral:
        tokenizer_kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(str(selector_path), **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = _resolve_model_cls(base_model).from_pretrained(
        base_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    llm = PeftModel.from_pretrained(base, str(selector_path))
    llm.eval()

    # Text backbone whose forward yields last_hidden_state (skips the lm_head):
    #   Qwen3    -> llm.base_model.model.model                 (Qwen3Model)
    #   Mistral3 -> llm.base_model.model.model.language_model  (VLM text decoder)
    text_model = llm.base_model.model.model
    if is_mistral:
        text_model = text_model.language_model

    hidden_size = base.config.text_config.hidden_size if is_mistral else base.config.hidden_size
    score_head = nn.Linear(hidden_size, 1)
    score_head.load_state_dict(torch.load(str(selector_path / "score_head.pt"),
                                          map_location="cpu", weights_only=True))
    score_head.eval()

    _selector_cache[model_name] = (tokenizer, llm, score_head, text_model)
    return tokenizer, llm, score_head, text_model


def m5_select(candidates: List[dict], gt_fol: str, sentence_id: int = 0) -> dict:
    """Score each candidate via M5 and return the best one."""
    result = {
        "sentence_id": sentence_id,
        "fol": None,
        "selected_idx": -1,
        "n_scored": 0,
        "best_score": 0.0,
        "scores": [],
        "error": None,
    }

    valid = [(i, c) for i, c in enumerate(candidates)
             if c.get("fol") and c.get("metrics", {}).get("execution_rate", 0.0) > 0.0]
    if not valid:
        result["error"] = "No valid candidates"
        return result

    tokenizer, llm, score_head, text_model = _load_selector(_current_model)
    device = next(llm.parameters()).device
    score_head.to(device)

    scores = []
    for idx, cand in valid:
        text = f"NL: {_current_nl}\nFOL: {cand['fol']}"
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=512).to(device)
        with torch.no_grad():
            outputs = text_model(**enc)
            last_hidden = outputs.last_hidden_state[:, -1, :].float()
            s = torch.sigmoid(score_head(last_hidden)).item()
        scores.append((idx, s))

    result["n_scored"] = len(scores)
    result["scores"] = [s for _, s in scores]

    best_idx, best_score = max(scores, key=lambda x: x[1])
    result["selected_idx"] = best_idx
    result["best_score"] = best_score
    result["fol"] = candidates[best_idx]["fol"]
    return result


_current_model = ""
_current_nl = ""


def run_m5(model: str = "Qwen4b", dataset: str = "test") -> None:
    global _current_model, _current_nl

    paths = _resolve_paths(model, dataset)
    _current_model = model

    if not paths["k10_metrics"].exists():
        raise FileNotFoundError(f"{paths['k10_metrics']} not found")
    if not paths["gt_file"].exists():
        raise FileNotFoundError(f"{paths['gt_file']} not found")

    with open(paths["k10_metrics"]) as f:
        k10_data = json.load(f)
    with open(paths["gt_file"]) as f:
        gt_data = json.load(f)

    gt_map = {i: item["FOL"] for i, item in enumerate(gt_data)}
    n_total = len(k10_data)

    out_path = paths["out_dir"] / f"{paths['tag']}_m5.json"

    print("=" * 55)
    print("  M5: Qwen/Ministral + LoRA Selector")
    print(f"  Model: {model}  |  Dataset: {dataset}")
    print("=" * 55)
    print(f"\n  Sentences:  {n_total}")
    print(f"  Output:     {out_path}\n")

    # Pre-load selector
    tokenizer, llm, score_head, _ = _load_selector(model)

    results = []
    t0 = time.perf_counter()

    sys.path.insert(0, str(_ROOT))
    from src.eval.eval import compute_all_metrics

    for sent_entry in k10_data:
        sid = sent_entry["sentence_id"]
        gt = gt_map.get(sid, "")
        _current_nl = sent_entry.get("nl", "")
        r = m5_select(sent_entry["candidates"], gt, sentence_id=sid)
        r["nl"] = _current_nl
        r["gt_fol"] = gt
        r["metrics"] = compute_all_metrics(
            r.get("fol", "") or "", gt,
            include_bleu=False, include_bertscore=False,
        )
        results.append(r)

        done = len(results)
        if done % 100 == 0 or done == n_total:
            elapsed = (time.perf_counter() - t0) / 60
            avg_score = sum(r2.get("best_score", 0) for r2 in results) / done
            print(f"  [{done:>4}/{n_total}]  avg_score={avg_score:.4f}  "
                  f"elapsed={elapsed:.1f}m")

    elapsed = (time.perf_counter() - t0) / 60
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  M5 COMPLETE — {elapsed:.1f} min  |  Saved: {out_path}")


def show_m5(model: str = "Qwen4b", dataset: str = "test") -> None:
    import numpy as np
    paths = _resolve_paths(model, dataset)
    m5_path = paths["out_dir"] / f"{paths['tag']}_m5.json"

    if not m5_path.exists():
        raise FileNotFoundError(f"{m5_path} not found — run M5 first")

    with open(m5_path) as f:
        data = json.load(f)

    valid = [e for e in data if "metrics" in e]
    n_total = len(data)

    metric_keys = [
        "execution_rate", "exact_match", "exact_match_pred_norm",
        "z3_le", "z3_le_pred_norm",
    ]

    n_exec_ok = sum(1 for e in valid if e["metrics"].get("execution_rate", 0.0) >= 1.0)
    n_z3_le = sum(1 for e in valid if e["metrics"].get("z3_le", 0.0) >= 1.0)

    print("=" * 55)
    print("  M5: Qwen/Ministral + LoRA Selector — RESULTS")
    print(f"  Model: {model}  |  Dataset: {dataset}")
    print("=" * 55)
    print(f"\n  Sentences:        {n_total}")
    print(f"  Z3 LE:            {n_z3_le}/{n_total} ({100*n_z3_le/n_total:.1f}%)")
    print(f"  Exec Rate:        {n_exec_ok}/{n_total} ({100*n_exec_ok/n_total:.1f}%)")
    print(f"  Avg Score:        {np.mean([e.get('best_score', 0) for e in valid]):.4f}")
    print()

    print(f"  {'Metric':<30s} {'Mean':>8s}")
    print("  " + "-" * 40)
    for key in metric_keys:
        vals = [e["metrics"].get(key, 0.0) for e in valid]
        mean_v = float(np.mean(vals))
        if key in ("execution_rate", "exact_match", "exact_match_pred_norm",
                    "z3_le", "z3_le_pred_norm"):
            print(f"  {key:<30s} {100*mean_v:>7.1f}%")
        else:
            print(f"  {key:<30s} {mean_v:>7.4f}")
    print(f"\n  Full results: {m5_path}")
