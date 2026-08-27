"""
M6: LR Scorer Selector
======================

Select the best FOL candidate by scoring each candidate with a trained
Logistic Regression model and picking the highest-scored one.

Pipeline:
  Step 1 — compute pairwise metrics (Z3 / BLEU / BertScore / back-translation)
           via `pw_metrics.py` → ``{tag}_pw_metrics.json``
  Step 2 — score with LR, select, save metrics (this module, cheap, re-runnable)

Usage::

    # Step 1 (once) — pairwise metrics
    from src.selectors.pw_metrics import run_pw_metrics
    run_pw_metrics(model="Qwen4b", dataset="test")

    # Step 2 (re-runnable)
    from src.selectors.m6 import run_m6, show_m6
    run_m6(model="Qwen4b", dataset="test")
    show_m6(model="Qwen4b", dataset="test")
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np

# -- project root ----------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent  # project/code/

# -- constants -------------------------------------------------------------
_DS_MAP = {"test": "malls", "test_folio": "folio", "test_willow": "willow"}
_DEFAULT_FEATURE_KEYS = [
    "n_unique", "z3_le_count", "mean_bleu", "mean_bertscore", "backtrans_sim",
]

# Zero metrics for sentences where no candidate survived the Z3 filter.
# These sentences must still count in the evaluation denominator.
_ZERO_METRICS = {
    "execution_rate": 0.0,
    "exact_match": 0.0, "exact_match_pred_norm": 0.0,
    "z3_le": 0.0, "z3_le_pred_norm": 0.0,
    "bleu": 0.0, "bleu_pred_norm": 0.0,
    "bertscore": 0.0, "bertscore_pred_norm": 0.0,
}


def _resolve_paths(model: str, dataset: str) -> Dict[str, Path]:
    ds_folder = _DS_MAP.get(dataset, dataset)
    suffix = dataset.replace("test_", "").replace("test", "")
    tag = f"{model}_k10_{suffix}" if suffix else f"{model}_k10"
    k10_dir = _ROOT / "data" / "results" / model / "k10"
    k10_metrics = k10_dir / f"{tag}_metrics.json"
    gt_file = _ROOT / "data" / f"{dataset}.json"
    out_dir = k10_dir / ds_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "k10_metrics": k10_metrics,
        "gt_file": gt_file,
        "out_dir": out_dir,
        "tag": tag,
    }


# ======================================================================
# Core: LR scoring + selection
# ======================================================================


def m6_select(
    features: List[dict],
    lr_model,
    feature_keys: List[str] = None,
) -> dict:
    """Score candidates with LR, pick the highest-scored one.

    Parameters
    ----------
    features : list of dicts
        Each dict has one entry per key in ``feature_keys``.
    lr_model : LogisticRegression
        Trained LR model with predict_proba.
    feature_keys : list of str, optional
        Feature columns to feed the scorer. Must match the trained model's
        columns and order. Defaults to the 5 pairwise-metric features.

    Returns
    -------
    dict with: selected_idx, best_score, scores, n_valid, n_unique
    """
    feature_keys = feature_keys or _DEFAULT_FEATURE_KEYS

    if not features:
        return {"selected_idx": -1, "best_score": 0.0, "scores": [], "n_valid": 0, "n_unique": 0}

    X = np.array([[r[k] for k in feature_keys] for r in features], dtype=np.float64)
    scores = lr_model.predict_proba(X)[:, 1].tolist()

    best_idx = int(np.argmax(scores))
    return {
        "selected_idx": features[best_idx]["candidate_idx"],
        "best_score": scores[best_idx],
        "scores": [round(s, 6) for s in scores],
        "n_valid": len(features),
        "n_unique": features[0]["n_unique"] if features else 0,
    }


def run_m6(
    model: str = "Qwen4b",
    dataset: str = "test",
    feature_keys: List[str] = None,
) -> None:
    """Score candidates with trained LR and save selection results.

    Loads ``{tag}_pw_metrics.json`` (from pw_metrics.py) and
    ``models/{model}_scorer_lr.pkl``.

    Parameters
    ----------
    feature_keys : list of str, optional
        Feature columns to feed the scorer. Must match the trained model's
        columns and order. Defaults to the 5 pairwise-metric features.
    """
    paths = _resolve_paths(model, dataset)
    features_path = paths["out_dir"] / f"{paths['tag']}_pw_metrics.json"
    lr_path = _ROOT / "models" / f"{model}_scorer_lr.pkl"

    if not features_path.exists():
        raise FileNotFoundError(f"{features_path} not found — run pw_metrics.py first")
    if not lr_path.exists():
        raise FileNotFoundError(f"{lr_path} not found — train LR scorer first")

    if not paths["gt_file"].exists():
        raise FileNotFoundError(f"{paths['gt_file']} not found")

    with open(features_path) as f:
        all_features = json.load(f)
    with open(paths["gt_file"]) as f:
        gt_data = json.load(f)

    gt_map = {i: item["FOL"] for i, item in enumerate(gt_data)}
    lr_model = joblib.load(str(lr_path))

    # Group features by sentence_id
    sent_groups: Dict[int, List[dict]] = defaultdict(list)
    for row in all_features:
        sent_groups[row["sentence_id"]].append(row)

    n_total = len(gt_map)
    n_covered = len(sent_groups)
    out_path = paths["out_dir"] / f"{paths['tag']}_m6.json"

    print("=" * 55)
    print("  M6: LR SCORER SELECTION")
    print(f"  Model:        {model}")
    print(f"  Dataset:      {dataset}")
    print(f"  LR model:     {lr_path.name}")
    print(f"  Sentences:    {n_total}")
    print(f"  With cands:   {n_covered}")
    print("=" * 55)
    print()

    # -- late import for metrics --
    sys.path.insert(0, str(_ROOT))
    from src.eval.eval import compute_all_metrics

    # Load k10 data to get FOL strings and original NL
    with open(paths["k10_metrics"]) as f:
        k10_data = json.load(f)
    k10_by_sid = {e["sentence_id"]: e for e in k10_data}

    results = []
    n_no_cand = 0
    t0 = time.perf_counter()

    # Iterate over ALL test sentences (gt_map) so sentences whose candidates
    # were all filtered out by Z3 still count in the evaluation denominator.
    for sid in sorted(gt_map.keys()):
        gt = gt_map[sid]
        k10_entry = k10_by_sid.get(sid, {})

        if sid not in sent_groups:
            # No candidate survived the Z3 filter — record a zero-score entry.
            n_no_cand += 1
            r = {
                "sentence_id": sid,
                "selected_idx": -1,
                "fol": "",
                "n_valid": 0,
                "n_unique": 0,
                "best_score": 0.0,
                "scores": [],
                "nl": k10_entry.get("nl", ""),
                "gt_fol": gt,
                "metrics": dict(_ZERO_METRICS),
            }
            results.append(r)
        else:
            features = sent_groups[sid]
            sel = m6_select(features, lr_model, feature_keys=feature_keys)

            candidates = k10_entry.get("candidates", [])
            best_idx = sel["selected_idx"]
            selected_fol = candidates[best_idx]["fol"] if 0 <= best_idx < len(candidates) else ""

            r = {
                "sentence_id": sid,
                "selected_idx": best_idx,
                "fol": selected_fol,
                "n_valid": sel["n_valid"],
                "n_unique": sel["n_unique"],
                "best_score": sel["best_score"],
                "scores": sel["scores"],
                "nl": k10_entry.get("nl", ""),
                "gt_fol": gt,
                "metrics": compute_all_metrics(
                    selected_fol, gt,
                    include_bleu=False, include_bertscore=False,
                ),
            }
            results.append(r)

        done = len(results)
        if done % 50 == 0 or done == n_total:
            elapsed = (time.perf_counter() - t0) / 60
            avg = elapsed / done
            rem = avg * (n_total - done)
            z3_ok = sum(1 for r2 in results if r2["metrics"].get("z3_le", 0.0) >= 1.0)
            print(
                f"  [{done:>4}/{n_total}]  "
                f"z3_le={z3_ok}/{done} ({100*z3_ok/done:.1f}%)  "
                f"elapsed={elapsed:.1f}m  ETA={rem:.1f}m",
                flush=True,
            )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = (time.perf_counter() - t0) / 60
    print()
    print("=" * 55)
    print(f"  M6 SELECTION COMPLETE — {elapsed:.1f} min")
    print(f"  Sentences:   {n_total}")
    print(f"  No cand:     {n_no_cand}")
    print(f"  Saved to:    {out_path}")


# ======================================================================
# Display
# ======================================================================


def show_m6(
    model: str = "Qwen4b",
    dataset: str = "test",
) -> None:
    """Print aggregate results from saved M6 metrics."""
    paths = _resolve_paths(model, dataset)
    m6_path = paths["out_dir"] / f"{paths['tag']}_m6.json"

    if not m6_path.exists():
        raise FileNotFoundError(f"{m6_path} not found — run M6 selection first")

    with open(m6_path) as f:
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
    print("  M6: LR SCORER — RESULTS")
    print(f"  Model: {model}  |  Dataset: {dataset}")
    print("=" * 55)
    print()
    print(f"  Sentences:        {n_total}")
    print(f"  Z3 LE:            {n_z3_le}/{n_total} ({100*n_z3_le/n_total:.1f}%)")
    print(f"  Exec Rate:        {n_exec_ok}/{n_total} ({100*n_exec_ok/n_total:.1f}%)")
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

    print()
    print(f"  Full results: {m6_path}")
