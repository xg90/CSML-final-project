"""
Feature-based selection (M1-M4 fast path)
=========================================

Read ``_pw_metrics.json`` (produced by pw_metrics.py) and select the best
candidate per sentence via a single argmax over one feature column.

This replaces the original heavy pipelines (Z3/BLEU/BertScore/backtrans)
for M1, M2, M3, M4.  M5 (LLM-FT) is unchanged.

Usage::

    from src.selectors.feature_select import run_from_features, show_from_features
    run_from_features(model="Qwen4b", dataset="test", feature_key="z3_le_count", method="1")
    show_from_features(model="Qwen4b", dataset="test", method="1")
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent  # project/code/

_DS_MAP = {"test": "malls", "test_folio": "folio", "test_willow": "willow"}

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
    features_path = k10_dir / ds_folder / f"{tag}_pw_metrics.json"
    out_dir = k10_dir / ds_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "k10_metrics": k10_metrics, "gt_file": gt_file,
        "features_path": features_path, "out_dir": out_dir, "tag": tag,
    }


def run_from_features(
    model: str = "Qwen4b",
    dataset: str = "test",
    feature_key: str = "z3_le_count",
    method: str = "1",
) -> None:
    """Select candidates from M6 features file by argmax over one column.

    Parameters
    ----------
    model : str
    dataset : str
    feature_key : str
        Column in _pw_metrics.json to argmax.
    method : str
        Method number ("1","2","3","4") for output filename and display.
    """
    paths = _resolve_paths(model, dataset)

    if not paths["features_path"].exists():
        raise FileNotFoundError(
            f"{paths['features_path']} not found — run pw_metrics.py first"
        )
    if not paths["gt_file"].exists():
        raise FileNotFoundError(f"{paths['gt_file']} not found")
    if not paths["k10_metrics"].exists():
        raise FileNotFoundError(f"{paths['k10_metrics']} not found")

    with open(paths["features_path"]) as f:
        all_features = json.load(f)
    with open(paths["gt_file"]) as f:
        gt_data = json.load(f)
    with open(paths["k10_metrics"]) as f:
        k10_data = json.load(f)

    gt_map = {i: item["FOL"] for i, item in enumerate(gt_data)}
    k10_by_sid = {e["sentence_id"]: e for e in k10_data}

    # Group features by sentence_id
    sent_groups: Dict[int, List[dict]] = defaultdict(list)
    for row in all_features:
        sent_groups[row["sentence_id"]].append(row)

    n_total = len(gt_map)
    n_covered = len(sent_groups)
    out_path = paths["out_dir"] / f"{paths['tag']}_m{method}.json"

    print("=" * 55)
    print(f"  M{method}: FEATURE-BASED SELECTION (argmax {feature_key})")
    print(f"  Model: {model}  |  Dataset: {dataset}")
    print(f"  Sentences:    {n_total}")
    print(f"  With cands:   {n_covered}")
    print("=" * 55)
    print()

    sys.path.insert(0, str(_ROOT))
    from src.eval.eval import compute_all_metrics

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
                feature_key: 0.0,
                "nl": k10_entry.get("nl", ""),
                "gt_fol": gt,
                "metrics": dict(_ZERO_METRICS),
            }
            results.append(r)
        else:
            rows = sent_groups[sid]
            best_row = max(rows, key=lambda r: r.get(feature_key, 0.0))
            best_idx = best_row["candidate_idx"]

            candidates = k10_entry.get("candidates", [])
            selected_fol = candidates[best_idx]["fol"] if 0 <= best_idx < len(candidates) else ""

            r = {
                "sentence_id": sid,
                "selected_idx": best_idx,
                "fol": selected_fol,
                "n_valid": best_row.get("n_unique", len(rows)),
                feature_key: best_row.get(feature_key, 0.0),
                "nl": k10_entry.get("nl", ""),
                "gt_fol": gt,
                "metrics": compute_all_metrics(
                    selected_fol, gt,
                    include_bleu=False, include_bertscore=False,
                ),
            }
            results.append(r)

        done = len(results)
        if done % 100 == 0 or done == n_total:
            elapsed = (time.perf_counter() - t0) / 60
            print(f"  [{done:>4}/{n_total}]  elapsed={elapsed:.1f}m", flush=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = (time.perf_counter() - t0) / 60
    z3_ok = sum(1 for r2 in results if r2["metrics"].get("z3_le", 0.0) >= 1.0)
    print()
    print(f"  M{method} COMPLETE — {elapsed:.1f} min")
    print(f"  Sentences:   {n_total}")
    print(f"  No cand:     {n_no_cand}")
    print(f"  Z3 LE:       {z3_ok}/{n_total} ({100*z3_ok/n_total:.1f}%)")
    print(f"  Saved to:    {out_path}")


def show_from_features(
    model: str = "Qwen4b",
    dataset: str = "test",
    method: str = "1",
) -> None:
    """Print aggregate metrics for a feature-based method."""
    paths = _resolve_paths(model, dataset)
    out_path = paths["out_dir"] / f"{paths['tag']}_m{method}.json"

    if not out_path.exists():
        raise FileNotFoundError(f"{out_path} not found — run selection first")

    with open(out_path) as f:
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
    print(f"  M{method}: FEATURE-BASED — RESULTS")
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
    print(f"  Full results: {out_path}")
