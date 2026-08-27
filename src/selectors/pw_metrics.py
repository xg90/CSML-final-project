"""
PW-Metrics: Pairwise Metrics Pipeline (M6 feature computation)
==============================================================

Independent pipeline for computing pairwise metrics on TEST k10 candidates.
Feeds the M6 LR scorer (Stage 2).  Six steps, each reading the previous step's
output and adding one feature.

Pipeline
--------
  Step 1 – Z3 filter        keep only Z3-parsable candidates
  Step 2 – n_unique         unique FOL count per sentence
  Step 3 – z3_le_count      pairwise Z3 LE with other filtered candidates
  Step 4 – mean_bleu        mean pairwise FOL-token BLEU  (filtered only)
  Step 5 – mean_bertscore   mean pairwise BertScore F1     (filtered only)
  Step 6 – dedup + backtrans  dedup per sentence, back-translate, cosine sim

Final output schema (no label — inference has no ground truth)
--------------------------------------------------------------
  sentence_id    : int   – sentence index
  candidate_idx  : int   – original candidate index in the k10 pool
  n_unique       : int   – unique FOL count among Z3-filtered candidates
  z3_le_count    : int   – # of other filtered candidates Z3-equivalent
  mean_bleu      : float – mean pairwise FOL-token BLEU
  mean_bertscore : float – mean pairwise BertScore F1
  backtrans_sim  : float – BERT cosine similarity (NL vs back-translated NL)

Usage
-----
    python src/selectors/pw_metrics.py --model Qwen4b --dataset test
    python src/selectors/pw_metrics.py --model Qwen4b --dataset test --step 6
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# -- project root ----------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent  # project/code/

# -- constants -------------------------------------------------------------
_DS_MAP = {"test": "malls", "test_folio": "folio", "test_willow": "willow"}
_DATASET_KEY_MAP = {"test": "malls", "test_folio": "folio", "test_willow": "willow"}
_FOL_TOKEN_PATTERN = re.compile(r"[∀∃]|\w+|¬|∧|∨|→|↔|⊕|[()]|=")

# -- lazy singletons -------------------------------------------------------
_bleu_metric = None
_bertscore_metric = None


def _get_bleu():
    global _bleu_metric
    if _bleu_metric is None:
        import evaluate
        _bleu_metric = evaluate.load("bleu")
    return _bleu_metric


def _get_bertscore():
    global _bertscore_metric
    if _bertscore_metric is None:
        import evaluate
        _bertscore_metric = evaluate.load("bertscore")
    return _bertscore_metric


def _fol_tokenize(s: str) -> List[str]:
    return _FOL_TOKEN_PATTERN.findall(s)


# ======================================================================
# Path / IO helpers
# ======================================================================


def _resolve_paths(model: str, dataset: str) -> Dict[str, Path]:
    """Resolve test k10 metrics + output dir for a model+dataset combo."""
    ds_folder = _DS_MAP.get(dataset, dataset)
    suffix = dataset.replace("test_", "").replace("test", "")
    tag = f"{model}_k10_{suffix}" if suffix else f"{model}_k10"
    k10_dir = _ROOT / "data" / "results" / model / "k10"
    k10_metrics = k10_dir / f"{tag}_metrics.json"
    out_dir = k10_dir / ds_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    return {"k10_metrics": k10_metrics, "out_dir": out_dir, "tag": tag}


def _pw_step_path(model: str, dataset: str, step: int) -> Path:
    """step 0 → k10_metrics source; step 6 → final _pw_metrics.json; else step{N}."""
    paths = _resolve_paths(model, dataset)
    tag = paths["tag"]
    if step == 0:
        return paths["k10_metrics"]
    if step == 6:
        return paths["out_dir"] / f"{tag}_pw_metrics.json"
    return paths["out_dir"] / f"{tag}_pw_step{step}.json"


def _cache_path(model: str, dataset: str) -> Path:
    paths = _resolve_paths(model, dataset)
    return paths["out_dir"] / f"{paths['tag']}_backtrans_cache.json"


def _load_rows(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_rows(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def _group_by_sentence(rows: List[dict]) -> Dict[int, List[dict]]:
    groups: Dict[int, List[dict]] = {}
    for r in rows:
        groups.setdefault(r["sentence_id"], []).append(r)
    return groups


def _read_cache(model: str, dataset: str) -> Dict[str, str]:
    path = _cache_path(model, dataset)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _write_cache(model: str, dataset: str, cache: Dict[str, str]) -> None:
    path = _cache_path(model, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


# ======================================================================
# Z3 helpers
# ======================================================================


def _try_parse_z3(fol: str):
    """Try to parse a FOL string into a Z3 expression. Returns expr or None."""
    sys.path.insert(0, str(_ROOT))
    from src.eval.z3parser import fol_to_z3
    try:
        return fol_to_z3(fol)
    except Exception:
        return None


def _z3_le_check(fol_a: str, fol_b: str) -> bool:
    """Check if two FOL strings are Z3-logically equivalent (returns bool)."""
    from src.eval.z3_equiv import check_equivalence
    try:
        return check_equivalence(fol_a, fol_b, timeout_ms=10_000).is_equivalent
    except Exception:
        return False


# ======================================================================
# Step 1 – Z3 filter
# ======================================================================


def pw_step1_z3_filter(model: str = "Qwen4b", dataset: str = "test",
                       in_path: Optional[Path] = None,
                       force: bool = False) -> Path:
    """Keep only candidates whose FOL can be parsed by Z3.

    Reads ``{tag}_metrics.json``, writes ``{tag}_pw_step1.json``.
    """
    src = in_path or _pw_step_path(model, dataset, 0)
    dst = _pw_step_path(model, dataset, 1)

    if dst.exists() and not force:
        print(f"  [SKIP] {dst.name} already exists (use --force to recompute)")
        return dst

    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    data = _load_rows(src)
    rows: List[dict] = []
    n_total, n_kept = 0, 0

    for idx, entry in enumerate(data):
        sid = entry.get("sentence_id", idx)
        nl = entry.get("nl", "")
        gt = entry.get("gt_fol", "")
        for i, c in enumerate(entry.get("candidates", [])):
            n_total += 1
            fol = (c.get("fol") or "").strip()
            if fol and _try_parse_z3(fol) is not None:
                rows.append({
                    "sentence_id": sid,
                    "candidate_idx": i,
                    "nl": nl,
                    "gt_fol": gt,
                    "fol": fol,
                })
                n_kept += 1

    _save_rows(dst, rows)
    print(f"  Z3 filter: {n_kept}/{n_total} candidates kept "
          f"({100 * n_kept / max(n_total, 1):.1f}%)")
    print(f"  Saved -> {dst}")
    return dst


# ======================================================================
# Step 2 – n_unique
# ======================================================================


def pw_step2_n_unique(model: str = "Qwen4b", dataset: str = "test",
                      in_path: Optional[Path] = None,
                      force: bool = False) -> Path:
    """Count unique FOL strings per sentence among Z3-filtered candidates."""
    src = in_path or _pw_step_path(model, dataset, 1)
    dst = _pw_step_path(model, dataset, 2)

    if dst.exists() and not force:
        print(f"  [SKIP] {dst.name} already exists")
        return dst

    rows = _load_rows(src)
    groups = _group_by_sentence(rows)
    out: List[dict] = []

    for group in groups.values():
        n = len({r["fol"] for r in group if r["fol"]})
        out.extend([{**r, "n_unique": n} for r in group])

    _save_rows(dst, out)
    print(f"  n_unique added -> {dst}  ({len(out)} rows)")
    return dst


# ======================================================================
# Step 3 – z3_le_count
# ======================================================================


def pw_step3_z3_le_count(model: str = "Qwen4b", dataset: str = "test",
                         in_path: Optional[Path] = None,
                         force: bool = False) -> Path:
    """Pairwise Z3 logical-equivalence check among filtered candidates."""
    src = in_path or _pw_step_path(model, dataset, 2)
    dst = _pw_step_path(model, dataset, 3)

    if dst.exists() and not force:
        print(f"  [SKIP] {dst.name} already exists")
        return dst

    rows = _load_rows(src)
    groups = _group_by_sentence(rows)
    out: List[dict] = []

    n_checks = 0
    n_sentences = len(groups)
    t0 = time.perf_counter()

    for si, (sid, group) in enumerate(groups.items()):
        m = len(group)
        counts = [0] * m
        for a in range(m):
            for b in range(a + 1, m):
                n_checks += 1
                if _z3_le_check(group[a]["fol"], group[b]["fol"]):
                    counts[a] += 1
                    counts[b] += 1
        out.extend([{**r, "z3_le_count": c} for r, c in zip(group, counts)])

        if (si + 1) % 100 == 0 or si + 1 == n_sentences:
            elapsed = time.perf_counter() - t0
            avg = elapsed / (si + 1)
            eta = avg * (n_sentences - si - 1)
            print(f"    [{si+1}/{n_sentences}] sentences  "
                  f"checks={n_checks}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                  flush=True)

    elapsed = time.perf_counter() - t0
    _save_rows(dst, out)
    print(f"  z3_le_count: {n_checks} pairwise checks in {elapsed:.1f}s")
    print(f"  Saved -> {dst}  ({len(out)} rows)")
    return dst


# ======================================================================
# Step 4 – mean_bleu
# ======================================================================


def pw_step4_mean_bleu(model: str = "Qwen4b", dataset: str = "test",
                       in_path: Optional[Path] = None,
                       force: bool = False) -> Path:
    """Mean pairwise FOL-token BLEU among Z3-filtered candidates only."""
    src = in_path or _pw_step_path(model, dataset, 3)
    dst = _pw_step_path(model, dataset, 4)

    if dst.exists() and not force:
        print(f"  [SKIP] {dst.name} already exists")
        return dst

    rows = _load_rows(src)
    groups = _group_by_sentence(rows)
    bleu = _get_bleu()
    out: List[dict] = []

    n_sentences = len(groups)
    t0 = time.perf_counter()

    for si, group in enumerate(groups.values()):
        fols = [r["fol"] for r in group]
        m = len(fols)
        mat = [[0.0] * m for _ in range(m)]

        for a in range(m):
            for b in range(m):
                if a == b:
                    mat[a][b] = 1.0
                    continue
                ta, tb = _fol_tokenize(fols[a]), _fol_tokenize(fols[b])
                mo = min(4, min(len(ta), len(tb)))
                if mo >= 1:
                    try:
                        res = bleu.compute(
                            predictions=[fols[a]], references=[[fols[b]]],
                            tokenizer=_fol_tokenize, max_order=mo,
                        )
                        mat[a][b] = float(res["bleu"] or 0.0)
                    except Exception:
                        mat[a][b] = 0.0

        for i, r in enumerate(group):
            others = [mat[i][j] for j in range(m) if j != i]
            mean = (sum(others) / len(others)) if others else 0.0
            out.append({**r, "mean_bleu": round(mean, 6)})

        if (si + 1) % 100 == 0 or si + 1 == n_sentences:
            elapsed = time.perf_counter() - t0
            avg = elapsed / (si + 1)
            eta = avg * (n_sentences - si - 1)
            print(f"    [{si+1}/{n_sentences}] sentences  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

    elapsed = time.perf_counter() - t0
    _save_rows(dst, out)
    print(f"  mean_bleu computed in {elapsed:.1f}s")
    print(f"  Saved -> {dst}  ({len(out)} rows)")
    return dst


# ======================================================================
# Step 5 – mean_bertscore
# ======================================================================


def pw_step5_mean_bertscore(model: str = "Qwen4b", dataset: str = "test",
                            in_path: Optional[Path] = None,
                            force: bool = False) -> Path:
    """Mean pairwise BertScore F1 among Z3-filtered candidates only."""
    src = in_path or _pw_step_path(model, dataset, 4)
    dst = _pw_step_path(model, dataset, 5)

    if dst.exists() and not force:
        print(f"  [SKIP] {dst.name} already exists")
        return dst

    rows = _load_rows(src)
    groups = _group_by_sentence(rows)
    bertscore = _get_bertscore()
    out: List[dict] = []

    n_sentences = len(groups)
    t0 = time.perf_counter()

    for si, group in enumerate(groups.values()):
        fols = [r["fol"] for r in group]
        m = len(fols)
        mat = [[0.0] * m for _ in range(m)]

        for a in range(m):
            for b in range(a + 1, m):
                fa, fb = fols[a], fols[b]
                if not fa or not fb:
                    continue
                try:
                    res = bertscore.compute(
                        predictions=[fa], references=[fb], lang="en",
                    )
                    v = float(res["f1"][0]) if "f1" in res else 0.0
                    mat[a][b] = v
                    mat[b][a] = v  # symmetric
                except Exception:
                    pass
            mat[a][a] = 1.0

        for i, r in enumerate(group):
            others = [mat[i][j] for j in range(m) if j != i]
            mean = (sum(others) / len(others)) if others else 0.0
            out.append({**r, "mean_bertscore": round(mean, 6)})

        if (si + 1) % 100 == 0 or si + 1 == n_sentences:
            elapsed = time.perf_counter() - t0
            avg = elapsed / (si + 1)
            eta = avg * (n_sentences - si - 1)
            print(f"    [{si+1}/{n_sentences}] sentences  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

    elapsed = time.perf_counter() - t0
    _save_rows(dst, out)
    print(f"  mean_bertscore computed in {elapsed:.1f}s")
    print(f"  Saved -> {dst}  ({len(out)} rows)")
    return dst


# ======================================================================
# Step 6 – dedup + back-translation similarity
# ======================================================================


def pw_step6_dedup_backtrans(model: str = "Qwen4b", dataset: str = "test",
                             in_path: Optional[Path] = None,
                             force: bool = False) -> Path:
    """Deduplicate per sentence by FOL, then back-translate unique FOLs globally.

    Final output drops ``nl``, ``gt_fol``, ``fol`` — keeps only the 5 features
    plus ``sentence_id`` and ``candidate_idx`` (what M6 Stage 2 consumes).
    """
    src = in_path or _pw_step_path(model, dataset, 5)
    dst = _pw_step_path(model, dataset, 6)

    if dst.exists() and not force:
        print(f"  [SKIP] {dst.name} already exists")
        return dst

    rows = _load_rows(src)
    n_before = len(rows)

    # -- (a) per-sentence dedup --------------------------------------------
    deduped: List[dict] = []
    for group in _group_by_sentence(rows).values():
        seen: Set[str] = set()
        for r in group:
            if r["fol"] in seen:
                continue
            seen.add(r["fol"])
            deduped.append(r)

    n_after = len(deduped)
    print(f"  Dedup: {n_before} -> {n_after} rows "
          f"({n_before - n_after} duplicates removed)")

    # -- (b) globally-unique FOLs (deterministic first-occurrence order) ---
    unique_fols: List[str] = []
    seen_g: Set[str] = set()
    for r in deduped:
        if r["fol"] not in seen_g:
            seen_g.add(r["fol"])
            unique_fols.append(r["fol"])
    print(f"  Unique FOLs (global): {len(unique_fols)}")

    # -- (c) load models ---------------------------------------------------
    from src.selectors.m4 import (
        _bert_embed, _cosine_similarity, _informalize,
        _load_bert, _load_few_shot_examples, _load_informalizer,
    )

    dataset_key = _DATASET_KEY_MAP.get(dataset, "malls")
    examples = _load_few_shot_examples(dataset_key)
    print(f"  Few-shot examples ({dataset_key}): {len(examples) if examples else 0}")

    print("  Loading informalization LLM ...", flush=True)
    _load_informalizer(model)
    print("  Loading BERT ...", flush=True)
    _load_bert()

    # -- (d) back-translate each globally-unique FOL once ------------------
    cache = _read_cache(model, dataset)
    print(f"  Back-translation cache: {len(cache)} entries loaded")

    n_new = 0
    t0 = time.perf_counter()
    for i, fol in enumerate(unique_fols):
        key = hashlib.md5(fol.encode()).hexdigest()
        if key in cache:
            continue
        back_nl = _informalize(fol, model_key=model, sentence_id=0,
                               examples=examples)
        cache[key] = back_nl if back_nl else fol
        n_new += 1
        if n_new % 10 == 0:
            _write_cache(model, dataset, cache)
            elapsed = time.perf_counter() - t0
            print(f"    [{i+1}/{len(unique_fols)}] "
                  f"new={n_new}  elapsed={elapsed:.1f}s", flush=True)

    _write_cache(model, dataset, cache)
    elapsed = time.perf_counter() - t0
    print(f"  Back-translation: {n_new} new, {len(cache)} total in cache "
          f"({elapsed:.1f}s)")

    # -- (e) compute backtrans_sim with embedding caches -------------------
    nl_emb_cache: Dict[int, np.ndarray] = {}
    back_emb_cache: Dict[str, np.ndarray] = {}

    n_sentences = len({r["sentence_id"] for r in deduped})
    out: List[dict] = []
    last_sid = None
    n_done = 0
    t0_phase = time.perf_counter()

    for r in deduped:
        sim = 0.0

        if r["nl"].strip():
            if r["sentence_id"] not in nl_emb_cache:
                nl_emb_cache[r["sentence_id"]] = _bert_embed(r["nl"])
            orig_emb = nl_emb_cache[r["sentence_id"]]

            key = hashlib.md5(r["fol"].encode()).hexdigest()
            back_nl = cache.get(key, r["fol"])
            if back_nl and back_nl != r["fol"]:
                if key not in back_emb_cache:
                    back_emb_cache[key] = _bert_embed(back_nl)
                sim = _cosine_similarity(orig_emb, back_emb_cache[key])

        # Drop nl/gt_fol/fol — keep only what M6 Stage 2 consumes
        out.append({
            "sentence_id": r["sentence_id"],
            "candidate_idx": r["candidate_idx"],
            "n_unique": r["n_unique"],
            "z3_le_count": r["z3_le_count"],
            "mean_bleu": r["mean_bleu"],
            "mean_bertscore": r["mean_bertscore"],
            "backtrans_sim": round(sim, 6),
        })

        if r["sentence_id"] != last_sid:
            last_sid = r["sentence_id"]
            n_done += 1
            if n_done % 20 == 0 or n_done == n_sentences:
                elapsed = time.perf_counter() - t0_phase
                print(f"    [{n_done}/{n_sentences}] sentences  "
                      f"elapsed={elapsed:.0f}s", flush=True)

    _save_rows(dst, out)
    print(f"  Saved -> {dst}  ({len(out)} rows)")
    return dst


# ======================================================================
# Orchestrator + CLI
# ======================================================================

_STEP_FNS = {
    1: pw_step1_z3_filter,
    2: pw_step2_n_unique,
    3: pw_step3_z3_le_count,
    4: pw_step4_mean_bleu,
    5: pw_step5_mean_bertscore,
    6: pw_step6_dedup_backtrans,
}

_STEP_NAMES = {
    1: "Z3 filter",
    2: "n_unique",
    3: "z3_le_count",
    4: "mean_bleu",
    5: "mean_bertscore",
    6: "dedup + backtrans_sim",
}


def run_pw_metrics(model: str = "Qwen4b", dataset: str = "test",
                   from_step: int = 1, to_step: int = 6,
                   force: bool = False) -> None:
    """Chain steps 1->6, each reading the previous step's output."""
    paths = _resolve_paths(model, dataset)

    if from_step <= 1 <= to_step and not paths["k10_metrics"].exists():
        raise FileNotFoundError(
            f"Source k10 metrics not found: {paths['k10_metrics']}\n"
            f"  Run k10 generation / inference first."
        )

    print("=" * 60)
    print("  PW-METRICS PIPELINE")
    print(f"  Model:       {model}")
    print(f"  Dataset:     {dataset}")
    print(f"  Steps:       {from_step} -> {to_step}")
    if force:
        print("  Force:       YES (overwrite existing)")
    print("=" * 60)
    print()

    t0 = time.perf_counter()

    for step in range(from_step, to_step + 1):
        print(f"--- Step {step}: {_STEP_NAMES[step]} ---")
        _STEP_FNS[step](model=model, dataset=dataset, force=force)
        print()

    elapsed = (time.perf_counter() - t0) / 60
    print("=" * 60)
    print(f"  PIPELINE COMPLETE - {elapsed:.1f} min")

    final = _pw_step_path(model, dataset, 6)
    if final.exists():
        rows = _load_rows(final)
        print(f"  Final rows: {len(rows)}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="PW-Metrics pipeline (steps 1-6)")
    ap.add_argument("--model", default="Qwen4b", help="Model key: Qwen4b or Qwen8b")
    ap.add_argument("--dataset", default="test",
                    help="Dataset: test | test_folio | test_willow")
    ap.add_argument("--step", type=int, choices=list(range(1, 7)), default=None,
                    help="Run only this one step (1-6).")
    ap.add_argument("--from-step", type=int, default=1)
    ap.add_argument("--to-step", type=int, default=6)
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if output already exists")
    args = ap.parse_args()

    if args.step is not None:
        print(f"--- Step {args.step}: {_STEP_NAMES[args.step]} ---")
        _STEP_FNS[args.step](model=args.model, dataset=args.dataset, force=args.force)
    else:
        run_pw_metrics(model=args.model, dataset=args.dataset,
                       from_step=args.from_step, to_step=args.to_step,
                       force=args.force)
