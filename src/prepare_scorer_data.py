"""
Scorer Training Data Generator (Multi-Step Pipeline)
=====================================================

Step-by-step pipeline to generate numeric feature vectors for m6 LR scorer
training.  Each step reads the previous step's output, adds one feature, and
saves its own JSON.  Steps are independently runnable via CLI.

Pipeline
--------
  Step 1 – Z3 filter        keep only Z3-parsable candidates
  Step 2 – n_unique         unique FOL count per sentence; drop n=1 sentences
  Step 3 – z3_le_count      pairwise Z3 LE with other filtered candidates
  Step 4 – mean_bleu        mean pairwise FOL-token BLEU  (filtered only)
  Step 5 – mean_bertscore   mean pairwise BertScore F1     (filtered only)
  Step 6 – dedup + backtrans  dedup per sentence, back-translate, cosine sim
  Step 7 – z3_le_label      Z3 LE with ground-truth FOL (final label)

Output schema (final)
---------------------
  sentence_id    : int      – index of the sentence in the dataset
  candidate_idx  : int      – original index of the candidate in the k10 pool
  n_unique       : int      – unique FOL count among Z3-filtered candidates
  z3_le_count    : int      – # of other filtered candidates Z3-equivalent
  mean_bleu      : float    – mean pairwise FOL-token BLEU
  mean_bertscore : float    – mean pairwise BertScore F1
  backtrans_sim  : float    – BERT cosine similarity (NL vs back-translated NL)
  z3_le_label    : int (0/1)– Z3 LE with ground-truth FOL

Usage
-----
    # Full pipeline
    python src/prepare_scorer_data.py --model Qwen4b

    # Single step
    python src/prepare_scorer_data.py --step 4 --model Qwen4b

    # Range
    python src/prepare_scorer_data.py --model Qwen4b --from-step 3 --to-step 6

    # Force recompute
    python src/prepare_scorer_data.py --step 6 --model Qwen4b --force
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
_ROOT = Path(__file__).resolve().parent.parent  # project/code/

# -- constants -------------------------------------------------------------
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


def _k10_dir(model: str) -> Path:
    return _ROOT / "data" / "results" / model / "k10"


def _step_path(model: str, step: int) -> Path:
    """step 0 → k10_val source;  step 7 → final scorer;  else step{N}.json."""
    if step == 0:
        return _k10_dir(model) / f"{model}_k10_val.json"
    if step == 7:
        return _k10_dir(model) / f"{model}_val_scorer.json"
    return _k10_dir(model) / f"{model}_val_scorer_step{step}.json"


def _cache_path(model: str) -> Path:
    return _k10_dir(model) / f"{model}_backtrans_cache.json"


def _load_rows(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_rows(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def _group_by_sentence(rows: List[dict]) -> Dict[int, List[dict]]:
    """Group rows by sentence_id, preserving insertion order."""
    groups: Dict[int, List[dict]] = {}
    for r in rows:
        groups.setdefault(r["sentence_id"], []).append(r)
    return groups


def _read_cache(model: str) -> Dict[str, str]:
    path = _cache_path(model)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _write_cache(model: str, cache: Dict[str, str]) -> None:
    path = _cache_path(model)
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
    """Check if two FOL strings are Z3-logically equivalent.

    Returns a real ``bool`` (uses ``.is_equivalent`` on the EquivResult),
    NOT the EquivResult object itself — the dataclass has no ``__bool__``
    and would be truthy for every call.
    """
    from src.eval.z3_equiv import check_equivalence
    try:
        return check_equivalence(fol_a, fol_b, timeout_ms=10_000).is_equivalent
    except Exception:
        return False


# ======================================================================
# Back-translation (reuse m4 internals)
# ======================================================================


def _load_m4_modules():
    """Late-import m4 internal helpers to avoid circular / heavy imports."""
    from src.selectors.m4 import (
        _bert_embed,
        _cosine_similarity,
        _informalize,
        _load_bert,
        _load_few_shot_examples,
        _load_informalizer,
    )
    return (
        _bert_embed, _cosine_similarity,
        _informalize, _load_bert,
        _load_few_shot_examples, _load_informalizer,
    )


# ======================================================================
# Step 1 – Z3 filter
# ======================================================================


def step1_z3_filter(model: str = "Qwen4b", in_path: Optional[Path] = None,
                    force: bool = False) -> Path:
    """Keep only candidates whose FOL can be parsed by Z3.

    Reads ``{model}_k10_val.json``, writes ``{model}_val_scorer_step1.json``.
    """
    src = in_path or _step_path(model, 0)
    dst = _step_path(model, 1)

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


def step2_n_unique(model: str = "Qwen4b", in_path: Optional[Path] = None,
                   force: bool = False) -> Path:
    """Count unique FOL strings per sentence among Z3-filtered candidates.

    Drops sentences whose candidates collapse to a single unique FOL
    (n_unique == 1) — they carry no candidate diversity for scorer training.

    Reads step1, writes step2.
    """
    src = in_path or _step_path(model, 1)
    dst = _step_path(model, 2)

    if dst.exists() and not force:
        print(f"  [SKIP] {dst.name} already exists")
        return dst

    rows = _load_rows(src)
    groups = _group_by_sentence(rows)
    out: List[dict] = []
    n_dropped = 0

    for group in groups.values():
        n = len({r["fol"] for r in group if r["fol"]})
        if n == 1:
            n_dropped += 1
            continue
        out.extend([{**r, "n_unique": n} for r in group])

    _save_rows(dst, out)
    print(f"  n_unique added -> {dst}  ({len(out)} rows, "
          f"{n_dropped} n=1 sentences dropped)")
    return dst


# ======================================================================
# Step 3 – z3_le_count
# ======================================================================


def step3_z3_le_count(model: str = "Qwen4b", in_path: Optional[Path] = None,
                      force: bool = False) -> Path:
    """Pairwise Z3 logical-equivalence check among filtered candidates.

    Per sentence: for N candidates, N*(N-1)/2 checks.  Counts are undirected
    (both sides incremented on equivalence).  Done BEFORE dedup.

    Reads step2, writes step3.
    """
    src = in_path or _step_path(model, 2)
    dst = _step_path(model, 3)

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


def step4_mean_bleu(model: str = "Qwen4b", in_path: Optional[Path] = None,
                    force: bool = False) -> Path:
    """Mean pairwise FOL-token BLEU among Z3-filtered candidates only.

    Computed BEFORE dedup.  Reads step3, writes step4.
    """
    src = in_path or _step_path(model, 3)
    dst = _step_path(model, 4)

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


def step5_mean_bertscore(model: str = "Qwen4b", in_path: Optional[Path] = None,
                         force: bool = False) -> Path:
    """Mean pairwise BertScore F1 among Z3-filtered candidates only.

    Computed BEFORE dedup.  Upper-triangle optimisation (F1 is symmetric).
    Reads step4, writes step5.
    """
    src = in_path or _step_path(model, 4)
    dst = _step_path(model, 5)

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


def step6_dedup_backtrans(model: str = "Qwen4b",
                          in_path: Optional[Path] = None,
                          force: bool = False) -> Path:
    """Deduplicate per sentence by FOL, then back-translate unique FOLs globally.

    - Per-sentence dedup: within each sentence, keep first occurrence per unique FOL.
    - Global back-translation: each unique FOL is informalized exactly once via LLM.
    - Persistent cache: ``{model}_backtrans_cache.json`` for crash-safe resume.
    - BERT cosine similarity between original NL embedding and back-translated NL.

    Reads step5, writes step6.
    """
    src = in_path or _step_path(model, 5)
    dst = _step_path(model, 6)

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
    (_bert_embed, _cosine_similarity, _informalize,
     _load_bert, _load_few_shot_examples, _load_informalizer) = _load_m4_modules()

    examples = _load_few_shot_examples("malls")
    print(f"  Few-shot examples: {len(examples) if examples else 0}")

    print("  Loading informalization LLM ...", flush=True)
    _load_informalizer(model)
    print("  Loading BERT ...", flush=True)
    _load_bert()

    # -- (d) back-translate each globally-unique FOL once ------------------
    cache = _read_cache(model)
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
        # incremental save every 10 new translations (crash-safe)
        if n_new % 10 == 0:
            _write_cache(model, cache)
            elapsed = time.perf_counter() - t0
            print(f"    [{i+1}/{len(unique_fols)}] "
                  f"new={n_new}  elapsed={elapsed:.1f}s", flush=True)

    _write_cache(model, cache)
    elapsed = time.perf_counter() - t0
    print(f"  Back-translation: {n_new} new, {len(cache)} total in cache "
          f"({elapsed:.1f}s)")

    # -- (e) compute backtrans_sim with embedding caches -------------------
    nl_emb_cache: Dict[int, np.ndarray] = {}
    back_emb_cache: Dict[str, np.ndarray] = {}

    # count unique sentences for progress
    n_sentences = len({r["sentence_id"] for r in deduped})

    out: List[dict] = []
    last_sid = None
    n_done = 0
    t0_phase = time.perf_counter()

    for r in deduped:
        sim = 0.0

        if r["nl"].strip():
            # NL embedding (cached per sentence_id)
            if r["sentence_id"] not in nl_emb_cache:
                nl_emb_cache[r["sentence_id"]] = _bert_embed(r["nl"])
            orig_emb = nl_emb_cache[r["sentence_id"]]

            # back-NL embedding (cached per FOL key)
            key = hashlib.md5(r["fol"].encode()).hexdigest()
            back_nl = cache.get(key, r["fol"])
            if back_nl and back_nl != r["fol"]:
                if key not in back_emb_cache:
                    back_emb_cache[key] = _bert_embed(back_nl)
                sim = _cosine_similarity(orig_emb, back_emb_cache[key])

        out.append({**r, "backtrans_sim": round(sim, 6)})

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
# Step 7 – z3_le_label
# ======================================================================


def step7_z3_le_label(model: str = "Qwen4b",
                      in_path: Optional[Path] = None,
                      force: bool = False) -> Path:
    """Z3 logical-equivalence check between each candidate FOL and ground-truth FOL.

    Drops ``nl``, ``gt_fol``, ``fol`` from output — restores the original
    8-field row schema expected by downstream training / inference.

    Reads step6, writes final ``{model}_val_scorer.json``.
    """
    src = in_path or _step_path(model, 6)
    dst = _step_path(model, 7)

    if dst.exists() and not force:
        print(f"  [SKIP] {dst.name} already exists")
        return dst

    rows = _load_rows(src)
    out: List[dict] = []

    # count unique sentences for progress
    n_sentences = len({r["sentence_id"] for r in rows})

    n_pos = 0
    last_sid = None
    n_done = 0
    t0 = time.perf_counter()

    for r in rows:
        label = 0
        if r.get("fol") and r.get("gt_fol"):
            try:
                label = 1 if _z3_le_check(r["fol"], r["gt_fol"]) else 0
            except Exception:
                label = 0
        if label:
            n_pos += 1

        out.append({
            "sentence_id": r["sentence_id"],
            "candidate_idx": r["candidate_idx"],
            "n_unique": r["n_unique"],
            "z3_le_count": r["z3_le_count"],
            "mean_bleu": r["mean_bleu"],
            "mean_bertscore": r["mean_bertscore"],
            "backtrans_sim": r["backtrans_sim"],
            "z3_le_label": label,
        })

        if r["sentence_id"] != last_sid:
            last_sid = r["sentence_id"]
            n_done += 1
            if n_done % 100 == 0 or n_done == n_sentences:
                elapsed = time.perf_counter() - t0
                avg = elapsed / n_done
                eta = avg * (n_sentences - n_done)
                print(f"    [{n_done}/{n_sentences}] sentences  "
                      f"pos={n_pos}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                      flush=True)

    elapsed = time.perf_counter() - t0
    _save_rows(dst, out)

    print(f"  z3_le_label computed in {elapsed:.1f}s")
    print(f"  Positive labels: {n_pos}/{len(out)} "
          f"({100 * n_pos / max(len(out), 1):.1f}%)")
    print(f"  Saved -> {dst}")
    return dst


# ======================================================================
# Orchestrator
# ======================================================================

_STEP_FNS = {
    1: step1_z3_filter,
    2: step2_n_unique,
    3: step3_z3_le_count,
    4: step4_mean_bleu,
    5: step5_mean_bertscore,
    6: step6_dedup_backtrans,
    7: step7_z3_le_label,
}

_STEP_NAMES = {
    1: "Z3 filter",
    2: "n_unique",
    3: "z3_le_count",
    4: "mean_bleu",
    5: "mean_bertscore",
    6: "dedup + backtrans_sim",
    7: "z3_le_label",
}


def run_scorer_data(model: str = "Qwen4b",
                    from_step: int = 1, to_step: int = 7,
                    force: bool = False) -> None:
    """Chain steps 1->7, each reading the previous step's output.

    Parameters
    ----------
    model : str
        Model key (``"Qwen4b"`` or ``"Qwen8b"``).
    from_step : int
        First step to run (1-7).
    to_step : int
        Last step to run (inclusive, >= from_step, <= 7).
    force : bool
        Recompute even if each step's output already exists.
    """
    # Validate source exists when step 1 is included
    if from_step <= 1 <= to_step:
        src = _step_path(model, 0)
        if not src.exists():
            raise FileNotFoundError(
                f"Source k10 file not found: {src}\n"
                f"  Run k10 generation first (e.g. generate_k10.py) "
                f"or provide a different model."
            )

    print("=" * 60)
    print("  SCORER TRAINING DATA PIPELINE")
    print(f"  Model:       {model}")
    print(f"  Steps:       {from_step} -> {to_step}")
    if force:
        print("  Force:       YES (overwrite existing)")
    print("=" * 60)
    print()

    t0 = time.perf_counter()

    for step in range(from_step, to_step + 1):
        print(f"--- Step {step}: {_STEP_NAMES[step]} ---")
        _STEP_FNS[step](model=model, force=force)
        print()

    elapsed = (time.perf_counter() - t0) / 60
    print("=" * 60)
    print(f"  PIPELINE COMPLETE - {elapsed:.1f} min")

    if to_step >= 7:
        final = _step_path(model, 7)
        if final.exists():
            rows = _load_rows(final)
            labels = [r["z3_le_label"] for r in rows]
            n_pos = sum(labels)
            print(f"  Final rows: {len(rows)}")
            print(f"  Positive labels (Z3 LE = 1): {n_pos}/{len(rows)} "
                  f"({100 * n_pos / max(len(rows), 1):.1f}%)")
    print("=" * 60)


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Scorer training data pipeline (steps 1-7)",
    )
    ap.add_argument("--model", default="Qwen4b",
                    help="Model key: Qwen4b or Qwen8b")
    ap.add_argument("--step", type=int, choices=list(range(1, 8)),
                    default=None,
                    help="Run only this one step (1-7).")
    ap.add_argument("--from-step", type=int, default=1,
                    help="First step when chaining (default: 1)")
    ap.add_argument("--to-step", type=int, default=7,
                    help="Last step when chaining (default: 7)")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if output already exists")
    args = ap.parse_args()

    if args.step is not None:
        # Single-step mode
        print(f"--- Step {args.step}: {_STEP_NAMES[args.step]} ---")
        _STEP_FNS[args.step](model=args.model, force=args.force)
    else:
        run_scorer_data(
            model=args.model,
            from_step=args.from_step,
            to_step=args.to_step,
            force=args.force,
        )
