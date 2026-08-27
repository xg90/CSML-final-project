"""
FOL Evaluation Metrics — 7 metrics.

    execution_rate         Z3 execution success rate (1 if FOL parses, 0 otherwise)
    exact_match            Whitespace-normalised string equality
    exact_match_pred_norm  Exact match after Levenshtein predicate alignment
    z3_le                  Z3-based logical equivalence (strict, 0/1)
    z3_le_pred_norm        Z3 LE after Levenshtein predicate alignment
    le                     Closeness truth-table LE — predicate-soft-binding, continuous [0,1]
    bleu                   FOL-tokenized BLEU (n-gram precision, 0~1)
    bleu_pred_norm         BLEU after Levenshtein predicate alignment
    bertscore              BERT embedding cosine F1 (RoBERTa-large, 0~1)
    bertscore_pred_norm    BertScore after Levenshtein predicate alignment

Dependencies:
    pip install evaluate python-Levenshtein z3-solver bert_score

Usage:
    from src.eval.eval import compute_all_metrics, batch_compute

    scores = compute_all_metrics("∀x (Bird(x) → Fly(x))", "∀x (Bird(x) → CanFly(x))")
    print(scores["bleu"], scores["bertscore"], scores["z3_le"])
"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional

try:
    from Levenshtein import distance as levenshtein_dist
except ImportError:
    levenshtein_dist = None

from .z3_equiv import check_equivalence

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
_bertscore = None
_bleu = None
_FOL_TOKEN_PATTERN = re.compile(r"[∀∃]|\w+|¬|∧|∨|→|↔|⊕|[()]|=")


def _fol_tokenize(s: str):
    """Tokenize a FOL string into tokens for BLEU n-gram matching."""
    return _FOL_TOKEN_PATTERN.findall(s)


def _get_bertscore():
    global _bertscore
    if _bertscore is None:
        import evaluate
        _bertscore = evaluate.load("bertscore")
    return _bertscore


def _get_bleu():
    global _bleu
    if _bleu is None:
        import evaluate
        _bleu = evaluate.load("bleu")
    return _bleu


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_METRIC_KEYS = [
    "execution_rate",
    "exact_match", "exact_match_pred_norm",
    "z3_le", "z3_le_pred_norm",
    "bleu", "bleu_pred_norm",
    "bertscore", "bertscore_pred_norm",
]


def align_predicates(gen_fol: str, gt_fol: str, max_dist: float = 0.6) -> str:
    """Map generated predicate names to GT predicate names via Levenshtein ratio."""
    if not gen_fol or not gt_fol:
        return gen_fol or ""
    aligned = str(gen_fol)
    gen_preds = re.findall(r"\b\w+(?=\()", aligned)
    gt_preds = re.findall(r"\b\w+(?=\()", str(gt_fol))
    if not gen_preds or not gt_preds:
        return aligned
    if levenshtein_dist is None:
        return aligned
    for gp in sorted(set(gen_preds), key=len, reverse=True):
        best = min(gt_preds, key=lambda p: levenshtein_dist(gp, p) / max(len(gp), len(p)))
        if levenshtein_dist(gp, best) / max(len(gp), len(best)) <= max_dist:
            aligned = aligned.replace(gp + "(", best + "(")
    return aligned


def compute_all_metrics(
    pred_fol: str,
    gt_fol: str,
    *,
    z3_timeout_ms: int = 10_000,
    pred_align_max_dist: float = 0.6,
    include_bleu: bool = True,
    include_bertscore: bool = True,
) -> Dict[str, float]:
    """Compute evaluation metrics between a generated FOL and ground-truth.

    Args:
        pred_fol: Generated FOL formula.
        gt_fol:   Ground-truth FOL formula.
        z3_timeout_ms: Timeout for Z3 checks (default 10s).
        pred_align_max_dist: Levenshtein threshold for predicate alignment.
        include_bleu: If False, skip BLEU computation (bleu/bleu_pred_norm = 0).
        include_bertscore: If False, skip BertScore computation
            (bertscore/bertscore_pred_norm = 0).

    Returns:
        Dict with all metric keys.  All values in [0, 1].
    """
    pred = str(pred_fol or "").strip()
    gt = str(gt_fol or "").strip()

    # ---- Predicate-aligned FOL ----
    pred_aligned = align_predicates(pred, gt, max_dist=pred_align_max_dist)

    # ---- execution_rate: Z3 parse success ----
    execution_rate = 0.0
    try:
        from .z3parser import fol_to_z3
        import z3 as _z3
        _exec_ctx = _z3.Context()
        try:
            fol_to_z3(pred, z3_timeout_ms, ctx=_exec_ctx)
            execution_rate = 1.0
        finally:
            del _exec_ctx
    except Exception:
        pass

    # ---- le_z3 (strict) ----
    le_z3 = 0.0
    try:
        eq = check_equivalence(pred, gt, sentence_id=0, candidate_id="",
                               timeout_ms=z3_timeout_ms)
        le_z3 = 1.0 if eq.is_equivalent else 0.0
    except Exception:
        pass

    # ---- z3_le_pred_norm: Z3 LE after Levenshtein pred-alignment ----
    le_z3_pred_norm = le_z3
    if pred_aligned != pred:
        try:
            eq_aligned = check_equivalence(pred_aligned, gt, sentence_id=0,
                                           candidate_id="", timeout_ms=z3_timeout_ms)
            le_z3_pred_norm = 1.0 if eq_aligned.is_equivalent else 0.0
        except Exception:
            pass

    # ---- exact_match ----
    exact_match = 1.0 if _normalise(pred) == _normalise(gt) else 0.0

    # ---- exact_match (predicate-aligned) ----
    exact_match_pred_norm = 1.0 if _normalise(pred_aligned) == _normalise(gt) else 0.0

    # ---- bleu (FOL-tokenized n-gram precision) ----
    bleu_val = 0.0
    if include_bleu:
        try:
            bl = _get_bleu()
            pred_tokens = _fol_tokenize(pred)
            gt_tokens = _fol_tokenize(gt)
            min_len = min(len(pred_tokens), len(gt_tokens))
            res_bl = bl.compute(predictions=[pred], references=[[gt]],
                                tokenizer=_fol_tokenize, max_order=min(4, min_len))
            bleu_val = float(res_bl["bleu"] or 0.0)
        except Exception:
            pass

    # ---- bertscore ----
    bertscore_val = 0.0
    if include_bertscore:
        try:
            bs = _get_bertscore()
            res = bs.compute(predictions=[pred], references=[gt], lang="en")
            bertscore_val = float(res["f1"][0]) if "f1" in res else 0.0
        except Exception:
            pass

    # ---- bleu (predicate-aligned) ----
    bleu_pred_norm_val = bleu_val  # fallback
    if include_bleu and pred_aligned != pred:
        try:
            bl = _get_bleu()
            pred_tokens_norm = _fol_tokenize(pred_aligned)
            min_len_norm = min(len(pred_tokens_norm), len(gt_tokens))
            res_bl_norm = bl.compute(predictions=[pred_aligned], references=[[gt]],
                                     tokenizer=_fol_tokenize, max_order=min(4, min_len_norm))
            bleu_pred_norm_val = float(res_bl_norm["bleu"] or 0.0)
        except Exception:
            pass

    # ---- bertscore (predicate-aligned) ----
    bertscore_pred_norm_val = bertscore_val  # fallback
    if include_bertscore and pred_aligned != pred:
        try:
            bs = _get_bertscore()
            res_norm = bs.compute(predictions=[pred_aligned], references=[gt], lang="en")
            bertscore_pred_norm_val = float(res_norm["f1"][0]) if "f1" in res_norm else 0.0
        except Exception:
            pass

    return {
        "execution_rate":           execution_rate,
        "exact_match":              exact_match,
        "exact_match_pred_norm":    exact_match_pred_norm,
        "z3_le":                    le_z3,
        "z3_le_pred_norm":          le_z3_pred_norm,

        "bleu":                     bleu_val,
        "bleu_pred_norm":           bleu_pred_norm_val,
        "bertscore":                bertscore_val,
        "bertscore_pred_norm":      bertscore_pred_norm_val,
    }


def batch_compute(
    pairs: List[tuple],
    *,
    z3_timeout_ms: int = 10_000,
    verbose: bool = True,
) -> List[Dict[str, float]]:
    """Compute all 6 metrics for a list of (pred_fol, gt_fol) pairs.

    At batch level, execution_rate is the mean across all samples.
    """
    results = []
    t0 = time.perf_counter()
    for i, (pred, gt) in enumerate(pairs):
        r = compute_all_metrics(pred, gt, z3_timeout_ms=z3_timeout_ms)
        results.append(r)
        if verbose and (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            cum_exec_rate = sum(1 for r2 in results if r2["execution_rate"] > 0.5) / len(results) * 100
            print(f"  [{i+1}/{len(pairs)}] {elapsed:.1f}s  "
                  f"exec_rate={cum_exec_rate:.1f}%  "
                  f"({elapsed/(i+1):.2f}s/item)")
    return results


def _normalise(s: str) -> str:
    return re.sub(r"\s+", "", str(s))
