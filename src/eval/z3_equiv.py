"""
Z3 logical equivalence checking.

Two input modes:
  1. FOLAST dict (preferred for generated formulas) → builds Z3 expression directly
  2. FOL string (for GT formulas or when AST unavailable) → parsed via fol_parser

    phi_gen = psi_gt   <=>  Z3.check(Not(phi_gen == psi_gt)) == UNSAT
"""

from __future__ import annotations
import time
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass

import z3

from .z3parser import fol_to_z3, FOLParseError


@dataclass
class EquivResult:
    """Result of a single equivalence check."""
    sentence_id: int
    candidate_id: str
    fol_generated: str
    fol_gt: str
    is_equivalent: bool
    parse_ok: bool
    parse_error: str = ""
    z3_time_ms: float = 0.0
    counter_model: Optional[Dict[str, bool]] = None

    @property
    def success(self) -> bool:
        return self.parse_ok and self.is_equivalent

    def summary(self) -> str:
        if not self.parse_ok:
            return f"PARSE_ERROR: {self.parse_error[:60]}"
        if self.is_equivalent:
            return f"EQUIVALENT ({self.z3_time_ms:.0f}ms)"
        return f"NOT EQUIVALENT ({self.z3_time_ms:.0f}ms)"


def check_equivalence(
    fol_gen: str,
    fol_gt: str,
    sentence_id: int = -1,
    candidate_id: str = "",
    timeout_ms: int = 30000,
    ast_gen: Optional[Dict[str, Any]] = None,
) -> EquivResult:
    """
    Check if a generated FOL is logically equivalent to ground truth.

    Uses FOLAST tree (ast_gen) when available for reliable Z3 conversion.
    Falls back to FOL string parsing for both.

    Args:
        fol_gen: Generated FOL string.
        fol_gt: Ground truth FOL string.
        sentence_id: Sentence identifier.
        candidate_id: Candidate identifier.
        timeout_ms: Z3 solver timeout.
        ast_gen: FOLAST dict from JSON (preferred for generated formula).

    Returns:
        EquivResult with verdict.
    """
    t0 = time.perf_counter()

    # Create a fresh Z3 context — ALL memory freed when this context is deleted
    ctx = z3.Context()

    try:
        if not fol_gen:
            raise ValueError("No FOL string available")
        phi_gen = fol_to_z3(fol_gen, timeout_ms, ctx=ctx)
    except (FOLParseError, ValueError, KeyError, AttributeError) as e:
        dt = (time.perf_counter() - t0) * 1000
        del ctx  # free Z3 memory
        return EquivResult(
            sentence_id=sentence_id, candidate_id=candidate_id,
            fol_generated=fol_gen, fol_gt=fol_gt,
            is_equivalent=False, parse_ok=False,
            parse_error=f"GEN parse error: {e}", z3_time_ms=dt,
        )

    # Build GT formula expression (within same ctx)
    try:
        phi_gt = fol_to_z3(fol_gt, timeout_ms, ctx=ctx)
    except FOLParseError as e:
        dt = (time.perf_counter() - t0) * 1000
        del ctx
        return EquivResult(
            sentence_id=sentence_id, candidate_id=candidate_id,
            fol_generated=fol_gen, fol_gt=fol_gt,
            is_equivalent=False, parse_ok=True,
            parse_error=f"GT parse error: {e}", z3_time_ms=dt,
        )

    # Check Not(phi_gen == phi_gt) — all within ctx
    s = z3.Solver(ctx=ctx)
    s.set("timeout", timeout_ms)
    s.add(z3.Not(phi_gen == phi_gt))

    result = s.check()
    dt = (time.perf_counter() - t0) * 1000

    if result == z3.unsat:
        equiv_result = EquivResult(
            sentence_id=sentence_id, candidate_id=candidate_id,
            fol_generated=fol_gen, fol_gt=fol_gt,
            is_equivalent=True, parse_ok=True, z3_time_ms=dt,
        )
    elif result == z3.sat:
        try:
            model = s.model()
            counter = {}
            for d in list(model.decls())[:10]:
                try:
                    counter[str(d)] = bool(model[d])
                except Exception:
                    pass
        except Exception:
            counter = None
        equiv_result = EquivResult(
            sentence_id=sentence_id, candidate_id=candidate_id,
            fol_generated=fol_gen, fol_gt=fol_gt,
            is_equivalent=False, parse_ok=True, z3_time_ms=dt,
            counter_model=counter,
        )
    else:
        equiv_result = EquivResult(
            sentence_id=sentence_id, candidate_id=candidate_id,
            fol_generated=fol_gen, fol_gt=fol_gt,
            is_equivalent=False, parse_ok=True, z3_time_ms=dt,
            parse_error=f"Z3 result: {result}",
        )

    # Free ALL Z3 C++ memory from this call
    del phi_gen, phi_gt, s
    del ctx
    return equiv_result


def check_equivalence_with_details(
    fol_gen: str, fol_gt: str,
    sentence_id: int = -1, candidate_id: str = "",
    timeout_ms: int = 30000,
    ast_gen: Optional[Dict[str, Any]] = None,
) -> EquivResult:
    """Alias for check_equivalence."""
    return check_equivalence(fol_gen, fol_gt, sentence_id, candidate_id, timeout_ms, ast_gen)


def batch_check_equivalence(
    pairs: List[Tuple[int, str, str, str, Optional[Dict[str, Any]]]],
    timeout_ms: int = 30000,
    verbose: bool = True,
) -> List[EquivResult]:
    """
    Batch-check multiple (sid, cid, fol_gen, fol_gt, ast_gen?) pairs.

    Args:
        pairs: List of (sentence_id, candidate_id, fol_gen, fol_gt, ast_gen_or_None).
        timeout_ms: Per-check timeout.
        verbose: Print progress every 50 checks.

    Returns:
        List of EquivResult in input order.
    """
    results = []
    total = len(pairs)
    for i, item in enumerate(pairs):
        if len(item) == 5:
            sid, cid, fol_gen, fol_gt, ast_gen = item
        else:
            sid, cid, fol_gen, fol_gt = item
            ast_gen = None

        r = check_equivalence(fol_gen, fol_gt, sid, cid, timeout_ms, ast_gen)
        results.append(r)
        if verbose and (i + 1) % 50 == 0:
            ok = sum(1 for r2 in results if r2.success)
            print(f"  [{i+1}/{total}] LE Rate so far: {ok}/{i+1} ({100*ok/(i+1):.1f}%)")
    return results
