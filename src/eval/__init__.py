"""
Evaluation module for NL-to-FOL translation.

Provides:
  - FOL string → Z3 expression parsing (z3parser)
  - Z3 logical equivalence checking (z3_equiv)
  - 7 evaluation metrics (eval)

Usage (from any notebook):
    from src.eval import (
        check_equivalence, batch_check_equivalence,
        fol_to_z3, normalize_fol,
        compute_all_metrics, batch_compute,
    )
"""

from .z3_equiv import (
    check_equivalence,
    check_equivalence_with_details,
    batch_check_equivalence,
    EquivResult,
)
from .z3parser import fol_to_z3, normalize_fol, FOLParseError
from .eval import compute_all_metrics, batch_compute

__all__ = [
    # Z3 equivalence
    "check_equivalence", "check_equivalence_with_details",
    "batch_check_equivalence", "EquivResult",
    # Parsers
    "fol_to_z3", "normalize_fol", "FOLParseError",
    # AlignmentFOL metrics
    "compute_all_metrics", "batch_compute",
]
