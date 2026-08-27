"""
Prepare Few-Shot Examples for M5 Informalization
=================================================

For each dataset (MALLS, FOLIO, WILLOW), sample 5 (FOL, NL) pairs from
the full data pool, EXCLUDING entries that appear in the test set.

Output:
    project/code/data/few_shot_examples.json

    {
      "malls":  [["FOL1", "NL1"], ... x5],
      "folio":  [["FOL1", "NL1"], ... x5],
      "willow": [["FOL1", "NL1"], ... x5]
    }

Usage:
    python src/prepare_few_shot.py
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project/code/
DATA_DIR = PROJECT_ROOT / "data"

SEED = 42
N_EXAMPLES = 5

# HF sources
FOLIO_HF_URL = "https://huggingface.co/datasets/yuan-yang/MALLS-v0/resolve/main/folio_parsed.json"

OUTPUT_PATH = DATA_DIR / "few_shot_examples.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def nl_fingerprint(nl_text: str) -> str:
    """Normalised NL hash for dedup (same as prepare_folio.py / prepare_willow.py)."""
    key = nl_text.strip().rstrip(".").strip().lower()
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def load_json(path: Path) -> List[dict]:
    """Load a JSON array file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_pairs(entries: List[dict]) -> List[Tuple[str, str]]:
    """Extract (FOL, NL) pairs from a list of {NL, FOL} dicts, skipping empties."""
    pairs: List[Tuple[str, str]] = []
    for e in entries:
        nl = (e.get("NL") or "").strip()
        fol = (e.get("FOL") or "").strip()
        if nl and fol:
            pairs.append((fol, nl))
    return pairs


def build_exclude_set(test_entries: List[dict]) -> Set[str]:
    """Build a set of FOL strings to exclude (from test data)."""
    exclude: Set[str] = set()
    for e in test_entries:
        fol = (e.get("FOL") or "").strip()
        if fol:
            exclude.add(fol)
    return exclude


def deduplicate_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Deduplicate by NL fingerprint, keeping first occurrence."""
    seen: Set[str] = set()
    result: List[Tuple[str, str]] = []
    for fol, nl in pairs:
        fp = nl_fingerprint(nl)
        if fp not in seen:
            seen.add(fp)
            result.append((fol, nl))
    return result


def sample_examples(
    pool: List[Tuple[str, str]],
    exclude_fols: Set[str],
    n: int = N_EXAMPLES,
    seed: int = SEED,
) -> List[Tuple[str, str]]:
    """Sample n (FOL, NL) pairs from pool, excluding test FOLs."""
    # Filter out test FOLs
    filtered = [(fol, nl) for fol, nl in pool if fol not in exclude_fols]

    rng = random.Random(seed)
    rng.shuffle(filtered)

    k = min(n, len(filtered))
    return filtered[:k]


# ---------------------------------------------------------------------------
# Dataset-specific loaders
# ---------------------------------------------------------------------------


def prepare_malls() -> Optional[List[Tuple[str, str]]]:
    """MALLS: load train.json + val.json, exclude test.json entries."""
    train_path = DATA_DIR / "train.json"
    val_path = DATA_DIR / "val.json"
    test_path = DATA_DIR / "test.json"

    if not train_path.exists() or not val_path.exists():
        print("  [MALLS] train.json or val.json missing — skipping.")
        return None

    test_entries = load_json(test_path) if test_path.exists() else []
    exclude_fols = build_exclude_set(test_entries)

    train_pairs = extract_pairs(load_json(train_path))
    val_pairs = extract_pairs(load_json(val_path))
    pool = train_pairs + val_pairs

    # Deduplicate
    pool = deduplicate_pairs(pool)

    examples = sample_examples(pool, exclude_fols)
    print(f"  [MALLS] pool={len(pool):,}  test_excluded={len(exclude_fols):,}  "
          f"sampled={len(examples)}")
    return examples


def prepare_folio() -> Optional[List[Tuple[str, str]]]:
    """FOLIO: download folio_parsed.json from HF, exclude test_folio.json entries."""
    test_path = DATA_DIR / "test_folio.json"

    if not test_path.exists():
        print("  [FOLIO] test_folio.json missing — skipping.")
        return None

    # Download full FOLIO data from HuggingFace
    print("  [FOLIO] Downloading folio_parsed.json from HuggingFace ...")
    try:
        import requests
        resp = requests.get(FOLIO_HF_URL, timeout=120)
        resp.raise_for_status()
        raw = resp.json()
        if not isinstance(raw, list):
            print(f"  [FOLIO] Expected JSON array, got {type(raw).__name__} — skipping.")
            return None
        full_entries = raw
    except Exception as e:
        print(f"  [FOLIO] Download failed: {e} — skipping.")
        return None

    test_entries = load_json(test_path)
    exclude_fols = build_exclude_set(test_entries)

    pool = extract_pairs(full_entries)
    pool = deduplicate_pairs(pool)

    examples = sample_examples(pool, exclude_fols)
    print(f"  [FOLIO] pool={len(pool):,}  test_excluded={len(exclude_fols):,}  "
          f"sampled={len(examples)}")
    return examples


def prepare_willow() -> Optional[List[Tuple[str, str]]]:
    """WILLOW: load from HF iedeveci/WillowNLtoFOL, exclude test_willow.json entries."""
    test_path = DATA_DIR / "test_willow.json"

    if not test_path.exists():
        print("  [WILLOW] test_willow.json missing — skipping.")
        return None

    # Load full WillowNLtoFOL from HuggingFace
    print("  [WILLOW] Loading iedeveci/WillowNLtoFOL from HuggingFace ...")
    try:
        from datasets import load_dataset
        ds = load_dataset("iedeveci/WillowNLtoFOL")
    except Exception as e:
        print(f"  [WILLOW] Failed to load: {e} — skipping.")
        return None

    test_entries = load_json(test_path)
    exclude_fols = build_exclude_set(test_entries)

    raw_pairs: List[Tuple[str, str]] = []
    for split_name, split_ds in ds.items():
        for row in split_ds:
            nl = (row.get("NL_sentence") or "").strip()
            fol = (row.get("FOL_expression") or "").strip()
            if nl and fol:
                raw_pairs.append((fol, nl))

    pool = deduplicate_pairs(raw_pairs)

    examples = sample_examples(pool, exclude_fols)
    print(f"  [WILLOW] pool={len(pool):,}  test_excluded={len(exclude_fols):,}  "
          f"sampled={len(examples)}")
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 55)
    print("  Prepare Few-Shot Examples for M5 Informalization")
    print(f"  N per dataset: {N_EXAMPLES}  |  seed: {SEED}")
    print("=" * 55)
    print()

    results: Dict[str, List[Tuple[str, str]]] = {}

    # MALLS
    print("[1/3] MALLS — train.json + val.json, excluding test.json")
    malls = prepare_malls()
    if malls:
        results["malls"] = malls

    print()

    # FOLIO
    print("[2/3] FOLIO — HF folio_parsed.json, excluding test_folio.json")
    folio = prepare_folio()
    if folio:
        results["folio"] = folio

    print()

    # WILLOW
    print("[3/3] WILLOW — HF WillowNLtoFOL, excluding test_willow.json")
    willow = prepare_willow()
    if willow:
        results["willow"] = willow

    print()

    # Save
    if not results:
        print("[ERROR] No examples generated — nothing to save.")
        sys.exit(1)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("=" * 55)
    print(f"  Saved to: {OUTPUT_PATH}")
    for key, pairs in results.items():
        print(f"    {key}: {len(pairs)} examples")
        for i, (fol, nl) in enumerate(pairs):
            print(f"      [{i+1}] FOL: {fol[:80]}{'...' if len(fol) > 80 else ''}")
            print(f"          NL:  {nl[:80]}{'...' if len(nl) > 80 else ''}")
    print("=" * 55)


if __name__ == "__main__":
    main()
