"""
Prepare WillowNLtoFOL Dataset — NL-to-FOL Translation Project
===============================================================

Downloads the iedeveci/WillowNLtoFOL dataset from Hugging Face, extracts
NL-FOL pairs from all splits, deduplicates by NL fingerprint, and saves
the processed data.

The dataset is human-constructed (~16K pairs, Deveci 2024), complementary
to MALLS (GPT-4 generated, ~28K pairs).

Usage:
    python src/prepare_willow.py

Output:
    project/code/data/test_willow.json   — 1,000 sampled NL-FOL pairs
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project/code/
DATA_DIR = PROJECT_ROOT / "data"

SEED = 42
SAMPLE_SIZE = 1_000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def nl_fingerprint(nl_text: str) -> str:
    """Normalised NL hash for dedup."""
    key = nl_text.strip().rstrip(".").strip().lower()
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def save_json(data: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  [OK] {path.name:<25s} {len(data):>6,d} entries  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    random.seed(SEED)

    # 1. Download / load from Hugging Face ---------------------------------------
    print("\n[1/5] Loading iedeveci/WillowNLtoFOL from Hugging Face ...")
    print("       (this may take a minute on first run)")

    from datasets import load_dataset

    ds = load_dataset("iedeveci/WillowNLtoFOL")
    print(f"       Splits: {list(ds.keys())}")

    total_rows = 0
    for split_name, split_ds in ds.items():
        print(f"       {split_name:<15s} {len(split_ds):>8,d} rows")
        total_rows += len(split_ds)
    print(f"       {'─' * 30}")
    print(f"       {'Total':<15s} {total_rows:>8,d} rows")

    # 2. Extract NL-FOL pairs ----------------------------------------------------
    print(f"\n[2/5] Extracting NL-FOL pairs ...")

    raw_pairs: List[Tuple[str, str]] = []
    missing = 0

    for split_name, split_ds in ds.items():
        for row in split_ds:
            nl = (row.get("NL_sentence") or "").strip()
            fol = (row.get("FOL_expression") or "").strip()
            if nl and fol:
                raw_pairs.append((nl, fol))
            else:
                missing += 1

    print(f"       Raw pairs extracted:  {len(raw_pairs):>14,d}")
    print(f"       Missing NL or FOL:    {missing:>14,d}")

    # 3. Deduplicate by NL fingerprint -------------------------------------------
    print(f"\n[3/5] Deduplicating by NL fingerprint ...")

    seen: Set[str] = set()
    unique_pairs: List[dict] = []
    n_dup = 0

    for nl_text, fol_text in raw_pairs:
        fp = nl_fingerprint(nl_text)
        if fp in seen:
            n_dup += 1
        else:
            seen.add(fp)
            unique_pairs.append({"NL": nl_text, "FOL": fol_text})

    print(f"       Duplicates removed:   {n_dup:>14,d}")
    print(f"       Unique pairs:         {len(unique_pairs):>14,d}")
    if len(raw_pairs) > 0:
        print(f"       Dedup ratio:          {len(unique_pairs) / len(raw_pairs) * 100:>13.1f}%")

    # 4. Check overlap with existing MALLS test set ------------------------------
    print(f"\n[4/5] Checking overlap with MALLS test set ...")
    malls_test_path = DATA_DIR / "test.json"
    n_overlap = 0
    if malls_test_path.exists():
        with open(malls_test_path, "r", encoding="utf-8") as f:
            malls_test = json.load(f)
        malls_fps: Set[str] = {nl_fingerprint(e["NL"]) for e in malls_test}
        n_overlap = sum(1 for p in unique_pairs if nl_fingerprint(p["NL"]) in malls_fps)
        print(f"       MALLS test entries:   {len(malls_test):>14,d}")
        print(f"       Overlap with MALLS:   {n_overlap:>14,d}")
    else:
        print(f"       MALLS test.json not found — skipping overlap check.")

    # 5. Shuffle and save --------------------------------------------------------
    print(f"\n[5/5] Shuffling (seed={SEED}) and saving ...")
    random.shuffle(unique_pairs)

    # Save 1K sample
    sampled = unique_pairs[:SAMPLE_SIZE]
    save_json(sampled, DATA_DIR / "test_willow.json")

    # 6. Summary ------------------------------------------------------------------
    print(f"\n{'='*55}")
    print("  SUMMARY")
    print(f"{'='*55}")
    print(f"  Total rows in WillowNLtoFOL:   {total_rows:>10,d}")
    print(f"  Raw NL-FOL pairs:              {len(raw_pairs):>10,d}")
    print(f"  Unique after NL dedup:         {len(unique_pairs):>10,d}")
    print(f"  Overlap with MALLS test:       {n_overlap:>10,d}")
    print(f"  ─────────────────────────────────────")
    print(f"  Human-constructed (Deveci 2024) — complementary to MALLS (GPT-4).")
    print(f"  Output: {DATA_DIR.resolve() / 'test_willow.json'}")


if __name__ == "__main__":
    main()
