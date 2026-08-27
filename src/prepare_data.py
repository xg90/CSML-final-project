"""
Data Preparation Script — NL-to-FOL Translation Project
========================================================

Produces four non-overlapping JSON files from MALLS-v0.1:

    test.json     1,000   MALLS-v0.1-test (human-verified, held-out)
    train.json    18,000  Training set (SFT)
    val.json      2,000   Validation set
    remain.json   ~7,000  Unused remainder

Key constraints:
    - No entry appears in more than one split.
    - test.json is the official MALLS human-verified test set — it MUST NOT
      overlap with train / val / remain.

Usage:
    python src/prepare_data.py

Output directory: project/code/data/
"""

from __future__ import annotations

import json
import hashlib
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Set

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project/code/
DATA_DIR = PROJECT_ROOT / "data"

# Hugging Face raw URL for MALLS-v0.1
MALLS_TEST_URL = (
    "https://huggingface.co/datasets/yuan-yang/MALLS-v0/resolve/main/"
    "MALLS-v0.1-test.json"
)
MALLS_TRAIN_URL = (
    "https://huggingface.co/datasets/yuan-yang/MALLS-v0/resolve/main/"
    "MALLS-v0.1-train.json"
)

SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nl_fingerprint(nl_text: str) -> str:
    """Normalised NL hash for dedup — strips trailing punctuation + whitespace."""
    key = nl_text.strip().rstrip(".").strip().lower()
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def load_json(path: str | Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(data: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  [OK] {path.name:<20s} {len(data):>6,d} entries  ({size_mb:.1f} MB)")


def download_malls_test(url: str, dest: Path) -> List[dict]:
    """Download MALLS-v0.1-test from Hugging Face, caching locally."""
    if dest.exists():
        print(f"  Using cached {dest.name} ({dest.stat().st_size / 1024:.0f} KB)")
        return load_json(dest)

    print(f"  Downloading MALLS-v0.1-test from Hugging Face ...")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved to {dest}")
        return load_json(dest)
    except Exception:
        print("  Falling back to Hugging Face datasets library ...")
        from datasets import load_dataset
        ds = load_dataset("yuan-yang/MALLS-v0", split="test")
        data = [{"NL": ex["NL"], "FOL": ex["FOL"]} for ex in ds]
        save_json(data, dest)
        return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load / download MALLS-v0.1-test (1K human-verified) -----------------
    print("\n[1/5] Loading MALLS-v0.1-test (human-verified 1K) ...")
    malls_test_raw_path = DATA_DIR / "_malls_test_raw.json"
    test_data = download_malls_test(MALLS_TEST_URL, malls_test_raw_path)
    print(f"       {len(test_data)} entries")

    # Normalise format — keep only NL + FOL
    test_entries: List[dict] = []
    for ex in test_data:
        test_entries.append({
            "NL": ex["NL"].strip(),
            "FOL": ex["FOL"].strip(),
            "split": "test",
            "source": "MALLS-v0.1-test",
        })

    # Build fingerprint set of test entries for dedup
    test_fps: Set[str] = {nl_fingerprint(ex["NL"]) for ex in test_entries}
    print(f"       {len(test_fps)} unique NL fingerprints")

    # 2. Load / download MALLS-v0.1-train (~27K) --------------------------------
    print(f"\n[2/5] Loading MALLS-v0.1-train ...")
    malls_train_raw_path = DATA_DIR / "_malls_train_raw.json"
    train_data = download_malls_test(MALLS_TRAIN_URL, malls_train_raw_path)
    print(f"       {len(train_data)} entries")

    # Remove entries whose NL fingerprint overlaps with test set.
    # Also deduplicate within the pool.
    seen_fps: Set[str] = set()
    pool: List[dict] = []
    n_overlap = 0
    n_dup = 0
    for ex in train_data:
        fp = nl_fingerprint(ex["NL"])
        if fp in test_fps:
            n_overlap += 1
        elif fp in seen_fps:
            n_dup += 1
        else:
            seen_fps.add(fp)
            pool.append({
                "NL": ex["NL"].strip(),
                "FOL": ex["FOL"].strip(),
                "split": None,
                "source": "MALLS-v0.1-train",
            })
    print(f"       {n_overlap} entries removed (overlap with test set)")
    print(f"       {n_dup} entries removed (duplicates within pool)")
    print(f"       {len(pool)} entries in pool after dedup")

    # 3. Shuffle the pool ---------------------------------------------------------------
    print(f"\n[3/5] Shuffling pool (seed={SEED}) ...")
    random.shuffle(pool)

    # 4. Assign splits ------------------------------------------------------------------
    print(f"\n[4/5] Assigning splits ...")

    N_TRAIN = 18_000
    N_VAL = 2_000

    total_needed = N_TRAIN + N_VAL
    if len(pool) < total_needed:
        raise RuntimeError(
            f"Pool size ({len(pool)}) < needed ({total_needed}). "
            f"Reduce split sizes or add more data."
        )

    idx = 0

    # train
    for i in range(N_TRAIN):
        pool[idx]["split"] = "train"
        idx += 1
    print(f"  train:    {N_TRAIN:>6,d}  [pool[{0}:{idx}]]")

    # val
    start = idx
    for i in range(N_VAL):
        pool[idx]["split"] = "val"
        idx += 1
    print(f"  val:      {N_VAL:>6,d}  [pool[{start}:{idx}]]")

    # remain
    start = idx
    n_remain = len(pool) - idx
    for i in range(idx, len(pool)):
        pool[i]["split"] = "remain"
    print(f"  remain:   {n_remain:>6,d}  [pool[{start}:{len(pool)}]]")

    # 5. Write output files -------------------------------------------------------------
    print(f"\n[5/5] Writing output files to {DATA_DIR}/ ...")

    split_map: Dict[str, List[dict]] = {
        "test": test_entries,
        "train": [e for e in pool if e["split"] == "train"],
        "val":   [e for e in pool if e["split"] == "val"],
        "remain": [e for e in pool if e["split"] == "remain"],
    }

    for name in ["test", "train", "val", "remain"]:
        entries = split_map[name]
        # Remove internal 'split' key for cleaner output
        output = [{"NL": e["NL"], "FOL": e["FOL"]} for e in entries]
        save_json(output, DATA_DIR / f"{name}.json")

    # 6. Sanity checks ------------------------------------------------------------------
    print(f"\n{'='*55}")
    print("  SANITY CHECKS")
    print(f"{'='*55}")

    all_fps: Dict[str, str] = {}  # fp → split name
    for name in ["test", "train", "val", "remain"]:
        for e in split_map[name]:
            fp = nl_fingerprint(e["NL"])
            if fp in all_fps:
                raise RuntimeError(
                    f"DUPLICATE: '{e['NL'][:60]}...' appears in both "
                    f"'{all_fps[fp]}' and '{name}'!"
                )
            all_fps[fp] = name

    total = sum(len(v) for v in split_map.values())
    print(f"  Total entries across all splits: {total:,}")
    print(f"  Unique NL fingerprints:         {len(all_fps):,}")
    print(f"  Overlaps:                       0")
    for name in ["train", "val", "remain"]:
        print(f"  {name}: {len(split_map[name]):>6,d} entries")

    print(f"\n  [OK] All checks passed.")
    print(f"  Data directory: {DATA_DIR.resolve()}")


if __name__ == "__main__":
    main()
