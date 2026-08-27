"""
Prepare FOLIO Test Set — NL-to-FOL Translation Project
========================================================

Downloads folio_parsed.json from HuggingFace (MALLS-v0 dataset),
deduplicates NL-FOL pairs by NL fingerprint, samples 1,000 entries,
and saves as test_folio.json.

Usage:
    python src/prepare_folio.py

Output:
    project/code/data/test_folio.json — up to 1,000 non-overlapping NL-FOL pairs
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Set

import requests

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project/code/
DATA_DIR = PROJECT_ROOT / "data"

HF_URL = "https://huggingface.co/datasets/yuan-yang/MALLS-v0/resolve/main/folio_parsed.json"
SEED = 42
TARGET_SIZE = 1_000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def nl_fingerprint(nl_text: str) -> str:
    """Normalised NL hash for dedup."""
    key = nl_text.strip().rstrip(".").strip().lower()
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def download_dataset(url: str) -> List[dict]:
    """Download and parse the FOLIO parsed JSON from HuggingFace."""
    print(f"  Downloading: {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")
    return data


def save_json(data: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  [OK] {path.name:<20s} {len(data):>6,d} entries  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    random.seed(SEED)

    # 1. Download dataset -----------------------------------------------------------
    print("\n[1/3] Downloading FOLIO parsed dataset from HuggingFace ...")
    entries = download_dataset(HF_URL)
    print(f"  Downloaded: {len(entries):,} entries")

    # 2. Deduplicate by NL fingerprint ----------------------------------------------
    print(f"\n[2/3] Deduplicating by NL fingerprint ...")
    seen: Set[str] = set()
    unique_pairs: List[dict] = []
    n_dup = 0

    for entry in entries:
        nl = entry.get("NL", "").strip()
        fol = entry.get("FOL", "").strip()
        if not nl or not fol:
            continue
        fp = nl_fingerprint(nl)
        if fp in seen:
            n_dup += 1
        else:
            seen.add(fp)
            unique_pairs.append({"NL": nl, "FOL": fol})

    total_pairs = len(unique_pairs) + n_dup
    print(f"  Valid pairs:         {total_pairs:>10,d}")
    print(f"  Duplicates removed:  {n_dup:>10,d}")
    print(f"  Unique pairs:        {len(unique_pairs):>10,d}")

    # 3. Shuffle and sample ---------------------------------------------------------
    print(f"\n[3/3] Shuffling (seed={SEED}) and sampling {TARGET_SIZE:,} ...")
    random.shuffle(unique_pairs)

    if len(unique_pairs) < TARGET_SIZE:
        print(f"  ⚠ Only {len(unique_pairs)} unique pairs — using all.")
        sampled = unique_pairs
    else:
        sampled = unique_pairs[:TARGET_SIZE]

    output_path = DATA_DIR / "test_folio.json"
    save_json(sampled, output_path)

    # Sanity checks
    fingerprints = [nl_fingerprint(e["NL"]) for e in sampled]
    n_unique = len(set(fingerprints))
    print(f"  Unique NL fingerprints:  {n_unique:>6,d}")
    assert n_unique == len(sampled), "Duplicate found in output!"
    print(f"\n  [OK] Output: {output_path.resolve()}")


if __name__ == "__main__":
    main()
