"""
K10 Multi-Candidate Analysis Module
====================================

Reusable analysis functions for k=10 FOL generation results.
Each function works across any model (Qwen4b/Qwen8b/Ministral8b)
and any dataset (test/test_folio/test_willow).

All functions are print-only — no return values.

Usage::

    from src.analysis_k10 import (
        analyze_pass_k_table,
        compare_k10_t0_table,
        plot_diversity_z3_distribution,
        plot_t0_vs_k10_quadrant_distributions,
        plot_methods_correct_heatmap,
    )

    analyze_pass_k_table(metric="z3_le_pred_norm")
    compare_k10_t0_table(metric="z3_le_pred_norm")
    plot_methods_correct_heatmap(model="Qwen8b", dataset="test")
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# -- project root ----------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent  # project/code/

# -- constants -------------------------------------------------------------
K = 10  # number of candidates

# ======================================================================
# Path helpers
# ======================================================================


def _build_model_tag(model: str, dataset: str, k10: bool = False) -> str:
    """Build the file tag like 'Qwen4b_k10_folio' or 'Qwen4b_folio'."""
    suffix = dataset.replace("test_", "").replace("test", "")
    if k10:
        return f"{model}_k10_{suffix}" if suffix else f"{model}_k10"
    else:
        return f"{model}_{suffix}" if suffix else model


def _derive_t0_metrics_path(k10_metrics_path: Path) -> Path:
    """Strip '_k10' from filename and go up one directory level.

    data/results/Qwen4b/k10/Qwen4b_k10_folio_metrics.json
    -> data/results/Qwen4b/Qwen4b_folio_metrics.json
    """
    parent = k10_metrics_path.parent.parent  # .../k10/ -> .../
    t0_filename = k10_metrics_path.name.replace("_k10", "")
    return parent / t0_filename


def _resolve_paths(
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    k10_data: Optional[str] = None,
    k10_metrics: Optional[str] = None,
    t0_metrics: Optional[str] = None,
    data_root: Optional[str] = None,
) -> Dict[str, Path]:
    """Resolve data paths from either explicit paths or model+dataset strings.

    Precedence: explicit paths > model+dataset auto-resolution.
    """
    root = Path(data_root) if data_root else _ROOT
    results = root / "data" / "results"

    if model and dataset:
        results_dir = results / model
        k10_dir = results_dir / "k10"
        tag_k10 = _build_model_tag(model, dataset, k10=True)
        tag_t0 = _build_model_tag(model, dataset, k10=False)

        return {
            "results_dir": results_dir,
            "k10_dir": k10_dir,
            "k10_data_path": k10_dir / f"{tag_k10}.json",
            "k10_metrics_path": k10_dir / f"{tag_k10}_metrics.json",
            "t0_metrics_path": results_dir / f"{tag_t0}_metrics.json",
        }
    else:
        resolved: Dict[str, Path] = {}
        if k10_data:
            p = Path(k10_data)
            resolved["k10_data_path"] = p
            resolved["k10_dir"] = p.parent
            resolved["results_dir"] = p.parent.parent
        if k10_metrics:
            p = Path(k10_metrics)
            resolved["k10_metrics_path"] = p
            resolved["k10_dir"] = p.parent
            resolved["results_dir"] = p.parent.parent
            resolved["t0_metrics_path"] = _derive_t0_metrics_path(p)
        if t0_metrics:
            resolved["t0_metrics_path"] = Path(t0_metrics)

        return resolved


# ======================================================================
# Cell 8/9: Pass@K upper-bound table across model × dataset combos
# ======================================================================

DEFAULT_MODELS = ["Qwen4b", "Qwen8b", "Ministral8b"]
DEFAULT_DATASETS = ["test", "test_folio", "test_willow"]


def _metric_k_upper_means(
    data: List[dict], key: str, ks: List[int],
    extra_hits: Optional[set] = None,
) -> Dict[int, float]:
    """Mean over sentences of max(key) across the first K candidates, per K.

    ``extra_hits``: optional set of (sentence_id, selected_idx) where a
    selector's m-file records its chosen candidate as correct (see
    ``_load_selector_hits``).  A sentence whose selector picked a correct
    candidate within the first K counts as a first-K hit even when the stored
    pool metric disagrees (pool and m-file metrics come from separate
    processes).  With no ``extra_hits`` this is the plain oracle upper bound.
    """
    means: Dict[int, float] = {}
    for kv in ks:
        per_sent = []
        for sent_entry in data:
            sid = sent_entry["sentence_id"]
            vals = [
                c["metrics"].get(key, 0.0)
                for c in sent_entry["candidates"][:kv]
                if c.get("fol") and "metrics" in c
            ]
            v = max(vals) if vals else 0.0
            if extra_hits and any((sid, idx) in extra_hits for idx in range(kv)):
                v = max(v, 1.0)
            per_sent.append(v)
        means[kv] = float(np.mean(per_sent))
    return means


def analyze_pass_k_table(
    *,
    metric: str,
    models: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    k_min: int = 3,
    k_max: int = 10,
    data_root: Optional[str] = None,
) -> None:
    """Pass@K upper-bound table across model × dataset combinations.

    One row per model+dataset combo, one column per K value.  Each cell is
    the mean over sentences of the max of ``metric`` across the first K
    candidates (oracle upper bound), formatted as a percentage.  A sentence
    whose selector's m-file records its chosen candidate as correct within
    the first K also counts as a hit (union), keeping this table consistent
    with the heatmap UB and the K10-vs-T0 quadrants.

    ``metric`` is a metric key, e.g. ``"execution_rate"`` or
    ``"z3_le_pred_norm"``.  Missing result files are printed as "n/a".
    """
    models = models or DEFAULT_MODELS
    datasets = datasets or DEFAULT_DATASETS
    ks = list(range(k_min, k_max + 1))
    col_w = 24 + 8 * len(ks)

    print("=" * col_w)
    print(f"  PASS@K UPPER BOUND — {metric}")
    print("=" * col_w)
    print()

    header = f"  {'Model / Dataset':<24s}"
    for kv in ks:
        header += f"  {'K=' + str(kv):>6s}"
    print(header)
    print("  " + "-" * (col_w - 2))

    for model in models:
        for dataset in datasets:
            label = f"{model} / {dataset}"
            paths = _resolve_paths(model=model, dataset=dataset, data_root=data_root)
            metrics_file = paths["k10_metrics_path"]

            if not metrics_file.exists():
                row = f"  {label:<24s}"
                for _ in ks:
                    row += f"  {'n/a':>6s}"
                print(row)
                continue

            with open(metrics_file, encoding="utf-8") as f:
                data = json.load(f)
            # Union the selectors' m-file hits into the oracle, so the pass@K
            # table agrees with the heatmap's UB and the K10-vs-T0 quadrants.
            means = _metric_k_upper_means(
                data, metric, ks,
                extra_hits=_load_selector_hits(model, dataset, metric),
            )

            row = f"  {label:<24s}"
            for kv in ks:
                row += f"  {100 * means[kv]:>5.1f}%"
            print(row)

    print()
    print(f"  Upper bound = mean over sentences of max({metric}) "
          f"across the first K candidates, unioned with the selectors' "
          f"m-file hits (pool and m-file metrics may disagree).")


# ======================================================================
# Cell 10: Z3-executable diversity distribution — bar chart
# ======================================================================


def plot_diversity_z3_distribution(
    *,
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    k10_metrics_path: Optional[str] = None,
    data_root: Optional[str] = None,
) -> None:
    """Bar chart of the Z3-executable unique-count distribution (cell 5).

    Per sentence, counts unique FOL strings among candidates with
    ``execution_rate == 1.0`` (Z3-executable).
    x = unique count (0..10), y = number of sentences.

    Requires matplotlib (no GPU / no Z3 import needed).
    """
    import matplotlib.pyplot as plt

    paths = _resolve_paths(
        model=model, dataset=dataset,
        k10_metrics=k10_metrics_path, data_root=data_root,
    )
    metrics_file = paths["k10_metrics_path"]
    if not metrics_file.exists():
        raise FileNotFoundError(f"{metrics_file} not found")

    with open(metrics_file, encoding="utf-8") as f:
        data = json.load(f)

    per_sent_unique = []
    for r in data:
        exec_fols = [
            c.get("fol", "")
            for c in r["candidates"]
            if c.get("fol") and c.get("metrics", {}).get("execution_rate", 0.0) >= 1.0
        ]
        per_sent_unique.append(len(set(exec_fols)))

    n_sent = len(per_sent_unique)
    dist = Counter(per_sent_unique)
    ks = list(range(0, K + 1))
    counts = [dist.get(k, 0) for k in ks]

    # -- single-hue styling (dataviz reference palette, validated) --
    BAR = "#2a78d6"     # sequential blue, step 450
    SURFACE = "#fcfcfb"  # chart surface
    INK_SEC = "#52514e"  # secondary ink
    AXIS = "#c3c2b7"    # baseline / axis
    GRID = "#e1e0d9"    # hairline gridline

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    bars = ax.bar(ks, counts, color=BAR, edgecolor=SURFACE, linewidth=1.5)

    ax.set_xticks(ks)
    ax.set_xlabel("Unique Z3-executable FOLs per sentence (out of 10)",
                  color=INK_SEC)
    ax.set_ylabel("Sentences", color=INK_SEC)

    # recessive horizontal grid, behind bars
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    # recessive chrome: only left + bottom axes, muted
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AXIS)

    ax.tick_params(colors=INK_SEC)

    # per-bar count labels (the exact distribution values)
    for bar, c in zip(bars, counts):
        if c == 0:
            continue
        pct = 100 * c / n_sent
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{c}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=8, color=INK_SEC,
        )

    ax.set_ylim(0, max(counts) * 1.18)
    plt.tight_layout()
    plt.show()


# ======================================================================
# Cell 8: T0 vs K10 GT-hit matrix + per-quadrant unique distribution
# ======================================================================


# ------------------------------------------------------------------
# FOL feature extraction (for quadrant structural analysis)
# ------------------------------------------------------------------

_PRED_RE = re.compile(r'\b([A-Z][a-zA-Z]*(?=\())')
_QUANT_RE = re.compile(r'[∀∃]')
_CONN_RE = re.compile(r'[∧∨→⊕]')


def _extract_fol_features(fol: str) -> Dict[str, float]:
    """Extract structural features from a single FOL string.

    Returns dict with: len, n_preds, n_quants, n_conns.
    """
    return {
        "len":       len(fol),
        "n_preds":   len(_PRED_RE.findall(fol)),
        "n_quants":  len(_QUANT_RE.findall(fol)),
        "n_conns":   len(_CONN_RE.findall(fol)),
    }


def _quadrant_fol_stats(label: str, fols: List[str]) -> None:
    """Print mean ± std of FOL structural features for one quadrant."""
    if not fols:
        print(f"\n  [{label}] FOL stats: 0 FOLs — skip")
        return

    features = [_extract_fol_features(f) for f in fols]
    keys = [
        ("len",        "FOL length"),
        ("n_preds",    "predicates"),
        ("n_quants",   "quantifiers"),
        ("n_conns",    "connectives"),
    ]

    print(f"\n  [{label}]  FOL structural features  (n={len(fols)} FOLs)")
    print(f"  {'Feature':<16s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>6s}  {'Max':>6s}")
    print("  " + "-" * 50)
    for key, display in keys:
        vals = [f[key] for f in features]
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        min_v = np.min(vals)
        max_v = np.max(vals)
        if key == "len":
            print(f"  {display:<16s}  {mean_v:>8.1f}  {std_v:>8.1f}  {min_v:>6.0f}  {max_v:>6.0f}")
        else:
            print(f"  {display:<16s}  {mean_v:>8.2f}  {std_v:>8.2f}  {min_v:>6.0f}  {max_v:>6.0f}")


QUADRANT_KEYS = [
    "T0 HIT + K10 HIT",
    "T0 HIT + K10 MISS",
    "T0 MISS + K10 HIT",
    "T0 MISS + K10 MISS",
]

# Paper-friendly panel titles for the 2×2 distribution plot (cell 11).
# Single = T0 greedy; Pool = the k=10 candidate pool.
# Cell 7's text printout keeps the compact QUADRANT_KEYS names.
QUADRANT_LABELS = {
    "T0 HIT + K10 HIT":   "Single Correct + Pool Correct",
    "T0 HIT + K10 MISS":  "Single Correct + Pool Incorrect",
    "T0 MISS + K10 HIT":  "Single Incorrect + Pool Correct",
    "T0 MISS + K10 MISS": "Single Incorrect + Pool Incorrect",
}


def _t0_k10_quadrants(
    k10_data: List[dict], t0_data: List[dict], metric: str = "z3_le_pred_norm",
    extra_hit: Optional[set] = None,
) -> Dict[str, List[int]]:
    """Bucket k10 sentences by T0 GT-hit × K10 GT-hit (shared by cell 7 + plots).

    T0 hit = T0 ``metric`` >= 1.0 for that sentence.  K10 hit = any
    Z3-executable candidate (execution_rate == 1.0) has ``metric`` >= 1.0
    (default ``z3_le_pred_norm``, predicate-normalised LE), OR the sentence
    appears in ``extra_hit`` (sentence_ids any selector's m-file records as
    correct — see ``_selector_correct_sids``).  The union keeps the K10 hit
    consistent with the heatmap's UB and the m-files.  Each sentence lands in
    one of QUADRANT_KEYS.

    Returns: (quadrants, quad_fols)
      quadrants: {label: [unique executable FOL count per sentence]}
      quad_fols: {label: [deduped executable FOL strings]}  (for structural stats)
    """
    t0_map = {}
    for e in t0_data:
        if "metrics" in e:
            t0_map[e["sentence_id"]] = e["metrics"].get(metric, 0.0)

    quadrants: Dict[str, List[int]] = {k: [] for k in QUADRANT_KEYS}
    quad_fols: Dict[str, List[str]] = {k: [] for k in QUADRANT_KEYS}

    for r in k10_data:
        sid = r["sentence_id"]
        t0_hit = t0_map.get(sid, 0.0) >= 1.0

        k10_exec = [
            c for c in r["candidates"]
            if c.get("fol") and c.get("metrics", {}).get("execution_rate", 0.0) >= 1.0
        ]
        k10_hit = (
            any(c["metrics"].get(metric, 0.0) >= 1.0 for c in k10_exec)
            or bool(extra_hit and sid in extra_hit)
        )
        unique_n = len(set(c["fol"] for c in k10_exec))
        sent_fols = list(set(c["fol"] for c in k10_exec))

        if t0_hit and k10_hit:
            quadrants["T0 HIT + K10 HIT"].append(unique_n)
            quad_fols["T0 HIT + K10 HIT"].extend(sent_fols)
        elif t0_hit and not k10_hit:
            quadrants["T0 HIT + K10 MISS"].append(unique_n)
            quad_fols["T0 HIT + K10 MISS"].extend(sent_fols)
        elif not t0_hit and k10_hit:
            quadrants["T0 MISS + K10 HIT"].append(unique_n)
            quad_fols["T0 MISS + K10 HIT"].extend(sent_fols)
        else:
            quadrants["T0 MISS + K10 MISS"].append(unique_n)
            quad_fols["T0 MISS + K10 MISS"].extend(sent_fols)

    return quadrants, quad_fols


def compare_k10_t0_table(
    *,
    metric: str = "z3_le_pred_norm",
    models: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    data_root: Optional[str] = None,
) -> None:
    """Per-combo K10 pool (T=0.7) vs T0 greedy (k=1) hit-rate table.

    For each model×dataset combo: K10 = fraction of sentences where any
    Z3-executable candidate hits ``metric`` (>= 1.0); T0 = fraction where the
    single T=0 output hits it.  Uses the same bucketing as the 2×2 matrix
    (``_t0_k10_quadrants``), so these rates match ``analyze_t0_vs_k10_matrix``.

    Prints a 3-column table: K10 (T=0.7) | T0 (k=1) | Δ (pp).
    """
    models = models or DEFAULT_MODELS
    datasets = datasets or DEFAULT_DATASETS

    print("=" * 64)
    print(f"  K10 POOL (T=0.7) vs T0 GREEDY (k=1) — {metric}")
    print("=" * 64)
    print()
    print(f"  {'Model / Dataset':<22s}  {'K10 (T=0.7)':>10s}  {'T0 (k=1)':>9s}  {'Δ (pp)':>8s}")
    print("  " + "-" * 60)

    for model in models:
        for dataset in datasets:
            label = f"{model} / {dataset}"
            paths = _resolve_paths(model=model, dataset=dataset, data_root=data_root)
            k10_file = paths["k10_metrics_path"]
            t0_file = paths.get("t0_metrics_path")

            if not k10_file.exists() or t0_file is None or not t0_file.exists():
                print(f"  {label:<22s}  {'n/a':>10s}  {'n/a':>9s}  {'n/a':>8s}")
                continue

            with open(k10_file, encoding="utf-8") as f:
                k10_data = json.load(f)
            with open(t0_file, encoding="utf-8") as f:
                t0_data = json.load(f)

            quadrants, _ = _t0_k10_quadrants(
                k10_data, t0_data, metric=metric,
                extra_hit=_selector_correct_sids(model, dataset, metric),
            )
            n = sum(len(v) for v in quadrants.values())
            k10_hit = len(quadrants["T0 HIT + K10 HIT"]) + len(quadrants["T0 MISS + K10 HIT"])
            t0_hit = len(quadrants["T0 HIT + K10 HIT"]) + len(quadrants["T0 HIT + K10 MISS"])
            delta = 100 * (k10_hit - t0_hit) / max(n, 1)

            print(
                f"  {label:<22s}  {100 * k10_hit / max(n, 1):>9.1f}%"
                f"  {100 * t0_hit / max(n, 1):>8.1f}%  {delta:>+7.1f}"
            )

    print("  " + "-" * 60)
    print(f"  K10 = any Z3-executable candidate has {metric} >= 1.0 (pool hit),")
    print(f"        or any selector's m-file records its chosen candidate correct.")
    print(f"  T0  = single greedy output has {metric} >= 1.0.  Δ = K10 − T0 (pp).")


def analyze_t0_vs_k10_matrix(
    *,
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    k10_metrics_path: Optional[str] = None,
    t0_metrics_path: Optional[str] = None,
) -> None:
    """2×2 matrix: T=0 GT-hit vs K=10 GT-hit, with unique-count dist + FOL stats.

    For each NL sentence, checks whether T=0 greedy AND/OR any k10 candidate
    is Z3-logically-equivalent to GT via predicate-normalised LE
    (z3_le_pred_norm == 1.0, executable only).

    Prints:
      - 2×2 matrix (T0 hit/miss × K10 hit/miss) with sentence counts
      - Unique (exec) candidate distribution for each of the 4 quadrants
      - FOL structural features (length, predicates, quantifiers, connectives)
        mean / std / min / max for each quadrant
    """
    paths = _resolve_paths(
        model=model, dataset=dataset,
        k10_metrics=k10_metrics_path, t0_metrics=t0_metrics_path,
    )
    k10_file = paths["k10_metrics_path"]
    t0_file = paths.get("t0_metrics_path")

    if not k10_file.exists():
        raise FileNotFoundError(f"{k10_file} not found")
    if t0_file is None or not t0_file.exists():
        t0_file = _derive_t0_metrics_path(k10_file)
        if not t0_file.exists():
            raise FileNotFoundError(
                f"T=0 metrics not found at {t0_file}. "
                f"Pass t0_metrics_path explicitly to override."
            )

    with open(k10_file) as f:
        k10_data = json.load(f)
    with open(t0_file) as f:
        t0_data = json.load(f)

    quadrants, quad_fols = _t0_k10_quadrants(
        k10_data, t0_data, extra_hit=_selector_correct_sids(model, dataset),
    )

    # -- print 2×2 matrix --
    hh = len(quadrants["T0 HIT + K10 HIT"])
    hm = len(quadrants["T0 HIT + K10 MISS"])
    mh = len(quadrants["T0 MISS + K10 HIT"])
    mm = len(quadrants["T0 MISS + K10 MISS"])
    n = hh + hm + mh + mm

    print("=" * 55)
    print("  T0 vs K10 — GT HIT MATRIX (Z3 LE pred-norm)")
    print("=" * 55)
    print()
    print(f"  {'':>18s}  {'K10 HIT':>12s}  {'K10 MISS':>12s}")
    print(f"  {'T0 HIT':>18s}  {hh:>6d} ({100*hh/n:>4.1f}%)  {hm:>6d} ({100*hm/n:>4.1f}%)")
    print(f"  {'T0 MISS':>18s}  {mh:>6d} ({100*mh/n:>4.1f}%)  {mm:>6d} ({100*mm/n:>4.1f}%)")
    print()
    print(f"  T0  hit rate:  {hh+hm}/{n} ({100*(hh+hm)/n:.1f}%)")
    print(f"  K10 hit rate:  {hh+mh}/{n} ({100*(hh+mh)/n:.1f}%)")
    print(f"  K10 gains:     {mh} sentences — T0 missed but K10 found GT")

    # -- print per-quadrant unique dist --
    for label, unique_list in quadrants.items():
        _quadrant_hist(label, unique_list)

    # -- print per-quadrant FOL structural features --
    print()
    print("=" * 55)
    print("  QUADRANT FOL STRUCTURAL FEATURES")
    print("=" * 55)
    for label in QUADRANT_KEYS:
        _quadrant_fol_stats(label, quad_fols[label])


def plot_t0_vs_k10_quadrant_distributions(
    *,
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    k10_metrics_path: Optional[str] = None,
    t0_metrics_path: Optional[str] = None,
    data_root: Optional[str] = None,
) -> None:
    """2×2 small-multiple bar charts of the four quadrant distributions (cell 7).

    Same quadrant bucketing as ``analyze_t0_vs_k10_matrix`` (T0 GT-hit × K10
    GT-hit via ``z3_le_pred_norm``, Z3-executable only).  Each panel is the
    unique executable-FOL count distribution (0..10) for one quadrant.

    Per-panel y-scales (quadrants have very different n); single hue, no legend
    (one series per panel), no figure title — caption lives in LaTeX.  Requires
    matplotlib.
    """
    import matplotlib.pyplot as plt

    paths = _resolve_paths(
        model=model, dataset=dataset,
        k10_metrics=k10_metrics_path, t0_metrics=t0_metrics_path,
        data_root=data_root,
    )
    k10_file = paths["k10_metrics_path"]
    t0_file = paths.get("t0_metrics_path")

    if not k10_file.exists():
        raise FileNotFoundError(f"{k10_file} not found")
    if t0_file is None or not t0_file.exists():
        t0_file = _derive_t0_metrics_path(k10_file)
        if not t0_file.exists():
            raise FileNotFoundError(
                f"T=0 metrics not found at {t0_file}. "
                f"Pass t0_metrics_path explicitly to override."
            )

    with open(k10_file, encoding="utf-8") as f:
        k10_data = json.load(f)
    with open(t0_file, encoding="utf-8") as f:
        t0_data = json.load(f)

    quadrants, _ = _t0_k10_quadrants(
        k10_data, t0_data, extra_hit=_selector_correct_sids(model, dataset),
    )

    # -- single-hue styling (dataviz reference palette, validated) --
    BAR = "#2a78d6"     # sequential blue, step 450
    SURFACE = "#fcfcfb"  # chart surface
    INK = "#0b0b0b"     # primary ink
    INK_SEC = "#52514e"  # secondary ink
    AXIS = "#c3c2b7"    # baseline / axis
    GRID = "#e1e0d9"    # hairline gridline

    ks = list(range(0, K + 1))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), dpi=110)
    fig.patch.set_facecolor(SURFACE)

    for row in range(2):
        for col in range(2):
            key = QUADRANT_KEYS[row * 2 + col]
            ax = axes[row, col]
            ax.set_facecolor(SURFACE)

            counts = [Counter(quadrants[key]).get(k, 0) for k in ks]
            n = len(quadrants[key])

            ax.bar(ks, counts, color=BAR, edgecolor=SURFACE, linewidth=1.5)
            ax.set_xticks(ks)
            title = ax.set_title(f"{QUADRANT_LABELS[key]}   (n = {n})",
                                 color=INK, fontsize=10)
            title.set_ha("left")   # loc='left' is unreliable in some mpl versions

            # each panel carries its own full axis labels + ticks (per-panel
            # y-scales differ, so both columns must show their coordinates)
            ax.set_xlabel("Unique Z3-executable FOLs", color=INK_SEC, fontsize=9)
            ax.set_ylabel("Sentences", color=INK_SEC, fontsize=9)

            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_color(AXIS)
            ax.tick_params(colors=INK_SEC, labelsize=8)
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

            if max(counts) > 0:
                ax.set_ylim(0, max(counts) * 1.15)

    fig.tight_layout()
    plt.show()


def _quadrant_hist(label: str, unique_list: List[int]) -> None:
    """Print a compact histogram for one quadrant."""
    total = len(unique_list)
    if total == 0:
        print(f"\n  {label}: 0 sentences — skip")
        return
    dist = Counter(unique_list)
    mean_u = np.mean(unique_list)
    print(f"\n  {label} (n={total}, mean unique={mean_u:.1f})")
    print(f"  {'u':>2s}  {'cnt':>5s}  {'pct':>6s}")
    print("  " + "-" * 20)
    for k in range(0, 11):
        cnt = dist.get(k, 0)
        if cnt:
            print(f"  {k:>2d}  {cnt:>5d}  {100*cnt/total:>5.1f}%")


# ======================================================================
# Cell 7: Method co-correctness heatmap (BL, M1..M6, UB)
# ======================================================================

# dataset → subfolder holding the per-method selection outputs (m1..m6 json)
MFILE_SUBFOLDER = {
    "test": "malls",
    "test_folio": "folio",
    "test_willow": "willow",
}

METHOD_ORDER = ["BL", "M1", "M2", "M3", "M4", "M5", "M6", "UB"]


# ------------------------------------------------------------------
# Pool-oracle helpers (shared by the heatmap UB, K10-vs-T0 quadrants
# and the pass@K upper bounds).  The stored pool metrics
# (``<tag>_k10_metrics.json``) and the m-file metrics are computed in
# separate processes, and ``z3_le_pred_norm`` (predicate-aligned LE) can
# disagree for borderline formulas — see ``align_predicates``
# nondeterminism.  Every pool-oracle consumer uses the union so all cells
# of the notebook agree and M1..M6 ⊆ UB holds by construction.
# ------------------------------------------------------------------


def _load_selector_hits(
    model: str, dataset: str, metric: str = "z3_le_pred_norm"
) -> set:
    """(sentence_id, selected_idx) pairs where a selector's m-file records its
    chosen candidate as correct (``metric`` >= 1.0)."""
    paths = _resolve_paths(model=model, dataset=dataset)
    sub = MFILE_SUBFOLDER.get(dataset, "")
    tag = _build_model_tag(model, dataset, k10=True)
    hits = set()
    for i in range(1, 7):
        mi_file = paths["k10_dir"] / sub / f"{tag}_m{i}.json"
        if not mi_file.exists():
            continue
        with open(mi_file, encoding="utf-8") as f:
            mi_data = json.load(f)
        for e in mi_data:
            if e.get("metrics", {}).get(metric, 0.0) >= 1.0:
                hits.add((e["sentence_id"], e.get("selected_idx", -1)))
    return hits


def _selector_correct_sids(
    model: str, dataset: str, metric: str = "z3_le_pred_norm"
) -> set:
    """Sentence_ids any selector's m-file records as correct (union over M1..M6)."""
    return {sid for sid, _ in _load_selector_hits(model, dataset, metric)}


def _pool_oracle_correct(
    model: str, dataset: str, metric: str = "z3_le_pred_norm"
) -> set:
    """Pool oracle: sentence_ids where the k10 pool contains a correct
    executable candidate, OR any selector's m-file records its chosen
    candidate as correct.  Since each M_i picks from the pool, the union is
    semantically right (the pool does contain that correct candidate) and
    guarantees M_i ⊆ this set for every selector."""
    paths = _resolve_paths(model=model, dataset=dataset)
    with open(paths["k10_metrics_path"], encoding="utf-8") as f:
        k10_data = json.load(f)
    pool_ub = {
        r["sentence_id"] for r in k10_data
        if any(
            c.get("metrics", {}).get(metric, 0.0) >= 1.0
            for c in r["candidates"]
            if c.get("fol")
            and c.get("metrics", {}).get("execution_rate", 0.0) >= 1.0
        )
    }
    return pool_ub | _selector_correct_sids(model, dataset, metric)


def _method_correct_sets(
    model: str, dataset: str, metric: str = "z3_le_pred_norm"
) -> Dict[str, set]:
    """Per-method sets of sentence_ids correct (``metric`` >= 1.0).

    BL = greedy (T=0) baseline output.  M1..M6 = selected candidate from the
    pool (files ``<tag>_m{i}.json``).  UB = pool contains an executable
    candidate correct under ``metric``.

    UB is defined as the union of (a) sentences whose stored pool metrics
    (``<tag>_k10_metrics.json``) contain a correct executable candidate and
    (b) every sentence any selector's m-file records as correct.  The pool
    metrics and the m-file metrics are computed in separate processes, and
    ``z3_le_pred_norm`` (predicate-aligned LE) can disagree for borderline
    formulas — see ``align_predicates`` nondeterminism.  Since each M_i picks
    from the pool, its correct sentences genuinely belong to the pool's UB, so
    the union is semantically right and guarantees M1..M6 ⊆ UB by construction.
    """
    paths = _resolve_paths(model=model, dataset=dataset)
    k10_dir = paths["k10_dir"]
    tag = _build_model_tag(model, dataset, k10=True)
    correct: Dict[str, set] = {}

    with open(paths["t0_metrics_path"], encoding="utf-8") as f:
        t0_data = json.load(f)
    correct["BL"] = {
        e["sentence_id"] for e in t0_data
        if e.get("metrics", {}).get(metric, 0.0) >= 1.0
    }

    sub = MFILE_SUBFOLDER.get(dataset, "")
    for i in range(1, 7):
        mi_file = k10_dir / sub / f"{tag}_m{i}.json"
        if not mi_file.exists():
            print(f"  [warn] {mi_file} not found — M{i} treated as empty")
            correct[f"M{i}"] = set()
            continue
        with open(mi_file, encoding="utf-8") as f:
            mi_data = json.load(f)
        correct[f"M{i}"] = {
            e["sentence_id"] for e in mi_data
            if e.get("metrics", {}).get(metric, 0.0) >= 1.0
        }

    # UB = pool oracle, computed with the same union as every other pool-oracle
    # consumer (see _pool_oracle_correct) so M_i ⊆ UB holds by construction.
    correct["UB"] = _pool_oracle_correct(model, dataset, metric)
    return correct


def plot_methods_correct_heatmap(
    *,
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    metric: str = "z3_le_pred_norm",
) -> None:
    """8×8 symmetric co-correctness heatmap (BL, M1..M6, UB).

    Cell (i, j) = number of sentences where method i AND method j are correct
    under ``metric`` (>= 1.0).  Diagonal = sentences correct under that method
    alone.  UB = pool contains at least one correct candidate, so every
    pool selector's correct set is a subset of it.  Matrix is symmetric by
    construction; counts are shown in every cell.

    Sequential single-hue blue + colorbar (magnitude scale).
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    correct = _method_correct_sets(model=model, dataset=dataset, metric=metric)
    names = METHOD_ORDER
    n = len(names)
    mat = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            mat[i, j] = len(correct[names[i]] & correct[names[j]])

    # -- styling (dataviz sequential blue) --
    SURFACE = "#fcfcfb"  # chart surface
    INK = "#0b0b0b"      # primary ink
    INK_SEC = "#52514e"  # secondary ink
    AXIS = "#c3c2b7"     # baseline / axis
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "fol_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#104281", "#0d366b"]
    )
    vmax = int(mat.max())

    fig, ax = plt.subplots(figsize=(8.5, 7.2), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            lum = v / vmax if vmax else 0.0
            color = "#ffffff" if lum > 0.55 else INK
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=9, color=color)

    ax.set_xticks(range(n))
    ax.set_xticklabels(names, color=INK, fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, color=INK, fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for spine in ax.spines.values():
        spine.set_color(AXIS)
    ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"# sentences both correct ({metric})", color=INK_SEC)
    cbar.ax.tick_params(colors=INK_SEC)
    cbar.outline.set_visible(False)

    plt.tight_layout()
    plt.show()
