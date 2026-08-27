# NL → FOL: Multi-Candidate Generation and Selection

Translate natural-language sentences into First-Order Logic (FOL) with small
decoder-only LLMs (Qwen3-4B / Qwen3-8B / Ministral-8B, LoRA fine-tuned).
The approach is **k=10 multi-candidate generation + candidate selection
(M1–M6)**, with analysis reporting pass@k upper bounds, pool-vs-greedy gains,
and method co-correctness.

---

## Repository structure

```
.
├── data/                     # Datasets + all generation/selection results
│   ├── test.json             # 1k MALLS test set (official human-verified)
│   ├── train.json            # 18k MALLS training set
│   ├── val.json              # 2k MALLS validation set
│   ├── remain.json           # ~7k unused MALLS remainder
│   ├── test_folio.json       # 1k FOLIO test set
│   ├── test_willow.json      # 1k WillowNLtoFOL test set
│   ├── few_shot_examples.json# FOL→NL few-shot pairs for back-translation
│   └── results/{model}/      # Per-model outputs (generation, metrics, selectors)
│       ├── {model}.json              # T=0 greedy generation
│       ├── {model}_metrics.json      # T=0 evaluation metrics
│       └── k10/                     # k=10 candidates
│           ├── {model}_k10.json              # raw candidate pool
│           ├── {model}_k10_metrics.json      # per-candidate metrics
│           ├── {model}_k10_{m1..m6}.json     # selection-method outputs
│           └── {model}_pw_metrics.json       # pairwise features (M6)
├── src/                      # Reusable Python modules
│   ├── prepare_data.py       # MALLS split generation
│   ├── prepare_folio.py      # FOLIO test-set preparation
│   ├── prepare_willow.py     # WillowNLtoFOL test-set preparation
│   ├── prepare_few_shot.py   # few-shot example generation
│   ├── prepare_k10.py        # k10 dedup + Z3 filtering (selector training data)
│   ├── prepare_scorer_data.py# M6 scorer training data
│   ├── finetune_selector.py  # M4/M5 selector fine-tuning
│   ├── run_inference.py      # T=0 greedy inference + evaluation
│   ├── generate_k10.py       # k=10 multi-candidate generation
│   ├── generate_k10_ministral.py / generate_t0_ministral.py  # Ministral variants
│   ├── selectors/            # Candidate selection
│   │   ├── feature_select.py # M1–M4: argmax over a consensus feature
│   │   ├── m4.py             # BERT embedding + FOL→NL back-translation utils
│   │   ├── m5.py             # M5: fine-tuned LLM selector
│   │   ├── m6.py             # M6: Logistic-Regression scorer
│   │   └── pw_metrics.py     # pairwise feature pipeline (Z3 filter → features)
│   ├── eval/                 # Metrics and Z3 tooling
│   │   ├── eval.py           # 7 evaluation metrics (execution, EM, Z3-LE, BLEU, BERTScore)
│   │   ├── z3_equiv.py       # Z3 logical-equivalence checking
│   │   └── z3parser.py       # FOL string → Z3 expression
│   └── analysis_k10.py       # pass@k, K10-vs-T0, quadrants, co-correctness heatmap
├── notebooks/                # Colab workflow, one notebook per stage
│   ├── finetune/             # LoRA fine-tuning of base models and the selector
│   ├── generate*.ipynb       # T=0 and k=10 generation
│   ├── evaluate*.ipynb       # metric computation
│   ├── selection/            # selection data, features, M1–M6, LR scorer
│   ├── ministral/            # Ministral-8B generation variants
│   └── analysis.ipynb        # main analysis (pass@k, K10 vs T0, heatmap)
└── figures/                  # charts produced by the analysis notebooks
```

> **Note:** `models/` (the LoRA adapters) is **not** included in this
> repository because the adapter weights exceed GitHub's per-file size limit.
> Model checkpoints must be stored separately (e.g., Hugging Face Hub).

---

## Pipeline

Run on **Google Colab** (GPU). Each stage maps to the notebooks in `notebooks/`.

### Stage 0 — Data preparation
```bash
python src/prepare_data.py        # MALLS → test/train/val/remain
python src/prepare_folio.py       # FOLIO test set
python src/prepare_willow.py      # WillowNLtoFOL test set
python src/prepare_few_shot.py    # few-shot examples for back-translation
```

### Stage 1 — Fine-tuning
`notebooks/finetune/` — LoRA fine-tune the base generation models
(`finetune_Qwen4b/8b`, `finetune_Ministral8b`) and the M4/M5 selector
(`finetune_selector*`).

### Stage 2 — Generation
| Notebook | Purpose |
|---|---|
| `generate.ipynb` | T=0 greedy output (`run_inference.run_inference`) |
| `generate_k10.ipynb` | k=10 candidate pool (`run_inference.run_inference_k10`) |
| `generate_k10_val.ipynb` | k=10 on the validation set (`generate_k10.run_k10`) |
| `ministral/generate_t0_ministral.ipynb` | Ministral T=0 |
| `ministral/generate_k10*.ipynb` | Ministral k=10 |

### Stage 3 — Evaluation
| Notebook | Purpose |
|---|---|
| `evaluate.ipynb` | metrics for the T=0 output (`run_eval_only`) |
| `evaluate_k10.ipynb` | per-candidate metrics for the k=10 pool (`run_eval_k10_only`) |
| `evaluate_k10_val.ipynb` | metrics on the validation pool (`compute_all_metrics`) |

### Stage 4 — Candidate selection (M1–M6)
1. `selection/selector_data.ipynb` — dedup + Z3-filter the k=10 pool
   (`prepare_k10_data`) and build selector training data.
2. `selection/scorer_data.ipynb` — build the M6 scorer features + labels
   (`prepare_scorer_data`).
3. `selection/pw_metrics.ipynb` — pairwise feature pipeline
   (`pw_metrics`: Z3 filter → unique count → pairwise LE → BLEU → BERTScore → back-translation similarity).
4. `selection/train_scorer_LR.ipynb` — train the M6 Logistic-Regression scorer.
5. `selection/select_m1..m6.ipynb` — run each selector and write
   `{model}_k10_m{i}.json`.

| Method | Strategy |
|---|---|
| M1 | LE-Vote: largest consensus class by pairwise Z3 equivalence |
| M2 | GSC-BLEU: highest mean pairwise FOL-token BLEU |
| M3 | GSC-BERTScore: highest mean pairwise RoBERTa F1 |
| M4 | Back-translation: FOL→NL informalization + BERT cosine similarity |
| M5 | Fine-tuned LLM selector (Qwen + LoRA) |
| M6 | Logistic regression over the 5 pairwise features |

### Stage 5 — Analysis
`notebooks/analysis.ipynb` drives `src/analysis_k10.py`:
- **Pass@K upper bounds** — oracle over the first K candidates, for all metrics.
- **K10 pool vs T0 greedy** — how often the pool contains a correct candidate
  that the greedy output missed.
- **T0 vs K10 quadrants** — correct/incorrect 2×2 with structural stats.
- **Method co-correctness heatmap** — 8×8 matrix of `BL, M1..M6, UB`, where
  `UB` is the pool oracle (a sentence counts as a pool hit if the pool
  contains a correct executable candidate **or** any selector's m-file
  records it as correct).
