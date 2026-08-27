# Multi-candidate FOL selectors
#   m1 — LE-Vote              (pairwise Z3 equivalence, largest consensus class)
#   m2 — GSC-BLEU             (pairwise FOL-token BLEU, highest mean consensus)
#   m3 — GSC-BertScore        (pairwise RoBERTa cosine F1, highest mean consensus)
#   m4 — Back-Translation     (FOL→NL informalization + BERT cosine similarity, D3)
#   m5 — LLM-FT               (Qwen + LoRA fine-tuned selector)
#   m6 — LR Scorer            (trained Logistic Regression on 5 consensus features)
