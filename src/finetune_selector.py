"""
Fine-tune FOL Selector Model (M4)
==================================

M4/M5: Qwen/Ministral + LoRA + score_head → BCE → [0,1]

Training data: data/results/{model}/k10/{model}_k10_{dataset}_selector_train.json
                             + {model}_k10_{dataset}_selector_val.json
Output model:  models/{model}_selector/
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel, AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent  # project/code/
DATA_DIR = _ROOT / "data"
RESULTS_BASE = DATA_DIR / "results"
MODEL_DIR = _ROOT / "models"

# ---------------------------------------------------------------------------
# Model registry — maps model tag → HuggingFace base model
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: Dict[str, str] = {
    "Qwen4b":  "Qwen/Qwen3-4B",
    "Qwen8b":  "Qwen/Qwen3-8B",
    "Ministral8b": "mistralai/Ministral-3-8B-Instruct-2512",
}


def _resolve_model_cls(base_model: str):
    """Return the Auto class that loads ``base_model``.

    Mistral3 (``model_type == "mistral3"``) must be loaded with
    ``Mistral3ForConditionalGeneration``; everything else (Qwen3, ...) keeps
    ``AutoModelForCausalLM``.
    """
    from transformers import AutoConfig
    if AutoConfig.from_pretrained(base_model).model_type == "mistral3":
        from transformers import Mistral3ForConditionalGeneration
        return Mistral3ForConditionalGeneration
    return AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class FOLSelectorDataset(Dataset):
    """(NL, FOL) → label dataset for BCE training."""

    def __init__(self, samples, tokenizer, max_length=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        text = f"NL: {s['nl']}\nFOL: {s['fol']}"
        enc = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(float(s["label"]), dtype=torch.float),
        }


# ---------------------------------------------------------------------------
# M4: Qwen3-4B + LoRA + score_head
# ---------------------------------------------------------------------------


class QwenSelector(nn.Module):
    """Qwen decoder-only + LoRA + score_head for regression."""

    def __init__(self, base_model_name: str, lora_config: LoraConfig):
        super().__init__()
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 60:
            base = _resolve_model_cls(base_model_name).from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            base = _resolve_model_cls(base_model_name).from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        base.config.use_cache = False
        base.gradient_checkpointing_enable()
        self.llm = get_peft_model(base, lora_config)
        hidden_size = base.config.hidden_size
        self.score_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.llm.model.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :].float()  # last token
        logit = self.score_head(last_hidden).squeeze(-1)
        return torch.sigmoid(logit)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def _load_selector_data(model_name: str, dataset: str) -> tuple[list[dict], list[dict]]:
    """Load k10 selector train/val data.

    Looks in data/results/{model_name}/k10/ for:
      {model_name}_k10_{dataset}_selector_train.json
      {model_name}_k10_{dataset}_selector_val.json
    """
    k10_dir = RESULTS_BASE / model_name / "k10"
    suffix = dataset.replace("test_", "").replace("test", "")
    tag = f"{model_name}_k10_{suffix}" if suffix else f"{model_name}_k10"

    train_path = k10_dir / f"{tag}_selector_train.json"
    val_path = k10_dir / f"{tag}_selector_val.json"

    for p in [train_path, val_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Selector data not found: {p}\n"
                f"Run notebooks/selection/selector_data.ipynb first."
            )
    with open(train_path, encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, encoding="utf-8") as f:
        val_data = json.load(f)
    return train_data["samples"], val_data["samples"]


def _train_epoch(model, loader, optimizer, scheduler, device, epoch, total_epochs,
                 accum_steps=1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        preds = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.BCELoss()(preds, labels) / accum_steps
        loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        if batch_idx % 200 == 0:
            print(f"    batch {batch_idx:>4}/{len(loader)}  loss={loss.item() * accum_steps:.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def _eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        preds = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.BCELoss()(preds, labels)
        total_loss += loss.item()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # Accuracy at threshold 0.5
    acc = np.mean((np.array(all_preds) >= 0.5) == np.array(all_labels))
    return total_loss / len(loader), float(acc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_m4(
    model_name: str = "Qwen4b",
    base_model: Optional[str] = None,
    *,
    dataset: str = "val",
    epochs: int = 6,
    batch_size: int = 4,
    lr: float = 1e-4,
    max_length: int = 512,
    gradient_accumulation_steps: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    warmup_ratio: float = 0.1,
    early_stopping_patience: int = 1,
    verbose: bool = True,
) -> str:
    """Fine-tune M4 (Qwen3-4B + LoRA + score_head).

    Reads selector data from data/results/{model_name}/k10/.
    Early-stops when val_loss fails to improve for *early_stopping_patience*+1
    consecutive epochs.  Returns path to saved (best) model directory.
    """
    if base_model is None:
        base_model = _MODEL_REGISTRY.get(model_name)
        if base_model is None:
            raise ValueError(
                f"Unknown model_name '{model_name}'; pass base_model explicitly."
            )

    train_samples, val_samples = _load_selector_data(model_name, dataset)

    out_dir = MODEL_DIR / f"{model_name}_selector"
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 55)
        print(f"  M4 FINE-TUNING  ({base_model} + LoRA + head)")
        print(f"  Data:    {model_name}_k10_{dataset}_selector_*.json")
        print(f"  Train samples: {len(train_samples)}")
        print(f"  Val samples:   {len(val_samples)}")
        print(f"  Epochs:        {epochs}")
        print(f"  Batch size:    {batch_size} x {gradient_accumulation_steps}"
              f" = {batch_size * gradient_accumulation_steps} effective")
        print(f"  LR:            {lr}")
        print(f"  Output:        {out_dir}")
        print("=" * 55)
        print()

    # Tokenizer
    if verbose:
        print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset — pre-split, no need to shuffle sentences
    train_ds = FOLSelectorDataset(train_samples, tokenizer, max_length=max_length)
    val_ds = FOLSelectorDataset(val_samples, tokenizer, max_length=max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model
    if verbose:
        print("  Loading model + LoRA...")
    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    model = QwenSelector(base_model, lora_config)
    device = next(model.llm.parameters()).device
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ---- resume from checkpoint ----
    ckpt_path = out_dir / "checkpoint.pt"
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if ckpt_path.exists():
        if verbose:
            print("  Checkpoint found — resuming...")
        ckpt = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        patience_counter = ckpt.get("patience_counter", 0)
        if verbose:
            print(f"  Resumed from epoch {start_epoch + 1}  |  "
                  f"best_val_loss={best_val_loss:.4f}  patience={patience_counter}")
            print()

    # ---- train ----
    if verbose:
        print(f"\n  Training ({len(train_ds)} train / {len(val_ds)} val)...")
        print(f"  Total steps: {total_steps}  warmup: {warmup_steps}")
        print()

    t0 = datetime.now()
    stopped_early = False

    for epoch in range(start_epoch, epochs):
        print(f"  Epoch {epoch + 1}/{epochs}")
        train_loss = _train_epoch(model, train_loader, optimizer, scheduler,
                                   device, epoch, epochs,
                                   accum_steps=gradient_accumulation_steps)
        val_loss, val_acc = _eval_epoch(model, val_loader, device)
        print(f"    train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save LoRA + score_head
            model.llm.save_pretrained(str(out_dir))
            torch.save(model.score_head.state_dict(), str(out_dir / "score_head.pt"))
            tokenizer.save_pretrained(str(out_dir))
            print(f"    -> saved (best)")
        else:
            patience_counter += 1
            print(f"    -> no improvement (patience {patience_counter}/{early_stopping_patience + 1})")
            if patience_counter > early_stopping_patience:
                print(f"\n  Early stopping at epoch {epoch + 1}")
                stopped_early = True
                break

        # ---- checkpoint after each epoch ----
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
        }, str(ckpt_path))

    # ---- cleanup checkpoint ----
    if ckpt_path.exists():
        ckpt_path.unlink()

    elapsed = (datetime.now() - t0).total_seconds() / 60
    if verbose:
        print(f"\n  Done: {elapsed:.1f} min  |  Best val_loss: {best_val_loss:.4f}"
              f"{' (early stop)' if stopped_early else ''}")
        print(f"  Model saved: {out_dir}")

    return str(out_dir)
