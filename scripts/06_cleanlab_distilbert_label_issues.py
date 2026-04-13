from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from cleanlab.filter import find_label_issues

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


TRAIN_PATH = Path("data/processed/imdb_train_clean.parquet")
OUT_CSV = Path("reports/cleanlab_distilbert_label_issues.csv")
OUT_MD = Path("reports/cleanlab_distilbert_label_issues.md")
OUT_JSON = Path("reports/cleanlab_distilbert_label_issues_summary.json")
MODEL_NAME = "distilbert-base-uncased"


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def build_hf_dataset(texts: list[str], labels: list[int]) -> Dataset:
    return Dataset.from_dict({"text_clean": texts, "label": labels})


def tokenize_dataset(ds: Dataset, tokenizer):
    def tok(batch):
        return tokenizer(
            batch["text_clean"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )

    return ds.map(tok, batched=True)


def compute_metrics_placeholder(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = (preds == labels).mean()
    return {"accuracy": float(acc)}


def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing {TRAIN_PATH}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(TRAIN_PATH)

    required = {"id", "text", "label", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    texts = df["text"].tolist()
    labels = df["label"].to_numpy()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    n_samples = len(df)
    n_classes = len(np.unique(labels))
    pred_probs = np.zeros((n_samples, n_classes), dtype=np.float32)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), start=1):
        print(f"\n[INFO] Fold {fold}/5")

        train_texts = [texts[i] for i in train_idx]
        train_labels = labels[train_idx].tolist()

        val_texts = [texts[i] for i in val_idx]
        val_labels = labels[val_idx].tolist()

        train_ds = build_hf_dataset(train_texts, train_labels)
        val_ds = build_hf_dataset(val_texts, val_labels)

        train_ds = tokenize_dataset(train_ds, tokenizer)
        val_ds = tokenize_dataset(val_ds, tokenizer)

        keep_cols = ["input_ids", "attention_mask", "label"]
        if "token_type_ids" in train_ds.column_names:
            keep_cols.append("token_type_ids")

        train_ds.set_format(type="torch", columns=keep_cols)
        val_ds.set_format(type="torch", columns=keep_cols)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=n_classes,
        )

        training_args = TrainingArguments(
            output_dir=f"tmp/distilbert_fold_{fold}",
            eval_strategy="epoch",
            save_strategy="no",
            logging_strategy="epoch",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            num_train_epochs=1,   # start light; increase later if you want
            learning_rate=2e-5,
            weight_decay=0.01,
            report_to="none",
            seed=42,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_placeholder,
        )

        trainer.train()

        preds = trainer.predict(val_ds)
        logits = preds.predictions
        probs = softmax_np(logits)

        pred_probs[val_idx] = probs

        # free memory between folds
        del model, trainer, train_ds, val_ds, preds, logits, probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # cleanlab: identify suspicious labels from out-of-fold probabilities
    issue_mask = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )

    # issue_mask here is ranked indices
    suspicious_idx = np.array(issue_mask, dtype=int)

    flagged = df.iloc[suspicious_idx].copy()
    flagged["pred_label"] = np.argmax(pred_probs[suspicious_idx], axis=1)
    flagged["pred_prob_neg"] = pred_probs[suspicious_idx, 0]
    flagged["pred_prob_pos"] = pred_probs[suspicious_idx, 1]
    flagged["label_quality"] = pred_probs[suspicious_idx, labels[suspicious_idx]]

    flagged = flagged.sort_values("label_quality", ascending=True)

    flagged.to_csv(OUT_CSV, index=False)

    summary = {
        "model_name": MODEL_NAME,
        "train_rows": int(len(df)),
        "flagged_issues": int(len(flagged)),
        "issue_rate": float(len(flagged) / len(df)),
        "device": device,
        "num_folds": 5,
        "num_epochs": 1,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    preview = flagged.head(20).copy()
    preview["text"] = preview["text"].astype(str).str.slice(0, 240)

    md_lines = [
        "# Cleanlab DistilBERT Label Issues Report (IMDb Train)",
        "",
        f"- Model: **{MODEL_NAME}**",
        f"- Device: **{device}**",
        f"- Train rows: **{summary['train_rows']}**",
        f"- Flagged label issues: **{summary['flagged_issues']}**",
        f"- Issue rate: **{summary['issue_rate']:.4%}**",
        "",
        f"CSV output: `{OUT_CSV.as_posix()}`",
        f"JSON summary: `{OUT_JSON.as_posix()}`",
        "",
    ]

    if len(preview) > 0:
        md_lines.append("## Top 20 suspicious examples")
        md_lines.append("")
        try:
            md_lines.append(preview.to_markdown(index=False))
        except Exception:
            md_lines.append(preview.to_string(index=False))
    else:
        md_lines.append("No suspicious label issues were flagged.")

    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[OK] Flagged issues: {len(flagged)} / {len(df)}")
    print(f"[OK] Wrote -> {OUT_CSV}")
    print(f"[OK] Wrote -> {OUT_MD}")
    print(f"[OK] Wrote -> {OUT_JSON}")


if __name__ == "__main__":
    main()
