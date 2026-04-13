from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from cleanlab.classification import CleanLearning


TRAIN_PATH = Path("data/processed/imdb_train_clean.parquet")
OUT_CSV = Path("reports/cleanlab_label_issues.csv")
OUT_MD = Path("reports/cleanlab_label_issues.md")
OUT_JSON = Path("reports/cleanlab_label_issues_summary.json")


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=30000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=300,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing {TRAIN_PATH}. Run ingestion first.")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(TRAIN_PATH)

    required = {"id", "text_clean", "label", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    X = df["text_clean"].astype(str)
    y = df["label"].astype(int).to_numpy()

    model = build_model()

    # CleanLearning will use cross-validation internally to estimate out-of-sample pred_probs
    cl = CleanLearning(
        clf=model,
        cv_n_folds=5,
        verbose=True,
        seed=42,
    )

    cl.fit(X, y)

    # DataFrame of label issues ranked by severity
    issues = cl.get_label_issues()

    # Keep only rows flagged as likely issues
    flagged = issues[issues["is_label_issue"]].copy()

    # Join back to original data
    flagged = flagged.join(df[["id", "text", "label", "source"]])

    # Some useful renamed columns if present
    rename_map = {}
    if "given_label" in flagged.columns:
        rename_map["given_label"] = "label_given"
    if "predicted_label" in flagged.columns:
        rename_map["predicted_label"] = "label_pred"
    flagged = flagged.rename(columns=rename_map)

    # Reorder columns to make report readable
    preferred_cols = [
        "id",
        "source",
        "label",
        "label_given",
        "label_pred",
        "is_label_issue",
        "label_quality",
        "text",
    ]
    ordered_cols = [c for c in preferred_cols if c in flagged.columns] + [
        c for c in flagged.columns if c not in preferred_cols
    ]
    flagged = flagged[ordered_cols]

    # Sort: worst quality first if available
    if "label_quality" in flagged.columns:
        flagged = flagged.sort_values("label_quality", ascending=True)

    flagged.to_csv(OUT_CSV, index=False)

    summary = {
        "train_rows": int(len(df)),
        "flagged_issues": int(len(flagged)),
        "issue_rate": float(len(flagged) / len(df)),
        "columns_in_output": list(flagged.columns),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Markdown summary
    md_lines = [
        "# Cleanlab Label Issues Report (IMDb Train)",
        "",
        f"- Train rows: **{summary['train_rows']}**",
        f"- Flagged label issues: **{summary['flagged_issues']}**",
        f"- Issue rate: **{summary['issue_rate']:.4%}**",
        "",
        f"CSV output: `{OUT_CSV.as_posix()}`",
        f"JSON summary: `{OUT_JSON.as_posix()}`",
        "",
    ]

    if len(flagged) > 0:
        preview = flagged.head(20).copy()
        # shorten text for markdown readability
        if "text" in preview.columns:
            preview["text"] = preview["text"].astype(str).str.slice(0, 240)
        md_lines.append("## Top 20 suspicious examples")
        md_lines.append("")
        try:
            md_lines.append(preview.to_markdown(index=False))
        except Exception:
            md_lines.append(preview.to_string(index=False))
    else:
        md_lines.append("No suspicious label issues were flagged.")

    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[OK] Flagged issues: {summary['flagged_issues']} / {summary['train_rows']}")
    print(f"[OK] Wrote -> {OUT_CSV}")
    print(f"[OK] Wrote -> {OUT_MD}")
    print(f"[OK] Wrote -> {OUT_JSON}")


if __name__ == "__main__":
    main()
