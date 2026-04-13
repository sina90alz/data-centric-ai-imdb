from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


TRAIN_PATH = Path("data/raw/imdb_train.parquet")
TEST_PATH = Path("data/raw/imdb_test.parquet")

OUT_CSV = Path("reports/train_test_near_similarity.csv")
OUT_JSON = Path("reports/train_test_near_similarity_summary.json")
OUT_MD = Path("reports/train_test_near_similarity.md")

TEXT_COL = "text"
TRAIN_ID_COL = "id"
TEST_ID_COL = "id"
LABEL_COL = "label"

TOP_K = 3
SIMILARITY_THRESHOLD = 0.90
SUMMARY_THRESHOLDS = [0.85, 0.90, 0.95]


def normalize_text(text: str) -> str:
    """
    Light normalization only.
    Keep it conservative to avoid creating false similarities.
    """
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def validate_columns(df: pd.DataFrame, required: set[str], df_name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {sorted(missing)}")


def shorten_text(text: str, max_len: int = 240) -> str:
    text = str(text)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing train file: {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test file: {TEST_PATH}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    required = {TEXT_COL, TRAIN_ID_COL, LABEL_COL}
    validate_columns(train, required, "train")
    validate_columns(test, {TEXT_COL, TEST_ID_COL, LABEL_COL}, "test")

    train = train.copy()
    test = test.copy()

    # Keep only rows with non-null text
    train = train[train[TEXT_COL].notna()].copy()
    test = test[test[TEXT_COL].notna()].copy()

    # Cast types defensively
    train[TEXT_COL] = train[TEXT_COL].astype(str)
    test[TEXT_COL] = test[TEXT_COL].astype(str)
    train[LABEL_COL] = train[LABEL_COL].astype(int)
    test[LABEL_COL] = test[LABEL_COL].astype(int)

    # Light normalization
    train["text_norm"] = train[TEXT_COL].map(normalize_text)
    test["text_norm"] = test[TEXT_COL].map(normalize_text)

    # Exact train/test overlap detection on normalized text
    train_text_set = set(train["text_norm"].values)
    test_text_set = set(test["text_norm"].values)

    exact_overlap_texts = train_text_set.intersection(test_text_set)
    exact_overlap_unique_count = len(exact_overlap_texts)

    exact_overlap_mask_test = test["text_norm"].isin(exact_overlap_texts)
    exact_overlap_test_rows = int(exact_overlap_mask_test.sum())

    # Exclude exact overlaps from near-similarity stage
    test_noleak = test.loc[~exact_overlap_mask_test].copy()

    if len(test_noleak) == 0:
        summary = {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "test_rows_after_exact_overlap_removal": 0,
            "exact_overlap_unique_texts": exact_overlap_unique_count,
            "exact_overlap_test_rows_removed": exact_overlap_test_rows,
            "top_k": TOP_K,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "message": "All test rows were exact overlaps after normalization.",
        }
        OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        OUT_CSV.write_text("", encoding="utf-8")
        OUT_MD.write_text(
            "\n".join(
                [
                    "# Train/Test Near Similarity Report",
                    "",
                    "All test rows were removed as exact overlaps after normalization.",
                    "",
                    f"- Train rows: **{len(train)}**",
                    f"- Test rows: **{len(test)}**",
                    f"- Exact overlap unique texts: **{exact_overlap_unique_count}**",
                    f"- Exact overlap test rows removed: **{exact_overlap_test_rows}**",
                ]
            ),
            encoding="utf-8",
        )
        print("[OK] No near-similarity stage executed because all test rows were exact overlaps.")
        return

    # Fit TF-IDF on train only
    vectorizer = TfidfVectorizer(
        lowercase=False,          # already normalized
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=30000,
    )

    X_train = vectorizer.fit_transform(train["text_norm"])
    X_test = vectorizer.transform(test_noleak["text_norm"])

    # Find nearest train neighbors for each test review
    nn = NearestNeighbors(
        n_neighbors=TOP_K,
        metric="cosine",
        algorithm="brute",
    )
    nn.fit(X_train)

    distances, indices = nn.kneighbors(X_test, return_distance=True)

    # Build pair table
    rows: list[dict] = []

    test_noleak = test_noleak.reset_index(drop=True)
    train = train.reset_index(drop=True)

    for test_row_idx in range(len(test_noleak)):
        test_row = test_noleak.iloc[test_row_idx]

        for rank in range(TOP_K):
            train_row_idx = int(indices[test_row_idx, rank])
            distance = float(distances[test_row_idx, rank])
            similarity = 1.0 - distance

            train_row = train.iloc[train_row_idx]

            rows.append(
                {
                    "test_id": test_row[TEST_ID_COL],
                    "train_id": train_row[TRAIN_ID_COL],
                    "neighbor_rank": rank + 1,
                    "similarity": similarity,
                    "cosine_distance": distance,
                    "test_label": int(test_row[LABEL_COL]),
                    "train_label": int(train_row[LABEL_COL]),
                    "same_label": bool(int(test_row[LABEL_COL]) == int(train_row[LABEL_COL])),
                    "test_text": test_row[TEXT_COL],
                    "train_text": train_row[TEXT_COL],
                    "test_text_norm": test_row["text_norm"],
                    "train_text_norm": train_row["text_norm"],
                }
            )

    pairs = pd.DataFrame(rows)

    # Summary counts across thresholds
    threshold_counts: dict[str, int] = {}
    conflicting_counts: dict[str, int] = {}

    for thr in SUMMARY_THRESHOLDS:
        thr_key = f"pairs_ge_{str(thr).replace('.', '_')}"
        conflict_key = f"conflicting_label_pairs_ge_{str(thr).replace('.', '_')}"

        mask = pairs["similarity"] >= thr
        threshold_counts[thr_key] = int(mask.sum())
        conflicting_counts[conflict_key] = int((mask & (~pairs["same_label"])).sum())

    # Final suspicious set
    suspicious = pairs.loc[pairs["similarity"] >= SIMILARITY_THRESHOLD].copy()
    suspicious = suspicious.sort_values(
        by=["similarity", "neighbor_rank"],
        ascending=[False, True],
    )

    suspicious.to_csv(OUT_CSV, index=False)

    summary = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_rows_after_exact_overlap_removal": int(len(test_noleak)),
        "exact_overlap_unique_texts": exact_overlap_unique_count,
        "exact_overlap_test_rows_removed": exact_overlap_test_rows,
        "top_k": TOP_K,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "vectorizer": {
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.95,
            "max_features": 30000,
            "strip_accents": "unicode",
        },
        "suspicious_pairs": int(len(suspicious)),
        **threshold_counts,
        **conflicting_counts,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Markdown report
    md_lines = [
        "# Train/Test Near Similarity Report",
        "",
        "This report checks for soft leakage by finding test reviews that are highly similar to train reviews using TF-IDF + cosine nearest neighbors.",
        "",
        f"- Train rows: **{summary['train_rows']}**",
        f"- Test rows: **{summary['test_rows']}**",
        f"- Test rows after exact-overlap removal: **{summary['test_rows_after_exact_overlap_removal']}**",
        f"- Exact overlap unique texts: **{summary['exact_overlap_unique_texts']}**",
        f"- Exact overlap test rows removed: **{summary['exact_overlap_test_rows_removed']}**",
        f"- Top-k neighbors per test row: **{summary['top_k']}**",
        f"- Suspicious similarity threshold: **{summary['similarity_threshold']:.2f}**",
        "",
        "## Counts by threshold",
        "",
    ]

    for thr in SUMMARY_THRESHOLDS:
        thr_key = f"pairs_ge_{str(thr).replace('.', '_')}"
        conflict_key = f"conflicting_label_pairs_ge_{str(thr).replace('.', '_')}"
        md_lines.append(
            f"- Pairs with similarity ≥ **{thr:.2f}**: **{summary[thr_key]}** "
            f"(conflicting labels: **{summary[conflict_key]}**)"
        )

    md_lines.extend(
        [
            "",
            f"CSV output: `{OUT_CSV.as_posix()}`",
            f"JSON summary: `{OUT_JSON.as_posix()}`",
            "",
        ]
    )

    if len(suspicious) > 0:
        preview = suspicious.head(20).copy()
        preview["test_text"] = preview["test_text"].map(shorten_text)
        preview["train_text"] = preview["train_text"].map(shorten_text)

        preview_cols = [
            "test_id",
            "train_id",
            "neighbor_rank",
            "similarity",
            "test_label",
            "train_label",
            "same_label",
            "test_text",
            "train_text",
        ]
        preview = preview[preview_cols]

        md_lines.append("## Top 20 suspicious pairs")
        md_lines.append("")
        try:
            md_lines.append(preview.to_markdown(index=False))
        except Exception:
            md_lines.append(preview.to_string(index=False))
    else:
        md_lines.append("No suspicious near-similar train/test pairs were found at the selected threshold.")

    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[OK] Exact overlap unique texts: {exact_overlap_unique_count}")
    print(f"[OK] Exact overlap test rows removed: {exact_overlap_test_rows}")
    print(f"[OK] Suspicious near-similar pairs: {len(suspicious)}")
    print(f"[OK] Wrote -> {OUT_CSV}")
    print(f"[OK] Wrote -> {OUT_JSON}")
    print(f"[OK] Wrote -> {OUT_MD}")


if __name__ == "__main__":
    main()