# Data-Centric AI — IMDb Data Pipeline

A data-centric ML pipeline for IMDb sentiment classification, emphasizing **data quality, validation, leakage detection, and label issue detection**.

## Goals

- Build a production-quality data pipeline
- Validate data schema and content
- Detect and fix data leakage
- Find and flag mislabeled samples
- Detect near-duplicates across train/test
- Generate reproducible audit trails

---

## Project Structure

```
data-centric-AI/
├── data/
│   ├── raw/
│   │   ├── imdb_train.parquet
│   │   └── imdb_test.parquet
│   └── processed/
│       ├── imdb_train_clean.parquet
│       ├── imdb_test_clean.parquet
│       └── imdb_test_noleak.parquet
├── gx/
│   └── expectations/
│       └── imdb_raw_suite.json
├── reports/
│   ├── validation_raw.md
│   ├── leakage_fix.md
│   ├── cleanlab_label_issues.md
│   ├── cleanlab_label_issues.csv
│   ├── train_test_near_similarity.md
│   └── ge_validation_raw.md
├── scripts/
│   ├── 01_ingest_imdb.py
│   ├── 02_validate_raw.py
│   ├── 03_clean_and_save.py
│   ├── 04_remove_leakage.py
│   ├── 05_cleanlab_label_issues.py
│   ├── 06_cleanlab_distilbert_label_issues.py
│   ├── 07_near_similarity_train_test.py
│   ├── ge_01_build_suite_raw.py
│   └── ge_02_validate_raw.py
└── src/data/
    └── cleaning.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install datasets pandas pyarrow great-expectations cleanlab scikit-learn transformers torch
```

## Pipeline Steps

| Step | Script | Purpose | Output |
|------|--------|---------|--------|
| 1 | `01_ingest_imdb.py` | Download IMDb from HuggingFace, add id/source columns | `imdb_train.parquet`, `imdb_test.parquet` |
| 2 | `02_validate_raw.py` | Check schema, stats, duplicates, leakage detection | `validation_raw.md` (123 leaked rows found) |
| 3 | `ge_01_build_suite_raw.py` | Create Great Expectations validation suite | `imdb_raw_suite.json` |
| 4 | `ge_02_validate_raw.py` | Validate with GX expectations | `ge_validation_raw.md`, `.json` |
| 5 | `03_clean_and_save.py` | Remove HTML tags/entities, normalize whitespace | `imdb_train_clean.parquet`, `imdb_test_clean.parquet` |
| 6 | `04_remove_leakage.py` | Remove exact text matches from test | `imdb_test_noleak.parquet`, `leakage_fix.md` |
| 7 | `05_cleanlab_label_issues.py` | TF-IDF + LogReg label detection (5-fold CV) | `cleanlab_label_issues.csv`, `.md` (559 issues found) |
| 8 | `06_cleanlab_distilbert_label_issues.py` | DistilBERT label detection (5-fold CV) | `cleanlab_distilbert_label_issues.csv`, `.md` |
| 9 | `07_near_similarity_train_test.py` | k-NN similarity check, configurable threshold | `train_test_near_similarity.csv`, `.md` |

## Dataset Insights

- **Size:** 25K train + 25K test
- **Label:** Balanced (50/50 pos/neg)
- **Text length:** Median ~174 words
- **Format:** Contains HTML artifacts (`<br>` tags)
- **Duplicates:** 96 train, 199 test (exact)
- **Leakage:** 123 test rows match train (removed)
- **Label issues:** 559 rows (~2.2%) flagged

## Design Decisions

| Feature | Approach | Rationale |
|---------|----------|-----------|
| **Text storage** | Keep `text` + `text_clean` | Auditability, reversibility |
| **Leakage handling** | Remove from test only | Conservative, preserves ground truth |
| **Label detection** | TF-IDF + DistilBERT | Speed vs. accuracy trade-off |
| **Validation** | Great Expectations suite | Reusable, contractual |

## Dataset

**IMDb Movie Reviews** (https://huggingface.co/datasets/stanfordnlp/imdb)  
Binary sentiment classification: 25K train + 25K test, balanced labels (0=negative, 1=positive)

## Tech Stack

| Library | Purpose |
|---------|---------|
| `datasets` | Download IMDb from HuggingFace |
| `pandas` | Data operations |
| `pyarrow` | Parquet I/O |
| `great-expectations` | Schema validation |
| `cleanlab` | Label issue detection |
| `scikit-learn` | TF-IDF, LogisticRegression, CV |
| `transformers` | DistilBERT fine-tuning |
| `torch` | Backend for transformers |
