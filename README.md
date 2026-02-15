# 📊 Data-Centric AI — IMDb Data Pipeline

A professional data-centric machine learning project focused on building a robust data pipeline using the IMDb reviews dataset.

This project emphasizes **data quality, validation, leakage detection, and reproducibility** rather than only model training.

---

## 🎯 Project Goals

- Build a realistic ML data pipeline
- Practice data validation and profiling
- Detect and correct data leakage
- Implement deterministic data cleaning
- Understand dataset characteristics before modeling
- Prepare data for reliable downstream ML

---

## 🧠 Key Concepts Practiced

- Data ingestion from Hugging Face
- Schema validation
- Distribution analysis
- Duplicate detection
- Data leakage detection
- Text normalization
- Reproducible pipelines
- Data versioning mindset

---

## 🗂️ Project Structure

```
data-centric-AI/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── eval/
│
├── reports/
│   └── img/
│
├── scripts/
│   ├── 01_ingest_imdb.py
│   ├── 02_validate_raw.py
│   ├── 03_clean_and_save.py
│
├── src/
│   └── data/
│       └── cleaning.py
│
└── README.md
```

---

## ⚙️ Setup

### Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
```

### Install dependencies

```bash
pip install datasets pandas pyarrow matplotlib
```

---

## 🚀 Pipeline Steps

### 1️⃣ Data Ingestion

```bash
python -m scripts.01_ingest_imdb
```

Downloads IMDb dataset and saves raw parquet snapshots.

---

### 2️⃣ Data Validation

```bash
python -m scripts.02_validate_raw
```

Generates validation report including:

- schema checks
- label distribution
- text length statistics
- duplicate detection
- leakage detection

---

### 3️⃣ Data Cleaning

```bash
python -m scripts.03_clean_and_save
```

Performs deterministic text normalization:

- removes HTML artifacts
- normalizes whitespace
- keeps original + cleaned text

---

## 📈 Current Dataset Insights

- Balanced dataset (50/50 sentiment)
- Long text distribution (median ~174 words)
- Formatting noise present (<br> tags)
- Minor duplicate presence
- Small cross-split leakage detected

---

## 🧪 Next Steps

- Leakage correction
- Distribution profiling plots
- Near-duplicate detection
- Baseline model
- Drift simulation
- Dataset versioning

---

## 📚 Dataset

IMDb Movie Reviews Dataset  
https://huggingface.co/datasets/stanfordnlp/imdb

---

## ⭐ Future Improvements

- Add Great Expectations validation
- Add DVC for dataset versioning
- Add automated data quality checks
- Add CI pipeline
- Add model training stage

---
