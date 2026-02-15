import argparse
from pathlib import Path
import pandas as pd


def basic_profile(df: pd.DataFrame, name: str):
    # lengths
    df["text_len_chars"] = df["text"].astype(str).str.len()
    df["text_len_words"] = df["text"].astype(str).str.split().apply(len)

    summary = {
        "name": name,
        "rows": len(df),
        "null_text": int(df["text"].isna().sum()),
        "empty_text": int((df["text"].astype(str).str.strip() == "").sum()),
        "label_values": df["label"].value_counts(dropna=False).to_dict(),
        "min_chars": int(df["text_len_chars"].min()),
        "p50_chars": int(df["text_len_chars"].median()),
        "p95_chars": int(df["text_len_chars"].quantile(0.95)),
        "max_chars": int(df["text_len_chars"].max()),
        "min_words": int(df["text_len_words"].min()),
        "p50_words": int(df["text_len_words"].median()),
        "p95_words": int(df["text_len_words"].quantile(0.95)),
        "max_words": int(df["text_len_words"].max()),
    }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", default="data/raw/imdb_train.parquet")
    p.add_argument("--test_path", default="data/raw/imdb_test.parquet")
    p.add_argument("--report_path", default="reports/validation_raw.md")
    args = p.parse_args()

    train_path = Path(args.train_path)
    test_path = Path(args.test_path)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    # --- Schema checks ---
    required = {"id", "text", "label", "source"}
    for name, df in [("train", train), ("test", test)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"[{name}] Missing columns: {sorted(missing)}")

    # --- Label checks (IMDb should be binary 0/1) ---
    for name, df in [("train", train), ("test", test)]:
        bad = set(df["label"].dropna().unique()) - {0, 1}
        if bad:
            raise ValueError(f"[{name}] Unexpected labels found: {bad}")

    # --- Duplicate checks (exact duplicates) ---
    train_dups = int(train["text"].duplicated().sum())
    test_dups = int(test["text"].duplicated().sum())

    # --- Leakage check: duplicates across train/test ---
    train_texts = set(train["text"].astype(str).tolist())
    test_texts = set(test["text"].astype(str).tolist())
    overlap = len(train_texts.intersection(test_texts))

    # --- Profiles ---
    train_sum = basic_profile(train.copy(), "train")
    test_sum = basic_profile(test.copy(), "test")

    # --- Write report ---
    md = []
    md.append("# Raw Data Validation Report (IMDb)\n")

    def fmt_summary(s):
        lines = []
        lines.append(f"**Split:** {s['name']}")
        lines.append(f"- Rows: {s['rows']}")
        lines.append(f"- Null text: {s['null_text']}")
        lines.append(f"- Empty text: {s['empty_text']}")
        lines.append(f"- Label distribution: {s['label_values']}")
        lines.append(f"- Chars (min/p50/p95/max): {s['min_chars']}/{s['p50_chars']}/{s['p95_chars']}/{s['max_chars']}")
        lines.append(f"- Words (min/p50/p95/max): {s['min_words']}/{s['p50_words']}/{s['p95_words']}/{s['max_words']}")
        return "\n".join(lines)

    md.append("## Split summaries\n")
    md.append(fmt_summary(train_sum) + "\n")
    md.append(fmt_summary(test_sum) + "\n")

    md.append("## Duplicates & leakage\n")
    md.append(f"- Exact duplicate texts in train: **{train_dups}**")
    md.append(f"- Exact duplicate texts in test: **{test_dups}**")
    md.append(f"- Exact text overlap train ↔ test (leakage risk): **{overlap}**\n")

    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[OK] Wrote report -> {report_path}")


if __name__ == "__main__":
    main()
