import argparse
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", default="data/raw/imdb_train.parquet")
    p.add_argument("--test_path", default="data/raw/imdb_test.parquet")
    p.add_argument("--out_test_path", default="data/processed/imdb_test_noleak.parquet")
    p.add_argument("--report_path", default="reports/leakage_fix.md")
    p.add_argument("--use_clean_text", action="store_true",
                   help="If set, use text_clean column instead of text (requires cleaned files).")
    args = p.parse_args()

    train_path = Path(args.train_path)
    test_path = Path(args.test_path)
    out_test_path = Path(args.out_test_path)
    report_path = Path(args.report_path)

    out_test_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    col = "text_clean" if args.use_clean_text else "text"

    if col not in train.columns or col not in test.columns:
        raise ValueError(f"Column '{col}' not found in train/test. "
                         f"Available train cols: {list(train.columns)}; test cols: {list(test.columns)}")

    # Drop missing texts (paranoia safety)
    train_text = train[col].dropna().astype(str)
    test_text = test[col].dropna().astype(str)

    train_set = set(train_text.values)
    test_set = set(test_text.values)

    overlap_texts = train_set.intersection(test_set)
    overlap_count = len(overlap_texts)

    # Remove overlapping rows from test
    before_rows = len(test)
    mask_leak = test[col].astype(str).isin(overlap_texts)
    removed_rows = int(mask_leak.sum())

    test_noleak = test.loc[~mask_leak].copy()
    after_rows = len(test_noleak)

    # Save
    test_noleak.to_parquet(out_test_path, index=False)

    # Report
    md = []
    md.append("# Leakage Removal Report\n")
    md.append(f"- Train file: `{train_path.as_posix()}`")
    md.append(f"- Test file: `{test_path.as_posix()}`")
    md.append(f"- Text column used: `{col}`\n")

    md.append("## Summary\n")
    md.append(f"- Unique texts in train: **{len(train_set)}**")
    md.append(f"- Unique texts in test: **{len(test_set)}**")
    md.append(f"- Exact overlap unique texts (train ∩ test): **{overlap_count}**")
    md.append(f"- Test rows removed (leakage rows): **{removed_rows}**")
    md.append(f"- Test rows before: **{before_rows}**")
    md.append(f"- Test rows after: **{after_rows}**\n")

    # Show a few examples for audit
    examples = list(overlap_texts)[:5]
    if examples:
        md.append("## Example overlapping texts (first 5)\n")
        for i, t in enumerate(examples, 1):
            # keep it short to avoid huge markdown
            snippet = (t[:200] + "…") if len(t) > 200 else t
            md.append(f"{i}. {snippet}")

    report_path.write_text("\n".join(md), encoding="utf-8")

    print(f"[OK] Overlap unique texts: {overlap_count}")
    print(f"[OK] Removed test rows: {removed_rows}")
    print(f"[OK] Wrote cleaned test -> {out_test_path}")
    print(f"[OK] Wrote report      -> {report_path}")


if __name__ == "__main__":
    main()
