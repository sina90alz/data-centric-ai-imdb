import argparse
from pathlib import Path
import pandas as pd

from src.data.cleaning import clean_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_train", default="data/raw/imdb_train.parquet")
    p.add_argument("--in_test", default="data/raw/imdb_test.parquet")
    p.add_argument("--out_dir", default="data/processed")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, in_path in [("train", args.in_train), ("test", args.in_test)]:
        df = pd.read_parquet(in_path)

        before_empty = int((df["text"].astype(str).str.strip() == "").sum())
        df["text_clean"] = df["text"].astype(str).apply(clean_text)
        after_empty = int((df["text_clean"].astype(str).str.strip() == "").sum())

        changed = int((df["text"].astype(str) != df["text_clean"].astype(str)).sum())

        # keep both columns for traceability
        out_path = out_dir / f"imdb_{split_name}_clean.parquet"
        df.to_parquet(out_path, index=False)

        print(f"[OK] {split_name}: wrote -> {out_path}")
        print(f"     changed rows: {changed} / {len(df)}")
        print(f"     empty before/after: {before_empty} -> {after_empty}")

    print("[DONE] Cleaning complete.")


if __name__ == "__main__":
    main()
