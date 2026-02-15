import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="stanfordnlp/imdb")  # or "imdb"
    p.add_argument("--out_dir", default="data/raw")
    p.add_argument("--revision", default=None)  # optional pin
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, revision=args.revision) if args.revision else load_dataset(args.dataset)

    # Expect splits: train/test (IMDb)
    for split in ds.keys():
        df = ds[split].to_pandas()

        # IMDb typically: text, label
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(f"Unexpected schema in split={split}. Columns: {list(df.columns)}")

        # Add id and source columns for evaluation/pipeline consistency
        df.insert(0, "id", range(1, len(df) + 1))
        df["source"] = split  # train/test

        out_path = out_dir / f"imdb_{split}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"[OK] Wrote {len(df)} rows -> {out_path}")

    print("[DONE] Raw snapshot created.")


if __name__ == "__main__":
    main()
