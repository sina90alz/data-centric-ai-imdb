from __future__ import annotations

from pathlib import Path
import pandas as pd
import great_expectations as ge

SUITE_NAME = "imdb_raw_suite"
TRAIN_PATH = Path("data/raw/imdb_train.parquet")


def _get_or_create_suite(context, suite_name: str):
    try:
        return context.get_expectation_suite(expectation_suite_name=suite_name)
    except Exception:
        # create new suite
        return context.add_expectation_suite(expectation_suite_name=suite_name)


def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing {TRAIN_PATH}. Run ingestion first.")

    context = ge.get_context()

    # Create or load suite
    suite = _get_or_create_suite(context, SUITE_NAME)

    df = pd.read_parquet(TRAIN_PATH)

    gdf = ge.from_pandas(df)
    gdf.set_default_expectation_argument("result_format", "SUMMARY")

    # --- Expectations ---
    gdf.expect_table_columns_to_match_ordered_list(["id", "text", "label", "source"])
    gdf.expect_column_values_to_not_be_null("text")
    gdf.expect_column_values_to_not_be_null("label")
    gdf.expect_column_values_to_be_in_set("label", [0, 1])

    gdf.expect_table_row_count_to_be_between(min_value=20000, max_value=30000)
    gdf.expect_column_value_lengths_to_be_between("text", min_value=20, max_value=20000)
    gdf.expect_column_values_to_be_unique("id")

    # Save suite (from gdf expectations)
    context.save_expectation_suite(
        expectation_suite=gdf.get_expectation_suite(discard_failed_expectations=False),
        expectation_suite_name=SUITE_NAME,
    )

    context.build_data_docs()

    print(f"[OK] Saved suite: {SUITE_NAME}")
    print(r"[OK] Data Docs: great_expectations\uncommitted\data_docs\local_site\index.html")


if __name__ == "__main__":
    main()
