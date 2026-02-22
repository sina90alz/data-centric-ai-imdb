from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import great_expectations as ge

SUITE_NAME = "imdb_raw_suite"
TRAIN_PATH = Path("data/raw/imdb_train.parquet")
TEST_PATH = Path("data/raw/imdb_test.parquet")

OUT_JSON = Path("reports/ge_validation_raw.json")
OUT_MD = Path("reports/ge_validation_raw.md")


def _validate_one(df: pd.DataFrame, suite, run_name: str) -> dict:
    gdf = ge.from_pandas(df)
    gdf.set_default_expectation_argument("result_format", "SUMMARY")
    res_obj = gdf.validate(expectation_suite=suite, run_name=run_name)

    # Convert GE result object to JSON-serializable dict
    try:
        return res_obj.to_json_dict()
    except Exception:
        # fallback: JSON string -> dict
        return json.loads(res_obj.to_json())


def _summarize(res: dict) -> dict:
    return {
        "success": bool(res.get("success")),
        "statistics": res.get("statistics", {}),
        "failed_expectations": [
            {
                "expectation_type": r["expectation_config"]["expectation_type"],
                "kwargs": r["expectation_config"].get("kwargs", {}),
                "result": r.get("result", {}),
                "exception_info": r.get("exception_info", {}),
            }
            for r in res.get("results", [])
            if r.get("success") is False
        ],
    }


def main():
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Missing raw parquet files. Run ingestion first.")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    context = ge.get_context()
    suite = context.get_expectation_suite(expectation_suite_name=SUITE_NAME)

    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    train_res = _validate_one(train, suite, run_name="imdb_raw_train")
    test_res = _validate_one(test, suite, run_name="imdb_raw_test")

    payload = {
        "suite_name": SUITE_NAME,
        "train": train_res,
        "test": test_res,
        "train_summary": _summarize(train_res),
        "test_summary": _summarize(test_res),
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def md_block(name: str, summary: dict) -> str:
        stats = summary.get("statistics", {})
        failed = summary.get("failed_expectations", [])
        lines = [
            f"## {name}",
            f"- Success: **{summary['success']}**",
            f"- Evaluated expectations: {stats.get('evaluated_expectations')}",
            f"- Successful expectations: {stats.get('successful_expectations')}",
            f"- Unsuccessful expectations: {stats.get('unsuccessful_expectations')}",
        ]
        if failed:
            lines.append("\n### Failed expectations")
            for f in failed[:10]:
                lines.append(f"- `{f['expectation_type']}` kwargs={f['kwargs']}")
        return "\n".join(lines)

    md = "\n\n".join(
        [
            "# GE Validation Report (Raw IMDb)\n",
            md_block("TRAIN", payload["train_summary"]),
            md_block("TEST", payload["test_summary"]),
            f"\nFull JSON: `{OUT_JSON.as_posix()}`\n",
        ]
    )
    OUT_MD.write_text(md, encoding="utf-8")

    print(f"[OK] Train success: {payload['train_summary']['success']}")
    print(f"[OK] Test  success: {payload['test_summary']['success']}")
    print(f"[OK] Wrote -> {OUT_JSON}")
    print(f"[OK] Wrote -> {OUT_MD}")

    # Still build docs for suite browsing
    context.build_data_docs()
    print(r"[OK] Data Docs: great_expectations\uncommitted\data_docs\local_site\index.html")


if __name__ == "__main__":
    main()
