from pathlib import Path
import yaml

BASE = Path("great_expectations")
GX_YML = BASE / "great_expectations.yml"
UNCOMMITTED = BASE / "uncommitted"

def main():
    BASE.mkdir(exist_ok=True)
    (BASE / "expectations").mkdir(parents=True, exist_ok=True)
    (BASE / "plugins").mkdir(parents=True, exist_ok=True)
    (BASE / "checkpoints").mkdir(parents=True, exist_ok=True)
    UNCOMMITTED.mkdir(parents=True, exist_ok=True)
    (UNCOMMITTED / "data_docs").mkdir(parents=True, exist_ok=True)
    (UNCOMMITTED / "validations").mkdir(parents=True, exist_ok=True)

    if not GX_YML.exists():
        cfg = {
            "config_version": 3,
            "datasources": {},
            "plugins_directory": "plugins",
            "stores": {
                "expectations_store": {
                    "class_name": "ExpectationsStore",
                    "store_backend": {
                        "class_name": "TupleFilesystemStoreBackend",
                        "base_directory": "expectations",
                    },
                },
                "validations_store": {
                    "class_name": "ValidationsStore",
                    "store_backend": {
                        "class_name": "TupleFilesystemStoreBackend",
                        "base_directory": "uncommitted/validations",
                    },
                },
                "evaluation_parameter_store": {"class_name": "EvaluationParameterStore"},
                "checkpoint_store": {
                    "class_name": "CheckpointStore",
                    "store_backend": {
                        "class_name": "TupleFilesystemStoreBackend",
                        "base_directory": "checkpoints",
                    },
                },
            },
            "expectations_store_name": "expectations_store",
            "validations_store_name": "validations_store",
            "evaluation_parameter_store_name": "evaluation_parameter_store",
            "checkpoint_store_name": "checkpoint_store",
            "data_docs_sites": {
                "local_site": {
                    "class_name": "SiteBuilder",
                    "store_backend": {
                        "class_name": "TupleFilesystemStoreBackend",
                        "base_directory": "uncommitted/data_docs/local_site",
                    },
                    "site_index_builder": {"class_name": "DefaultSiteIndexBuilder"},
                }
            },
            "anonymous_usage_statistics": {"enabled": False},
        }
        GX_YML.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(f"[OK] Created {GX_YML}")

    print("[DONE] GE bootstrap complete.")
    print("Next: run suite builder + validation scripts.")

if __name__ == "__main__":
    main()
