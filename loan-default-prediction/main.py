"""Run Phase 1 database and raw dataset checks."""

from src.data.loader import (
    DATASET_PATH,
    EXPECTED_COLUMNS,
    load_dataset,
    print_dataset_summary,
    validate_dataset,
)
from src.database import verify_database_connection


def main() -> int:
    print("=" * 40)
    print("LOAN DEFAULT PREDICTION SYSTEM\nPHASE 1")
    print("=" * 40)
    print("\nDatabase\n--------")
    database_connected = True
    try:
        verify_database_connection()
        print("PostgreSQL: Connected")
    except RuntimeError as exc:
        database_connected = False
        print(f"PostgreSQL: Unavailable\n{exc}")

    print(f"\nFile: {DATASET_PATH.name}")
    try:
        dataframe = load_dataset()
        validate_dataset(dataframe)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Dataset validation failed: {exc}")
        return 1
    print_dataset_summary(dataframe)
    print("\nValidation\n----------")
    print(f"Required columns: {'PASS' if list(dataframe.columns) == EXPECTED_COLUMNS else 'FAIL'}")
    print("Target validation: PASS")
    print("\nPhase 1 dataset validation completed successfully.")
    if not database_connected:
        print("PostgreSQL check failed; start/configure PostgreSQL to complete database verification.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
