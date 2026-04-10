"""
ingest.py

Downloads all 7 Home Credit CSV files from Kaggle into data/raw/,
loads them with explicit dtype mappings, and registers with DVC.
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Dtype maps for all 7 tables
# ---------------------------------------------------------------------------

APPLICATION_TRAIN_DTYPES = {
    "SK_ID_CURR": "int32",
    "TARGET": "int8",
    "NAME_CONTRACT_TYPE": "category",
    "CODE_GENDER": "category",
    "FLAG_OWN_CAR": "category",
    "FLAG_OWN_REALTY": "category",
    "CNT_CHILDREN": "int8",
    "AMT_INCOME_TOTAL": "float32",
    "AMT_CREDIT": "float32",
    "AMT_ANNUITY": "float32",
    "AMT_GOODS_PRICE": "float32",
    "NAME_TYPE_SUITE": "category",
    "NAME_INCOME_TYPE": "category",
    "NAME_EDUCATION_TYPE": "category",
    "NAME_FAMILY_STATUS": "category",
    "NAME_HOUSING_TYPE": "category",
    "REGION_POPULATION_RELATIVE": "float32",
    "DAYS_BIRTH": "int32",
    "DAYS_EMPLOYED": "int32",
    "DAYS_REGISTRATION": "float32",
    "DAYS_ID_PUBLISH": "int32",
    "OWN_CAR_AGE": "float32",
    "FLAG_MOBIL": "int8",
    "FLAG_EMP_PHONE": "int8",
    "FLAG_WORK_PHONE": "int8",
    "FLAG_CONT_MOBILE": "int8",
    "FLAG_PHONE": "int8",
    "FLAG_EMAIL": "int8",
    "OCCUPATION_TYPE": "category",
    "CNT_FAM_MEMBERS": "float32",
    "REGION_RATING_CLIENT": "int8",
    "REGION_RATING_CLIENT_W_CITY": "int8",
    "WEEKDAY_APPR_PROCESS_START": "category",
    "HOUR_APPR_PROCESS_START": "int8",
    "REG_REGION_NOT_LIVE_REGION": "int8",
    "REG_REGION_NOT_WORK_REGION": "int8",
    "LIVE_REGION_NOT_WORK_REGION": "int8",
    "REG_CITY_NOT_LIVE_CITY": "int8",
    "REG_CITY_NOT_WORK_CITY": "int8",
    "LIVE_CITY_NOT_WORK_CITY": "int8",
    "ORGANIZATION_TYPE": "category",
    "EXT_SOURCE_1": "float32",
    "EXT_SOURCE_2": "float32",
    "EXT_SOURCE_3": "float32",
}

APPLICATION_TEST_DTYPES = {
    k: v for k, v in APPLICATION_TRAIN_DTYPES.items() if k != "TARGET"
}

BUREAU_DTYPES = {
    "SK_ID_CURR": "int32",
    "SK_ID_BUREAU": "int32",
    "CREDIT_ACTIVE": "category",
    "CREDIT_CURRENCY": "category",
    "DAYS_CREDIT": "int32",
    "CREDIT_DAY_OVERDUE": "int16",
    "DAYS_CREDIT_ENDDATE": "float32",
    "DAYS_ENDDATE_FACT": "float32",
    "AMT_CREDIT_MAX_OVERDUE": "float32",
    "CNT_CREDIT_PROLONG": "int8",
    "AMT_CREDIT_SUM": "float32",
    "AMT_CREDIT_SUM_DEBT": "float32",
    "AMT_CREDIT_SUM_LIMIT": "float32",
    "AMT_CREDIT_SUM_OVERDUE": "float32",
    "CREDIT_TYPE": "category",
    "DAYS_CREDIT_UPDATE": "int32",
    "AMT_ANNUITY": "float32",
}

BUREAU_BALANCE_DTYPES = {
    "SK_ID_BUREAU": "int32",
    "MONTHS_BALANCE": "int16",
    "STATUS": "category",
}

PREVIOUS_APPLICATION_DTYPES = {
    "SK_ID_CURR": "int32",
    "SK_ID_PREV": "int32",
    "NAME_CONTRACT_TYPE": "category",
    "AMT_ANNUITY": "float32",
    "AMT_APPLICATION": "float32",
    "AMT_CREDIT": "float32",
    "AMT_DOWN_PAYMENT": "float32",
    "AMT_GOODS_PRICE": "float32",
    "WEEKDAY_APPR_PROCESS_START": "category",
    "HOUR_APPR_PROCESS_START": "int8",
    "FLAG_LAST_APPL_PER_CONTRACT": "category",
    "NFLAG_LAST_APPL_IN_DAY": "int8",
    "RATE_DOWN_PAYMENT": "float32",
    "RATE_INTEREST_PRIMARY": "float32",
    "RATE_INTEREST_PRIVILEGED": "float32",
    "NAME_CASH_LOAN_PURPOSE": "category",
    "NAME_CONTRACT_STATUS": "category",
    "DAYS_DECISION": "int32",
    "NAME_PAYMENT_TYPE": "category",
    "CODE_REJECT_REASON": "category",
    "NAME_CLIENT_TYPE": "category",
    "NAME_GOODS_CATEGORY": "category",
    "NAME_PORTFOLIO": "category",
    "NAME_PRODUCT_TYPE": "category",
    "CHANNEL_TYPE": "category",
    "SELLERPLACE_AREA": "int32",
    "NAME_SELLER_INDUSTRY": "category",
    "CNT_PAYMENT": "float32",
    "NAME_YIELD_GROUP": "category",
    "PRODUCT_COMBINATION": "category",
    "DAYS_FIRST_DRAWING": "float32",
    "DAYS_FIRST_DUE": "float32",
    "DAYS_LAST_DUE_1ST_VERSION": "float32",
    "DAYS_LAST_DUE": "float32",
    "DAYS_TERMINATION": "float32",
    "NFLAG_INSURED_ON_APPROVAL": "float32",
}

POS_CASH_DTYPES = {
    "SK_ID_CURR": "int32",
    "SK_ID_PREV": "int32",
    "MONTHS_BALANCE": "int16",
    "CNT_INSTALMENT": "float32",
    "CNT_INSTALMENT_FUTURE": "float32",
    "NAME_CONTRACT_STATUS": "category",
    "SK_DPD": "int16",
    "SK_DPD_DEF": "int16",
}

INSTALLMENTS_DTYPES = {
    "SK_ID_CURR": "int32",
    "SK_ID_PREV": "int32",
    "NUM_INSTALMENT_VERSION": "float32",
    "NUM_INSTALMENT_NUMBER": "int16",
    "DAYS_INSTALMENT": "float32",
    "DAYS_ENTRY_PAYMENT": "float32",
    "AMT_INSTALMENT": "float32",
    "AMT_PAYMENT": "float32",
}

DTYPE_MAPS = {
    "application_train.csv": APPLICATION_TRAIN_DTYPES,
    "application_test.csv": APPLICATION_TEST_DTYPES,
    "bureau.csv": BUREAU_DTYPES,
    "bureau_balance.csv": BUREAU_BALANCE_DTYPES,
    "previous_application.csv": PREVIOUS_APPLICATION_DTYPES,
    "POS_CASH_balance.csv": POS_CASH_DTYPES,
    "installments_payments.csv": INSTALLMENTS_DTYPES,
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_DATA_DIR = Path("data/raw")
COMPETITION_NAME = "home-credit-default-risk"
TABLE_NAMES = list(DTYPE_MAPS.keys())

CREDENTIALS_HELP = """\
Kaggle credentials not found.

To get your API key:
1. Go to https://www.kaggle.com/settings/account
2. Scroll to "API" section
3. Click "Create New Token" — downloads kaggle.json
4. Copy the values into your .env file:
   KAGGLE_USERNAME=<username from kaggle.json>
   KAGGLE_KEY=<key from kaggle.json>

You must also accept the competition rules at:
https://www.kaggle.com/competitions/home-credit-default-risk/rules
(one-time, required before API download works)
"""

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def download_data() -> None:
    """
    Downloads all 7 Home Credit CSVs from Kaggle into data/raw/.
    Idempotent — skips files that already exist.
    Requires KAGGLE_USERNAME and KAGGLE_KEY in .env or environment.
    """
    # Load .env before any Kaggle calls
    load_dotenv()

    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if not username or not key:
        print(CREDENTIALS_HELP)
        sys.exit(1)

    # Set env vars so the kaggle API picks them up
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    # Import kaggle after setting credentials
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename in TABLE_NAMES:
        filepath = RAW_DATA_DIR / filename

        if filepath.exists():
            print(f"Skipping {filename} (already exists)")
            continue

        print(f"Downloading {filename}...")
        api.competition_download_file(
            COMPETITION_NAME,
            filename,
            path=str(RAW_DATA_DIR),
        )

        # Unzip if the API downloaded a zip file
        zip_path = RAW_DATA_DIR / f"{filename}.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(RAW_DATA_DIR)
            zip_path.unlink()

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Filename':<30} {'Size (MB)':>10} {'Rows':>12}")
    print("-" * 60)
    for filename in TABLE_NAMES:
        filepath = RAW_DATA_DIR / filename
        size_mb = filepath.stat().st_size / (1024 * 1024)
        # Count rows without loading entire file into memory
        with open(filepath) as f:
            row_count = sum(1 for _ in f) - 1  # subtract header
        print(f"{filename:<30} {size_mb:>10.1f} {row_count:>12,}")
    print("=" * 60)


def load_table(filename: str, nrows: int | None = None) -> pd.DataFrame:
    """
    Loads a single Home Credit table with explicit dtypes.
    Args:
        filename: one of the 7 table filenames (e.g. 'application_train.csv')
        nrows: if provided, load only this many rows (useful for development)
    Returns:
        DataFrame with correct dtypes applied
    Raises:
        FileNotFoundError: if the CSV does not exist in data/raw/
        ValueError: if filename is not in DTYPE_MAPS
    """
    if filename not in DTYPE_MAPS:
        raise ValueError(
            f"Unknown table '{filename}'. " f"Must be one of: {', '.join(TABLE_NAMES)}"
        )

    filepath = RAW_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"File not found: {filepath}. "
            f"Run 'python src/data/ingest.py --download' first."
        )

    df = pd.read_csv(
        filepath,
        dtype=DTYPE_MAPS[filename],
        nrows=nrows,
        low_memory=False,
    )

    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"{filename}: {df.shape}, memory usage: {mem_mb:.1f} MB")

    return df


def load_all_tables(
    nrows: int | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Loads all 7 tables and returns them as a dict keyed by filename.
    """
    tables: dict[str, pd.DataFrame] = {}
    for filename in TABLE_NAMES:
        tables[filename] = load_table(filename, nrows=nrows)
    return tables


def register_with_dvc() -> None:
    """
    Runs `dvc add` on every CSV in data/raw/ and stages the
    resulting .dvc files for git commit.
    """
    for filename in TABLE_NAMES:
        filepath = RAW_DATA_DIR / filename
        if not filepath.exists():
            print(f"WARNING: {filepath} does not exist, skipping DVC add")
            continue

        print(f"Running: dvc add {filepath}")
        result = subprocess.run(
            ["dvc", "add", str(filepath)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"dvc add failed for {filepath}:\n{result.stderr}")

    print("\nDVC registration complete. Run:")
    print("  git add data/raw/*.dvc data/raw/.gitignore")
    print("  git commit -m 'data: register raw tables with DVC'")


def main() -> None:
    """CLI entry point with --download and --load flags."""
    parser = argparse.ArgumentParser(
        description="Home Credit data ingestion pipeline",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download all 7 CSVs from Kaggle and register with DVC",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load all 7 tables and print shape/memory info",
    )

    args = parser.parse_args()

    if not args.download and not args.load:
        parser.print_help()
        sys.exit(1)

    if args.download:
        download_data()
        register_with_dvc()

    if args.load:
        load_all_tables()


if __name__ == "__main__":
    main()
