"""
merge.py

Aggregates all 7 Home Credit tables into a single wide DataFrame
with one row per applicant (SK_ID_CURR). Joins bureau, previous
application, POS_CASH, and installments aggregates onto the main
application_train table. Saves to data/processed/application_merged.parquet.
"""

import gc
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so `python src/data/merge.py` works
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.ingest import load_table

PROCESSED_DATA_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DATA_DIR / "application_merged.parquet"


# ---------------------------------------------------------------------------
# Helper: flatten MultiIndex columns
# ---------------------------------------------------------------------------


def _flatten_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Flatten MultiIndex columns to single-level strings with prefix."""
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            new_cols.append(f"{prefix}_{col[0]}_{col[1]}".upper())
        else:
            new_cols.append(f"{prefix}_{col}".upper())
    df.columns = new_cols
    return df


# ---------------------------------------------------------------------------
# Aggregation functions
# ---------------------------------------------------------------------------


def aggregate_bureau() -> pd.DataFrame:
    """
    Aggregates bureau.csv and bureau_balance.csv into per-applicant features.
    Returns DataFrame indexed by SK_ID_CURR with bureau aggregate columns.
    """
    # --- Bureau aggregations ---
    print("  Loading bureau.csv...")
    bureau = load_table("bureau.csv")

    bureau_agg = {
        "DAYS_CREDIT": ["mean", "max", "min", "std"],
        "CREDIT_DAY_OVERDUE": ["mean", "max"],
        "DAYS_CREDIT_ENDDATE": ["mean", "max"],
        "AMT_CREDIT_MAX_OVERDUE": ["mean", "max"],
        "CNT_CREDIT_PROLONG": ["sum", "mean"],
        "AMT_CREDIT_SUM": ["mean", "max", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["mean", "max", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean", "sum"],
        "DAYS_CREDIT_UPDATE": ["mean", "max"],
        "AMT_ANNUITY": ["mean", "max"],
    }

    bureau_grouped = bureau.groupby("SK_ID_CURR").agg(bureau_agg)
    bureau_grouped = _flatten_columns(bureau_grouped, "BUREAU")

    # Loan counts
    bureau_grouped["BUREAU_LOAN_COUNT"] = (
        bureau.groupby("SK_ID_CURR").size().astype("float32")
    )
    bureau_grouped["BUREAU_ACTIVE_LOAN_COUNT"] = (
        bureau[bureau["CREDIT_ACTIVE"] == "Active"]
        .groupby("SK_ID_CURR")
        .size()
        .astype("float32")
    )
    bureau_grouped["BUREAU_CLOSED_LOAN_COUNT"] = (
        bureau[bureau["CREDIT_ACTIVE"] == "Closed"]
        .groupby("SK_ID_CURR")
        .size()
        .astype("float32")
    )

    # Keep SK_ID_BUREAU mapping before deleting bureau
    bureau_id_map = bureau[["SK_ID_BUREAU", "SK_ID_CURR"]].copy()
    del bureau
    gc.collect()

    # --- Bureau Balance aggregations ---
    print("  Loading bureau_balance.csv...")
    bb = load_table("bureau_balance.csv")

    # Merge to get SK_ID_CURR
    bb = bb.merge(bureau_id_map, on="SK_ID_BUREAU", how="left")
    del bureau_id_map
    gc.collect()

    bb_agg = {
        "MONTHS_BALANCE": ["mean", "min", "max", "size"],
    }

    bb_grouped = bb.groupby("SK_ID_CURR").agg(bb_agg)
    bb_grouped = _flatten_columns(bb_grouped, "BB")

    # Status rates
    total_per_applicant = bb.groupby("SK_ID_CURR").size()
    bb_grouped["BB_STATUS_0_RATE"] = (
        bb[bb["STATUS"] == "0"].groupby("SK_ID_CURR").size() / total_per_applicant
    ).astype("float32")
    bb_grouped["BB_STATUS_C_RATE"] = (
        bb[bb["STATUS"] == "C"].groupby("SK_ID_CURR").size() / total_per_applicant
    ).astype("float32")
    bb_grouped["BB_DPD_RATE"] = (
        bb[~bb["STATUS"].isin(["0", "C", "X"])].groupby("SK_ID_CURR").size()
        / total_per_applicant
    ).astype("float32")

    del bb, total_per_applicant
    gc.collect()

    # Merge bureau agg with bureau_balance agg
    result = bureau_grouped.merge(bb_grouped, on="SK_ID_CURR", how="left")
    del bureau_grouped, bb_grouped
    gc.collect()

    # Cast all numeric columns to float32
    for col in result.columns:
        if result[col].dtype in ["float64", "int64", "int32", "int16", "int8"]:
            result[col] = result[col].astype("float32")

    print(f"  Bureau aggregation done: {result.shape}")
    return result


def aggregate_previous_applications() -> pd.DataFrame:
    """
    Aggregates previous_application.csv into per-applicant features.
    Returns DataFrame indexed by SK_ID_CURR.
    """
    print("  Loading previous_application.csv...")
    prev = load_table("previous_application.csv")

    prev_agg = {
        "AMT_ANNUITY": ["mean", "max", "min"],
        "AMT_APPLICATION": ["mean", "max", "sum"],
        "AMT_CREDIT": ["mean", "max", "sum"],
        "AMT_DOWN_PAYMENT": ["mean", "max"],
        "AMT_GOODS_PRICE": ["mean", "max"],
        "RATE_DOWN_PAYMENT": ["mean", "max"],
        "DAYS_DECISION": ["mean", "max", "min"],
        "CNT_PAYMENT": ["mean", "sum"],
        "DAYS_FIRST_DRAWING": ["mean", "max"],
        "DAYS_FIRST_DUE": ["mean", "max"],
        "DAYS_LAST_DUE": ["mean", "max"],
        "DAYS_TERMINATION": ["mean", "max"],
    }

    result = prev.groupby("SK_ID_CURR").agg(prev_agg)
    result = _flatten_columns(result, "PREV")

    # Loan counts
    result["PREV_LOAN_COUNT"] = prev.groupby("SK_ID_CURR").size().astype("float32")
    result["PREV_APPROVED_COUNT"] = (
        prev[prev["NAME_CONTRACT_STATUS"] == "Approved"]
        .groupby("SK_ID_CURR")
        .size()
        .astype("float32")
    )
    result["PREV_REFUSED_COUNT"] = (
        prev[prev["NAME_CONTRACT_STATUS"] == "Refused"]
        .groupby("SK_ID_CURR")
        .size()
        .astype("float32")
    )
    result["PREV_APPROVAL_RATE"] = (
        result["PREV_APPROVED_COUNT"] / result["PREV_LOAN_COUNT"]
    ).astype("float32")

    del prev
    gc.collect()

    # Cast all numeric columns to float32
    for col in result.columns:
        if result[col].dtype in ["float64", "int64", "int32", "int16", "int8"]:
            result[col] = result[col].astype("float32")

    print(f"  Previous applications aggregation done: {result.shape}")
    return result


def aggregate_pos_cash() -> pd.DataFrame:
    """
    Aggregates POS_CASH_balance.csv into per-applicant features.
    Returns DataFrame indexed by SK_ID_CURR.
    """
    print("  Loading POS_CASH_balance.csv...")
    pos = load_table("POS_CASH_balance.csv")

    pos_agg = {
        "MONTHS_BALANCE": ["mean", "min", "max", "size"],
        "CNT_INSTALMENT": ["mean", "max"],
        "CNT_INSTALMENT_FUTURE": ["mean", "max"],
        "SK_DPD": ["mean", "max", "sum"],
        "SK_DPD_DEF": ["mean", "max", "sum"],
    }

    result = pos.groupby("SK_ID_CURR").agg(pos_agg)
    result = _flatten_columns(result, "POS")

    # Unique loan count
    result["POS_LOAN_COUNT"] = (
        pos.groupby("SK_ID_CURR")["SK_ID_PREV"].nunique().astype("float32")
    )

    # Completed rate
    total_per_applicant = pos.groupby("SK_ID_CURR").size()
    result["POS_COMPLETED_RATE"] = (
        pos[pos["NAME_CONTRACT_STATUS"] == "Completed"].groupby("SK_ID_CURR").size()
        / total_per_applicant
    ).astype("float32")

    del pos, total_per_applicant
    gc.collect()

    # Cast all numeric columns to float32
    for col in result.columns:
        if result[col].dtype in ["float64", "int64", "int32", "int16", "int8"]:
            result[col] = result[col].astype("float32")

    print(f"  POS Cash aggregation done: {result.shape}")
    return result


def aggregate_installments() -> pd.DataFrame:
    """
    Aggregates installments_payments.csv into per-applicant features.
    Returns DataFrame indexed by SK_ID_CURR.
    """
    print("  Loading installments_payments.csv...")
    inst = load_table("installments_payments.csv")

    inst_agg = {
        "NUM_INSTALMENT_VERSION": ["mean", "max"],
        "NUM_INSTALMENT_NUMBER": ["mean", "max"],
        "DAYS_INSTALMENT": ["mean", "max", "min"],
        "DAYS_ENTRY_PAYMENT": ["mean", "max", "min"],
        "AMT_INSTALMENT": ["mean", "max", "sum"],
        "AMT_PAYMENT": ["mean", "max", "sum"],
    }

    result = inst.groupby("SK_ID_CURR").agg(inst_agg)
    result = _flatten_columns(result, "INST")

    # Total installment records per applicant
    result["INST_COUNT"] = inst.groupby("SK_ID_CURR").size().astype("float32")

    # Payment ratio: AMT_PAYMENT / AMT_INSTALMENT (mean per applicant)
    inst["PAYMENT_RATIO"] = np.where(
        inst["AMT_INSTALMENT"] > 0,
        inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"],
        np.nan,
    )
    result["INST_PAYMENT_RATIO"] = (
        inst.groupby("SK_ID_CURR")["PAYMENT_RATIO"].mean().astype("float32")
    )

    # Days past due (positive means late)
    inst["DAYS_PAST_DUE"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]

    # Mean of late payments only
    late_mask = inst["DAYS_PAST_DUE"] > 0
    late_dpd = inst.loc[late_mask].groupby("SK_ID_CURR")["DAYS_PAST_DUE"].mean()
    result["INST_DAYS_PAST_DUE"] = late_dpd.astype("float32")

    # Late payment rate
    total_per_applicant = inst.groupby("SK_ID_CURR").size()
    late_per_applicant = inst.loc[late_mask].groupby("SK_ID_CURR").size()
    result["INST_LATE_PAYMENT_RATE"] = (
        late_per_applicant / total_per_applicant
    ).astype("float32")

    del inst, total_per_applicant, late_per_applicant
    gc.collect()

    # Cast all numeric columns to float32
    for col in result.columns:
        if result[col].dtype in ["float64", "int64", "int32", "int16", "int8"]:
            result[col] = result[col].astype("float32")

    print(f"  Installments aggregation done: {result.shape}")
    return result


# ---------------------------------------------------------------------------
# Build merged dataset
# ---------------------------------------------------------------------------


def build_merged_dataset(save: bool = True) -> pd.DataFrame:
    """
    Orchestrates all aggregations and joins them onto application_train.

    Args:
        save: if True, saves to data/processed/application_merged.parquet
              and registers with DVC.
    Returns:
        Merged DataFrame with 307,511 rows.
    """
    # Step 1: load main table
    print("Loading application_train.csv...")
    app = load_table("application_train.csv")
    n_applicants = len(app)
    print(f"  {n_applicants:,} applicants")

    # Step 2: run each aggregation and join
    aggregations = [
        ("Bureau", aggregate_bureau),
        ("Previous applications", aggregate_previous_applications),
        ("POS Cash", aggregate_pos_cash),
        ("Installments", aggregate_installments),
    ]

    for name, agg_fn in aggregations:
        print(f"\nAggregating {name}...")
        agg_df = agg_fn()
        print(f"  {len(agg_df):,} applicants, {agg_df.shape[1]} features")
        app = app.merge(agg_df, on="SK_ID_CURR", how="left")
        print(f"  Merged. Shape: {app.shape}")
        del agg_df
        gc.collect()

    # Step 3: assert row count preserved
    assert (
        len(app) == n_applicants
    ), f"Row count changed after joins: expected {n_applicants}, got {len(app)}"

    # Step 4: print summary
    print("\nMerge complete:")
    print(f"  Rows:    {len(app):,}")
    print(f"  Columns: {app.shape[1]}")
    print(f"  Memory:  {app.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    null_rate = app.isnull().mean().mean()
    print(f"  Overall null rate: {null_rate:.1%}")

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving to {OUTPUT_PATH}...")
        app.to_parquet(OUTPUT_PATH, index=False, compression="snappy")
        print(f"  Saved. File size: {OUTPUT_PATH.stat().st_size / 1e6:.1f} MB")
        _register_with_dvc()

    return app


def _register_with_dvc() -> None:
    """Registers the merged Parquet file with DVC."""
    # Find dvc binary relative to the Python interpreter (same venv)
    dvc_bin = Path(sys.executable).parent / "dvc"
    dvc_cmd = str(dvc_bin) if dvc_bin.exists() else "dvc"

    result = subprocess.run(
        [dvc_cmd, "add", str(OUTPUT_PATH)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Warning: DVC registration failed: {result.stderr}")
    else:
        print("  Registered with DVC.")
        print("  Run: git add data/processed/application_merged.parquet.dvc")


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------


def load_merged() -> pd.DataFrame:
    """
    Loads the merged Parquet file. Raises FileNotFoundError if not built yet.
    This is the function all downstream steps use to load data.
    """
    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"Merged dataset not found at {OUTPUT_PATH}. "
            "Run `python src/data/merge.py` to build it."
        )
    print(f"Loading merged dataset from {OUTPUT_PATH}...")
    df = pd.read_parquet(OUTPUT_PATH)
    print(f"  Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_merged_dataset(save=True)
