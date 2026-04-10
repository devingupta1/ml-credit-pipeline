"""
validate.py

Typed, validated ingestion layer using Pandera schema contracts and
Great Expectations quality gates. If schema changes or null rates
spike, the pipeline crashes here before corrupting downstream steps.
"""

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so `python src/data/validate.py` works
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import great_expectations as ge
import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

from src.data.ingest import load_table

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class ValidationError(Exception):
    """Raised when data validation fails. Contains details of failed checks."""

    pass


# ---------------------------------------------------------------------------
# Pandera schema for application_train
# ---------------------------------------------------------------------------

APPLICATION_TRAIN_SCHEMA = DataFrameSchema(
    columns={
        "SK_ID_CURR": Column(
            "int32",
            nullable=False,
            unique=True,
            checks=Check.gt(0),
            description="Unique loan application ID",
        ),
        "TARGET": Column(
            "int8",
            nullable=False,
            checks=Check.isin([0, 1]),
            description="1 = defaulted, 0 = repaid",
        ),
        "AMT_INCOME_TOTAL": Column(
            "float32",
            nullable=False,
            checks=Check.gt(0),
            description="Applicant annual income",
        ),
        "AMT_CREDIT": Column(
            "float32",
            nullable=True,
            checks=Check.gt(0),
            description="Loan credit amount",
        ),
        "AMT_ANNUITY": Column(
            "float32",
            nullable=True,
            checks=Check.gt(0),
            description="Loan annuity",
        ),
        "DAYS_BIRTH": Column(
            "int32",
            nullable=False,
            checks=Check.lt(0),
            description="Days since birth (negative = past)",
        ),
        "DAYS_EMPLOYED": Column(
            "int32",
            nullable=False,
            description="Days employed (365243 = unemployed sentinel)",
        ),
        "CODE_GENDER": Column(
            "category",
            nullable=False,
            checks=Check.isin(["M", "F", "XNA"]),
            description="Applicant gender",
        ),
        "NAME_CONTRACT_TYPE": Column(
            "category",
            nullable=False,
            checks=Check.isin(["Cash loans", "Revolving loans"]),
            description="Loan contract type",
        ),
        "EXT_SOURCE_1": Column(
            "float32",
            nullable=True,
            checks=Check.in_range(0.0, 1.0),
            description="Normalized external score 1",
        ),
        "EXT_SOURCE_2": Column(
            "float32",
            nullable=True,
            checks=Check.in_range(0.0, 1.0),
            description="Normalized external score 2",
        ),
        "EXT_SOURCE_3": Column(
            "float32",
            nullable=True,
            checks=Check.in_range(0.0, 1.0),
            description="Normalized external score 3",
        ),
        "REGION_RATING_CLIENT": Column(
            "int8",
            nullable=False,
            checks=Check.isin([1, 2, 3]),
            description="Region risk rating 1-3",
        ),
        "CNT_CHILDREN": Column(
            "int8",
            nullable=False,
            checks=Check.ge(0),
            description="Number of children",
        ),
    },
    coerce=False,
    strict=False,
)

# ---------------------------------------------------------------------------
# Reports directory
# ---------------------------------------------------------------------------

REPORTS_DIR = Path("reports")

# ---------------------------------------------------------------------------
# Pandera validation
# ---------------------------------------------------------------------------


def validate_pandera(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Runs Pandera schema validation against application_train DataFrame.

    Returns:
        (passed: bool, message: str)
        passed=True if all checks pass
        passed=False with descriptive message if any check fails
    """
    try:
        APPLICATION_TRAIN_SCHEMA.validate(df, lazy=True)
        return (True, "Pandera validation passed — all checks green")
    except pa.errors.SchemaErrors as exc:
        failure_cases = exc.failure_cases
        # Build a human-readable summary of failures
        lines = ["Pandera validation FAILED:"]
        for _, row in failure_cases.iterrows():
            col = row.get("column", "N/A")
            check = row.get("check", "N/A")
            count = row.get("failure_case", "N/A")
            lines.append(f"  Column '{col}' — check '{check}' — value: {count}")
        msg = "\n".join(lines)
        return (False, msg)
    except pa.errors.SchemaError as exc:
        return (False, f"Pandera validation FAILED: {exc}")


# ---------------------------------------------------------------------------
# Great Expectations checks
# ---------------------------------------------------------------------------


def build_ge_suite(df: pd.DataFrame):
    """
    Builds a Great Expectations suite on the application_train DataFrame.
    Returns the GE dataset with all expectations attached.
    """
    ge_df = ge.from_pandas(df)

    # Completeness checks
    ge_df.expect_column_to_exist("SK_ID_CURR")
    ge_df.expect_column_to_exist("TARGET")
    ge_df.expect_column_values_to_not_be_null("SK_ID_CURR")
    ge_df.expect_column_values_to_not_be_null("TARGET")
    ge_df.expect_column_values_to_not_be_null("AMT_INCOME_TOTAL")
    ge_df.expect_column_values_to_not_be_null("DAYS_BIRTH")
    ge_df.expect_column_values_to_not_be_null("CODE_GENDER")

    # Uniqueness
    ge_df.expect_column_values_to_be_unique("SK_ID_CURR")

    # Target distribution — expect class imbalance (8-10% positive)
    ge_df.expect_column_mean_to_be_between("TARGET", min_value=0.05, max_value=0.15)

    # Value set membership
    ge_df.expect_column_values_to_be_in_set("TARGET", value_set=[0, 1])
    ge_df.expect_column_values_to_be_in_set("CODE_GENDER", value_set=["M", "F", "XNA"])
    ge_df.expect_column_values_to_be_in_set(
        "NAME_CONTRACT_TYPE", value_set=["Cash loans", "Revolving loans"]
    )
    ge_df.expect_column_values_to_be_in_set("REGION_RATING_CLIENT", value_set=[1, 2, 3])

    # Range checks
    ge_df.expect_column_values_to_be_between(
        "AMT_INCOME_TOTAL", min_value=0, strict_min=True
    )
    ge_df.expect_column_values_to_be_between("DAYS_BIRTH", max_value=0, strict_max=True)
    ge_df.expect_column_values_to_be_between("CNT_CHILDREN", min_value=0)

    # Null rate checks — these columns should have low null rates
    ge_df.expect_column_values_to_not_be_null("EXT_SOURCE_2", mostly=0.90)
    ge_df.expect_column_values_to_not_be_null("AMT_CREDIT", mostly=0.99)
    ge_df.expect_column_values_to_not_be_null("AMT_ANNUITY", mostly=0.99)

    # Row count sanity check
    ge_df.expect_table_row_count_to_be_between(min_value=300000, max_value=320000)

    # Column count sanity check
    ge_df.expect_table_column_count_to_be_between(min_value=100, max_value=130)

    return ge_df


def validate_ge(df: pd.DataFrame) -> tuple[bool, dict]:
    """
    Runs Great Expectations suite and saves HTML report.

    Returns:
        (passed: bool, results: dict)
        passed=True only if ALL expectations pass
        results contains per-expectation pass/fail details
    """
    ge_df = build_ge_suite(df)
    results = ge_df.validate()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save JSON results
    json_path = REPORTS_DIR / "ge_validation_results.json"
    with open(json_path, "w") as f:
        json.dump(results.to_json_dict(), f, indent=2, default=str)

    # Generate HTML report
    html_path = REPORTS_DIR / "ge_validation_report.html"
    _generate_html_report(results, html_path)

    # Print summary table
    all_results = results.results
    print("\nGreat Expectations Results")
    print("=" * 70)

    passed_count = 0
    failed_count = 0

    for r in all_results:
        exp_type = r.expectation_config.expectation_type
        kwargs = r.expectation_config.kwargs
        col = kwargs.get("column", "—")
        status = "PASS" if r.success else "FAIL"

        if r.success:
            passed_count += 1
        else:
            failed_count += 1

        print(f"  {exp_type:<45} {str(col):<20} {status}")

    total = passed_count + failed_count
    print("=" * 70)
    print(
        f"  Total: {total} checks | Passed: {passed_count} | " f"Failed: {failed_count}"
    )
    print(f"\n  HTML report: {html_path}")
    print(f"  JSON results: {json_path}")

    all_passed = results.success
    return (all_passed, results.to_json_dict())


def _generate_html_report(results, output_path: Path) -> None:
    """Generate a basic HTML report from GE validation results."""
    all_results = results.results
    passed = sum(1 for r in all_results if r.success)
    failed = sum(1 for r in all_results if not r.success)
    total = passed + failed
    overall = "PASSED" if results.success else "FAILED"
    status_color = "#22c55e" if results.success else "#ef4444"

    rows_html = []
    for r in all_results:
        exp_type = r.expectation_config.expectation_type
        kwargs = r.expectation_config.kwargs
        col = kwargs.get("column", "—")
        status = "PASS" if r.success else "FAIL"
        row_color = "#dcfce7" if r.success else "#fecaca"
        text_color = "#166534" if r.success else "#991b1b"

        # Build details from observed value
        observed = r.result if r.result else {}
        details_parts = []
        if "observed_value" in observed:
            details_parts.append(f"observed: {observed['observed_value']}")
        if "unexpected_percent" in observed:
            details_parts.append(f"unexpected: {observed['unexpected_percent']:.2f}%")
        if "element_count" in observed:
            details_parts.append(f"rows: {observed['element_count']:,}")
        details = " | ".join(details_parts) if details_parts else "—"

        rows_html.append(
            f"<tr style='background:{row_color}'>"
            f"<td style='color:{text_color};font-weight:600'>{status}</td>"
            f"<td><code>{exp_type}</code></td>"
            f"<td><code>{col}</code></td>"
            f"<td>{details}</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Great Expectations Validation Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
         sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem;
         background: #f9fafb; color: #1f2937; }}
  h1 {{ color: #111827; }}
  .badge {{ display: inline-block; padding: 4px 12px; border-radius: 4px;
            font-weight: 700; color: white; background: {status_color}; }}
  .summary {{ margin: 1rem 0; font-size: 1.1rem; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
  th {{ background: #1f2937; color: white; text-align: left;
       padding: 8px 12px; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #e5e7eb; }}
  code {{ font-size: 0.85em; }}
  .footer {{ margin-top: 2rem; color: #6b7280; font-size: 0.85rem; }}
</style>
</head>
<body>
<h1>Great Expectations — Validation Report</h1>
<p class="summary">
  Overall: <span class="badge">{overall}</span> &nbsp;
  {passed} passed &middot; {failed} failed &middot; {total} total
</p>
<table>
<thead>
  <tr><th>Status</th><th>Expectation</th><th>Column</th><th>Details</th></tr>
</thead>
<tbody>
{"".join(rows_html)}
</tbody>
</table>
<p class="footer">
  Table: application_train.csv &middot;
  Generated by src/data/validate.py
</p>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------


def validate_application_train(
    raise_on_failure: bool = True,
) -> tuple[bool, dict]:
    """
    Runs the full validation pipeline on application_train.csv.

    Loads the table, runs Pandera schema validation, runs Great
    Expectations quality checks, saves HTML report.

    Args:
        raise_on_failure: if True, raises ValidationError on any failure.
                         if False, returns results without raising.
    Returns:
        (all_passed: bool, report: dict with 'pandera' and 'ge' keys)
    Raises:
        ValidationError: if raise_on_failure=True and any check fails
    """
    print("=" * 70)
    print("Running validation on application_train.csv...")
    print("=" * 70)

    # Load the data
    df = load_table("application_train.csv")

    # Layer 1: Pandera schema validation
    print("\n--- Layer 1: Pandera Schema Validation ---")
    pandera_passed, pandera_msg = validate_pandera(df)
    if pandera_passed:
        print(f"  ✓ {pandera_msg}")
    else:
        print(f"  ✗ {pandera_msg}")

    # Layer 2: Great Expectations quality gates
    print("\n--- Layer 2: Great Expectations Quality Gates ---")
    ge_passed, ge_results = validate_ge(df)

    # Combine results
    report = {
        "pandera": {"passed": pandera_passed, "message": pandera_msg},
        "ge": {"passed": ge_passed, "results": ge_results},
        "overall": pandera_passed and ge_passed,
    }

    # Final status
    print("\n" + "=" * 70)
    if report["overall"]:
        print("  ✓ VALIDATION PASSED")
    else:
        print("  ✗ VALIDATION FAILED")
        failures = []
        if not pandera_passed:
            failures.append("Pandera schema")
        if not ge_passed:
            failures.append("Great Expectations")
        print(f"    Failed layers: {', '.join(failures)}")
    print("=" * 70)

    if raise_on_failure and not report["overall"]:
        failure_summary = []
        if not pandera_passed:
            failure_summary.append(f"Pandera: {pandera_msg}")
        if not ge_passed:
            failed_exps = [
                r["expectation_config"]["expectation_type"]
                for r in ge_results.get("results", [])
                if not r.get("success", True)
            ]
            failure_summary.append(
                f"GE: {len(failed_exps)} expectations failed — "
                + ", ".join(failed_exps[:5])
            )
        raise ValidationError("\n".join(failure_summary))

    return (report["overall"], report)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run data validation")
    parser.add_argument(
        "--no-raise",
        action="store_true",
        help="Print results without raising on failure",
    )
    args = parser.parse_args()

    passed, report = validate_application_train(raise_on_failure=not args.no_raise)
    sys.exit(0 if passed else 1)
