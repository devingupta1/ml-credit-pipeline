"""
leakage_audit.py

Automated single-feature AUC check on all numeric columns in the
merged dataset. Flags any column with AUC > 0.80 as a potential data
leak that must be investigated before modeling.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

# Ensure project root is on sys.path so `python src/data/leakage_audit.py` works
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.merge import load_merged

REPORTS_DIR = Path("reports")
PLOTS_DIR = REPORTS_DIR / "plots"

# Columns to always exclude from the audit
EXCLUDE_COLS = {"SK_ID_CURR", "TARGET"}


# ---------------------------------------------------------------------------
# Single-feature AUC
# ---------------------------------------------------------------------------


def single_feature_auc(
    df: pd.DataFrame,
    target_col: str = "TARGET",
) -> pd.DataFrame:
    """
    Computes single-feature AUC for every numeric column using a depth-1
    DecisionTree. Returns a sorted DataFrame with columns: feature, auc, flag.

    Flag definitions:
        LEAK:        AUC > 0.80  — likely data leakage
        INVESTIGATE: 0.70 < AUC <= 0.80 — suspiciously strong
        OK:          AUC <= 0.70 — normal
    """
    y = df[target_col].values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in EXCLUDE_COLS]

    results = []
    total = len(numeric_cols)
    print(f"Running single-feature AUC on {total} numeric columns...")

    for i, col in enumerate(numeric_cols):
        if (i + 1) % 20 == 0 or i == 0 or i == total - 1:
            print(f"  [{i + 1}/{total}] {col}")

        x = df[[col]].fillna(-999).values
        dt = DecisionTreeClassifier(max_depth=1, random_state=42)
        dt.fit(x, y)
        proba = dt.predict_proba(x)[:, 1]

        try:
            auc = roc_auc_score(y, proba)
        except ValueError:
            auc = 0.5  # constant predictions

        # Take max(auc, 1-auc) to handle inverted relationships
        auc = max(auc, 1 - auc)

        if auc > 0.80:
            flag = "LEAK"
        elif auc > 0.70:
            flag = "INVESTIGATE"
        else:
            flag = "OK"

        results.append({"feature": col, "auc": round(auc, 6), "flag": flag})

    results_df = (
        pd.DataFrame(results).sort_values("auc", ascending=False).reset_index(drop=True)
    )
    return results_df


# ---------------------------------------------------------------------------
# Run full audit
# ---------------------------------------------------------------------------


def run_audit() -> pd.DataFrame:
    """
    Loads the merged dataset, runs single-feature AUC audit, saves
    results CSV, generates summary plot.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Leakage Audit — Single-Feature AUC Check")
    print("=" * 60)

    df = load_merged()
    results = single_feature_auc(df)

    # Print flagged features
    leaks = results[results["flag"] == "LEAK"]
    investigate = results[results["flag"] == "INVESTIGATE"]

    print("\n" + "=" * 60)
    if len(leaks) > 0:
        print(f"⚠  LEAK flags: {len(leaks)}")
        print(leaks[["feature", "auc"]].to_string(index=False))
    else:
        print("✓  No LEAK flags detected")

    if len(investigate) > 0:
        print(f"\n?  INVESTIGATE flags: {len(investigate)}")
        print(investigate[["feature", "auc"]].to_string(index=False))
    else:
        print("✓  No INVESTIGATE flags")

    print(f"\nTotal features checked: {len(results)}")
    print(f"  OK: {len(results[results['flag'] == 'OK'])}")
    print(f"  INVESTIGATE: {len(investigate)}")
    print(f"  LEAK: {len(leaks)}")

    # Save results CSV
    csv_path = REPORTS_DIR / "leakage_audit.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Generate plot — top 30 by AUC
    _plot_audit_results(results)

    print("=" * 60)
    return results


def _plot_audit_results(results: pd.DataFrame) -> None:
    """Plot top 30 features by AUC, colored by flag."""
    top30 = results.head(30).sort_values("auc", ascending=True)

    color_map = {"OK": "#2ecc71", "INVESTIGATE": "#f39c12", "LEAK": "#e74c3c"}
    colors = [color_map.get(f, "#3498db") for f in top30["flag"]]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(range(len(top30)), top30["auc"].values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30["feature"].values, fontsize=8)
    ax.set_xlabel("Single-Feature AUC")
    ax.set_title("Leakage Audit — Top 30 Features by AUC", fontweight="bold")
    ax.axvline(
        x=0.80, color="red", linestyle="--", alpha=0.7, label="LEAK threshold (0.80)"
    )
    ax.axvline(
        x=0.70,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label="INVESTIGATE threshold (0.70)",
    )
    ax.axvline(x=0.50, color="gray", linestyle=":", alpha=0.5, label="Random (0.50)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0.45, max(0.85, top30["auc"].max() + 0.05))

    fig.tight_layout()
    plot_path = PLOTS_DIR / "09_leakage_audit.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {plot_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_audit()
    leaks = results[results["flag"] == "LEAK"]
    if len(leaks) > 0:
        print(f"\n⚠  Exiting with code 1 — {len(leaks)} LEAK flag(s) found")
        sys.exit(1)
    else:
        sys.exit(0)
