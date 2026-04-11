"""
eda.py

Exploratory Data Analysis on the merged Home Credit dataset.
Produces 8 diagnostic plots, a ydata-profiling report, and a
plain-text summary. All outputs saved to reports/.
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — no plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure project root is on sys.path so `python src/data/eda.py` works
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.merge import load_merged

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

REPORTS_DIR = Path("reports")
PLOTS_DIR = REPORTS_DIR / "plots"

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
    }
)


# ---------------------------------------------------------------------------
# Analysis 1 — Target distribution
# ---------------------------------------------------------------------------


def plot_target_distribution(df: pd.DataFrame) -> None:
    """Plot the class distribution of TARGET (bar chart + percentage labels)."""
    print("\n--- Analysis 1: Target Distribution ---")
    counts = df["TARGET"].value_counts().sort_index()
    pct = df["TARGET"].value_counts(normalize=True).sort_index() * 100

    print(f"  Class 0: {pct[0]:.2f}% | Class 1: {pct[1]:.2f}%")
    print(f"  Counts — 0: {counts[0]:,} | 1: {counts[1]:,}")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        ["Class 0 (Repaid)", "Class 1 (Default)"],
        counts.values,
        color=["#2ecc71", "#e74c3c"],
        edgecolor="white",
        linewidth=1.5,
    )

    for bar, p in zip(bars, pct.values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1000,
            f"{p:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=13,
        )

    ax.set_ylabel("Count")
    ax.set_title("Target Distribution — Loan Default", fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.savefig(PLOTS_DIR / "01_target_distribution.png")
    plt.close(fig)
    print("  Saved: 01_target_distribution.png")


# ---------------------------------------------------------------------------
# Analysis 2 — Null rate per feature
# ---------------------------------------------------------------------------


def plot_null_rates(df: pd.DataFrame) -> None:
    """Plot top 40 features by null rate."""
    print("\n--- Analysis 2: Null Rates ---")
    null_rate = df.isnull().mean().sort_values(ascending=False)

    n_50 = (null_rate > 0.50).sum()
    n_20 = (null_rate > 0.20).sum()
    n_05 = (null_rate > 0.05).sum()
    print(f"  Features with null rate > 50%: {n_50}")
    print(f"  Features with null rate > 20%: {n_20}")
    print(f"  Features with null rate >  5%: {n_05}")

    top40 = null_rate.head(40)

    fig, ax = plt.subplots(figsize=(10, 12))
    colors = [
        "#e74c3c" if v > 0.5 else "#f39c12" if v > 0.2 else "#3498db"
        for v in top40.values
    ]
    ax.barh(range(len(top40)), top40.values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(top40)))
    ax.set_yticklabels(top40.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Null Rate")
    ax.set_title("Top 40 Features by Null Rate", fontweight="bold")
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label=">50%")
    ax.axvline(x=0.2, color="orange", linestyle="--", alpha=0.5, label=">20%")
    ax.legend()
    fig.savefig(PLOTS_DIR / "02_null_rates.png")
    plt.close(fig)
    print("  Saved: 02_null_rates.png")

    return null_rate


# ---------------------------------------------------------------------------
# Analysis 3 — Numeric feature distributions split by TARGET
# ---------------------------------------------------------------------------


def plot_feature_distributions(df: pd.DataFrame) -> None:
    """Plot top 12 numeric features by correlation with TARGET as histograms."""
    print("\n--- Analysis 3: Feature Distributions (by TARGET) ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "TARGET"]

    corr = df[numeric_cols].corrwith(df["TARGET"]).abs().sort_values(ascending=False)
    top12 = corr.head(12).index.tolist()
    print(f"  Top 12 features: {', '.join(top12[:6])}...")

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    for i, col in enumerate(top12):
        ax = axes[i]
        for target_val, color, label in [
            (0, "#2ecc71", "Repaid"),
            (1, "#e74c3c", "Default"),
        ]:
            subset = df.loc[df["TARGET"] == target_val, col].dropna()
            # Clip outliers for better visualization
            low, high = subset.quantile(0.01), subset.quantile(0.99)
            subset = subset.clip(low, high)
            ax.hist(subset, bins=50, alpha=0.5, color=color, label=label, density=True)
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    fig.suptitle(
        "Top 12 Features by |Correlation with TARGET|", fontweight="bold", fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PLOTS_DIR / "03_feature_distributions.png")
    plt.close(fig)
    print("  Saved: 03_feature_distributions.png")


# ---------------------------------------------------------------------------
# Analysis 4 — Correlation with TARGET
# ---------------------------------------------------------------------------


def plot_target_correlations(df: pd.DataFrame) -> dict:
    """Plot top 30 features by correlation with TARGET."""
    print("\n--- Analysis 4: Correlation with TARGET ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "TARGET"]

    corr = df[numeric_cols].corrwith(df["TARGET"]).dropna().sort_values()
    top_pos = corr.tail(15)
    top_neg = corr.head(15)
    top30 = pd.concat([top_neg, top_pos])

    print(f"  Strongest positive: {corr.idxmax()} ({corr.max():.4f})")
    print(f"  Strongest negative: {corr.idxmin()} ({corr.min():.4f})")

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in top30.values]
    ax.barh(range(len(top30)), top30.values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30.index, fontsize=8)
    ax.set_xlabel("Pearson Correlation with TARGET")
    ax.set_title("Top 30 Features — Correlation with TARGET", fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)
    fig.savefig(PLOTS_DIR / "04_target_correlations.png")
    plt.close(fig)
    print("  Saved: 04_target_correlations.png")

    # Return full correlation for summary
    full_corr = df[numeric_cols].corrwith(df["TARGET"]).dropna()
    top3 = full_corr.abs().sort_values(ascending=False).head(3)
    return {"top3_features": top3.index.tolist(), "top3_values": top3.values.tolist()}


# ---------------------------------------------------------------------------
# Analysis 5 — Missing data heatmap
# ---------------------------------------------------------------------------


def plot_missing_heatmap(df: pd.DataFrame) -> None:
    """Plot a heatmap of null patterns for columns with 5-80% null rate."""
    print("\n--- Analysis 5: Missing Data Heatmap ---")
    null_rate = df.isnull().mean()
    cols = (
        null_rate[(null_rate > 0.05) & (null_rate < 0.80)]
        .sort_values(ascending=False)
        .index.tolist()
    )

    if len(cols) == 0:
        print("  No columns with null rate between 5% and 80%. Skipping.")
        return

    # Cap columns for readability
    cols = cols[:30]
    print(f"  Plotting {len(cols)} columns with null rate 5-80%")

    sample = df[cols].sample(n=min(5000, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        sample.isnull().astype(int).T,
        cbar=False,
        cmap="YlOrRd",
        yticklabels=True,
        xticklabels=False,
        ax=ax,
    )
    ax.set_title("Missing Data Patterns (5,000 row sample)", fontweight="bold")
    ax.set_ylabel("Feature")
    ax.set_xlabel("Rows (sampled)")
    ax.tick_params(axis="y", labelsize=7)
    fig.savefig(PLOTS_DIR / "05_missing_data_heatmap.png")
    plt.close(fig)
    print("  Saved: 05_missing_data_heatmap.png")


# ---------------------------------------------------------------------------
# Analysis 6 — Class imbalance deep dive (default rate by category)
# ---------------------------------------------------------------------------


def plot_default_by_category(df: pd.DataFrame) -> None:
    """Plot default rate per category for top 5 categorical features."""
    print("\n--- Analysis 6: Default Rate by Category ---")
    cat_features = [
        "NAME_CONTRACT_TYPE",
        "CODE_GENDER",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
    ]

    fig, axes = plt.subplots(len(cat_features), 1, figsize=(12, 4 * len(cat_features)))

    for i, col in enumerate(cat_features):
        ax = axes[i]
        if col not in df.columns:
            print(f"  Skipping {col} — not in DataFrame")
            continue

        rates = df.groupby(col)["TARGET"].mean().sort_values(ascending=True)
        colors = plt.cm.RdYlGn_r(rates.values / rates.max())
        ax.barh(range(len(rates)), rates.values, color=colors, edgecolor="white")
        ax.set_yticks(range(len(rates)))
        ax.set_yticklabels(rates.index, fontsize=9)
        ax.set_xlabel("Default Rate")
        ax.set_title(f"Default Rate by {col}", fontweight="bold")

        # Add percentage labels
        for j, v in enumerate(rates.values):
            ax.text(v + 0.002, j, f"{v:.1%}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "06_default_rate_by_category.png")
    plt.close(fig)
    print("  Saved: 06_default_rate_by_category.png")


# ---------------------------------------------------------------------------
# Analysis 7 — External scores
# ---------------------------------------------------------------------------


def plot_external_scores(df: pd.DataFrame) -> None:
    """Plot EXT_SOURCE distributions and scatter matrix."""
    print("\n--- Analysis 7: External Scores ---")
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

    # Top row: distributions split by TARGET
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i, col in enumerate(ext_cols):
        ax = axes[0, i]
        for target_val, color, label in [
            (0, "#2ecc71", "Repaid"),
            (1, "#e74c3c", "Default"),
        ]:
            subset = df.loc[df["TARGET"] == target_val, col].dropna()
            ax.hist(subset, bins=50, alpha=0.5, color=color, label=label, density=True)
        ax.set_title(f"{col} by TARGET", fontweight="bold")
        ax.set_xlabel(col)
        ax.legend(fontsize=8)

    # Bottom row: scatter plots (pairwise)
    sample = (
        df[ext_cols + ["TARGET"]]
        .dropna()
        .sample(n=min(10000, len(df)), random_state=42)
    )
    scatter_pairs = [(0, 1), (0, 2), (1, 2)]
    for idx, (a, b) in enumerate(scatter_pairs):
        ax = axes[1, idx]
        colors = sample["TARGET"].map({0: "#2ecc71", 1: "#e74c3c"})
        ax.scatter(sample[ext_cols[a]], sample[ext_cols[b]], c=colors, alpha=0.2, s=5)
        ax.set_xlabel(ext_cols[a], fontsize=9)
        ax.set_ylabel(ext_cols[b], fontsize=9)
        ax.set_title(f"{ext_cols[a]} vs {ext_cols[b]}", fontweight="bold")

    fig.suptitle(
        "External Scores — Distribution & Scatter Matrix",
        fontweight="bold",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PLOTS_DIR / "07_external_scores.png")
    plt.close(fig)
    print("  Saved: 07_external_scores.png")


# ---------------------------------------------------------------------------
# Analysis 8 — DAYS_EMPLOYED anomaly
# ---------------------------------------------------------------------------


def plot_days_employed_anomaly(df: pd.DataFrame) -> dict:
    """Plot DAYS_EMPLOYED distribution, highlighting the 365243 sentinel."""
    print("\n--- Analysis 8: DAYS_EMPLOYED Anomaly ---")
    sentinel = 365243
    sentinel_mask = df["DAYS_EMPLOYED"] == sentinel
    n_sentinel = sentinel_mask.sum()
    pct_sentinel = n_sentinel / len(df) * 100
    default_rate_sentinel = df.loc[sentinel_mask, "TARGET"].mean() * 100
    default_rate_normal = df.loc[~sentinel_mask, "TARGET"].mean() * 100

    print(f"  DAYS_EMPLOYED == 365243: {n_sentinel:,} rows ({pct_sentinel:.1f}%)")
    print(f"  Default rate (sentinel): {default_rate_sentinel:.2f}%")
    print(f"  Default rate (normal):   {default_rate_normal:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: full distribution showing the anomaly
    ax = axes[0]
    ax.hist(
        df["DAYS_EMPLOYED"].values,
        bins=100,
        color="#3498db",
        edgecolor="white",
        alpha=0.7,
    )
    ax.set_title("DAYS_EMPLOYED — Full Distribution (with anomaly)", fontweight="bold")
    ax.set_xlabel("DAYS_EMPLOYED")
    ax.set_ylabel("Count")
    ax.axvline(
        x=sentinel,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Sentinel={sentinel:,}",
    )
    ax.legend()

    # Right: distribution without sentinel
    ax = axes[1]
    normal = df.loc[~sentinel_mask, "DAYS_EMPLOYED"]
    ax.hist(normal, bins=100, color="#3498db", edgecolor="white", alpha=0.7)
    ax.set_title("DAYS_EMPLOYED — Without Sentinel", fontweight="bold")
    ax.set_xlabel("DAYS_EMPLOYED")
    ax.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "08_days_employed_anomaly.png")
    plt.close(fig)
    print("  Saved: 08_days_employed_anomaly.png")

    return {
        "sentinel_count": n_sentinel,
        "sentinel_pct": pct_sentinel,
        "sentinel_default_rate": default_rate_sentinel,
    }


# ---------------------------------------------------------------------------
# ydata-profiling report
# ---------------------------------------------------------------------------


def generate_profile_report(df: pd.DataFrame) -> None:
    """Generate a ydata-profiling HTML report with minimal=True."""
    print("\n--- Generating ydata-profiling Report ---")
    from ydata_profiling import ProfileReport

    profile = ProfileReport(
        df,
        title="Home Credit — Merged Dataset EDA",
        minimal=True,
        explorative=True,
        progress_bar=True,
    )

    report_path = REPORTS_DIR / "eda_profile.html"
    profile.to_file(report_path)
    size_mb = report_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {report_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# EDA summary text file
# ---------------------------------------------------------------------------


def write_summary(
    df: pd.DataFrame,
    corr_info: dict,
    days_info: dict,
    null_rate: pd.Series,
) -> None:
    """Write the EDA summary text file."""
    print("\n--- Writing EDA Summary ---")

    rows, cols = df.shape
    pct_0 = (df["TARGET"] == 0).mean() * 100
    pct_1 = (df["TARGET"] == 1).mean() * 100
    n_null_50 = (null_rate > 0.50).sum()
    n_null_20 = (null_rate > 0.20).sum()
    top3 = ", ".join(corr_info["top3_features"])
    ext2_null = df["EXT_SOURCE_2"].isnull().mean() * 100

    summary = f"""EDA Summary — ml-credit-pipeline
==================================
Dataset shape: {rows:,} rows x {cols} columns
Target distribution: {pct_0:.2f}% class 0 | {pct_1:.2f}% class 1
Null rate > 50%: {n_null_50} features
Null rate > 20%: {n_null_20} features
Top 3 features by correlation with TARGET: {top3}
DAYS_EMPLOYED sentinel (365243) count: {days_info['sentinel_count']:,} ({days_info['sentinel_pct']:.1f}% of rows)
DAYS_EMPLOYED sentinel default rate: {days_info['sentinel_default_rate']:.2f}%
EXT_SOURCE_2 null rate: {ext2_null:.2f}%
Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    summary_path = REPORTS_DIR / "eda_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)

    print(f"  Saved: {summary_path}")
    print(summary)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all EDA analyses and produce all outputs."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EDA — Home Credit Merged Dataset")
    print("=" * 60)

    # Load data
    df = load_merged()

    # Run all analyses
    plot_target_distribution(df)
    null_rate = plot_null_rates(df)
    plot_feature_distributions(df)
    corr_info = plot_target_correlations(df)
    plot_missing_heatmap(df)
    plot_default_by_category(df)
    plot_external_scores(df)
    days_info = plot_days_employed_anomaly(df)

    # ydata-profiling report
    generate_profile_report(df)

    # Summary
    write_summary(df, corr_info, days_info, null_rate)

    print("\n" + "=" * 60)
    print("EDA complete. All outputs saved to reports/")
    print("=" * 60)


if __name__ == "__main__":
    main()
