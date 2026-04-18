"""
evaluate.py

Evaluation script to determine the optimal threshold using a cost matrix,
calibrate the model probabilities, run SHAP explainability, and log results.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.merge import load_merged
from src.features.pipeline import fit_pipeline, get_feature_names


def load_champion_model():
    """Loads the champion model from MLflow Model Registry."""
    model_uri = "models:/credit-default-risk-champion/Staging"
    print(f"Loading champion model from {model_uri}...")
    model = mlflow.lightgbm.load_model(model_uri)
    return model


def compute_cost(y_true, y_pred, fn_cost=10, fp_cost=1) -> float:
    """Computes total cost based on false negatives and false positives."""
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(fn * fn_cost + fp * fp_cost)


def find_optimal_threshold(y_true, y_proba, fn_cost=10, fp_cost=1) -> dict:
    """Sweeps thresholds and finds the one that minimizes cost."""
    thresholds = np.arange(0.01, 1.0, 0.01)
    costs = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        c = compute_cost(y_true, y_pred, fn_cost, fp_cost)
        costs.append(c)

    opt_idx = np.argmin(costs)
    opt_t = thresholds[opt_idx]

    opt_y_pred = (y_proba >= opt_t).astype(int)

    return {
        "optimal_threshold": opt_t,
        "min_cost": costs[opt_idx],
        "precision_at_threshold": precision_score(y_true, opt_y_pred, zero_division=0),
        "recall_at_threshold": recall_score(y_true, opt_y_pred, zero_division=0),
        "f1_at_threshold": f1_score(y_true, opt_y_pred),
        "all_thresholds": thresholds.tolist(),
        "all_costs": costs,
    }


def plot_threshold_sweep(results: dict) -> None:
    """Plots cost vs threshold curve and saves it."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results["all_thresholds"], results["all_costs"], lw=2, color="#2c3e50")
    ax.axvline(
        x=results["optimal_threshold"],
        color="#e74c3c",
        linestyle="--",
        label=f"Optimal: {results['optimal_threshold']:.2f}",
    )
    ax.set_title("Cost vs. Classification Threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Total Cost")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    reports_dir = Path("reports/plots")
    reports_dir.mkdir(parents=True, exist_ok=True)
    filepath = reports_dir / "11_threshold_sweep.png"

    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    mlflow.log_artifact(str(filepath))


def calibrate_model(model, X_val, y_val):
    """Wraps model with isotonic regression calibration."""
    calibrated = CalibratedClassifierCV(
        estimator=model, method="isotonic", cv=None, ensemble=False
    )
    calibrated.fit(X_val, y_val)
    return calibrated


def plot_reliability_diagram(model_raw, model_calibrated, X_val, y_val) -> None:
    """Plots reliability diagrams (calibration curve) before and after calibration."""
    prob_raw = model_raw.predict_proba(X_val)[:, 1]
    prob_cal = model_calibrated.predict_proba(X_val)[:, 1]

    frac_pos_raw, mean_pred_raw = calibration_curve(y_val, prob_raw, n_bins=10)
    frac_pos_cal, mean_pred_cal = calibration_curve(y_val, prob_cal, n_bins=10)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    ax.plot(mean_pred_raw, frac_pos_raw, "s-", color="#e74c3c", label="Raw model")
    ax.plot(
        mean_pred_cal,
        frac_pos_cal,
        "o-",
        color="#27ae60",
        label="Calibrated (Isotonic)",
    )

    ax.set_title("Reliability Diagram (Calibration Curve)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="lower right")

    reports_dir = Path("reports/plots")
    reports_dir.mkdir(parents=True, exist_ok=True)
    filepath = reports_dir / "10_reliability_diagram.png"

    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    mlflow.log_artifact(str(filepath))


def save_threshold_config(threshold: float, metrics: dict) -> None:
    """Saves the optimal threshold and associated performance metrics."""
    config = {
        "optimal_threshold": threshold,
        "fn_cost": 10,
        "fp_cost": 1,
        "precision": metrics["precision_at_threshold"],
        "recall": metrics["recall_at_threshold"],
        "f1": metrics["f1_at_threshold"],
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    out_path = Path("reports/threshold_config.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)


def run_shap_analysis(model, X_val, y_val, feature_names, n_sample=2000):
    """Runs SHAP analysis on a subset of data and produces summary and waterfall plots."""
    np.random.seed(42)
    n_sample = min(n_sample, len(X_val))
    idx = np.random.choice(len(X_val), n_sample, replace=False)
    X_sample = X_val[idx]
    y_sample = y_val[idx]

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_sample)

    if len(explanation.values.shape) == 3:
        exp_pos = explanation[:, :, 1]
    elif isinstance(explanation.values, list):
        exp_pos = explanation[1]
    else:
        exp_pos = explanation

    exp_pos.feature_names = feature_names
    shap_values_pos = exp_pos.values

    reports_dir = Path("reports/plots")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Global summary plot
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values_pos, X_sample, feature_names=feature_names, show=False
    )
    plt.tight_layout()
    plt.savefig(reports_dir / "12_shap_global.png", dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(str(reports_dir / "12_shap_global.png"))

    # Top 10 Features
    mean_abs_shap = np.abs(shap_values_pos).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:10]
    print("\nTop 10 SHAP features:")
    top_10 = []
    for i, idx_ in enumerate(top_indices):
        feature = feature_names[idx_]
        val = mean_abs_shap[idx_]
        print(f"{i+1}. {feature}: {val:.4f}")
        top_10.append(f"{i+1}. {feature}: {val:.4f}")

    # 2. Local waterfall explanations
    preds = model.predict(X_sample)

    tp_idx = np.where((y_sample == 1) & (preds == 1))[0]
    tn_idx = np.where((y_sample == 0) & (preds == 0))[0]
    fp_idx = np.where((y_sample == 0) & (preds == 1))[0]

    def save_waterfall(idx_list, filename):
        if len(idx_list) > 0:
            target_idx = idx_list[0]
            fig = plt.figure(figsize=(8, 6))
            shap.waterfall_plot(exp_pos[target_idx], show=False)
            plt.tight_layout()
            plt.savefig(reports_dir / filename, dpi=150, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(str(reports_dir / filename))

    save_waterfall(tp_idx, "13_shap_waterfall_tp.png")
    save_waterfall(tn_idx, "14_shap_waterfall_tn.png")
    save_waterfall(fp_idx, "15_shap_waterfall_fp.png")

    return "\n".join(top_10)


def run_subgroup_analysis(model, pipeline, df_val, threshold) -> pd.DataFrame:
    """Computes AUC-PR per demographic subgroup correctly handled via calibrated probabilities."""
    results = []

    print("Transforming validation data for subgroup analysis...")
    X_val = pipeline.transform(df_val)
    y_val = df_val["TARGET"].values

    proba = model.predict_proba(X_val)[:, 1]

    def evaluate_group(group_name, category, mask):
        if mask.sum() > 0:
            y_sub = y_val[mask]
            prob_sub = proba[mask]
            auc_pr = (
                average_precision_score(y_sub, prob_sub)
                if len(np.unique(y_sub)) > 1
                else np.nan
            )
            auc_roc = (
                roc_auc_score(y_sub, prob_sub) if len(np.unique(y_sub)) > 1 else np.nan
            )
            default_rate = y_sub.mean()
            results.append(
                {
                    "Group": group_name,
                    "Category": category,
                    "AUC-PR": auc_pr,
                    "AUC-ROC": auc_roc,
                    "Default Rate": default_rate,
                    "Size": mask.sum(),
                }
            )

    # 1. CODE_GENDER
    if "CODE_GENDER" in df_val.columns:
        for g in ["M", "F"]:
            mask = df_val["CODE_GENDER"] == g
            evaluate_group("Gender", g, mask)

    # 2. Age Bucket from DAYS_BIRTH
    if "DAYS_BIRTH" in df_val.columns:
        age_years = -df_val["DAYS_BIRTH"] / 365.25
        evaluate_group("Age", "Young (<30)", age_years < 30)
        evaluate_group("Age", "Middle (30-50)", (age_years >= 30) & (age_years < 50))
        evaluate_group("Age", "Senior (>=50)", age_years >= 50)

    # 3. NAME_CONTRACT_TYPE
    if "NAME_CONTRACT_TYPE" in df_val.columns:
        for c in ["Cash loans", "Revolving loans"]:
            mask = df_val["NAME_CONTRACT_TYPE"] == c
            evaluate_group("Contract", c, mask)

    df_results = pd.DataFrame(results)

    if not df_results.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = df_results["Group"] + ": " + df_results["Category"]
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, df_results["AUC-PR"], align="center", color="#3498db")
        ax.set_yticks(y_pos, labels=labels)
        ax.invert_yaxis()
        ax.set_xlabel("AUC-PR")
        ax.set_title("AUC-PR by Demographic Subgroup")
        ax.set_xlim([0, 1.0])
        ax.grid(True, linestyle="--", alpha=0.7)

        reports_dir = Path("reports/plots")
        reports_dir.mkdir(parents=True, exist_ok=True)
        filepath = reports_dir / "16_subgroup_auc.png"

        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        mlflow.log_artifact(str(filepath))

    return df_results


def run_evaluation() -> dict:
    """Orchestrates the evaluation pipeline."""
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))

    exp_name = "home-credit-default-risk"
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        exp_id = mlflow.create_experiment(exp_name)
    else:
        exp_id = experiment.experiment_id

    print("Loading merged dataset...")
    df = load_merged()

    y = df["TARGET"].values
    X_df = df.drop(columns=["TARGET"]) if "TARGET" in df.columns else df.copy()

    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_df, y, test_size=0.2, stratify=y, random_state=42
    )

    df_train = X_tr_raw.copy()
    df_train["TARGET"] = y_tr

    print("Fitting preprocessing pipeline on training data...")
    pipeline, _ = fit_pipeline(df_train, "TARGET")
    print("Transforming validation data...")
    X_val = pipeline.transform(X_val_raw)
    model_raw = load_champion_model()

    # Get feature names from the pipeline utility
    feature_names = get_feature_names(pipeline)
    if len(feature_names) == 0:
        feature_names = model_raw.feature_name_

    prob_raw = model_raw.predict_proba(X_val)[:, 1]

    with mlflow.start_run(experiment_id=exp_id, run_name="evaluation"):
        print("Calibrating model using isotonic regression...")
        model_calibrated = calibrate_model(model_raw, X_val, y_val)

        prob_cal = model_calibrated.predict_proba(X_val)[:, 1]

        print("Running threshold sweep...")
        eval_results = find_optimal_threshold(y_val, prob_cal, fn_cost=10, fp_cost=1)
        plot_threshold_sweep(eval_results)

        plot_reliability_diagram(model_raw, model_calibrated, X_val, y_val)

        save_threshold_config(eval_results["optimal_threshold"], eval_results)
        mlflow.log_artifact("reports/threshold_config.json")

        mlflow.sklearn.log_model(model_calibrated, "model_calibrated")

        auc_roc = roc_auc_score(y_val, prob_cal)
        auc_pr = average_precision_score(y_val, prob_cal)
        eval_results["auc_roc"] = auc_roc
        eval_results["auc_pr"] = auc_pr

        summary = f"""
Evaluation Summary
================================
AUC-ROC:            {auc_roc:.4f}
AUC-PR:             {auc_pr:.4f}
Optimal threshold:  {eval_results["optimal_threshold"]:.2f}
Precision at threshold: {eval_results["precision_at_threshold"]*100:.2f}%
Recall at threshold:    {eval_results["recall_at_threshold"]*100:.2f}%
F1 at threshold:        {eval_results["f1_at_threshold"]*100:.2f}%
Calibration: isotonic regression applied
================================
"""
        print(summary)

        # New steps for prompt 12: SHAP and Subgroup analysis
        print("Running SHAP analysis on a subset (n=2000)...")
        top_10 = run_shap_analysis(
            model_raw, X_val, y_val, feature_names, n_sample=2000
        )

        print("Running Subgroup Analysis...")
        df_val = X_val_raw.copy()
        df_val["TARGET"] = y_val
        subgroup_results = run_subgroup_analysis(
            model_calibrated, pipeline, df_val, eval_results["optimal_threshold"]
        )
        print("\nSubgroup AUC-PR Results:")
        print(subgroup_results.to_markdown(index=False))

        return {
            "eval_results": eval_results,
            "top_10": top_10,
            "subgroup_results": subgroup_results,
        }


if __name__ == "__main__":
    run_evaluation()
