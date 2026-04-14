"""
evaluate.py

Evaluation script to determine the optimal threshold using a cost matrix,
calibrate the model probabilities, and log results.
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
from dotenv import load_dotenv
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.merge import load_merged
from src.features.pipeline import fit_pipeline


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

    # 1. Load data
    print("Loading merged dataset...")
    df = load_merged()

    y = df["TARGET"].values
    X_df = df.drop(columns=["TARGET"]) if "TARGET" in df.columns else df.copy()

    # Stratified split: 80% train to fit pipeline, 20% val for evaluation & calibration
    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_df, y, test_size=0.2, stratify=y, random_state=42
    )

    # Restore target for pipeline fitting
    df_train = X_tr_raw.copy()
    df_train["TARGET"] = y_tr

    # 2. Fit pipeline on train, transform validation
    print("Fitting preprocessing pipeline on training data...")
    pipeline, _ = fit_pipeline(df_train, "TARGET")
    print("Transforming validation data...")
    X_val = pipeline.transform(X_val_raw)

    # 3. Load champion model
    model_raw = load_champion_model()

    # 4. Predict raw probabilities
    prob_raw = model_raw.predict_proba(X_val)[:, 1]

    with mlflow.start_run(experiment_id=exp_id, run_name="evaluation"):
        # 6. Callibrate model
        print("Calibrating model using isotonic regression...")
        model_calibrated = calibrate_model(model_raw, X_val, y_val)

        prob_cal = model_calibrated.predict_proba(X_val)[:, 1]

        # 5. Threshold sweep
        print("Running threshold sweep...")
        eval_results = find_optimal_threshold(y_val, prob_cal, fn_cost=10, fp_cost=1)
        plot_threshold_sweep(eval_results)

        # 7. Reliability diagram
        plot_reliability_diagram(model_raw, model_calibrated, X_val, y_val)

        # 8. Save threshold
        save_threshold_config(eval_results["optimal_threshold"], eval_results)
        mlflow.log_artifact("reports/threshold_config.json")

        # Optionally, log the calibrated model back to MLflow or a new run
        mlflow.sklearn.log_model(model_calibrated, "model_calibrated")

        # Print summary
        summary = f"""
Evaluation Summary
================================
Optimal threshold:  {eval_results["optimal_threshold"]:.2f}
Precision at threshold: {eval_results["precision_at_threshold"]*100:.2f}%
Recall at threshold:    {eval_results["recall_at_threshold"]*100:.2f}%
F1 at threshold:        {eval_results["f1_at_threshold"]*100:.2f}%
Calibration: isotonic regression applied
================================
"""
        print(summary)

        return eval_results


if __name__ == "__main__":
    run_evaluation()
