"""
train.py

Baseline Logistic Regression model and MLflow experiment setup
for the Home Credit Machine Learning problem.
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import seaborn as sns
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.merge import load_merged
from src.features.pipeline import fit_pipeline


def get_experiment_id(experiment_name: str) -> str:
    """Retrieves or creates the MLflow experiment, returning its ID."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return mlflow.create_experiment(experiment_name)
    return experiment.experiment_id


def log_cv_metrics(metrics: dict, prefix: str = "") -> None:
    """Logs a dictionary of metrics to active MLflow run."""
    # MLflow expects key-value pairs
    to_log = {f"{prefix}{k}" if prefix else k: v for k, v in metrics.items()}
    mlflow.log_metrics(to_log)


def log_confusion_matrix(y_true, y_pred, filename: str) -> None:
    """Generates and logs a confusion matrix plot to MLflow."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    filepath = reports_dir / filename
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)

    mlflow.log_artifact(str(filepath))


def log_pr_curve(y_true, y_proba, filename: str) -> None:
    """Generates and logs a precision-recall curve plot to MLflow."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AUC-PR = {ap:.4f}", color="#3498db", lw=2)
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.7)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    filepath = reports_dir / filename
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)

    mlflow.log_artifact(str(filepath))


def train_baseline() -> dict:
    """
    Trains logistic regression baseline using StratifiedKFold and logs to MLflow.
    """
    # 1. Setup MLflow
    load_dotenv()
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(mlflow_uri)
    exp_name = "home-credit-default-risk"
    exp_id = get_experiment_id(exp_name)

    # 2. Load data and setup preprocessing
    df = load_merged()

    # Take a representative sample for the baseline to speed up prompt-execution if needed,
    # but the prompt implies we run on full dataset. Logistic regression fits fast though.
    # Note: On 300k rows with 200+ features, LogisticRegression might take a minute or two.
    y = df["TARGET"].values

    # Fit the preprocessing pipeline on our data.
    # Since fit_pipeline creates the pipeline and calls fit, we use it directly:
    print("Fitting preprocessing pipeline...")
    pipeline, pipeline_y = fit_pipeline(df, "TARGET")

    X_df = df.drop(columns=["TARGET"]) if "TARGET" in df.columns else df.copy()
    print("Transforming data...")
    X_transformed = pipeline.transform(X_df)

    # LogisticRegression initialization
    model_params = {"max_iter": 1000, "class_weight": "balanced", "random_state": 42}
    lr = LogisticRegression(**model_params)

    # Cross validation structure
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_auc_pr = []
    cv_auc_roc = []
    cv_f1 = []
    cv_precision = []
    cv_recall = []

    oof_proba = np.zeros(len(y))
    oof_pred = np.zeros(len(y))

    print(f"Starting {n_splits}-fold cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_transformed, y)):
        X_tr, y_tr = X_transformed[train_idx], y[train_idx]
        X_va, y_va = X_transformed[val_idx], y[val_idx]

        lr.fit(X_tr, y_tr)

        # Predict on validation data
        proba = lr.predict_proba(X_va)[:, 1]
        preds = lr.predict(X_va)

        # We record out-of-fold directly
        oof_proba[val_idx] = proba
        oof_pred[val_idx] = preds

        # Find optimal threshold on validation set
        prec, rec, thresholds = precision_recall_curve(y_va, proba)
        # Handle cases where denominator is 0 safely:
        fscores = np.divide(
            2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) != 0
        )
        opt_idx = np.argmax(fscores)
        opt_thresh = thresholds[opt_idx] if opt_idx < len(thresholds) else 0.5
        opt_preds = (proba >= opt_thresh).astype(int)

        cv_auc_pr.append(average_precision_score(y_va, proba))
        cv_auc_roc.append(roc_auc_score(y_va, proba))
        cv_f1.append(f1_score(y_va, opt_preds))
        cv_precision.append(precision_score(y_va, opt_preds, zero_division=0))
        cv_recall.append(recall_score(y_va, opt_preds, zero_division=0))

        print(f"  Fold {fold+1}: AUC-PR = {cv_auc_pr[-1]:.4f}")

    cv_metrics = {
        "auc_pr_mean": np.mean(cv_auc_pr),
        "auc_pr_std": np.std(cv_auc_pr),
        "auc_roc_mean": np.mean(cv_auc_roc),
        "auc_roc_std": np.std(cv_auc_roc),
        "f1_mean": np.mean(cv_f1),
        "f1_std": np.std(cv_f1),
        "precision_mean": np.mean(cv_precision),
        "precision_std": np.std(cv_precision),
        "recall_mean": np.mean(cv_recall),
        "recall_std": np.std(cv_recall),
    }

    print("\nStarting MLflow run...")
    with mlflow.start_run(experiment_id=exp_id) as run:
        # Log parameters
        mlflow.log_params(model_params)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("cv_folds", n_splits)

        # Log metrics
        log_cv_metrics(cv_metrics)

        # Optimal threshold across Out-Of-Fold
        prec_oof, rec_oof, thresh_oof = precision_recall_curve(y, oof_proba)
        fscores_oof = np.divide(
            2 * prec_oof * rec_oof,
            prec_oof + rec_oof,
            out=np.zeros_like(prec_oof),
            where=(prec_oof + rec_oof) != 0,
        )
        opt_idx_oof = np.argmax(fscores_oof)
        opt_thresh_oof = (
            thresh_oof[opt_idx_oof] if opt_idx_oof < len(thresh_oof) else 0.5
        )
        oof_opt_preds = (oof_proba >= opt_thresh_oof).astype(int)

        # Log plots using out-of-fold predictions
        log_confusion_matrix(y, oof_opt_preds, "baseline_confusion_matrix.png")
        log_pr_curve(y, oof_proba, "baseline_pr_curve.png")

        # Fit final model on full training set
        print("Fitting final model on full dataset...")
        lr.fit(X_transformed, y)

        # Log fitted model
        mlflow.sklearn.log_model(lr, "model")

        run_id = run.info.run_id

        # Summary
        print("\nBaseline — Logistic Regression")
        print("================================")
        print(
            f"AUC-PR  (CV mean ± std): {cv_metrics['auc_pr_mean']:.4f} ± {cv_metrics['auc_pr_std']:.4f}"
        )
        print(
            f"AUC-ROC (CV mean ± std): {cv_metrics['auc_roc_mean']:.4f} ± {cv_metrics['auc_roc_std']:.4f}"
        )
        print(f"MLflow run ID: {run_id}")
        print(f"View at: {mlflow_uri}")

    return {"metrics": cv_metrics, "run_id": run_id}


def main():
    train_baseline()


if __name__ == "__main__":
    main()
