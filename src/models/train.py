"""
train.py

Baseline Logistic Regression model, LightGBM/XGBoost Optuna tuning,
and MLflow experiment setup for the Home Credit Machine Learning problem.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import optuna
import seaborn as sns
import xgboost as xgb
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
    """Trains logistic regression baseline using StratifiedKFold and logs to MLflow."""
    load_dotenv()
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(mlflow_uri)
    exp_name = "home-credit-default-risk"
    exp_id = get_experiment_id(exp_name)

    df = load_merged()
    y = df["TARGET"].values

    print("Fitting preprocessing pipeline...")
    pipeline, pipeline_y = fit_pipeline(df, "TARGET")

    X_df = df.drop(columns=["TARGET"]) if "TARGET" in df.columns else df.copy()
    print("Transforming data...")
    X_transformed = pipeline.transform(X_df)

    model_params = {"max_iter": 1000, "class_weight": "balanced", "random_state": 42}
    lr = LogisticRegression(**model_params)

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

        proba = lr.predict_proba(X_va)[:, 1]
        preds = lr.predict(X_va)

        oof_proba[val_idx] = proba
        oof_pred[val_idx] = preds

        prec, rec, thresholds = precision_recall_curve(y_va, proba)
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
        mlflow.log_params(model_params)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("cv_folds", n_splits)
        log_cv_metrics(cv_metrics)

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

        log_confusion_matrix(y, oof_opt_preds, "baseline_confusion_matrix.png")
        log_pr_curve(y, oof_proba, "baseline_pr_curve.png")

        print("Fitting final model on full dataset...")
        lr.fit(X_transformed, y)

        mlflow.sklearn.log_model(lr, "model")

        run_id = run.info.run_id

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


def get_lgbm_search_space(trial) -> dict:
    """Returns LightGBM hyperparameter search space for an Optuna trial."""
    return {
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
    }


def get_xgb_search_space(trial) -> dict:
    """Returns XGBoost hyperparameter search space for an Optuna trial."""
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
    }


def run_optuna_study(
    model_class, search_space_fn, study_name, X, y, n_trials=50
) -> tuple:
    """Runs a full Optuna study for the given model class."""
    exp_name = "home-credit-default-risk"
    exp_id = get_experiment_id(exp_name)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    with mlflow.start_run(experiment_id=exp_id, run_name=study_name) as parent_run:

        def objective(trial):
            with mlflow.start_run(experiment_id=exp_id, nested=True):
                params = search_space_fn(trial)

                model_params = params.copy()
                model_params["random_state"] = 42

                if model_class.__name__ == "LGBMClassifier":
                    model_params["is_unbalance"] = True
                    model_params["verbosity"] = -1
                elif model_class.__name__ == "XGBClassifier":
                    scale_pos_weight = np.sum(y == 0) / np.sum(y == 1)
                    model_params["scale_pos_weight"] = scale_pos_weight
                    model_params["verbosity"] = 0

                model = model_class(**model_params)

                cv_auc_pr = []
                for train_idx, val_idx in skf.split(X, y):
                    X_tr, y_tr = X[train_idx], y[train_idx]
                    X_va, y_va = X[val_idx], y[val_idx]

                    model.fit(X_tr, y_tr)

                    proba = model.predict_proba(X_va)[:, 1]
                    cv_auc_pr.append(average_precision_score(y_va, proba))

                mean_auc_pr = np.mean(cv_auc_pr)

                mlflow.log_params(params)
                mlflow.log_metric("auc_pr_mean", mean_auc_pr)

                return mean_auc_pr

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
            direction="maximize", sampler=sampler, study_name=study_name
        )
        print(f"Starting {n_trials} trials for {study_name}...")
        study.optimize(objective, n_trials=n_trials)

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_auc_pr", study.best_value)

        return study, study.best_params, study.best_value


def register_champion(
    run_id: str, model_name: str = "credit-default-risk-champion"
) -> None:
    """Registers the best model run to MLflow Model Registry at Staging."""
    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)

    client = mlflow.MlflowClient()
    client.transition_model_version_stage(
        name=model_name, version=registered_model.version, stage="Staging"
    )
    print(
        f"Champion registered to MLflow Model Registry (Staging) - Version {registered_model.version}"
    )


def train_boosting_models() -> dict:
    """Orchestrates the full tuning pipeline for LightGBM and XGBoost, registers champion."""
    load_dotenv()
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(mlflow_uri)

    df = load_merged()
    y = df["TARGET"].values

    print("Fitting preprocessing pipeline for tuning...")
    pipeline, pipeline_y = fit_pipeline(df, "TARGET")
    X_df = df.drop(columns=["TARGET"]) if "TARGET" in df.columns else df.copy()
    print("Transforming data...")
    X_transformed = pipeline.transform(X_df)

    # 1. Run LightGBM Optuna study
    print("\n--- LightGBM Tuning ---")
    lgbm_study, lgbm_best_params, lgbm_best_auc = run_optuna_study(
        model_class=lgb.LGBMClassifier,
        search_space_fn=get_lgbm_search_space,
        study_name="lgbm_optuna_study",
        X=X_transformed,
        y=y,
        n_trials=50,
    )

    # 2. Run XGBoost Optuna study
    print("\n--- XGBoost Tuning ---")
    xgb_study, xgb_best_params, xgb_best_auc = run_optuna_study(
        model_class=xgb.XGBClassifier,
        search_space_fn=get_xgb_search_space,
        study_name="xgboost_optuna_study",
        X=X_transformed,
        y=y,
        n_trials=50,
    )

    # 3. Compare models
    is_lgbm_champion = lgbm_best_auc >= xgb_best_auc
    champion_name = "LightGBM" if is_lgbm_champion else "XGBoost"
    champion_best_params = lgbm_best_params if is_lgbm_champion else xgb_best_params
    champion_best_auc = lgbm_best_auc if is_lgbm_champion else xgb_best_auc

    # 4. Retrain Champion on full dataset
    print(
        f"\nRetraining champion ({champion_name}) on full dataset with best params..."
    )
    champion_params = champion_best_params.copy()
    champion_params["random_state"] = 42

    if is_lgbm_champion:
        champion_params["is_unbalance"] = True
        champion_params["verbosity"] = -1
        final_model = lgb.LGBMClassifier(**champion_params)
    else:
        champion_params["scale_pos_weight"] = np.sum(y == 0) / np.sum(y == 1)
        champion_params["verbosity"] = 0
        final_model = xgb.XGBClassifier(**champion_params)

    exp_id = get_experiment_id("home-credit-default-risk")

    with mlflow.start_run(
        experiment_id=exp_id, run_name=f"{champion_name}_champion"
    ) as champion_run:
        mlflow.log_params(champion_params)
        mlflow.log_param("model_type", champion_name)
        mlflow.log_metric("cv_auc_pr", champion_best_auc)

        final_model.fit(X_transformed, y)

        # We need to log it with MLflow sklearn module or specific module
        if is_lgbm_champion:
            mlflow.lightgbm.log_model(final_model, "model")
        else:
            mlflow.xgboost.log_model(final_model, "model")

        run_id = champion_run.info.run_id

    # 5. Register champion
    register_champion(run_id, model_name="credit-default-risk-champion")

    # 6. Print Summary
    print("\nModel Comparison")
    print("==========================================")
    print(
        "Baseline  (LogReg):   AUC-PR = 0.2470"
    )  # Hardcoded baseline from previous step 9
    print(f"LightGBM  (best CV):  AUC-PR = {lgbm_best_auc:.4f}")
    print(f"XGBoost   (best CV):  AUC-PR = {xgb_best_auc:.4f}")
    print(f"Champion:             {champion_name}")
    print("==========================================")

    return {
        "lgbm_best_auc": lgbm_best_auc,
        "xgb_best_auc": xgb_best_auc,
        "champion": champion_name,
        "champion_run_id": run_id,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train Baseline and/or Tune Boosting Models"
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "tune", "all"],
        default="all",
        help="Mode to run: baseline | tune | all",
    )
    args = parser.parse_args()

    if args.mode in ["baseline", "all"]:
        print("\n--- Running Baseline ---")
        train_baseline()

    if args.mode in ["tune", "all"]:
        print("\n--- Running Tuning ---")
        train_boosting_models()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
