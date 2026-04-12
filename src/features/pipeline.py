"""
pipeline.py

Builds and serializes the complete sklearn preprocessing pipeline
for the Home Credit Machine Learning problem.
"""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.merge import load_merged
from src.features.transformers import (
    CyclicalEncoder,
    DropColumns,
    ExternalSourceAggregator,
    IsUnemployedFlagger,
    RatioFeatureCreator,
    SentinelImputer,
)


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Helper to dynamically get numeric columns."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Helper to dynamically get categorical columns."""
    return df.select_dtypes(include=["category", "object"]).columns.tolist()


def build_pipeline(drop_columns: list[str]) -> Pipeline:
    """Constructs the unfitted preprocessing pipeline."""
    # We use make_column_selector dynamically, so the ColumnTransformer figures out
    # at fit time which output columns from the prior stages are numeric/categorical.
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                make_column_selector(dtype_include=[np.number]),
            ),
            (
                "cat",
                categorical_transformer,
                make_column_selector(dtype_exclude=[np.number]),
            ),
        ],
        remainder="drop",  # Although conceptually everything should be matched.
    )

    pipeline = Pipeline(
        steps=[
            ("sentinel", SentinelImputer()),
            ("unemployed_flag", IsUnemployedFlagger()),
            ("ratios", RatioFeatureCreator()),
            ("external_agg", ExternalSourceAggregator()),
            ("cyclical", CyclicalEncoder()),
            ("drop_cols", DropColumns(drop_columns)),
            ("preprocessor", preprocessor),
        ]
    )

    return pipeline


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Retrieves feature names from the fitted pipeline."""
    try:
        # Get preprocessor step
        preprocessor = pipeline.named_steps["preprocessor"]
        return preprocessor.get_feature_names_out().tolist()
    except AttributeError:
        # Fallback if get_feature_names_out is unavailable
        print("Warning: get_feature_names_out() unavailable, returning generic names")
        try:
            n_features = pipeline.transform(pd.DataFrame()).shape[1]
            return [f"feature_{i}" for i in range(n_features)]
        except Exception:
            return []


def fit_pipeline(
    df: pd.DataFrame, target_col: str = "TARGET"
) -> tuple[Pipeline, pd.Series]:
    """Fits the pipeline on the dataframe, returning the pipeline and the target series."""
    y = df[target_col].copy() if target_col in df.columns else None
    X = df.drop(columns=[target_col]) if target_col in df.columns else df.copy()

    columns_to_drop = [
        "SK_ID_CURR",
        "DAYS_EMPLOYED",
        "WEEKDAY_APPR_PROCESS_START",
        "COMMONAREA_MODE",
        "COMMONAREA_MEDI",
        "NONLIVINGAPARTMENTS_MODE",
        "NONLIVINGAPARTMENTS_MEDI",
        "LIVINGAPARTMENTS_MODE",
        "LIVINGAPARTMENTS_MEDI",
        "FLOORSMIN_MODE",
        "FLOORSMIN_MEDI",
        "YEARS_BUILD_MODE",
        "YEARS_BUILD_MEDI",
        "LANDAREA_MODE",
        "LANDAREA_MEDI",
        "BASEMENTAREA_MODE",
        "BASEMENTAREA_MEDI",
        "NONLIVINGAREA_MODE",
        "NONLIVINGAREA_MEDI",
        "ELEVATORS_MODE",
        "ELEVATORS_MEDI",
        "APARTMENTS_MODE",
        "APARTMENTS_MEDI",
        "ENTRANCES_MODE",
        "ENTRANCES_MEDI",
        "LIVINGAREA_MODE",
        "LIVINGAREA_MEDI",
        "FLOORSMAX_MODE",
        "FLOORSMAX_MEDI",
        "TOTALAREA_MODE",
        "FLAG_MOBIL",
        "FLAG_DOCUMENT_2",
        "FLAG_DOCUMENT_10",
        "FLAG_DOCUMENT_12",
        "FLAG_DOCUMENT_17",
        "FLAG_DOCUMENT_19",
        "FLAG_DOCUMENT_20",
        "FLAG_DOCUMENT_21",
    ]

    pipeline = build_pipeline(columns_to_drop)
    print(f"Fitting pipeline on X of shape {X.shape}...")
    pipeline.fit(X, y)
    return pipeline, y


def save_pipeline(pipeline: Pipeline, path: str = "models/pipeline.pkl") -> None:
    """Saves the pipeline to path using joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    size = os.path.getsize(path) / (1024 * 1024)
    print(f"Pipeline saved to {path} ({size:.2f} MB)")


def load_pipeline(path: str = "models/pipeline.pkl") -> Pipeline:
    """Loads the pipeline from path using joblib."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pipeline file not found at {path}")
    return joblib.load(path)


def main():
    print("=" * 60)
    print("Building Full Preprocessing Pipeline")
    print("=" * 60)

    try:
        df = load_merged()
    except Exception as e:
        print(f"Dataset load error: {e}")
        return

    pipeline, y = fit_pipeline(df)
    save_pipeline(pipeline)

    out_names = get_feature_names(pipeline)
    print("Pipeline summary:")
    print(f"  Features in : {df.shape[1] - 1}")
    print(f"  Features out: {len(out_names)}")

    print("\nRunning serialization roundtrip test...")
    sample = df.head(100).drop(columns=["TARGET"])
    out_original = pipeline.transform(sample)

    loaded_pipe = load_pipeline()
    out_loaded = loaded_pipe.transform(sample)

    if np.allclose(out_original, out_loaded, equal_nan=True):
        print("Serialization roundtrip: PASS")
    else:
        print("Serialization roundtrip: FAIL")


if __name__ == "__main__":
    main()
