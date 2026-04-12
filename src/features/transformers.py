"""
transformers.py

Custom scikit-learn transformers for the Home Credit ML pipeline.
Each transformer implements fit() and transform() and returns a modified copy
of the input DataFrame.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SentinelImputer(BaseEstimator, TransformerMixin):
    """Replaces DAYS_EMPLOYED == 365243 with NaN."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "DAYS_EMPLOYED" in X.columns:
            X["DAYS_EMPLOYED"] = X["DAYS_EMPLOYED"].replace(365243, np.nan)
        return X


class IsUnemployedFlagger(BaseEstimator, TransformerMixin):
    """Creates a binary IS_UNEMPLOYED column where DAYS_EMPLOYED is NaN."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "DAYS_EMPLOYED" in X.columns:
            X["IS_UNEMPLOYED"] = X["DAYS_EMPLOYED"].isna().astype(int)
        else:
            X["IS_UNEMPLOYED"] = 0
        return X


class RatioFeatureCreator(BaseEstimator, TransformerMixin):
    """Calculates custom ratio and domain-specific features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "AMT_CREDIT" in X.columns and "AMT_INCOME_TOTAL" in X.columns:
            X["CREDIT_INCOME_RATIO"] = X["AMT_CREDIT"] / (X["AMT_INCOME_TOTAL"] + 1)

        if "AMT_ANNUITY" in X.columns and "AMT_INCOME_TOTAL" in X.columns:
            X["ANNUITY_INCOME_RATIO"] = X["AMT_ANNUITY"] / (X["AMT_INCOME_TOTAL"] + 1)

        if "AMT_ANNUITY" in X.columns and "AMT_CREDIT" in X.columns:
            X["CREDIT_TERM"] = X["AMT_ANNUITY"] / (X["AMT_CREDIT"] + 1)

        if "DAYS_EMPLOYED" in X.columns and "DAYS_BIRTH" in X.columns:
            # -1 to avoid division by zero
            X["DAYS_EMPLOYED_RATIO"] = X["DAYS_EMPLOYED"] / (X["DAYS_BIRTH"] - 1)

        if "DAYS_BIRTH" in X.columns:
            X["AGE_YEARS"] = X["DAYS_BIRTH"].abs() / 365.0

        return X


class ExternalSourceAggregator(BaseEstimator, TransformerMixin):
    """Computes mean, std, and missing count for external source features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        ext_cols = [
            c
            for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
            if c in X.columns
        ]
        if len(ext_cols) > 0:
            X["EXT_SOURCE_MEAN"] = X[ext_cols].mean(axis=1)
            X["EXT_SOURCE_STD"] = X[ext_cols].std(axis=1)
            X["EXT_SOURCE_MISSING_COUNT"] = X[ext_cols].isna().sum(axis=1)

        return X


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """Encodes WEEKDAY_APPR_PROCESS_START using sine and cosine."""

    def __init__(self):
        self.period_ = 7
        self.day_mapping_ = {
            "MONDAY": 0,
            "TUESDAY": 1,
            "WEDNESDAY": 2,
            "THURSDAY": 3,
            "FRIDAY": 4,
            "SATURDAY": 5,
            "SUNDAY": 6,
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "WEEKDAY_APPR_PROCESS_START" in X.columns:
            # Map string/categorical days to numbers if they are strings
            if X["WEEKDAY_APPR_PROCESS_START"].dtype.name in ["category", "object"]:
                mapped = (
                    X["WEEKDAY_APPR_PROCESS_START"]
                    .astype(str)
                    .str.upper()
                    .map(self.day_mapping_)
                )
            else:
                mapped = X["WEEKDAY_APPR_PROCESS_START"]

            X["WEEKDAY_SIN"] = np.sin(2 * np.pi * mapped / self.period_)
            X["WEEKDAY_COS"] = np.cos(2 * np.pi * mapped / self.period_)

        return X


class DropColumns(BaseEstimator, TransformerMixin):
    """Drops specified columns if they exist in the DataFrame."""

    def __init__(self, columns_to_drop: list):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cols_to_drop = [c for c in self.columns_to_drop if c in X.columns]
        X = X.drop(columns=cols_to_drop)
        return X
