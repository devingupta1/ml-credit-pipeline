import numpy as np
import pandas as pd
import pytest

from src.features.transformers import (
    CyclicalEncoder,
    DropColumns,
    ExternalSourceAggregator,
    IsUnemployedFlagger,
    RatioFeatureCreator,
    SentinelImputer,
)


def test_sentinel_imputer():
    df = pd.DataFrame({"DAYS_EMPLOYED": [100, 365243, -500], "OTHER_COL": [1, 2, 3]})

    imputer = SentinelImputer()
    result = imputer.fit_transform(df)

    assert pd.isna(result.loc[1, "DAYS_EMPLOYED"])
    assert result.loc[0, "DAYS_EMPLOYED"] == 100
    assert result.loc[2, "DAYS_EMPLOYED"] == -500
    assert "OTHER_COL" in result.columns
    # original df unchanged
    assert df.loc[1, "DAYS_EMPLOYED"] == 365243

    # Test column absent
    df2 = pd.DataFrame({"A": [1, 2]})
    result2 = imputer.transform(df2)
    assert "A" in result2.columns


def test_is_unemployed_flagger():
    df = pd.DataFrame({"DAYS_EMPLOYED": [100, np.nan, -500]})

    flagger = IsUnemployedFlagger()
    result = flagger.fit_transform(df)

    assert list(result["IS_UNEMPLOYED"]) == [0, 1, 0]
    # original df unchanged
    assert "IS_UNEMPLOYED" not in df.columns

    # Test column absent
    df2 = pd.DataFrame({"A": [1]})
    result2 = flagger.transform(df2)
    assert result2.loc[0, "IS_UNEMPLOYED"] == 0


def test_ratio_feature_creator():
    df = pd.DataFrame(
        {
            "AMT_CREDIT": [100000.0, 0.0],
            "AMT_INCOME_TOTAL": [50000.0, 0.0],
            "AMT_ANNUITY": [5000.0, 100.0],
            "DAYS_EMPLOYED": [-1000.0, np.nan],
            "DAYS_BIRTH": [-10000.0, -15000.0],
        }
    )

    creator = RatioFeatureCreator()
    result = creator.fit_transform(df)

    # Check CREDIT_INCOME_RATIO
    assert result.loc[0, "CREDIT_INCOME_RATIO"] == pytest.approx(100000.0 / 50001.0)
    assert result.loc[1, "CREDIT_INCOME_RATIO"] == 0.0  # (0 / 1)

    # Check zero/NaN safety
    assert not np.isinf(result["CREDIT_INCOME_RATIO"]).any()
    assert pd.isna(result.loc[1, "DAYS_EMPLOYED_RATIO"])

    # Check AGE_YEARS
    assert result.loc[0, "AGE_YEARS"] == pytest.approx(10000.0 / 365.0)


def test_external_source_aggregator():
    df = pd.DataFrame(
        {
            "EXT_SOURCE_1": [0.5, np.nan, np.nan],
            "EXT_SOURCE_2": [0.4, 0.5, np.nan],
            "EXT_SOURCE_3": [0.6, np.nan, np.nan],
        }
    )

    agg = ExternalSourceAggregator()
    result = agg.fit_transform(df)

    assert result.loc[0, "EXT_SOURCE_MEAN"] == pytest.approx(0.5)
    assert result.loc[0, "EXT_SOURCE_MISSING_COUNT"] == 0

    assert result.loc[1, "EXT_SOURCE_MEAN"] == pytest.approx(0.5)
    assert result.loc[1, "EXT_SOURCE_MISSING_COUNT"] == 2

    assert pd.isna(result.loc[2, "EXT_SOURCE_MEAN"])
    assert result.loc[2, "EXT_SOURCE_MISSING_COUNT"] == 3


def test_cyclical_encoder():
    df = pd.DataFrame({"WEEKDAY_APPR_PROCESS_START": ["MONDAY", "SUNDAY", "WEDNESDAY"]})

    enc = CyclicalEncoder()
    result = enc.fit_transform(df)

    assert "WEEKDAY_SIN" in result.columns
    assert "WEEKDAY_COS" in result.columns

    # MONDAY -> 0
    assert result.loc[0, "WEEKDAY_SIN"] == pytest.approx(0.0)
    assert result.loc[0, "WEEKDAY_COS"] == pytest.approx(1.0)

    val_sin = result["WEEKDAY_SIN"]
    val_cos = result["WEEKDAY_COS"]
    assert val_sin.between(-1.0, 1.0).all()
    assert val_cos.between(-1.0, 1.0).all()


def test_drop_columns():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})

    dropper = DropColumns(columns_to_drop=["B", "D"])
    result = dropper.fit_transform(df)

    assert "B" not in result.columns
    assert "A" in result.columns
    assert "C" in result.columns
    # Doesn't fail if D is absent
