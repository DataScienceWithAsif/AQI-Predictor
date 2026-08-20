"""
Correctness tests for feature_pipeline/features.py using a small,
deterministic dataset (aqi = 100, 101, 102, ...) so every expected value is
hand-checkable, plus a version with a deliberate gap to verify gap handling.

Run with:
    python feature_pipeline/test_features.py
"""

import numpy as np
import pandas as pd

import features


def _make_clean_hourly_data(n_hours: int = 50) -> pd.DataFrame:
    """AQI increases by exactly 1 each hour: 100, 101, 102, ... — makes every
    lag/rolling/target value predictable by hand."""
    timestamps = pd.date_range("2026-08-01", periods=n_hours, freq="h", tz="UTC")
    aqi = 100 + np.arange(n_hours)
    return pd.DataFrame({
        "timestamp": timestamps.astype(str),
        "city": "TestCity",
        "temp": 25.0, "humidity": 50.0, "pressure": 1010.0, "wind_speed": 2.0,
        "pm25": aqi * 0.5, "pm10": aqi * 0.7,
        "co": 300.0, "no2": 20.0, "so2": 5.0, "o3": 50.0,
        "aqi": aqi.astype(float),
    })


def test_lag_features_are_correct():
    raw_df = _make_clean_hourly_data(n_hours=50)
    features_df, _ = features.compute_features(raw_df)

    # Row 30: aqi=130. aqi 1h ago should be 129, 3h ago 127, 24h ago 106.
    row = features_df[features_df["aqi"] == 130].iloc[0]
    assert row["aqi_lag_1h"] == 129
    assert row["aqi_lag_3h"] == 127
    assert row["aqi_lag_24h"] == 106
    print("PASS: lag features match hand-calculated values")


def test_change_rate_is_correct():
    raw_df = _make_clean_hourly_data(n_hours=50)
    features_df, _ = features.compute_features(raw_df)

    # AQI rises by exactly 1 every hour on a clean hourly grid -> change rate = 1.0
    non_null = features_df["aqi_change_rate"].dropna()
    assert (non_null == 1.0).all(), f"Expected all 1.0, got {non_null.unique()}"
    print("PASS: aqi_change_rate == 1.0 for every row on a clean +1/hour series")


def test_targets_are_correct():
    raw_df = _make_clean_hourly_data(n_hours=100)
    _, targets_df = features.compute_features(raw_df)

    # Row 10 (aqi=110): aqi_next_1d should be aqi at hour 34 = 134, etc.
    row = targets_df.iloc[10]
    assert row["aqi_next_1d"] == 100 + 10 + 24
    assert row["aqi_next_2d"] == 100 + 10 + 48
    assert row["aqi_next_3d"] == 100 + 10 + 72
    print("PASS: target columns correctly shifted 24h/48h/72h into the future")


def test_rolling_mean_matches_manual_pandas_calc():
    raw_df = _make_clean_hourly_data(n_hours=50)
    features_df, _ = features.compute_features(raw_df)

    manual_series = pd.Series(100 + np.arange(50))
    manual_roll_mean = manual_series.rolling(24, min_periods=6).mean()

    pd.testing.assert_series_equal(
        features_df["aqi_roll_mean_24h"].reset_index(drop=True),
        manual_roll_mean.reset_index(drop=True),
        check_names=False,
    )
    print("PASS: aqi_roll_mean_24h matches an independent manual pandas calculation")


def test_gap_produces_nan_not_wrong_values():
    """Drop hour index 20 entirely, then check the row right after the gap
    has a NaN lag_1h (since the row 1h before it is now missing/NaN) rather
    than silently pulling in a value from 2 hours ago and mislabeling it."""
    raw_df = _make_clean_hourly_data(n_hours=50)
    raw_df_with_gap = raw_df.drop(raw_df.index[20]).reset_index(drop=True)

    features_df, _ = features.compute_features(raw_df_with_gap)

    # hour 20 (aqi=120) is missing entirely; hour 21 (aqi=121) should now
    # have NaN aqi_lag_1h, NOT silently reuse aqi=119 from hour 19.
    row_after_gap = features_df[features_df["aqi"] == 121].iloc[0]
    assert pd.isna(row_after_gap["aqi_lag_1h"]), (
        f"Expected NaN across the gap, got {row_after_gap['aqi_lag_1h']}"
    )
    # and the gap hour itself should exist as a NaN row, not be silently dropped
    assert len(features_df) == 50, f"Expected reindexed length 50, got {len(features_df)}"
    print("PASS: a missing hour becomes an explicit NaN row instead of a silently wrong lag")


if __name__ == "__main__":
    test_lag_features_are_correct()
    test_change_rate_is_correct()
    test_targets_are_correct()
    test_rolling_mean_matches_manual_pandas_calc()
    test_gap_produces_nan_not_wrong_values()
    print("\nAll Day 3 correctness tests passed.")