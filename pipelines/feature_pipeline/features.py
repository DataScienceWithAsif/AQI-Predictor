"""
feature_pipeline/features.py

Turns the raw hourly rows collected by fetch.py (one row per hour per city)
into model-ready features and prediction targets:

    features_df, targets_df = compute_features(raw_df)

Run standalone to see it work on either your real accumulated Day 2 data,
or synthetic demo data if you don't have enough real history yet:
    python feature_pipeline/features.py
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd

# Columns expected on the raw input (produced by fetch.build_feature_row)
RAW_COLUMNS = [
    "timestamp", "city", "temp", "humidity", "pressure", "wind_speed",
    "pm25", "pm10", "co", "no2", "so2", "o3", "aqi",
]

LAG_HOURS = [1, 3, 24]
ROLLING_WINDOW_HOURS = 24
TARGET_HORIZONS_HOURS = {"aqi_next_1d": 24, "aqi_next_2d": 48, "aqi_next_3d": 72}


# --------------------------------------------------------------------------
# Step-by-step feature builders (each takes/returns a timestamp-indexed df)
# --------------------------------------------------------------------------

def _reindex_to_complete_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lag/rolling features only mean what their names say ('1 hour ago') if
    rows are actually one hour apart. Real hourly collection will have gaps
    (API downtime, a missed GitHub Actions run) — this reindexes onto a
    complete hourly DatetimeIndex, inserting NaN rows for any missing hours,
    so 'lag_1h' is always genuinely 1 hour, never silently 3.
    """
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
    full_index.name = "timestamp"
    return df.reindex(full_index)


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features, plus sin/cos cyclical encodings so e.g. hour 23 and
    hour 0 are recognised as adjacent instead of maximally far apart."""
    df = df.copy()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek  # 0=Monday
    df["month"] = df.index.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def _add_change_rate(df: pd.DataFrame) -> pd.DataFrame:
    """AQI change rate = (aqi_now - aqi_prev) / hours_since_prev_reading.
    Dividing by the actual elapsed hours (rather than assuming 1) keeps this
    correct even if reindexing logic ever changes upstream."""
    df = df.copy()
    hours_since_prev = df.index.to_series().diff().dt.total_seconds() / 3600
    df["aqi_change_rate"] = df["aqi"].diff() / hours_since_prev
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for h in LAG_HOURS:
        df[f"aqi_lag_{h}h"] = df["aqi"].shift(h)
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling stats use the current row and the past — that's fine, no
    leakage, since 'current AQI' is legitimately known at prediction time.
    min_periods=6 lets early rows get a (noisier) rolling stat instead of
    forcing a hard 24h warm-up before any value appears."""
    df = df.copy()
    window = ROLLING_WINDOW_HOURS
    df["aqi_roll_mean_24h"] = df["aqi"].rolling(window, min_periods=6).mean()
    df["aqi_roll_std_24h"] = df["aqi"].rolling(window, min_periods=6).std()
    df["pm25_roll_mean_24h"] = df["pm25"].rolling(window, min_periods=6).mean()
    df["pm25_roll_std_24h"] = df["pm25"].rolling(window, min_periods=6).std()
    return df


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Ground-truth AQI N hours in the future, built by shifting the raw AQI
    column *backward*. Computed from the original untouched 'aqi' column,
    not from any lag/rolling-derived version of it."""
    df = df.copy()
    for target_col, horizon_hours in TARGET_HORIZONS_HOURS.items():
        df[target_col] = df["aqi"].shift(-horizon_hours)
    return df


def _compute_features_single_city(df: pd.DataFrame) -> pd.DataFrame:
    # Note: pandas excludes the groupby key column ('city') from what this
    # function receives, so 'city' isn't available (or needed) in here —
    # it's restored afterwards from the groupby key in compute_features().
    df = df.set_index("timestamp").sort_index()
    df = _reindex_to_complete_hourly_grid(df)

    df = _add_time_features(df)
    df = _add_change_rate(df)
    df = _add_lag_features(df)
    df = _add_rolling_features(df)
    df = _add_targets(df)
    return df


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def compute_features(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Takes raw rows produced by fetch.py (one row per hour per city) and
    returns (features_df, targets_df), both indexed by timestamp with a
    'city' column, aligned row-for-row.

    Rows at the start of the series (or right after a data gap) will have
    NaN lag/rolling/target values where there isn't enough history yet —
    that's expected and left in place on purpose. Drop or impute NaNs at
    training time (Day 5), not here, so this function stays a pure,
    inspectable transform you can unit test in isolation.
    """
    missing_cols = set(RAW_COLUMNS) - set(raw_df.columns)
    if missing_cols:
        raise ValueError(f"raw_df is missing expected columns: {missing_cols}")

    df = raw_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # group_keys=True (the default) attaches 'city' back as an outer index
    # level, giving us a (city, timestamp) MultiIndex on the result.
    # include_groups=False: the per-group function no longer needs 'city'
    # itself (it's restored from the group key below), so exclude it from
    # what each group receives — silences a pandas FutureWarning and is
    # also just the correct semantics here.
    processed = df.groupby("city").apply(_compute_features_single_city, include_groups=False)
    processed.index = processed.index.set_names(["city", "timestamp"])

    feature_cols = [
        "temp", "humidity", "pressure", "wind_speed",
        "pm25", "pm10", "co", "no2", "so2", "o3", "aqi",
        "hour", "day_of_week", "month", "is_weekend",
        "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
        "aqi_change_rate",
        "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_24h",
        "aqi_roll_mean_24h", "aqi_roll_std_24h",
        "pm25_roll_mean_24h", "pm25_roll_std_24h",
    ]
    target_cols = list(TARGET_HORIZONS_HOURS.keys())

    features_df = processed[feature_cols].reset_index()
    targets_df = processed[target_cols].reset_index()

    return features_df, targets_df


# --------------------------------------------------------------------------
# Demo / local test runner
# --------------------------------------------------------------------------

def _make_synthetic_demo_data(n_hours: int = 24 * 14, city: str = "DemoCity") -> pd.DataFrame:
    """
    Generates a plausible synthetic hourly AQI dataset (with a simulated
    5-hour outage) so you can exercise compute_features() today, even
    before you have several real days of accumulated history from fetch.py.
    Do NOT use this for actual model training — swap in your real
    backfilled data on Day 4.
    """
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2026-08-01", periods=n_hours, freq="h", tz="UTC")

    hour_of_day = timestamps.hour
    diurnal = 20 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)  # peaks in the evening
    trend = np.linspace(0, 10, n_hours)  # slow drift over the 2 weeks
    noise = rng.normal(0, 8, n_hours)
    aqi = np.clip(90 + diurnal + trend + noise, 5, 400)
    pm25 = np.clip(aqi * 0.5 + rng.normal(0, 5, n_hours), 1, 300)

    df = pd.DataFrame({
        "timestamp": timestamps.astype(str),
        "city": city,
        "temp": 25 + 8 * np.sin(2 * np.pi * (hour_of_day - 9) / 24) + rng.normal(0, 1, n_hours),
        "humidity": np.clip(50 + rng.normal(0, 10, n_hours), 10, 100),
        "pressure": 1008 + rng.normal(0, 2, n_hours),
        "wind_speed": np.clip(rng.normal(3, 1.5, n_hours), 0, None),
        "pm25": pm25,
        "pm10": pm25 * 1.4 + rng.normal(0, 5, n_hours),
        "co": rng.normal(300, 30, n_hours),
        "no2": rng.normal(20, 5, n_hours),
        "so2": rng.normal(5, 2, n_hours),
        "o3": rng.normal(50, 10, n_hours),
        "aqi": aqi,
    })

    # simulate a realistic outage in the middle, to exercise gap handling
    df = df.drop(df.index[200:205]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    demo_path = "day2_feature_pipeline_log.csv"
    if os.path.exists(demo_path):
        print(f"Loading real data from {demo_path}...")
        raw_df = pd.read_csv(demo_path)
        if len(raw_df) < 48:
            print(
                f"Only {len(raw_df)} real rows collected so far — too few to see "
                f"24h lag / 72h target features populate. Using synthetic demo "
                f"data instead so you can inspect the full pipeline today."
            )
            raw_df = _make_synthetic_demo_data()
    else:
        print("No real data found yet — generating synthetic demo data instead.")
        raw_df = _make_synthetic_demo_data()

    features_df, targets_df = compute_features(raw_df)

    print(f"\nfeatures_df shape: {features_df.shape}")
    print(f"targets_df shape:  {targets_df.shape}")

    print("\nfeatures_df.tail(3):")
    print(features_df.tail(3).to_string())

    print("\ntargets_df.tail(3):")
    print(targets_df.tail(3).to_string())

    print(f"\nNaN counts per column in features_df (expected near the start/gaps):")
    print(features_df.isna().sum())