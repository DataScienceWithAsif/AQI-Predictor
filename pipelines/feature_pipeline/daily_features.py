"""
feature_pipeline/daily_features.py

Your project needs the AVERAGE AQI per calendar day for the next 3 days —
not an hourly point value 24h/48h/72h out. This module aggregates the
hourly raw data (same schema fetch.py/backfill.py produce) into daily
means first, then engineers day-level features and targets on top:

    daily_features_df, daily_targets_df = compute_daily_features(raw_df)

This replaces features.py's aqi_next_1d/2d/3d (hourly point-forecast
targets) with aqi_avg_next_1d/2d/3d (daily-mean targets) — train your
Day 5+ models against THIS module's targets. features.py's hourly output
is still fine to keep around for EDA or an hourly nowcast, just not for
the actual 3-day forecast deliverable.
"""

from typing import Tuple

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "timestamp", "city", "temp", "humidity", "pressure", "wind_speed",
    "pm25", "pm10", "co", "no2", "so2", "o3", "aqi",
]
MEAN_COLUMNS = [
    "temp", "humidity", "pressure", "wind_speed",
    "pm25", "pm10", "co", "no2", "so2", "o3", "aqi",
]

LAG_DAYS = [1, 2, 3, 7]
ROLLING_WINDOW_DAYS = 7
TARGET_HORIZONS_DAYS = {
    "aqi_avg_next_1d": 1,
    "aqi_avg_next_2d": 2,
    "aqi_avg_next_3d": 3,
}

# A day built from fewer than this many hourly readings is too sparse to
# trust as a real daily average (a missed cron run for 20 of 24 hours
# shouldn't quietly become "today's AQI"). Such days are flagged NaN
# rather than averaged over 2-3 lucky readings.
MIN_HOURS_PER_VALID_DAY = 18


# --------------------------------------------------------------------------
# Step 1: hourly -> daily aggregation
# --------------------------------------------------------------------------

def aggregate_to_daily(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses hourly rows into one row per (city, calendar day): the mean
    of every numeric column, plus 'hours_reported' — a coverage count so
    thin/incomplete days can be identified instead of silently trusted.
    """
    df = raw_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.floor("D")

    grouped = df.groupby(["city", "date"])
    daily = grouped[MEAN_COLUMNS].mean()
    daily["hours_reported"] = grouped.size()

    incomplete = daily["hours_reported"] < MIN_HOURS_PER_VALID_DAY
    daily.loc[incomplete, MEAN_COLUMNS] = np.nan

    return daily.reset_index()


# --------------------------------------------------------------------------
# Step 2: day-level feature builders (mirrors features.py's hourly logic)
# --------------------------------------------------------------------------

def _reindex_to_complete_daily_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Same reasoning as the hourly version: a missing calendar day should
    become an explicit NaN row, not silently shift what 'lag_1d' means."""
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
    full_index.name = "date"
    return df.reindex(full_index)


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["day_of_year"] = df.index.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    # day-of-year cyclical encoding captures seasonal AQI patterns
    # (e.g. winter smog) that pure month/week features miss
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df


def _add_change_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    days_since_prev = df.index.to_series().diff().dt.total_seconds() / 86400
    df["aqi_change_rate"] = df["aqi"].diff() / days_since_prev
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for d in LAG_DAYS:
        df[f"aqi_lag_{d}d"] = df["aqi"].shift(d)
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    w = ROLLING_WINDOW_DAYS
    df["aqi_roll_mean_7d"] = df["aqi"].rolling(w, min_periods=3).mean()
    df["aqi_roll_std_7d"] = df["aqi"].rolling(w, min_periods=3).std()
    df["pm25_roll_mean_7d"] = df["pm25"].rolling(w, min_periods=3).mean()
    df["pm25_roll_std_7d"] = df["pm25"].rolling(w, min_periods=3).std()
    return df


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Ground-truth daily-mean AQI 1/2/3 days ahead, from the untouched
    per-day 'aqi' column (itself already a daily mean from Step 1)."""
    df = df.copy()
    for target_col, horizon_days in TARGET_HORIZONS_DAYS.items():
        df[target_col] = df["aqi"].shift(-horizon_days)
    return df


def _compute_daily_features_single_city(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("date").sort_index()
    df = _reindex_to_complete_daily_grid(df)

    df = _add_time_features(df)
    df = _add_change_rate(df)
    df = _add_lag_features(df)
    df = _add_rolling_features(df)
    df = _add_targets(df)
    return df


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def compute_daily_features(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Takes raw hourly rows (fetch.py/backfill.py schema) and returns
    (daily_features_df, daily_targets_df), both indexed by 'date' with a
    'city' column, aligned row-for-row — one row per city per calendar day.
    """
    missing_cols = set(RAW_COLUMNS) - set(raw_df.columns)
    if missing_cols:
        raise ValueError(f"raw_df is missing expected columns: {missing_cols}")

    daily_raw = aggregate_to_daily(raw_df)

    processed = daily_raw.groupby("city").apply(
        _compute_daily_features_single_city, include_groups=False
    )
    processed.index = processed.index.set_names(["city", "date"])

    feature_cols = [
        "temp", "humidity", "pressure", "wind_speed",
        "pm25", "pm10", "co", "no2", "so2", "o3", "aqi",
        "hours_reported",
        "day_of_week", "month", "day_of_year", "is_weekend",
        "month_sin", "month_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
        "aqi_change_rate",
        "aqi_lag_1d", "aqi_lag_2d", "aqi_lag_3d", "aqi_lag_7d",
        "aqi_roll_mean_7d", "aqi_roll_std_7d",
        "pm25_roll_mean_7d", "pm25_roll_std_7d",
    ]
    target_cols = list(TARGET_HORIZONS_DAYS.keys())

    daily_features_df = processed[feature_cols].reset_index()
    daily_targets_df = processed[target_cols].reset_index()

    return daily_features_df, daily_targets_df


if __name__ == "__main__":
    import os

    demo_path = "day2_feature_pipeline_log.csv"
    backfill_path = "day4_backfilled_raw.csv"
    if os.path.exists(backfill_path):
        print(f"Loading backfilled data from {backfill_path}...")
        raw_df = pd.read_csv(backfill_path)
    elif os.path.exists(demo_path):
        print(f"Loading real data from {demo_path}...")
        raw_df = pd.read_csv(demo_path)
    else:
        raise SystemExit(
            "No data found — run backfill.py (Day 4) first, or fetch.py a few times."
        )

    daily_features_df, daily_targets_df = compute_daily_features(raw_df)
    print(f"\ndaily_features_df shape: {daily_features_df.shape}")
    print(f"daily_targets_df shape:  {daily_targets_df.shape}")
    print("\ndaily_features_df.tail(5):")
    print(daily_features_df.tail(5).to_string())
    print("\ndaily_targets_df.tail(5):")
    print(daily_targets_df.tail(5).to_string())