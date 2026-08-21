"""
feature_pipeline/final_features.py

Combines the two prior approaches into the design that satisfies BOTH
constraints on this project:

  1. The brief explicitly asks for hour-level time features and an
     hourly-granularity AQI change rate/lag/rolling features (Day 3 slide:
     "Include time-based features (hour, day, month)...").
  2. The actual deliverable is the AVERAGE AQI per calendar day for the
     next 3 days — a daily target, not an hourly point value.

So: keep hourly-granularity FEATURES (rich signal, matches the brief
literally, ~2,000+ training rows from a 90-day backfill instead of ~90),
but compute TARGETS as the daily-mean AQI for day+1/2/3 relative to each
row's own calendar day.

    features_df, targets_df = compute_features_with_daily_targets(raw_df)

Train your Day 5 model against THIS module's output.
"""

from typing import Tuple

import pandas as pd

import features as hourly_mod
import daily_features as daily_mod


def compute_features_with_daily_targets(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 1. Rich hourly features (discard the hourly point-forecast targets —
    #    we only want features.py's feature engineering, not its targets).
    hourly_features_df, _ = hourly_mod.compute_features(raw_df)
    hourly_features_df = hourly_features_df.copy()
    hourly_features_df["date"] = hourly_features_df["timestamp"].dt.floor("D")

    # 2. Daily mean AQI per city/day (with the incomplete-day NaN guard
    #    from daily_features.py's aggregate_to_daily).
    daily_avg = daily_mod.aggregate_to_daily(raw_df)[["city", "date", "aqi"]]
    daily_avg = daily_avg.rename(columns={"aqi": "daily_avg_aqi"})

    # 3. For each hourly row, join in the daily mean AQI 1/2/3 calendar
    #    days after that row's OWN date (not 24/48/72 hours after its
    #    exact hour — a 11pm row and a 1am row on the same day should get
    #    the identical "tomorrow's average" target).
    targets_df = hourly_features_df[["timestamp", "city", "date"]].copy()

    for target_col, horizon_days in daily_mod.TARGET_HORIZONS_DAYS.items():
        join_key = pd.DataFrame({
            "city": targets_df["city"],
            "date": targets_df["date"] + pd.Timedelta(days=horizon_days),
        })
        matched = join_key.merge(
            daily_avg.rename(columns={"daily_avg_aqi": target_col}),
            on=["city", "date"],
            how="left",
        )
        targets_df[target_col] = matched[target_col].values

    features_df = hourly_features_df.drop(columns=["date"])
    targets_df = targets_df.drop(columns=["date"])

    return features_df, targets_df


if __name__ == "__main__":
    import os

    backfill_path = "day4_backfilled_raw.csv"
    demo_path = "day2_feature_pipeline_log.csv"
    if os.path.exists(backfill_path):
        raw_df = pd.read_csv(backfill_path)
    elif os.path.exists(demo_path):
        raw_df = pd.read_csv(demo_path)
    else:
        raise SystemExit("No data found — run backfill.py (Day 4) first, or fetch.py a few times.")

    features_df, targets_df = compute_features_with_daily_targets(raw_df)
    print(f"features_df shape: {features_df.shape}")
    print(f"targets_df shape:  {targets_df.shape}")
    print("\ntargets_df.tail(10):")
    print(targets_df.tail(10).to_string())