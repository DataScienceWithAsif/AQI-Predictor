"""
feature_pipeline/verify_hopsworks_upload.py

Confirms backfill.py's data actually landed in Hopsworks correctly:
expected columns present, matching row counts between aqi_features and
aqi_targets, and — the check the UI alone won't show you — that every
features row actually joins to a targets row on (city, timestamp).

Run:
    python feature_pipeline/verify_hopsworks_upload.py
"""

import logging

from dotenv import load_dotenv

import hopsworks_io

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("feature_pipeline.verify")

EXPECTED_FEATURE_COLUMNS = {
    "timestamp", "city", "temp", "humidity", "pressure", "wind_speed",
    "pm25", "pm10", "co", "no2", "so2", "o3", "aqi",
    "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "aqi_change_rate", "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_24h",
    "aqi_roll_mean_24h", "aqi_roll_std_24h", "pm25_roll_mean_24h", "pm25_roll_std_24h",
}
EXPECTED_TARGET_COLUMNS = {
    "timestamp", "city", "aqi_avg_next_1d", "aqi_avg_next_2d", "aqi_avg_next_3d",
}


def main():
    fs = hopsworks_io.connect()
    # Deliberately NOT hopsworks_io.get_features_feature_group()/get_targets_feature_group()
    # here — those call get_or_create_feature_group(), which is built for the
    # write path (create-then-insert) and can hand back an unsaved local
    # placeholder (id=None) if its existence check doesn't resolve cleanly on
    # a pure read. fs.get_feature_group() always fetches real server metadata
    # for something that should already exist, and fails loudly if it doesn't.
    features_fg = fs.get_feature_group(name="aqi_features", version=hopsworks_io.DEFAULT_FEATURE_GROUP_VERSION)
    targets_fg = fs.get_feature_group(name="aqi_targets", version=hopsworks_io.DEFAULT_FEATURE_GROUP_VERSION)

    logger.info("Reading aqi_features back from Hopsworks (can take a minute)...")
    features_df = features_fg.read()
    logger.info("Reading aqi_targets back from Hopsworks...")
    targets_df = targets_fg.read()

    print(f"\naqi_features: {len(features_df)} rows, {len(features_df.columns)} columns")
    print(f"aqi_targets:  {len(targets_df)} rows, {len(targets_df.columns)} columns")

    missing_feature_cols = EXPECTED_FEATURE_COLUMNS - set(features_df.columns)
    missing_target_cols = EXPECTED_TARGET_COLUMNS - set(targets_df.columns)
    assert not missing_feature_cols, f"aqi_features is missing columns: {missing_feature_cols}"
    assert not missing_target_cols, f"aqi_targets is missing columns: {missing_target_cols}"
    print("PASS: both feature groups have all expected columns")

    assert len(features_df) == len(targets_df), (
        f"Row count mismatch: aqi_features has {len(features_df)}, aqi_targets has {len(targets_df)}"
    )
    print("PASS: aqi_features and aqi_targets have matching row counts")

    merged = features_df.merge(targets_df, on=["city", "timestamp"], how="inner")
    assert len(merged) == len(features_df), (
        f"Only {len(merged)}/{len(features_df)} rows joined on (city, timestamp) — "
        f"the two feature groups' primary keys may not actually match"
    )
    print("PASS: every features row joins cleanly to a targets row on (city, timestamp)")

    print("\nSample joined row (most recent):")
    sample_cols = [
        "timestamp", "city", "aqi", "aqi_lag_24h",
        "aqi_avg_next_1d", "aqi_avg_next_2d", "aqi_avg_next_3d",
    ]
    print(merged.sort_values("timestamp").tail(1)[sample_cols].to_string(index=False))

    print(f"\nDate range: {merged['timestamp'].min()} to {merged['timestamp'].max()}")
    print(f"Cities present: {merged['city'].unique().tolist()}")

    null_target_frac = merged[["aqi_avg_next_1d", "aqi_avg_next_2d", "aqi_avg_next_3d"]].isna().mean()
    print("\nNull fraction per target column (expect small, only near the very end of the data):")
    print(null_target_frac.to_string())

    print("\nAll checks passed — your Hopsworks upload is verified correct.")


if __name__ == "__main__":
    main()