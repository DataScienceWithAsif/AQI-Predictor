"""
feature_pipeline/check_automation_health.py

Day 9: a concrete, automatic answer to "is the CI/CD automation actually
running on its own?" — rather than manually re-reading timestamps in the
Hopsworks UI, or trusting a green GitHub Actions checkmark alone (which
proves a run happened, not that it happened ON SCHEDULE, unattended).

Checks:
    - aqi_raw_hourly: every city's newest row should be < ~2 hours old
      (the hourly workflow should have refreshed it recently)
    - aqi_features / aqi_targets: newest row overall should be < ~30 hours
      old (the daily workflow refreshes these once every 24h)

Run any time:
    python feature_pipeline/check_automation_health.py
"""

import sys
import logging
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

import hopsworks_io

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("feature_pipeline.check_automation_health")

RAW_HOURLY_MAX_AGE_HOURS = 2       # hourly job should refresh well within this
DAILY_FEATURES_MAX_AGE_HOURS = 30  # daily job should refresh well within this


def _age_hours(ts: pd.Timestamp) -> float:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return (datetime.now(timezone.utc) - ts).total_seconds() / 3600


def check_raw_hourly(fs) -> list:
    """Returns a list of (city, age_hours) for any city whose latest
    aqi_raw_hourly reading is staler than expected."""
    fg = fs.get_feature_group(name="aqi_raw_hourly", version=1)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    latest_per_city = df.sort_values("timestamp").groupby("city")["timestamp"].max()

    stale = []
    print("\naqi_raw_hourly — latest reading per city:")
    for city, ts in latest_per_city.items():
        age = _age_hours(ts)
        status = "OK" if age <= RAW_HOURLY_MAX_AGE_HOURS else "STALE"
        print(f"  {city:12s} {ts}  ({age:.1f}h ago)  [{status}]")
        if age > RAW_HOURLY_MAX_AGE_HOURS:
            stale.append((city, age))
    return stale


def check_daily_refresh(fs, fg_name: str, version: int) -> float:
    """Returns the age in hours of the newest row in the given feature group."""
    fg = fs.get_feature_group(name=fg_name, version=version)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    newest = df["timestamp"].max()
    age = _age_hours(newest)
    status = "OK" if age <= DAILY_FEATURES_MAX_AGE_HOURS else "STALE"
    print(f"\n{fg_name} — newest row: {newest} ({age:.1f}h ago) [{status}]")
    return age


def main():
    fs = hopsworks_io.connect()

    print("=" * 70)
    print("AUTOMATION HEALTH CHECK")
    print("=" * 70)

    stale_cities = check_raw_hourly(fs)
    features_age = check_daily_refresh(fs, "aqi_features", hopsworks_io.DEFAULT_FEATURE_GROUP_VERSION)
    targets_age = check_daily_refresh(fs, "aqi_targets", hopsworks_io.DEFAULT_FEATURE_GROUP_VERSION)

    print("\n" + "=" * 70)
    problems = []
    if stale_cities:
        problems.append(
            f"{len(stale_cities)} cit(ies) have stale aqi_raw_hourly data "
            f"(> {RAW_HOURLY_MAX_AGE_HOURS}h old): {[c for c, _ in stale_cities]}. "
            f"Check the 'Hourly Feature Pipeline' workflow's recent runs for failures."
        )
    if features_age > DAILY_FEATURES_MAX_AGE_HOURS:
        problems.append(
            f"aqi_features is {features_age:.1f}h old (expected < {DAILY_FEATURES_MAX_AGE_HOURS}h). "
            f"Check the 'Daily Training Pipeline' workflow's recent runs."
        )
    if targets_age > DAILY_FEATURES_MAX_AGE_HOURS:
        problems.append(f"aqi_targets is {targets_age:.1f}h old (expected < {DAILY_FEATURES_MAX_AGE_HOURS}h).")

    if problems:
        print("RESULT: ISSUES FOUND\n")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    else:
        print("RESULT: ALL CHECKS PASSED — automation is running on schedule.")


if __name__ == "__main__":
    main()