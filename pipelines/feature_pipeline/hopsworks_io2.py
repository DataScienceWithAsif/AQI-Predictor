"""
feature_pipeline/hopsworks_io2.py

Shared helpers for connecting to the Hopsworks Feature Store and
getting/creating the two feature groups used by this project:

    aqi_features  — model inputs (weather, pollutants, engineered features)
    aqi_targets   — model outputs (aqi_next_1d/2d/3d)

These are kept as two separate feature groups on purpose: at real
prediction time (the Day 8 web app), you'll have fresh *features* for the
current hour but you will never have the *targets* — those only exist for
historical rows where the future has already happened. Keeping them apart
now avoids a confusing "why are there real values in my target column at
inference time" bug later, and matches how Hopsworks Feature Views expect
inputs and labels to be joined from separate feature groups.

Used by backfill.py (Day 4), and later by the hourly pipeline (Day 7) and
the training pipeline (Day 5).
"""

import os
import logging

import pandas as pd
import hopsworks
from hsfs.feature_store import FeatureStore
from hsfs.feature_group import FeatureGroup

logger = logging.getLogger("feature_pipeline.hopsworks_io")


def connect_project():
    """
    Connects to Hopsworks using HOPSWORKS_API_KEY (and optionally
    HOPSWORKS_PROJECT) from the environment, and returns the raw Project
    object — use this when you need more than just the Feature Store (e.g.
    the training pipeline also needs project.get_model_registry()).
    """
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "HOPSWORKS_API_KEY is not set. Copy .env.example to .env, "
            "paste your key (Hopsworks -> your profile icon -> Settings -> "
            "API Keys -> New API Key), and make sure load_dotenv() has run "
            "before calling connect()."
        )
    project = hopsworks.login(
        api_key_value=api_key,
        project=os.environ.get("HOPSWORKS_PROJECT"),  # optional; None = your default project
    )
    logger.info(f"Connected to Hopsworks project: {project.name}")
    return project


def connect() -> FeatureStore:
    """Connects and returns just the Feature Store handle (most callers only need this)."""
    return connect_project().get_feature_store()


def get_features_feature_group(fs: FeatureStore, version: int = 1) -> FeatureGroup:
    return fs.get_or_create_feature_group(
        name="aqi_features",
        version=version,
        description=(
            "Hourly weather + pollutant model-input features, per city "
            "(hour/day/month time features, lag_1h/3h/24h, 24h rolling stats)."
        ),
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        online_enabled=False,  # batch-only is enough for an hourly/daily pipeline
    )


def get_targets_feature_group(fs: FeatureStore, version: int = 1) -> FeatureGroup:
    return fs.get_or_create_feature_group(
        name="aqi_targets",
        version=version,
        description=(
            "Average AQI for the next 1/2/3 CALENDAR DAYS (aqi_avg_next_1d/2d/3d), "
            "joined onto each hourly row by its own calendar date — only populated "
            "for historical rows where the future day's outcome is already known."
        ),
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        online_enabled=False,
    )


def prepare_for_hopsworks(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Hopsworks/Hive columns behave most predictably with timezone-naive
    datetimes. Our 'timestamp' column comes out of pandas as tz-aware
    (UTC) — this strips the tz info explicitly (values stay UTC, just the
    label is dropped) rather than letting Hopsworks silently reinterpret it.
    """
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    df[timestamp_col] = ts
    return df


def insert_features_and_targets(
    features_df: pd.DataFrame, targets_df: pd.DataFrame, version: int = 1
) -> None:
    """One-call convenience wrapper: connect, get both feature groups, and insert."""
    fs = connect()
    features_fg = get_features_feature_group(fs, version=version)
    targets_fg = get_targets_feature_group(fs, version=version)

    features_df = prepare_for_hopsworks(features_df)
    targets_df = prepare_for_hopsworks(targets_df)

    logger.info(f"Inserting {len(features_df)} rows into aqi_features...")
    features_fg.insert(features_df)

    logger.info(f"Inserting {len(targets_df)} rows into aqi_targets...")
    targets_fg.insert(targets_df)