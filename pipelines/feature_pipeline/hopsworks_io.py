"""
feature_pipeline/hopsworks_io.py

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


def _resolve_cert_folder() -> str:
    """
    Hopsworks' Python client defaults to /tmp for certificates, which is invalid
    on Windows. The official login contract accepts a custom cert_folder, so we
    override it explicitly to a Windows-safe path before the client tries to
    materialize TLS certs.
    """
    cert_folder = os.environ.get("HOPSWORKS_CERT_FOLDER")
    if cert_folder and not cert_folder.startswith("/"):
        os.makedirs(cert_folder, exist_ok=True)
        return cert_folder

    if os.name == "nt":
        cert_folder = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp", "hopsworks-certs")
        os.makedirs(cert_folder, exist_ok=True)
        os.environ["HOPSWORKS_CERT_FOLDER"] = cert_folder
        return cert_folder

    return "/tmp/hopsworks-certs"


def connect() -> FeatureStore:
    """
    Connects to Hopsworks using HOPSWORKS_API_KEY (and optionally
    HOPSWORKS_PROJECT) from the environment, and returns the project's
    Feature Store handle.
    """
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "HOPSWORKS_API_KEY is not set. Copy .env.example to .env, "
            "paste your key (Hopsworks -> your profile icon -> Settings -> "
            "API Keys -> New API Key), and make sure load_dotenv() has run "
            "before calling connect()."
        )

    host = os.environ.get("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
    port = int(os.environ.get("HOPSWORKS_PORT", "443"))
    project_name = os.environ.get("HOPSWORKS_PROJECT")
    cert_folder = _resolve_cert_folder()

    logger.info("Connecting to Hopsworks host=%s port=%s project=%s cert_folder=%s", host, port, project_name, cert_folder)
    project = hopsworks.login(
        host=host,
        port=port,
        project=project_name,
        api_key_value=api_key,
        cert_folder=cert_folder,
        hostname_verification=False,
    )
    logger.info(f"Connected to Hopsworks project: {project.name}")
    return project.get_feature_store()


def get_features_feature_group(fs: FeatureStore, version: int = 2) -> FeatureGroup:
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
        time_travel_format="DELTA",  # supported by the server and now installed in this environment
    )


def get_targets_feature_group(fs: FeatureStore, version: int = 2) -> FeatureGroup:
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
        time_travel_format="DELTA",
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