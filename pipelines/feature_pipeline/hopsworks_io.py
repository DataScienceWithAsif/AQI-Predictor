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

Used by backfill.py (Day 4), the training pipeline (Day 5, which needs the
Model Registry too — hence connect_project() below), and later by the
hourly pipeline (Day 7).
"""

import os
import logging

import pandas as pd
import hopsworks
from hopsworks.project import Project
from hsfs.feature_store import FeatureStore
from hsfs.feature_group import FeatureGroup

logger = logging.getLogger("feature_pipeline.hopsworks_io")

# Single source of truth for which feature group version is "current" —
# read this from callers (train.py, verify_hopsworks_upload.py) instead of
# hardcoding a version number, so bumping this in one place can't silently
# desync from what other scripts try to read.
#
# NOTE: version history —
#   v1: original single-city (Islamabad only) backfill. Superseded, but
#       left in place server-side; safe to ignore or delete later.
#   v2: an abandoned attempt (DELTA format bump) that was never actually
#       backfilled — exists as an empty/unused feature group if you look
#       in the Hopsworks UI. Safe to delete whenever convenient.
#   v3: CURRENT — multi-city backfill (Islamabad, Karachi, Lahore, Multan,
#       Peshawar), 10,920 rows, verified working with real R²=0.70-0.81
#       results in train.py.
DEFAULT_FEATURE_GROUP_VERSION = 3


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


def connect_project() -> Project:
    """
    Connects to Hopsworks using HOPSWORKS_API_KEY (and optionally
    HOPSWORKS_PROJECT/HOST/PORT) from the environment, and returns the raw
    Project object — use this when you need more than just the Feature
    Store (e.g. the training pipeline also needs project.get_model_registry()).
    """
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "HOPSWORKS_API_KEY is not set. Copy .env.example to .env, "
            "paste your key (Hopsworks -> your profile icon -> Settings -> "
            "API Keys -> New API Key), and make sure load_dotenv() has run "
            "before calling connect_project()."
        )

    host = os.environ.get("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
    port = int(os.environ.get("HOPSWORKS_PORT", "443"))
    project_name = os.environ.get("HOPSWORKS_PROJECT")
    cert_folder = _resolve_cert_folder()

    logger.info(
        "Connecting to Hopsworks host=%s port=%s project=%s cert_folder=%s",
        host, port, project_name, cert_folder,
    )
    project = hopsworks.login(
        host=host,
        port=port,
        project=project_name,
        api_key_value=api_key,
        cert_folder=cert_folder,
        hostname_verification=False,
    )
    logger.info(f"Connected to Hopsworks project: {project.name}")
    return project


def connect() -> FeatureStore:
    """Connects and returns just the Feature Store handle (most callers only need this)."""
    return connect_project().get_feature_store()


def get_features_feature_group(fs: FeatureStore, version: int = DEFAULT_FEATURE_GROUP_VERSION) -> FeatureGroup:
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


def get_targets_feature_group(fs: FeatureStore, version: int = DEFAULT_FEATURE_GROUP_VERSION) -> FeatureGroup:
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


def get_raw_hourly_feature_group(fs: FeatureStore, version: int = 1) -> FeatureGroup:
    """
    A THIRD feature group, distinct from aqi_features/aqi_targets: stores
    raw live readings (AQICN + OpenWeather current APIs) exactly as
    fetch.py's hourly job collects them, with no lag/rolling engineering
    applied.

    Why this exists as its own feature group rather than feeding straight
    into aqi_features: fetch.py's hourly job only ever has ONE fresh hour
    of live data — there's no way to compute aqi_lag_24h or a 24h rolling
    mean from a single row. aqi_features/aqi_targets are instead refreshed
    daily by re-running backfill.py (which pulls enough trailing history
    from Open-Meteo to compute those properly). This raw feed exists for
    live monitoring / a "current AQI right now" display (Day 8), and as
    a foundation you could extend later into full incremental feature
    engineering if you want to go beyond this project's 10-day scope.
    """
    return fs.get_or_create_feature_group(
        name="aqi_raw_hourly",
        version=version,
        description=(
            "Raw live hourly readings (AQICN + OpenWeather current APIs), per city, "
            "exactly as collected — no lag/rolling features applied. Used for live "
            "monitoring, not directly for model training (see aqi_features/aqi_targets)."
        ),
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        online_enabled=True,  # a live dashboard wants fast point-lookups of the latest row
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


def insert_raw_hourly(raw_df: pd.DataFrame) -> None:
    """
    Connects and upserts fetch.py's live hourly rows into aqi_raw_hourly.
    Safe to call every hour indefinitely — insert()'s default upsert
    behavior means re-running for the same (city, timestamp) just
    overwrites in place rather than duplicating.
    """
    fs = connect()
    raw_fg = get_raw_hourly_feature_group(fs)
    raw_df = prepare_for_hopsworks(raw_df)
    logger.info(f"Inserting {len(raw_df)} row(s) into aqi_raw_hourly...")
    raw_fg.insert(raw_df)