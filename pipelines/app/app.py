"""
app/app.py

Day 8 dashboard: loads the best registered model per horizon from the
Hopsworks Model Registry, the most recent feature row per city from
aqi_features, and the most recent live reading per city from
aqi_raw_hourly — then shows a 3-day AQI forecast with hazard alerts.

Run locally:
    streamlit run app/app.py

Deploy: Streamlit Community Cloud, main file path = app/app.py. Add
HOPSWORKS_API_KEY (and HOPSWORKS_PROJECT if not your default project) under
the app's Settings -> Secrets, in TOML format:
    HOPSWORKS_API_KEY = "..."
    HOPSWORKS_PROJECT = "AQI_Features"
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).resolve().parent.parent / "feature_pipeline"))
import hopsworks_io  # noqa: E402
from dotenv import load_dotenv
load_dotenv()

# Must stay in sync with training_pipeline/train.py's FEATURE_COLUMNS —
# duplicated here (rather than importing train.py) so the app doesn't need
# shap/matplotlib's heavier import chain just to read a constant.
NUMERIC_FEATURE_COLUMNS = [
    "temp", "humidity", "pressure", "wind_speed",
    "pm25", "pm10", "co", "no2", "so2", "o3", "aqi",
    "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "aqi_change_rate",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_24h",
    "aqi_roll_mean_24h", "aqi_roll_std_24h",
    "pm25_roll_mean_24h", "pm25_roll_std_24h",
]
FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + ["city"]
TARGET_COLUMNS = ["aqi_avg_next_1d", "aqi_avg_next_2d", "aqi_avg_next_3d"]
HORIZON_LABELS = {
    "aqi_avg_next_1d": "Tomorrow",
    "aqi_avg_next_2d": "In 2 days",
    "aqi_avg_next_3d": "In 3 days",
}
CITIES = ["Islamabad", "Karachi", "Lahore", "Multan", "Peshawar"]


# --------------------------------------------------------------------------
# AQI categorisation (US EPA breakpoints) — pure functions, unit tested
# --------------------------------------------------------------------------

AQI_CATEGORIES = [
    (0, 50, "Good", "#00e400", "\U0001F7E2"),
    (51, 100, "Moderate", "#dddd00", "\U0001F7E1"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00", "\U0001F7E0"),
    (151, 200, "Unhealthy", "#ff0000", "\U0001F534"),
    (201, 300, "Very Unhealthy", "#8f3f97", "\U0001F7E3"),
    (301, 500, "Hazardous", "#7e0023", "\U0001F7E4"),
]
HAZARD_THRESHOLD = 150  # "Unhealthy" and above triggers the alert banner


def categorize(aqi) -> tuple:
    """Returns (label, hex_color, emoji) for a given AQI value. Pure
    function — no Streamlit/Hopsworks dependency, safe to unit test."""
    if aqi is None or (isinstance(aqi, float) and np.isnan(aqi)):
        return ("Unknown", "#999999", "\u26AA")
    if aqi > 500:
        return ("Hazardous", "#7e0023", "\U0001F7E4")
    if aqi < 0:
        return ("Unknown", "#999999", "\u26AA")
    for lo, hi, label, color, emoji in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return (label, color, emoji)
    return ("Unknown", "#999999", "\u26AA")


def is_hazardous(*aqi_values) -> bool:
    """True if ANY of the given AQI values meets/exceeds HAZARD_THRESHOLD.
    Ignores None/NaN values rather than erroring on missing data."""
    valid = [v for v in aqi_values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return any(v >= HAZARD_THRESHOLD for v in valid)


# --------------------------------------------------------------------------
# Cached Hopsworks connections / loads
# --------------------------------------------------------------------------

def _load_secrets_into_env():
    """Streamlit Community Cloud provides secrets via st.secrets; local dev
    uses .env via python-dotenv (already loaded inside hopsworks_io). This
    bridges st.secrets into os.environ so hopsworks_io.connect_project()
    works unchanged in both environments."""
    for key in ["HOPSWORKS_API_KEY", "HOPSWORKS_PROJECT", "HOPSWORKS_HOST", "HOPSWORKS_PORT"]:
        try:
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = str(st.secrets[key])
        except Exception:
            pass  # st.secrets raises if no secrets.toml exists at all locally — fine, .env covers that case


@st.cache_resource(ttl=3600)
def get_feature_store():
    _load_secrets_into_env()
    return hopsworks_io.connect()


@st.cache_resource(ttl=3600)
def load_models() -> dict:
    """Loads the best (lowest test RMSE) registered model per horizon,
    across every version train.py has ever registered — so the app always
    reflects the current best model without needing a version bump here."""
    _load_secrets_into_env()
    project = hopsworks_io.connect_project()
    mr = project.get_model_registry()

    models = {}
    for target_col in TARGET_COLUMNS:
        model_name = f"aqi_{target_col}_model"
        hw_model = mr.get_best_model(name=model_name, metric="rmse", direction="min")
        if hw_model is None:
            st.error(f"No registered model found for '{model_name}'. Has training_pipeline/train.py run yet?")
            st.stop()
        local_dir = hw_model.download()
        pipeline = joblib.load(Path(local_dir) / "model.pkl")
        models[target_col] = {
            "pipeline": pipeline,
            "version": hw_model.version,
            "metrics": hw_model.training_metrics,
        }
    return models


@st.cache_data(ttl=600)
def load_latest_features() -> pd.DataFrame:
    """Latest row per city from aqi_features — the basis for predictions.
    NOTE: refreshed daily (training_pipeline.yml), and Open-Meteo's archive
    has ~6 days of latency, so 'latest' here is dated, not live — see
    load_latest_raw() for the truly live reading."""
    fs = get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=hopsworks_io.DEFAULT_FEATURE_GROUP_VERSION)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").groupby("city", as_index=False).tail(1)


@st.cache_data(ttl=600)
def load_latest_raw() -> pd.DataFrame:
    """Latest row per city from aqi_raw_hourly — genuinely live (refreshed
    hourly by feature_pipeline.yml), used only for the 'current AQI' display."""
    fs = get_feature_store()
    fg = fs.get_feature_group(name="aqi_raw_hourly", version=1)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").groupby("city", as_index=False).tail(1)


def compute_predictions(models: dict, feature_row: pd.DataFrame) -> dict:
    """Runs each horizon's model on a single-row feature DataFrame.
    Separated from the UI code so it's independently testable."""
    X_pred = feature_row[FEATURE_COLUMNS]
    return {
        target_col: float(model_info["pipeline"].predict(X_pred)[0])
        for target_col, model_info in models.items()
    }


# --------------------------------------------------------------------------
# UI (only runs when this file is executed by `streamlit run`)
# --------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="AQI Predictor", page_icon="\U0001F32B", layout="centered")
    st.title("\U0001F32B AQI Predictor")
    st.caption("3-day average AQI forecast for major Pakistani cities")

    city = st.selectbox("City", CITIES)

    with st.spinner("Loading models and latest data..."):
        models = load_models()
        features_df = load_latest_features()
        raw_df = load_latest_raw()

    city_features = features_df[features_df["city"] == city]
    city_raw = raw_df[raw_df["city"] == city]

    if city_features.empty:
        st.warning(
            f"No feature data found yet for {city}. The daily training pipeline "
            f"may not have run for this city yet — check back after the next scheduled run."
        )
        st.stop()

    feature_row = city_features.iloc[[0]]
    feature_timestamp = feature_row["timestamp"].iloc[0]
    predictions = compute_predictions(models, feature_row)

    # --- Current AQI (live) ---
    st.subheader("Current AQI")
    current_aqi = None
    if not city_raw.empty:
        current_aqi = float(city_raw["aqi"].iloc[0])
        current_ts = city_raw["timestamp"].iloc[0]
        if current_ts.tzinfo is None:
            current_ts = current_ts.tz_localize("UTC")
        label, color, emoji = categorize(current_aqi)
        age_minutes = int((datetime.now(timezone.utc) - current_ts).total_seconds() // 60)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("AQI right now", f"{current_aqi:.0f}", label)
        with col2:
            st.markdown(
                f"{emoji} **{label}** — as of {age_minutes} minute(s) ago "
                f"({current_ts.strftime('%Y-%m-%d %H:%M UTC')})"
            )
    else:
        st.info("No live reading available yet for this city — the hourly pipeline hasn't collected one.")

    # --- Hazard alert ---
    if is_hazardous(current_aqi, *predictions.values()):
        worst = max([v for v in [current_aqi, *predictions.values()] if v is not None])
        worst_label = categorize(worst)[0]
        st.error(
            f"\u26A0\uFE0F **Hazardous air quality alert** — AQI is expected to reach "
            f"**{worst:.0f}** ({worst_label}) within the next 3 days. "
            f"Consider limiting outdoor activity."
        )

    # --- 3-day forecast ---
    st.subheader("3-Day Forecast")
    st.caption(
        f"Based on the most recent available feature data for {city}, "
        f"as of **{feature_timestamp.strftime('%Y-%m-%d')}** "
        f"(training data has ~6 days of natural latency — see note below)."
    )

    cols = st.columns(3)
    for col, target_col in zip(cols, TARGET_COLUMNS):
        pred = predictions[target_col]
        label, color, emoji = categorize(pred)
        with col:
            st.metric(HORIZON_LABELS[target_col], f"{pred:.0f}")
            st.markdown(f"{emoji} {label}")

    # --- Trend chart ---
    chart_x = ["Last known"] + [HORIZON_LABELS[t] for t in TARGET_COLUMNS]
    chart_y = [float(feature_row["aqi"].iloc[0])] + [predictions[t] for t in TARGET_COLUMNS]
    chart_colors = [categorize(y)[1] for y in chart_y]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_x, y=chart_y, mode="lines+markers",
        line=dict(color="#4a90d9", width=2),
        marker=dict(size=14, color=chart_colors, line=dict(width=1, color="black")),
    ))
    for lo, hi, label, color, _ in AQI_CATEGORIES:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=color, opacity=0.08, line_width=0)
    fig.update_layout(yaxis_title="AQI", height=350, margin=dict(t=20, b=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- Transparency footer ---
    with st.expander("About these predictions"):
        model_lines = "\n".join(
            f"- **{HORIZON_LABELS[t]}**: {models[t]['pipeline'].steps[-1][0]} "
            f"(registry v{models[t]['version']}, test RMSE={models[t]['metrics'].get('rmse', float('nan')):.2f})"
            for t in TARGET_COLUMNS
        )
        st.markdown(
            f"""
Models used for this forecast:

{model_lines}

- Predictions are the **average AQI for the calendar day**, not an hour-by-hour value.
- Feature data refreshes daily; the live "current AQI" above refreshes hourly from a separate feed.
- AQI categories follow the US EPA 0–500 scale.
            """
        )


if __name__ == "__main__":
    main()