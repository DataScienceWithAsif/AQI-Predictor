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


def badge_html(label: str, color: str, text_color: str = "#ffffff") -> str:
    """Renders a solid-color pill badge as inline HTML — used everywhere a
    plain-text category label was previously the only visual cue. Pure
    function (just string formatting), safe to unit test independent of
    whether it actually renders correctly in a browser."""
    return (
        f'<span style="background-color:{color}; color:{text_color}; '
        f'padding:3px 12px; border-radius:12px; font-weight:600; '
        f'font-size:0.85em; white-space:nowrap;">{label}</span>'
    )


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
            "description": hw_model.description or "",
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
# Design system — dark emerald/teal glassmorphism for structural chrome.
# NOTE: AQI category hex codes (from categorize()/AQI_CATEGORIES) are never
# overridden by this palette — they're injected per-value at render time
# below, so the data itself always keeps its true EPA color meaning.
# --------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* ---------- Base app ---------- */
.stApp {
    background: radial-gradient(circle at 15% -10%, #123128 0%, #0a1614 55%, #060d0c 100%);
    color: #e6f4ef;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2b24 0%, #081512 100%);
    border-right: 1px solid rgba(52, 211, 153, 0.15);
}
section[data-testid="stSidebar"] * { 
    color: #d7f3e8 !important; 
}

h1, h2, h3, h4 { color: #6ee7b7 !important; font-weight: 700; letter-spacing: -0.02em; }
hr { border-color: rgba(52, 211, 153, 0.2) !important; }

/* Shrink header */
[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0) !important;
    height: 2.2rem;
    min-height: 2.2rem;
}
.block-container { padding-top: 0.5rem; padding-bottom: 3rem; }

/* ---------- Sidebar Selectbox Overrides ---------- */
/* 1. Box container */
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: rgba(10, 30, 24, 0.95) !important;
    border: 1px solid rgba(110, 231, 183, 0.4) !important;
    border-radius: 8px !important;
}

/* 2. Selected text styling (overrides -webkit-text-fill-color) */
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"] *,
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"] span,
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"] div {
    color: #7ed130 !important;
    -webkit-text-fill-color: #7ed130 !important;
    font-weight: 700 !important;
}

/* 3. Dropdown arrow icon */
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div[data-baseweb="select"] svg {
    fill: #7ed130 !important;
}

/* 4. Preserve sidebar 'City' label color */
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] label p {
    color: #8fd9c4 !important;
    -webkit-text-fill-color: #8fd9c4 !important;
    font-weight: 600 !important;
}

/* ---------- Expanders & Alerts ---------- */
[data-testid="stExpander"] {
    background: rgba(16, 60, 48, 0.22);
    border: 1px solid rgba(110, 231, 183, 0.15);
    border-radius: 14px;
}
[data-testid="stAlert"] {
    background: rgba(16, 60, 48, 0.25);
    border: 1px solid rgba(110, 231, 183, 0.2);
    border-radius: 14px;
    color: #e6f4ef;
}

/* ---------- Glass cards ---------- */
.glass-card {
    background: rgba(16, 60, 48, 0.35);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(110, 231, 183, 0.18);
    border-radius: 20px;
    padding: 1.5rem 1.75rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    margin-bottom: 0.5rem;
    height: 100%;
}
.hero-card { text-align: center; }
.hero-label {
    font-size: 0.8rem; letter-spacing: 0.12em; color: #8fd9c4;
    font-weight: 600; margin-bottom: 0.4rem; text-transform: uppercase;
}
.hero-number { font-size: 5.2rem; font-weight: 800; line-height: 1; margin-bottom: 0.7rem; }
.hero-side { display: flex; flex-direction: column; justify-content: center; }
.hero-side-title { font-size: 1.4rem; font-weight: 700; color: #e6f4ef; margin-bottom: 0.5rem; }
.hero-side-sub { font-size: 0.9rem; color: #9fd9c6; line-height: 1.6; }

.section-title { font-size: 1.3rem; font-weight: 700; color: #6ee7b7; margin: 1.8rem 0 0.7rem 0; }

.forecast-card { text-align: center; }
.forecast-label { font-size: 0.95rem; font-weight: 600; color: #d7f3e8; }
.forecast-date { font-size: 0.8rem; color: #8fd9c4; margin-bottom: 0.7rem; }
.forecast-number { font-size: 2.6rem; font-weight: 800; line-height: 1; margin-bottom: 0.7rem; }

.alert-card {
    background: rgba(255, 30, 30, 0.12);
    border: 1px solid rgba(255, 90, 90, 0.4);
    border-radius: 16px;
    padding: 1rem 1.25rem;
    margin: 1rem 0 1.5rem 0;
    color: #ffe4e4;
    font-size: 0.98rem;
}
</style>
"""


# --------------------------------------------------------------------------
# UI (only runs when this file is executed by `streamlit run`)
# --------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="AQI Predictor", page_icon="\U0001F32B", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # --- Sidebar: city selector + About ---
    with st.sidebar:
        st.markdown("### \U0001F32B AQI Predictor")
        city = st.selectbox("City", CITIES)
        st.divider()
        st.markdown("#### About")
        st.caption(
            "3-day average AQI forecasts for major Pakistani cities, powered by a "
            "live Hopsworks feature store and daily-retrained ML models "
            "(Ridge / Random Forest / XGBoost, whichever performs best per horizon)."
        )
        st.caption(f"Dashboard loaded: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

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

    st.markdown(f"# \U0001F32B {city}")

    # --- Hero: current AQI ---
    current_aqi = None
    if not city_raw.empty:
        current_aqi = float(city_raw["aqi"].iloc[0])
        current_ts = city_raw["timestamp"].iloc[0]
        if current_ts.tzinfo is None:
            current_ts = current_ts.tz_localize("UTC")
        label, color, emoji = categorize(current_aqi)
        age_minutes = int((datetime.now(timezone.utc) - current_ts).total_seconds() // 60)
        freshness_note = "just now" if age_minutes < 1 else f"{age_minutes} minute(s) ago"

        hero_col1, hero_col2 = st.columns([1.2, 2])
        with hero_col1:
            st.markdown(
                f"""
                <div class="glass-card hero-card">
                    <div class="hero-label">Current AQI</div>
                    <div class="hero-number" style="color:{color};">{current_aqi:.0f}</div>
                    {badge_html(label, color)}
                </div>
                """,
                unsafe_allow_html=True,
            )
        with hero_col2:
            st.markdown(
                f"""
                <div class="glass-card hero-side">
                    <div class="hero-side-title">{emoji} {label}</div>
                    <div class="hero-side-sub">
                        As of {freshness_note}<br>
                        {current_ts.strftime('%Y-%m-%d %H:%M UTC')}<br><br>
                        Live reading, refreshed hourly from a separate feed —
                        see the forecast note below for how this differs from
                        the data behind the 3-day predictions.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No live reading available yet for this city — the hourly pipeline hasn't collected one.")

    # --- Hazard alert ---
    if is_hazardous(current_aqi, *predictions.values()):
        worst = max([v for v in [current_aqi, *predictions.values()] if v is not None])
        worst_label = categorize(worst)[0]
        st.markdown(
            f"""
            <div class="alert-card">
                \u26A0\uFE0F <b>Hazardous air quality alert</b> — AQI is expected to reach
                <b>{worst:.0f}</b> ({worst_label}) within the next 3 days.
                Consider limiting outdoor activity.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- 3-day forecast grid ---
    st.markdown('<div class="section-title">3-Day Forecast</div>', unsafe_allow_html=True)
    st.caption(
        f"Based on the most recent available feature data for {city}, "
        f"as of **{feature_timestamp.strftime('%Y-%m-%d')}** "
        f"(training data has ~6 days of natural latency — see note below)."
    )

    cols = st.columns(3)
    for col, target_col in zip(cols, TARGET_COLUMNS):
        pred = predictions[target_col]
        label, color, emoji = categorize(pred)
        forecast_date = feature_timestamp + pd.Timedelta(days=int(target_col[-2]))
        with col:
            st.markdown(
                f"""
                <div class="glass-card forecast-card">
                    <div class="forecast-label">{HORIZON_LABELS[target_col]}</div>
                    <div class="forecast-date">{forecast_date.strftime('%a, %b %d')}</div>
                    <div class="forecast-number" style="color:{color};">{pred:.0f}</div>
                    {badge_html(label, color)}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --- Trend chart ---
    st.markdown('<div class="section-title">Trend</div>', unsafe_allow_html=True)
    chart_x = ["Last known"] + [HORIZON_LABELS[t] for t in TARGET_COLUMNS]
    chart_y = [float(feature_row["aqi"].iloc[0])] + [predictions[t] for t in TARGET_COLUMNS]
    chart_colors = [categorize(y)[1] for y in chart_y]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_x, y=chart_y, mode="lines+markers",
        line=dict(color="#34d399", width=3, shape="spline", smoothing=0.65),
        marker=dict(size=16, color=chart_colors, line=dict(width=2, color="#0a1614")),
    ))
    for lo, hi, label, color, _ in AQI_CATEGORIES:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=color, opacity=0.06, line_width=0)
    fig.update_layout(
        yaxis_title="AQI",
        height=380,
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d9f2ea"),
        xaxis=dict(showgrid=False, zeroline=False, color="#9fd9c6"),
        yaxis=dict(showgrid=False, zeroline=False, color="#9fd9c6"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- AQI scale legend ---
    with st.expander("AQI scale reference"):
        legend_html = " &nbsp; ".join(
            badge_html(f"{lo}\u2013{hi} {label}", color) for lo, hi, label, color, _ in AQI_CATEGORIES
        )
        st.markdown(legend_html, unsafe_allow_html=True)

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

    # --- What drives this prediction (SHAP, computed at training time) ---
    with st.expander("What drives this prediction?"):
        any_shown = False
        for t in TARGET_COLUMNS:
            description = models[t].get("description", "")
            if "Top SHAP features" in description:
                shap_part = description.split("Top SHAP features:", 1)[1].strip().rstrip(".")
                st.markdown(f"**{HORIZON_LABELS[t]}**: {shap_part}")
                any_shown = True
        if not any_shown:
            st.caption(
                "SHAP feature-importance data isn't available for the currently "
                "registered models yet — re-run training_pipeline/train.py to generate it."
            )


if __name__ == "__main__":
    main()