"""
eda/eda.py

Day 10: Exploratory Data Analysis on the real, backfilled multi-city AQI
dataset (aqi_features from Hopsworks) — trend over time, city comparison,
diurnal/weekly seasonality, pollutant correlations, and AQI category
breakdown. Saves plots ready to drop straight into the final report.

Run:
    python eda/eda.py
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — this runs from the command line
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent / "feature_pipeline"))
import hopsworks_io  # noqa: E402

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eda")

OUTPUT_DIR = Path("eda_plots")
WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

AQI_CATEGORIES = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#dddd00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]
CORRELATION_COLUMNS = [
    "temp", "humidity", "pressure", "wind_speed",
    "pm25", "pm10", "co", "no2", "so2", "o3", "aqi",
]


def categorize(aqi: float) -> str:
    """Same US EPA breakpoints as app.py's categorize() — duplicated here
    (not imported) since eda.py is a standalone script that shouldn't
    depend on the Streamlit app module."""
    for lo, hi, label, _ in AQI_CATEGORIES:
        if lo <= aqi <= hi:
            return label
    return "Hazardous" if aqi > 500 else "Unknown"


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    fs = hopsworks_io.connect()
    fg = fs.get_feature_group(name="aqi_features", version=hopsworks_io.DEFAULT_FEATURE_GROUP_VERSION)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["city", "timestamp"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_trend_over_time(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for city, group in df.groupby("city"):
        daily = group.set_index("timestamp")["aqi"].resample("1D").mean()
        ax.plot(daily.index, daily.values, label=city, linewidth=1.5, alpha=0.85)
    ax.set_title("Daily Average AQI Over Time, by City")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "trend_over_time.png", dpi=150)
    plt.close(fig)


def plot_city_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    order = df.groupby("city")["aqi"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="city", y="aqi", order=order, ax=ax)
    ax.set_title("AQI Distribution by City")
    ax.set_xlabel("")
    ax.set_ylabel("AQI")
    fig.tight_layout()
    fig.savefig(output_dir / "city_comparison_boxplot.png", dpi=150)
    plt.close(fig)


def plot_diurnal_pattern(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    hourly = df.groupby(["city", "hour"])["aqi"].mean().reset_index()
    for city, group in hourly.groupby("city"):
        ax.plot(group["hour"], group["aqi"], marker="o", markersize=3, label=city)
    ax.set_title("Average AQI by Hour of Day (Diurnal Pattern)")
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Average AQI")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "diurnal_pattern.png", dpi=150)
    plt.close(fig)


def plot_weekday_pattern(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    weekly = df.groupby(["city", "day_of_week"])["aqi"].mean().reset_index()
    for city, group in weekly.groupby("city"):
        ax.plot(group["day_of_week"], group["aqi"], marker="o", label=city)
    ax.set_xticks(range(7))
    ax.set_xticklabels(WEEKDAY_NAMES)
    ax.set_title("Average AQI by Day of Week")
    ax.set_ylabel("Average AQI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "weekday_pattern.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    corr = df[CORRELATION_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Correlation Between Pollutants & Weather")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_aqi_category_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    df = df.copy()
    df["category"] = df["aqi"].apply(categorize)
    order = [c[2] for c in AQI_CATEGORIES]
    colors = {c[2]: c[3] for c in AQI_CATEGORIES}

    counts = df.groupby(["city", "category"]).size().reset_index(name="count")
    pivot = counts.pivot(index="city", columns="category", values="count").fillna(0)
    pivot = pivot.reindex(columns=[c for c in order if c in pivot.columns])

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=[colors[c] for c in pivot.columns])
    ax.set_title("Hours Spent in Each AQI Category, by City")
    ax.set_ylabel("Hour count")
    ax.set_xlabel("")
    ax.legend(title="AQI Category", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "aqi_category_distribution.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Summary stats (printed + saved as text for the report appendix)
# --------------------------------------------------------------------------

def build_summary_text(df: pd.DataFrame) -> str:
    lines = []
    lines.append("=== Dataset Summary ===")
    lines.append(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    lines.append(f"Cities: {sorted(df['city'].unique())}")
    lines.append(f"Total rows: {len(df)}")
    lines.append("")
    lines.append("AQI summary stats by city:")
    lines.append(df.groupby("city")["aqi"].describe()[["mean", "std", "min", "max"]].round(1).to_string())
    lines.append("")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    lines.append("Missing values per column:")
    lines.append(missing.to_string() if len(missing) > 0 else "None")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data from Hopsworks aqi_features...")
    df = load_data()
    logger.info(f"Loaded {len(df)} rows across {df['city'].nunique()} cities")

    summary = build_summary_text(df)
    print("\n" + summary)
    (OUTPUT_DIR / "summary_stats.txt").write_text(summary)

    logger.info("Generating plots...")
    plot_trend_over_time(df, OUTPUT_DIR)
    plot_city_comparison(df, OUTPUT_DIR)
    plot_diurnal_pattern(df, OUTPUT_DIR)
    plot_weekday_pattern(df, OUTPUT_DIR)
    plot_correlation_heatmap(df, OUTPUT_DIR)
    plot_aqi_category_distribution(df, OUTPUT_DIR)

    logger.info(f"All EDA plots + summary_stats.txt saved under {OUTPUT_DIR}/ — ready for your report.")


if __name__ == "__main__":
    main()