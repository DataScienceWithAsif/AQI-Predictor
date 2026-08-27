"""
training_pipeline/train.py

Day 5+6: trains and evaluates AQI forecasting models against the Hopsworks
feature store, explains the winner with SHAP, and registers the best model
per horizon (1/2/3-day average AQI) in the Hopsworks Model Registry.

    1. Fetches (features, targets) from aqi_features / aqi_targets
    2. Splits by CALENDAR DATE (not row) into train/test
    3. Trains Ridge + Random Forest + XGBoost per horizon, evaluates RMSE/MAE/R^2
    4. Computes SHAP feature importance for the winning model per horizon
    5. Registers the better model per horizon in the Model Registry

Run:
    python training_pipeline/train.py
"""

import os
import sys
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — this script runs from the command line, not a notebook
import matplotlib.pyplot as plt
import shap
from dotenv import load_dotenv
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# feature_pipeline is a sibling directory of training_pipeline — adjust this
# if your repo nests them differently (e.g. pipelines/feature_pipeline vs
# pipelines/training_pipeline still resolves correctly since both hang off
# the same parent; a different layout may need a different relative path).
sys.path.append(str(Path(__file__).resolve().parent.parent / "feature_pipeline"))
import hopsworks_io  # noqa: E402

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("training_pipeline.train")

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
CATEGORICAL_FEATURE_COLUMNS = ["city"]
# 'city' matters now that backfill.py can pull multiple cities: without it,
# the model has no way to tell Lahore's baseline AQI from Islamabad's, and
# mixing cities together would look like unexplained noise rather than a
# real, learnable signal.
FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
TARGET_COLUMNS = ["aqi_avg_next_1d", "aqi_avg_next_2d", "aqi_avg_next_3d"]

TEST_SIZE_DAYS = 14  # last 14 CALENDAR DAYS held out — never split mid-day
MODEL_DIR = Path("trained_models")
SHAP_DIR = Path("day6_shap_plots")
SHAP_BACKGROUND_SIZE = 50
SHAP_EXPLAIN_SIZE = 50


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_merged_data(fs) -> pd.DataFrame:
    """Reads aqi_features/aqi_targets from Hopsworks and joins them on
    (city, timestamp) — the same read pattern verify_hopsworks_upload.py
    already proved works against your real project."""
    features_fg = fs.get_feature_group(name="aqi_features", version=hopsworks_io.DEFAULT_FEATURE_GROUP_VERSION)
    targets_fg = fs.get_feature_group(name="aqi_targets", version=hopsworks_io.DEFAULT_FEATURE_GROUP_VERSION)

    logger.info("Reading aqi_features...")
    features_df = features_fg.read()
    logger.info("Reading aqi_targets...")
    targets_df = targets_fg.read()

    merged = features_df.merge(targets_df, on=["city", "timestamp"], how="inner")
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Merged: {len(merged)} rows")
    return merged


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------

def date_based_split(df: pd.DataFrame, test_size_days: int = TEST_SIZE_DAYS, timestamp_col: str = "timestamp"):
    """
    Splits by CALENDAR DATE, never by row. This matters specifically here
    because every hour within the same day shares the identical daily-
    average target (see final_features.py) — a random row-level split
    would put different hours of the SAME day's target into both train
    and test, which is leakage, not a real generalisation test.
    """
    last_date = df[timestamp_col].max().normalize()
    # -1 so the test window is exactly `test_size_days` days inclusive of
    # the last day (e.g. test_size_days=14 -> the last 14 calendar days,
    # not 15).
    cutoff_date = last_date - pd.Timedelta(days=test_size_days - 1)

    train_df = df[df[timestamp_col] < cutoff_date].copy()
    test_df = df[df[timestamp_col] >= cutoff_date].copy()

    train_days = set(train_df[timestamp_col].dt.normalize())
    test_days = set(test_df[timestamp_col].dt.normalize())
    overlap = train_days & test_days
    if overlap:
        raise RuntimeError(f"Date leakage detected — {len(overlap)} day(s) appear in both train and test!")

    logger.info(
        f"Split: train={len(train_df)} rows/{len(train_days)} days, "
        f"test={len(test_df)} rows/{len(test_days)} days, cutoff={cutoff_date.date()}"
    )
    return train_df, test_df


# --------------------------------------------------------------------------
# Train + evaluate
# --------------------------------------------------------------------------

def compute_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def naive_persistence_baseline(test_df: pd.DataFrame, target_col: str) -> dict:
    """
    The evaluation your brief asks for isn't complete without this: a model
    is only actually useful if it beats a trivial 'no ML at all' guess.
    Here, the naive guess is 'tomorrow's average AQI will be about the same
    as the last 24h rolling average right now' (aqi_roll_mean_24h) — no
    fitting, no features besides what's already directly observed.

    If Ridge/RF can't beat this, they aren't adding real value yet.
    """
    clean = test_df.dropna(subset=["aqi_roll_mean_24h", target_col])
    if len(clean) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    return compute_metrics(clean[target_col], clean["aqi_roll_mean_24h"])


def _build_preprocessor() -> ColumnTransformer:
    """
    Scales numeric features and one-hot encodes 'city' — shared by both
    models so a multi-city dataset is handled identically for each.
    handle_unknown='ignore' means a city the model never saw in training
    (e.g. if you add a 6th city later without retraining) won't crash
    prediction, it'll just get an all-zero city encoding.
    """
    return ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURE_COLUMNS),
        ("city", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURE_COLUMNS),
    ])


def train_and_evaluate_horizon(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> dict:
    """Trains Ridge and Random Forest for one horizon, evaluates both on
    the held-out test days, and returns everything needed to pick and
    register the winner."""
    cols_needed = FEATURE_COLUMNS + [target_col]
    train_clean = train_df.dropna(subset=cols_needed)
    test_clean = test_df.dropna(subset=cols_needed)

    X_train, y_train = train_clean[FEATURE_COLUMNS], train_clean[target_col]
    X_test, y_test = test_clean[FEATURE_COLUMNS], test_clean[target_col]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            f"No usable rows for {target_col} after dropping NaNs "
            f"(train={len(X_train)}, test={len(X_test)}) — check your date range/backfill size."
        )

    n_cities = X_train["city"].nunique()
    logger.info(f"[{target_col}] Training on {n_cities} cit{'y' if n_cities == 1 else 'ies'}: "
                f"{sorted(X_train['city'].unique())}")

    ridge = Pipeline([
        ("preprocess", _build_preprocessor()),
        ("ridge", RidgeCV(alphas=np.logspace(-2, 3, 20))),
    ])
    ridge.fit(X_train, y_train)
    ridge_metrics = compute_metrics(y_test, ridge.predict(X_test))

    rf = Pipeline([
        ("preprocess", _build_preprocessor()),
        ("rf", RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    rf.fit(X_train, y_train)
    rf_metrics = compute_metrics(y_test, rf.predict(X_test))

    xgb = Pipeline([
        ("preprocess", _build_preprocessor()),
        ("xgb", XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    xgb.fit(X_train, y_train)
    xgb_metrics = compute_metrics(y_test, xgb.predict(X_test))

    candidates = {
        "ridge": (ridge, ridge_metrics),
        "random_forest": (rf, rf_metrics),
        "xgboost": (xgb, xgb_metrics),
    }
    best_name = min(candidates, key=lambda name: candidates[name][1]["rmse"])
    best_model, best_metrics = candidates[best_name]

    baseline_metrics = naive_persistence_baseline(test_clean, target_col)
    beats_baseline = best_metrics["rmse"] < baseline_metrics["rmse"]

    logger.info(
        f"[{target_col}] Ridge:        RMSE={ridge_metrics['rmse']:.2f} "
        f"MAE={ridge_metrics['mae']:.2f} R2={ridge_metrics['r2']:.3f}"
    )
    logger.info(
        f"[{target_col}] RandomForest: RMSE={rf_metrics['rmse']:.2f} "
        f"MAE={rf_metrics['mae']:.2f} R2={rf_metrics['r2']:.3f}"
    )
    logger.info(
        f"[{target_col}] XGBoost:      RMSE={xgb_metrics['rmse']:.2f} "
        f"MAE={xgb_metrics['mae']:.2f} R2={xgb_metrics['r2']:.3f}"
    )
    logger.info(
        f"[{target_col}] Naive baseline (persist last 24h avg): "
        f"RMSE={baseline_metrics['rmse']:.2f} MAE={baseline_metrics['mae']:.2f} R2={baseline_metrics['r2']:.3f}"
    )
    logger.info(
        f"[{target_col}] Winner: {best_name} "
        f"({'BEATS' if beats_baseline else 'DOES NOT BEAT'} the naive baseline)"
    )

    return {
        "target_col": target_col,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "ridge_metrics": ridge_metrics,
        "random_forest_metrics": rf_metrics,
        "xgboost_metrics": xgb_metrics,
        "baseline_metrics": baseline_metrics,
        "beats_baseline": beats_baseline,
        "best_model_name": best_name,
        "best_model": best_model,
        "best_metrics": best_metrics,
        "X_train_sample": X_train.head(3),
        "X_train_clean": X_train,
        "X_test_clean": X_test,
    }


# --------------------------------------------------------------------------
# SHAP explainability (Day 6)
# --------------------------------------------------------------------------

def compute_shap_importance(
    pipeline: Pipeline,
    X_train_clean: pd.DataFrame,
    X_test_clean: pd.DataFrame,
    target_col: str,
    output_dir: Path = SHAP_DIR,
) -> list:
    """
    Explains the winning model's feature importance with SHAP.

    Explains the TRANSFORMED feature matrix (after the pipeline's
    StandardScaler + OneHotEncoder step), not the raw input — SHAP's
    default masker can't perturb a mixed numeric/string DataFrame (the raw
    'city' column would break it). This has a nice side effect: each city
    gets its own individual importance (e.g. 'city_Lahore') rather than one
    combined 'city' score, which is more informative anyway.

    Works uniformly whether the winning model is Ridge, RandomForest, or
    XGBoost, since it explains via the fitted estimator's .predict on the
    already-transformed matrix — no model-specific code path needed.

    Saves a horizontal bar chart of the top 15 features by mean |SHAP
    value| and returns the full ranked (feature_name, importance) list.
    """
    preprocessor = pipeline.named_steps["preprocess"]
    final_estimator = pipeline[-1]  # whichever step comes after "preprocess"

    feature_names = preprocessor.get_feature_names_out()

    background = X_train_clean.sample(
        min(SHAP_BACKGROUND_SIZE, len(X_train_clean)), random_state=42
    )
    explain_sample = X_test_clean.sample(
        min(SHAP_EXPLAIN_SIZE, len(X_test_clean)), random_state=42
    )

    background_t = pd.DataFrame(preprocessor.transform(background), columns=feature_names)
    explain_t = pd.DataFrame(preprocessor.transform(explain_sample), columns=feature_names)

    with warnings.catch_warnings():
        # Cosmetic-only: the underlying estimator was fit on a plain numpy
        # array (from ColumnTransformer.transform), so sklearn warns when
        # it's later called with a named DataFrame during SHAP explanation.
        warnings.simplefilter("ignore")
        explainer = shap.Explainer(final_estimator.predict, background_t)
        shap_values = explainer(explain_t)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    ranked = sorted(zip(feature_names, mean_abs_shap), key=lambda t: -t[1])

    output_dir.mkdir(parents=True, exist_ok=True)
    top_n = 15
    top_features = ranked[:top_n]

    fig, ax = plt.subplots(figsize=(8, 6))
    names = [f[0] for f in reversed(top_features)]
    values = [f[1] for f in reversed(top_features)]
    ax.barh(names, values, color="#2b6cb0")
    ax.set_xlabel("Mean |SHAP value| (impact on predicted AQI)")
    ax.set_title(f"Top {top_n} feature importances — {target_col}")
    fig.tight_layout()
    plot_path = output_dir / f"shap_{target_col}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    logger.info(f"[{target_col}] SHAP plot saved to {plot_path}")
    logger.info(f"[{target_col}] Top 5 features by SHAP importance: {[f[0] for f in ranked[:5]]}")

    return ranked


# --------------------------------------------------------------------------
# Model Registry
# --------------------------------------------------------------------------

def register_model(project, result: dict) -> None:
    """Saves the winning model locally, then uploads it to the Hopsworks
    Model Registry with its evaluation metrics attached."""
    from hsml.model_schema import ModelSchema
    from hsml.schema import Schema

    target_col = result["target_col"]
    model_name = f"aqi_{target_col}_model"

    model_dir = MODEL_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["best_model"], model_dir / "model.pkl")

    mr = project.get_model_registry()
    schema = ModelSchema(
        input_schema=Schema(result["X_train_sample"]),
        output_schema=Schema(pd.Series([0.0], name=target_col)),
    )

    top_features = result.get("shap_top_features")
    shap_note = (
        f" Top SHAP features: {', '.join(f[0] for f in top_features[:5])}."
        if top_features else ""
    )

    hw_model = mr.sklearn.create_model(
        name=model_name,
        metrics=result["best_metrics"],
        description=(
            f"Predicts {target_col} (daily-average AQI) from hourly features. "
            f"Best of Ridge/RandomForest/XGBoost by test RMSE: {result['best_model_name']}."
            f"{shap_note}"
        ),
        input_example=result["X_train_sample"],
        model_schema=schema,
    )
    hw_model.save(str(model_dir))
    logger.info(f"Registered '{model_name}' (version {hw_model.version}) in the Model Registry.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    project = hopsworks_io.connect_project()
    fs = project.get_feature_store()

    merged = load_merged_data(fs)
    train_df, test_df = date_based_split(merged)

    all_results = [
        train_and_evaluate_horizon(train_df, test_df, target_col)
        for target_col in TARGET_COLUMNS
    ]

    logger.info("Computing SHAP feature importance for each horizon's winning model...")
    for r in all_results:
        r["shap_top_features"] = compute_shap_importance(
            r["best_model"], r["X_train_clean"], r["X_test_clean"], r["target_col"]
        )

    summary_df = pd.DataFrame([
        {
            "target": r["target_col"],
            "n_train": r["n_train"], "n_test": r["n_test"],
            "ridge_rmse": r["ridge_metrics"]["rmse"],
            "ridge_mae": r["ridge_metrics"]["mae"],
            "ridge_r2": r["ridge_metrics"]["r2"],
            "rf_rmse": r["random_forest_metrics"]["rmse"],
            "rf_mae": r["random_forest_metrics"]["mae"],
            "rf_r2": r["random_forest_metrics"]["r2"],
            "xgb_rmse": r["xgboost_metrics"]["rmse"],
            "xgb_mae": r["xgboost_metrics"]["mae"],
            "xgb_r2": r["xgboost_metrics"]["r2"],
            "baseline_rmse": r["baseline_metrics"]["rmse"],
            "baseline_mae": r["baseline_metrics"]["mae"],
            "winner": r["best_model_name"],
            "beats_baseline": r["beats_baseline"],
            "top_shap_feature": r["shap_top_features"][0][0],
        }
        for r in all_results
    ])
    print("\n=== Day 5+6 Model Comparison (Ridge vs RandomForest vs XGBoost) ===")
    print(summary_df.to_string(index=False))
    summary_df.to_csv("day6_model_comparison.csv", index=False)
    logger.info("Saved day6_model_comparison.csv (use this table directly in your report)")
    logger.info(f"SHAP bar charts saved under {SHAP_DIR}/ — one per horizon, ready to drop into your report")

    for r in all_results:
        register_model(project, r)

    logger.info(
        "Done. Check Hopsworks UI -> Model Registry for "
        "aqi_aqi_avg_next_1d_model / _2d_model / _3d_model."
    )


if __name__ == "__main__":
    main()