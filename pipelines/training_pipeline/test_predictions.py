"""
training_pipeline/test_predictions.py

Post-training sanity checks: verifies the registered AQI models are
predicting as intended, not just "beating a bad baseline" on paper.

Loads the same (features, targets) split train.py used, re-loads each
horizon's trained model from trained_models/, and runs a battery of
targeted checks:

    1. Predictions are finite and within a physically plausible AQI range
    2. Each model still beats the naive persistence baseline (regression
       guard — catches a silent regression if someone retrains later)
    3. R^2 isn't catastrophically negative (soft check — flags but doesn't
       fail the whole run unless --strict)
    4. Feature/target columns never overlap (leakage guard)
    5. pm25 sensitivity: raising pm25, holding everything else fixed,
       should raise the predicted AQI on average — a "does the model point
       the right direction" check, not an accuracy check
    6. Predictions are deterministic given identical input

Run:
    python training_pipeline/test_predictions.py
    python training_pipeline/test_predictions.py --strict --r2-floor -0.2

Exits 0 if everything passes (or only WARNs), 1 if anything FAILs — safe
to wire into CI right after train.py in the same workflow step.
"""

import sys
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# train.py lives in this same directory (training_pipeline/) — reuse its
# constants and helpers instead of redefining them, so this script can't
# silently drift out of sync with how the models were actually trained.
sys.path.append(str(Path(__file__).resolve().parent))
from train import (  # noqa: E402
    FEATURE_COLUMNS, TARGET_COLUMNS, MODEL_DIR,
    load_merged_data, date_based_split, compute_metrics,
    naive_persistence_baseline,
)

# Same sibling-directory assumption train.py makes for feature_pipeline.
sys.path.append(str(Path(__file__).resolve().parent.parent / "feature_pipeline"))
import hopsworks_io  # noqa: E402


PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


class CheckResult:
    def __init__(self, name, status, detail=""):
        self.name = name
        self.status = status
        self.detail = detail

    def __str__(self):
        return f"  [{self.status}] {self.name}" + (f" — {self.detail}" if self.detail else "")


# --------------------------------------------------------------------------
# Individual checks — each takes what it needs and returns one CheckResult
# --------------------------------------------------------------------------

def check_finite_and_bounded(preds: np.ndarray, lower: float = 0, upper: float = 500) -> CheckResult:
    """AQI has a defined index range; a prediction outside it (or NaN/inf)
    means something upstream broke, not that the model is just 'a bit off'."""
    finite = bool(np.isfinite(preds).all())
    if not finite:
        n_bad = int((~np.isfinite(preds)).sum())
        return CheckResult("predictions finite", FAIL, f"{n_bad} non-finite prediction(s)")

    bounded = bool(((preds >= lower) & (preds <= upper)).all())
    if not bounded:
        n_bad = int(((preds < lower) | (preds > upper)).sum())
        return CheckResult(
            "predictions in plausible AQI range", FAIL,
            f"{n_bad}/{len(preds)} outside [{lower}, {upper}] "
            f"(min={preds.min():.1f}, max={preds.max():.1f})",
        )
    return CheckResult("predictions finite & in plausible range", PASS,
                        f"range [{preds.min():.1f}, {preds.max():.1f}]")


def check_beats_baseline(model_metrics: dict, baseline_metrics: dict) -> CheckResult:
    beats = model_metrics["rmse"] < baseline_metrics["rmse"]
    detail = f"model RMSE={model_metrics['rmse']:.2f} vs baseline RMSE={baseline_metrics['rmse']:.2f}"
    return CheckResult("beats naive persistence baseline", PASS if beats else FAIL, detail)


def check_r2_floor(model_metrics: dict, floor: float, strict: bool) -> CheckResult:
    """R^2 below 0 means 'worse than predicting the mean.' This is a soft
    check by default (WARN) since a low-but-positive-progress R^2 can still
    be an improvement over a previous run — pass --strict once you have a
    real quality bar to enforce."""
    r2 = model_metrics["r2"]
    ok = r2 >= floor
    detail = f"R^2={r2:.3f} (floor={floor})"
    if ok:
        return CheckResult("R^2 above floor", PASS, detail)
    return CheckResult("R^2 above floor", FAIL if strict else WARN, detail)


def check_no_leakage() -> CheckResult:
    overlap = set(FEATURE_COLUMNS) & set(TARGET_COLUMNS)
    if overlap:
        return CheckResult("no target columns leaked into features", FAIL, f"overlap: {overlap}")
    return CheckResult("no target columns leaked into features", PASS)


def check_all_cities_present(merged: pd.DataFrame, test_df: pd.DataFrame) -> CheckResult:
    """Catches a city silently dropping out of live collection (API outage,
    a bad city name, a quota issue) before it reaches the metrics below —
    a missing city won't lower the aggregate R^2, it'll just vanish."""
    if "city" not in merged.columns:
        return CheckResult("all cities present in test window", WARN, "no 'city' column found")
    all_cities = set(merged["city"].unique())
    test_cities = set(test_df["city"].unique())
    missing = all_cities - test_cities
    if missing:
        return CheckResult("all cities present in test window", FAIL,
                            f"missing from test window: {sorted(missing)}")
    return CheckResult("all cities present in test window", PASS,
                        f"{len(test_cities)} cities: {sorted(test_cities)}")


def check_per_city_metrics(model, test_clean: pd.DataFrame, target_col: str,
                            r2_floor: float, strict: bool, min_rows: int) -> list:
    """Aggregate R^2 across pooled cities can hide one city the model fits
    poorly — a city with more predictable AQI can carry the average while
    another city's predictions are close to useless. This breaks metrics
    out per city so that failure mode can't hide."""
    if "city" not in test_clean.columns:
        return [CheckResult("per-city fit", WARN, "no 'city' column found — skipped")]

    results = []
    for city, group in test_clean.groupby("city"):
        if len(group) < min_rows:
            results.append(CheckResult(f"[{city}] per-city fit", WARN,
                                        f"only {len(group)} test rows, skipping"))
            continue
        preds = model.predict(group[FEATURE_COLUMNS])
        m = compute_metrics(group[target_col], preds)
        status = PASS if m["r2"] >= r2_floor else (FAIL if strict else WARN)
        results.append(CheckResult(f"[{city}] per-city fit", status,
                                    f"R^2={m['r2']:.3f} RMSE={m['rmse']:.2f} (n={len(group)})"))
    return results


def check_pm25_sensitivity(model, X_test: pd.DataFrame, bump: float = 25.0,
                            sample_size: int = 200, seed: int = 42) -> CheckResult:
    """pm25 is one of the strongest known AQI drivers. Raising it while
    holding everything else fixed should raise predicted AQI on average.
    This does NOT test accuracy — a model can pass this and still have bad
    RMSE. It only catches a model that learned an inverted or dead
    relationship to its most important input, which RMSE alone won't show."""
    if "pm25" not in X_test.columns:
        return CheckResult("pm25 sensitivity (directional)", WARN, "pm25 not in feature set")

    sample = X_test.sample(min(sample_size, len(X_test)), random_state=seed)
    base_pred = model.predict(sample)
    bumped = sample.copy()
    bumped["pm25"] = bumped["pm25"] + bump
    bumped_pred = model.predict(bumped)

    delta = float(bumped_pred.mean() - base_pred.mean())
    detail = f"mean prediction change for +{bump} pm25: {delta:+.2f}"
    return CheckResult("pm25 sensitivity (directional)", PASS if delta > 0 else FAIL, detail)


def check_deterministic(model, X_test: pd.DataFrame) -> CheckResult:
    p1 = model.predict(X_test)
    p2 = model.predict(X_test)
    same = bool(np.allclose(p1, p2))
    return CheckResult("predictions are deterministic", PASS if same else FAIL)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def load_model_for_target(target_col: str):
    model_path = MODEL_DIR / f"aqi_{target_col}_model" / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No local model found at {model_path} — run train.py first "
            f"(this script reads the .pkl train.py saves locally before "
            f"uploading to the Model Registry, not the registry itself)."
        )
    return joblib.load(model_path)


def run_checks_for_horizon(target_col: str, test_clean: pd.DataFrame, r2_floor: float,
                            strict: bool, min_city_rows: int) -> list:
    model = load_model_for_target(target_col)
    X_test = test_clean[FEATURE_COLUMNS]
    y_test = test_clean[target_col]

    preds = model.predict(X_test)
    metrics = compute_metrics(y_test, preds)
    baseline_metrics = naive_persistence_baseline(test_clean, target_col)

    results = [
        check_finite_and_bounded(preds),
        check_beats_baseline(metrics, baseline_metrics),
        check_r2_floor(metrics, r2_floor, strict),
        check_pm25_sensitivity(model, X_test),
        check_deterministic(model, X_test),
    ]
    results.extend(check_per_city_metrics(model, test_clean, target_col, r2_floor, strict, min_city_rows))
    return results


def main():
    parser = argparse.ArgumentParser(description="Sanity-check registered AQI models.")
    parser.add_argument("--strict", action="store_true",
                         help="Fail (not just warn) when R^2 is below --r2-floor.")
    parser.add_argument("--r2-floor", type=float, default=-0.5,
                         help="R^2 below this is flagged (default: -0.5).")
    parser.add_argument("--min-city-rows", type=int, default=10,
                         help="Skip per-city R^2 check for cities with fewer test rows than this (default: 10).")
    args = parser.parse_args()

    project = hopsworks_io.connect_project()
    fs = project.get_feature_store()
    merged = load_merged_data(fs)
    _, test_df = date_based_split(merged)

    overall_pass = True
    print("\n=== Model prediction sanity checks ===\n")

    leakage_result = check_no_leakage()
    print(leakage_result)
    overall_pass &= leakage_result.status != FAIL

    cities_result = check_all_cities_present(merged, test_df)
    print(cities_result)
    overall_pass &= cities_result.status != FAIL

    for target_col in TARGET_COLUMNS:
        cols_needed = FEATURE_COLUMNS + [target_col]
        test_clean = test_df.dropna(subset=cols_needed)
        print(f"\n[{target_col}] ({len(test_clean)} usable test rows)")

        if len(test_clean) == 0:
            print(f"  [{FAIL}] no usable test rows — skipping remaining checks for this horizon")
            overall_pass = False
            continue

        for result in run_checks_for_horizon(target_col, test_clean, args.r2_floor, args.strict,
                                              args.min_city_rows):
            print(result)
            if result.status == FAIL:
                overall_pass = False

    print("\n" + ("All checks passed." if overall_pass else "One or more checks FAILED.") + "\n")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()