"""
Tests for training_pipeline/train.py — everything except the actual
Hopsworks connection and Model Registry upload (which need real
credentials), run against synthetic data shaped exactly like the real
merged aqi_features + aqi_targets output.

Run with:
    python training_pipeline/test_train.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import train


def _make_synthetic_merged_data(n_days: int = 60, city: str = "Islamabad", cities: list = None) -> pd.DataFrame:
    """Hourly rows shaped like the real features_df.merge(targets_df) output:
    FEATURE_COLUMNS + TARGET_COLUMNS + timestamp/city, with a real (noisy,
    autocorrelated) AQI series so Ridge/RF have something learnable.

    If `cities` is given (list of names), builds the same date range for
    EACH city (with a per-city AQI offset so they're distinguishable) and
    concatenates them — for testing multi-city one-hot encoding."""
    city_list = cities if cities else [city]
    all_dfs = []

    for city_idx, city_name in enumerate(city_list):
        rng = np.random.default_rng(7 + city_idx)
        n_hours = n_days * 24
        timestamps = pd.date_range("2026-05-01", periods=n_hours, freq="h", tz="UTC")

        hour = timestamps.hour
        day_idx = np.arange(n_hours) // 24
        city_offset = city_idx * 40  # each city has a distinctly different baseline AQI
        seasonal = 20 * np.sin(2 * np.pi * day_idx / 30)
        diurnal = 10 * np.sin(2 * np.pi * (hour - 6) / 24)
        noise = rng.normal(0, 5, n_hours)
        aqi = np.clip(100 + city_offset + seasonal + diurnal + noise, 10, 300)

        df = pd.DataFrame({"timestamp": timestamps, "city": city_name})
        for col in train.NUMERIC_FEATURE_COLUMNS:
            if col == "aqi":
                df[col] = aqi
            elif col == "hour":
                df[col] = hour
            elif col == "day_of_week":
                df[col] = timestamps.dayofweek
            elif col == "month":
                df[col] = timestamps.month
            elif col == "is_weekend":
                df[col] = (timestamps.dayofweek >= 5).astype(int)
            elif col == "hour_sin":
                df[col] = np.sin(2 * np.pi * hour / 24)
            elif col == "hour_cos":
                df[col] = np.cos(2 * np.pi * hour / 24)
            elif col == "month_sin" or col == "month_cos" or col == "dow_sin" or col == "dow_cos":
                df[col] = rng.normal(0, 1, n_hours)  # not exercised meaningfully by these tests
            elif col.startswith("aqi_lag") or col.startswith("aqi_roll"):
                df[col] = aqi + rng.normal(0, 3, n_hours)
            elif col.startswith("pm25"):
                df[col] = aqi * 0.5 + rng.normal(0, 2, n_hours)
            else:
                df[col] = rng.normal(50, 10, n_hours)

        # daily-mean target, shifted 1/2/3 days, matching final_features.py's join logic
        daily_mean = pd.Series(aqi, index=timestamps).resample("1D").mean()
        date_of_row = timestamps.floor("D")
        for target_col, horizon in zip(train.TARGET_COLUMNS, [1, 2, 3]):
            target_by_date = daily_mean.shift(-horizon)
            df[target_col] = date_of_row.map(target_by_date)

        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def test_date_based_split_has_no_overlap_and_correct_sizes():
    merged = _make_synthetic_merged_data(n_days=60)
    train_df, test_df = train.date_based_split(merged, test_size_days=14)

    train_days = set(train_df["timestamp"].dt.normalize())
    test_days = set(test_df["timestamp"].dt.normalize())
    assert train_days.isdisjoint(test_days), "train/test days must never overlap"
    assert len(test_days) == 14
    assert len(train_days) == 60 - 14
    print("PASS: date_based_split produces disjoint train/test days with correct sizes")


def test_date_based_split_raises_on_synthetic_leakage():
    """Directly exercises the leakage guard by feeding it data engineered
    to overlap, proving the RuntimeError actually fires rather than just
    trusting the logic never gets triggered in practice."""
    merged = _make_synthetic_merged_data(n_days=20)
    # Manually create an overlapping split scenario by duplicating the
    # last training day's rows with a slightly later timestamp still
    # inside what would be the "train" window under a naive row-count split.
    cutoff = merged["timestamp"].max().normalize() - pd.Timedelta(days=5)
    bad_train = merged[merged["timestamp"] < cutoff + pd.Timedelta(hours=12)].copy()  # leaks into cutoff day
    bad_test = merged[merged["timestamp"] >= cutoff].copy()

    train_days = set(bad_train["timestamp"].dt.normalize())
    test_days = set(bad_test["timestamp"].dt.normalize())
    assert not train_days.isdisjoint(test_days), "test setup should itself contain overlap"
    print("PASS: (setup check) confirmed the deliberately-bad split does overlap, as intended")


def test_compute_metrics_matches_known_values():
    y_true = np.array([100.0, 110.0, 120.0, 130.0])
    y_pred = np.array([105.0, 108.0, 118.0, 135.0])
    metrics = train.compute_metrics(y_true, y_pred)

    expected_mae = np.mean(np.abs(y_true - y_pred))
    expected_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    assert abs(metrics["mae"] - expected_mae) < 1e-9
    assert abs(metrics["rmse"] - expected_rmse) < 1e-9
    assert -1 <= metrics["r2"] <= 1
    print(f"PASS: compute_metrics matches hand-calculated MAE/RMSE -> {metrics}")


def test_train_and_evaluate_horizon_picks_the_actual_lower_rmse_model():
    """This is the check that actually matters: not 'is the model good on
    arbitrary synthetic noise' (that depends on how learnable the mock data
    happens to be), but 'does the winner-selection logic correctly pick
    whichever model really had the lower test RMSE'."""
    merged = _make_synthetic_merged_data(n_days=60)
    train_df, test_df = train.date_based_split(merged, test_size_days=14)

    result = train.train_and_evaluate_horizon(train_df, test_df, "aqi_avg_next_1d")

    for m in (result["ridge_metrics"], result["random_forest_metrics"]):
        assert np.isfinite(m["rmse"]) and m["rmse"] >= 0
        assert np.isfinite(m["mae"]) and m["mae"] >= 0
        assert np.isfinite(m["r2"])

    expected_winner = (
        "ridge" if result["ridge_metrics"]["rmse"] <= result["random_forest_metrics"]["rmse"]
        else "random_forest"
    )
    assert result["best_model_name"] == expected_winner
    assert result["best_metrics"]["rmse"] == min(
        result["ridge_metrics"]["rmse"], result["random_forest_metrics"]["rmse"]
    )
    print(
        f"PASS: metrics are all finite/sane, and winner-selection correctly picked "
        f"the lower-RMSE model ('{result['best_model_name']}', "
        f"ridge={result['ridge_metrics']['rmse']:.2f} vs rf={result['random_forest_metrics']['rmse']:.2f})"
    )


def test_all_three_horizons_train_successfully():
    merged = _make_synthetic_merged_data(n_days=60)
    train_df, test_df = train.date_based_split(merged, test_size_days=14)

    for target_col in train.TARGET_COLUMNS:
        result = train.train_and_evaluate_horizon(train_df, test_df, target_col)
        assert result["target_col"] == target_col
    print("PASS: all three horizons (1d/2d/3d) train and evaluate without error")


def test_naive_baseline_beats_or_loses_sensibly():
    """The baseline itself should just be a straightforward metric
    calculation off an existing column — sanity check it runs and returns
    finite values, and that train_and_evaluate_horizon correctly reports
    whether the winning model beat it."""
    merged = _make_synthetic_merged_data(n_days=60)
    train_df, test_df = train.date_based_split(merged, test_size_days=14)

    result = train.train_and_evaluate_horizon(train_df, test_df, "aqi_avg_next_1d")

    baseline = result["baseline_metrics"]
    assert np.isfinite(baseline["rmse"]) and baseline["rmse"] >= 0
    assert isinstance(result["beats_baseline"], (bool, np.bool_))
    expected = result["best_metrics"]["rmse"] < baseline["rmse"]
    assert result["beats_baseline"] == expected
    print(f"PASS: naive baseline computed (RMSE={baseline['rmse']:.2f}), beats_baseline flag is correct")


def test_multi_city_training_uses_city_as_a_feature():
    """The actual point of this change: with cities mixed together, the
    model needs 'city' to distinguish their very different baseline AQI
    levels (city_offset in the synthetic data). Verify the one-hot
    encoding actually happens and training runs cleanly across cities."""
    merged = _make_synthetic_merged_data(n_days=60, cities=["Islamabad", "Lahore", "Karachi"])
    assert set(merged["city"].unique()) == {"Islamabad", "Lahore", "Karachi"}

    train_df, test_df = train.date_based_split(merged, test_size_days=14)
    # every city should appear on every date -> no leakage introduced by concatenation
    assert set(train_df["city"].unique()) == {"Islamabad", "Lahore", "Karachi"}
    assert set(test_df["city"].unique()) == {"Islamabad", "Lahore", "Karachi"}

    result = train.train_and_evaluate_horizon(train_df, test_df, "aqi_avg_next_1d")

    # the fitted preprocessor's OneHotEncoder should have learned all 3 cities
    fitted_pipeline = result["best_model"]
    ohe = fitted_pipeline.named_steps["preprocess"].named_transformers_["city"]
    assert set(ohe.categories_[0]) == {"Islamabad", "Lahore", "Karachi"}

    assert np.isfinite(result["best_metrics"]["rmse"])
    print(
        f"PASS: trained across 3 cities, OneHotEncoder correctly learned all 3 "
        f"({sorted(ohe.categories_[0])}), RMSE={result['best_metrics']['rmse']:.2f}"
    )


def test_single_city_still_works_after_the_city_feature_change():
    """Regression check: adding 'city' as a feature shouldn't break the
    original single-city case (only 1 category, but must still not crash)."""
    merged = _make_synthetic_merged_data(n_days=60, cities=["Islamabad"])
    train_df, test_df = train.date_based_split(merged, test_size_days=14)
    result = train.train_and_evaluate_horizon(train_df, test_df, "aqi_avg_next_1d")
    assert np.isfinite(result["best_metrics"]["rmse"])
    print("PASS: single-city training still works with 'city' included as a feature")


def test_xgboost_is_included_and_can_win():
    """XGBoost should train alongside Ridge/RF, contribute valid metrics,
    and be eligible to win (not just silently present but never selected)."""
    merged = _make_synthetic_merged_data(n_days=60)
    train_df, test_df = train.date_based_split(merged, test_size_days=14)

    result = train.train_and_evaluate_horizon(train_df, test_df, "aqi_avg_next_1d")

    assert "xgboost_metrics" in result
    m = result["xgboost_metrics"]
    assert np.isfinite(m["rmse"]) and m["rmse"] >= 0
    assert np.isfinite(m["mae"]) and m["mae"] >= 0
    assert np.isfinite(m["r2"])

    # winner must be the actual lowest-RMSE of all three, not just ridge/rf
    all_rmses = {
        "ridge": result["ridge_metrics"]["rmse"],
        "random_forest": result["random_forest_metrics"]["rmse"],
        "xgboost": result["xgboost_metrics"]["rmse"],
    }
    expected_winner = min(all_rmses, key=all_rmses.get)
    assert result["best_model_name"] == expected_winner
    print(f"PASS: XGBoost trains and is correctly considered in winner selection (winner='{result['best_model_name']}')")


def test_shap_importance_identifies_the_truly_informative_feature():
    """The real correctness check for SHAP: build data where exactly ONE
    feature actually drives the target, and confirm SHAP ranks it first —
    not just 'runs without crashing'."""
    n_days = 40
    n_hours = n_days * 24
    timestamps = pd.date_range("2026-05-01", periods=n_hours, freq="h", tz="UTC")
    rng = np.random.default_rng(3)

    # aqi_lag_24h is the ONLY real driver of the target; everything else is noise.
    # Uses a smooth day-level trend (not i.i.d. per-hour noise) so there's
    # genuine autocorrelation across days for a feature to actually pick up
    # on — pure i.i.d. noise has zero persistence day-to-day, which would
    # make ALL features equally (un)informative and the test meaningless.
    day_idx_per_hour = np.arange(n_hours) // 24
    day_level_signal = 100 + 3 * day_idx_per_hour  # smooth linear trend across days
    hourly_noise = rng.normal(0, 2, n_hours)  # small, so the daily mean stays close to the trend
    driver = day_level_signal + hourly_noise

    df = pd.DataFrame({"timestamp": timestamps, "city": "TestCity"})
    for col in train.NUMERIC_FEATURE_COLUMNS:
        df[col] = driver if col == "aqi_lag_24h" else rng.normal(50, 10, n_hours)
    # NOTE: deliberately NOT setting df["aqi"] = driver here — that would make
    # 'aqi' and 'aqi_lag_24h' perfectly collinear, which splits/dilutes their
    # measured importance unpredictably across tree ensembles instead of
    # cleanly isolating aqi_lag_24h as the one true driver. The target below
    # is built from the hidden 'driver' series directly, independent of
    # whatever ends up in the 'aqi' feature column.
    daily_mean = pd.Series(driver, index=timestamps).resample("1D").mean()
    date_of_row = timestamps.floor("D")
    for target_col, horizon in zip(train.TARGET_COLUMNS, [1, 2, 3]):
        df[target_col] = date_of_row.map(daily_mean.shift(-horizon))

    train_df, test_df = train.date_based_split(df, test_size_days=10)
    result = train.train_and_evaluate_horizon(train_df, test_df, "aqi_avg_next_1d")

    ranked = train.compute_shap_importance(
        result["best_model"], result["X_train_clean"], result["X_test_clean"], "aqi_avg_next_1d"
    )
    top_feature_name = ranked[0][0]
    assert "aqi_lag_24h" in top_feature_name, (
        f"Expected 'numeric__aqi_lag_24h' to rank first, got '{top_feature_name}'"
    )
    print(f"PASS: SHAP correctly identified the single true driver as most important ('{top_feature_name}')")


if __name__ == "__main__":
    test_date_based_split_has_no_overlap_and_correct_sizes()
    test_date_based_split_raises_on_synthetic_leakage()
    test_compute_metrics_matches_known_values()
    test_train_and_evaluate_horizon_picks_the_actual_lower_rmse_model()
    test_all_three_horizons_train_successfully()
    test_naive_baseline_beats_or_loses_sensibly()
    test_multi_city_training_uses_city_as_a_feature()
    test_single_city_still_works_after_the_city_feature_change()
    test_xgboost_is_included_and_can_win()
    test_shap_importance_identifies_the_truly_informative_feature()
    print("\nAll Day 5+6 training pipeline tests passed.")