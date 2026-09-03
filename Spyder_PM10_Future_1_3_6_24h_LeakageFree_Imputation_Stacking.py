# -*- coding: utf-8 -*-
"""
Leakage-controlled PM10 forecasting for 1, 3, 6 and 24 h horizons.

For each station, the final calendar year is reserved as a common,
strictly chronological test period.  Model selection, multivariate
imputation and stacking are learned without using that test period.

Forecast definition
-------------------
Predictors observed at issue time t are used to estimate measured PM10 at
t+h, where h is 1, 3, 6 or 24 hours.  Targets are matched by exact timestamp,
not by row position.  Consequently a missing clock hour cannot silently turn a
1-hour problem into a longer-horizon problem.

Imputation definition
---------------------
``find_best_impute_model`` selects an imputation regressor on the pre-test
period with forward-chaining TimeSeriesSplit.  A 24-hour safety boundary is
placed before the target-test cutoff so even the issue time of the first 24-h
test forecast lies outside the imputer-fitting block.  The selected model is then used
by ``impute_train_test_leakage_free``: every target-specific imputation model
is fitted only on pre-test observations and the already-fitted model is applied
to gaps in the final-year test block.  Predictor medians are fitted inside the
corresponding training-only pipeline.

The imputed current PM10 may be used as PM10(t), an input available at issue
time.  Future PM10(t+h) labels are NEVER imputed: training and test metrics use
only genuinely observed future PM10 measurements.

Forecast figures
----------------
Instead of same-hour-oriented SHAP, scatter and residual-violin panels, this
script produces diagnostics designed for multi-horizon forecasting: metric
curves across lead times, full-test-year measured/predicted daily trajectories,
and hourly trajectories for a transparent 30-day window centred on the largest
measured PM10 event.  A persistence forecast, PM10(t+h)=PM10(t), is included as
an operational benchmark.

Required packages (install in the environment used by Spyder):
    python -m pip install pandas numpy openpyxl scikit-learn scikit-optimize \
        xgboost lightgbm matplotlib
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import warnings
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    TimeSeriesSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor
    from skopt import BayesSearchCV
    from skopt.space import Integer, Real
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "Eksik paket var. PM10 ortaminda su komutu calistirin: "
        "python -m pip install scikit-optimize xgboost lightgbm openpyxl "
        "matplotlib"
    ) from exc


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42
FORECAST_HORIZONS_H = (1, 3, 6, 24)
TEST_YEARS = 1
INCLUDE_CURRENT_PM10_AS_PREDICTOR = True

CV_SPLITS = 5
IMPUTE_SEARCH_ITER = 20
BAYES_N_ITER = 30
N_JOBS_SEARCH = -1
BAYES_VERBOSE = 2
PRE_DISPATCH = "2*n_jobs"
MIN_MODEL_ROWS = 100
MIN_IMPUTE_OBS = 20

RIDGE_ALPHA_GRID = np.logspace(-4, 6, 101)

RUN_FIGURES = True
SHOW_FIGURES_IN_NOTEBOOK = False
FIGURE_DPI = 600
FORECAST_EVENT_WINDOW_DAYS = 30
HORIZON_FIGURE_SIZE_INCHES = (10.8, 8.0)
TIMESERIES_FIGURE_SIZE_INCHES = (11.2, 8.2)
PANEL_TITLE_FONTSIZE = 9.0
AXIS_LABEL_FONTSIZE = 9.0
AXIS_TICK_FONTSIZE = 7.5
LEGEND_FONTSIZE = 8.0
MAIN_TITLE_FONTSIZE = 11.0

SAVE_CHECKPOINT_AFTER_EACH_HORIZON = True
RESUME_FROM_CHECKPOINT = True
CONFIG_SCHEMA_VERSION = "2026-08-26-v2-future-specific-figures-persistence"

SCRIPT_DIR = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd().resolve()
)
BASE_DIR = SCRIPT_DIR
FILES = OrderedDict(
    {
        "Aliaga": BASE_DIR / "Aliaga_Meteor_Son.xlsx",
        "Bornova": BASE_DIR / "Bornova_Meteor_Son.xlsx",
        "Menemen": BASE_DIR / "Menemen_Meteor_Son.xlsx",
    }
)
OUTPUT_FILE = BASE_DIR / "PM10_Future_1_3_6_24h_Forecast_Diagnostics.xlsx"
FIGURE_DIR = BASE_DIR / "PM10_Future_1_3_6_24h_Forecast_Figures"

TARGET = "PM10"
OBSERVED_TARGET = "PM10_observed"
FORECAST_TARGET = "Target_PM10"

BASE_FEATURE_CANDIDATES = [
    "SO2",
    "CO",
    "NO2",
    "NOX",
    "NO",
    "O3",
    "Temp",
    "Humidity",
    "Pressure",
    "Windspeed",
    "Winddirection",
]

CYCLIC_FEATURE_CANDIDATES = [
    "Hour_sin",
    "Hour_cos",
    "DOW_sin",
    "DOW_cos",
    "DOY_sin",
    "DOY_cos",
    "WindDir_sin",
    "WindDir_cos",
    "Wind_u",
    "Wind_v",
]

COLUMN_ALIASES = {
    "time": "Tarih",
    "Date": "Tarih",
    "Datetime": "Tarih",
    "temperature_2m": "Temp",
    "relative_humidity_2m": "Humidity",
    "surface_pressure": "Pressure",
    "windspeed_10m": "Windspeed",
    "wind_speed_10m": "Windspeed",
    "winddirection_10m": "Winddirection",
    "wind_direction_10m": "Winddirection",
    "NOx": "NOX",
}

NONNEGATIVE_COLUMNS = [
    "PM10",
    "SO2",
    "CO",
    "NO2",
    "NOX",
    "NO",
    "O3",
    "Humidity",
    "Pressure",
    "Windspeed",
]


# =============================================================================
# REPRODUCIBILITY AND DATA PREPARATION
# =============================================================================


def set_reproducibility(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert numeric strings, including decimal-comma values, to float."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype("string").str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def rename_known_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        old: new
        for old, new in COLUMN_ALIASES.items()
        if old in df.columns and new not in df.columns
    }
    return df.rename(columns=rename_map)


def load_and_clean_station(file_path: Path) -> tuple[pd.DataFrame, dict]:
    """Load one station while retaining PM10 gaps for training-only imputation."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dosya bulunamadi: {file_path}")

    raw = rename_known_columns(pd.read_excel(file_path))
    if "Tarih" not in raw.columns or TARGET not in raw.columns:
        raise KeyError(f"{file_path.name}: Tarih ve PM10 sutunlari zorunludur.")

    raw_rows = len(raw)
    raw["Tarih"] = pd.to_datetime(raw["Tarih"], errors="coerce").dt.floor("h")
    invalid_date_rows = int(raw["Tarih"].isna().sum())
    raw = raw.dropna(subset=["Tarih"]).copy()

    numeric_columns = [TARGET] + [
        column for column in BASE_FEATURE_CANDIDATES if column in raw.columns
    ]
    for column in numeric_columns:
        raw[column] = coerce_numeric(raw[column])

    for column in NONNEGATIVE_COLUMNS:
        if column in raw.columns:
            raw.loc[raw[column] < 0, column] = np.nan
    if "Humidity" in raw.columns:
        raw.loc[~raw["Humidity"].between(0, 100), "Humidity"] = np.nan
    if "Winddirection" in raw.columns:
        raw["Winddirection"] = raw["Winddirection"] % 360.0

    duplicate_mask = raw.duplicated(subset=["Tarih"], keep=False)
    duplicate_hour_rows = int(duplicate_mask.sum())
    if duplicate_hour_rows:
        examples = (
            raw.loc[duplicate_mask, "Tarih"]
            .drop_duplicates()
            .sort_values()
            .head(5)
            .dt.strftime("%Y-%m-%d %H:%M:%S")
            .tolist()
        )
        raise ValueError(
            f"{file_path.name}: saat normalizasyonundan sonra "
            f"{duplicate_hour_rows} yinelenen satir bulundu. Ornekler: {examples}"
        )

    cleaned = raw.sort_values("Tarih").reset_index(drop=True)
    cleaned[OBSERVED_TARGET] = cleaned[TARGET].copy()
    if len(cleaned) < MIN_MODEL_ROWS:
        raise ValueError(f"{file_path.name}: yalnizca {len(cleaned)} tarihli satir var.")

    summary = {
        "Raw_rows": raw_rows,
        "Removed_invalid_date": invalid_date_rows,
        "Duplicate_hour_rows": duplicate_hour_rows,
        "Rows_after_date_filter": len(cleaned),
        "Observed_PM10_rows": int(cleaned[OBSERVED_TARGET].notna().sum()),
        "Missing_PM10_rows": int(cleaned[OBSERVED_TARGET].isna().sum()),
        "First_timestamp": cleaned["Tarih"].min(),
        "Last_timestamp": cleaned["Tarih"].max(),
    }
    return cleaned, summary


def add_cyclic_time_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add issue-time cyclic and wind-vector features after imputation."""
    out = df.copy()
    dt = out["Tarih"]
    hour = dt.dt.hour + dt.dt.minute / 60.0
    dow = dt.dt.dayofweek
    doy = dt.dt.dayofyear

    out["Hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["Hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["DOW_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    out["DOW_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    out["DOY_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    out["DOY_cos"] = np.cos(2.0 * np.pi * doy / 365.25)

    if "Winddirection" in out.columns:
        radians = np.deg2rad(out["Winddirection"] % 360.0)
        out["WindDir_sin"] = np.sin(radians)
        out["WindDir_cos"] = np.cos(radians)
        if "Windspeed" in out.columns:
            out["Wind_u"] = out["Windspeed"] * np.cos(radians)
            out["Wind_v"] = out["Windspeed"] * np.sin(radians)

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def calculate_test_start(station_df: pd.DataFrame) -> pd.Timestamp:
    """Return the start of the final calendar-year target test block."""
    last_timestamp = pd.Timestamp(station_df["Tarih"].max())
    test_start = last_timestamp - pd.DateOffset(years=TEST_YEARS)
    if test_start <= station_df["Tarih"].min():
        raise ValueError("Son bir yil ayrildiginda egitim donemi bos kaliyor.")
    return pd.Timestamp(test_start)


def safe_time_series_cv(n_samples: int) -> TimeSeriesSplit:
    """Choose a valid forward-chaining split count for the available rows."""
    if n_samples < 6:
        raise ValueError(f"TimeSeriesSplit icin yetersiz satir: {n_samples}")
    n_splits = min(CV_SPLITS, max(2, n_samples // 25), n_samples - 1)
    return TimeSeriesSplit(n_splits=n_splits)


# =============================================================================
# LEAKAGE-CONTROLLED MODEL-BASED IMPUTATION
# =============================================================================


def imputation_model_candidates() -> OrderedDict:
    return OrderedDict(
        {
            "HistGradientBoosting": (
                HistGradientBoostingRegressor(random_state=RANDOM_STATE),
                {
                    "max_iter": [100, 200, 300],
                    "learning_rate": np.linspace(0.01, 0.20, 10),
                    "max_depth": [None, 5, 10, 15],
                },
            ),
            "RandomForest": (
                RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
                {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                },
            ),
            "ExtraTrees": (
                ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=1),
                {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                },
            ),
            "XGBoost": (
                XGBRegressor(
                    random_state=RANDOM_STATE,
                    objective="reg:squarederror",
                    tree_method="hist",
                    n_jobs=1,
                    verbosity=0,
                ),
                {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": np.linspace(0.01, 0.20, 10),
                    "max_depth": [3, 5, 7, 9],
                },
            ),
        }
    )


def find_best_impute_model(
    df: pd.DataFrame,
) -> tuple[str, dict, float, object]:
    """Select the PM10 imputer using training-period forward-chaining CV only."""
    target = TARGET
    features = [
        column
        for column in df.columns
        if column != target and df[column].notna().sum() >= 2
    ]
    train_data = df.loc[df[target].notna()].copy()
    if len(train_data) < MIN_IMPUTE_OBS:
        raise ValueError(
            f"Imputasyon modeli secimi icin yalnizca {len(train_data)} PM10 gozlemi var."
        )
    if not features:
        raise ValueError("Imputasyon modeli icin kullanilabilir predictor yok.")

    X_train = train_data[features]
    y_train = train_data[target]
    cv = safe_time_series_cv(len(train_data))
    best_name = None
    best_rmse = np.inf
    best_params = None
    best_estimator = None

    for name, (model, distributions) in imputation_model_candidates().items():
        print(f"  Imputasyon aday modeli: {name}")
        pipeline = Pipeline(
            [
                ("feature_imputer", SimpleImputer(strategy="median")),
                ("regressor", model),
            ]
        )
        pipeline_params = {
            f"regressor__{key}": value for key, value in distributions.items()
        }
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=pipeline_params,
            n_iter=IMPUTE_SEARCH_ITER,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=N_JOBS_SEARCH,
            random_state=RANDOM_STATE,
            refit=True,
            error_score="raise",
            pre_dispatch=PRE_DISPATCH,
        )
        search.fit(X_train, y_train)
        candidate_rmse = -float(search.best_score_)
        print(f"    CV RMSE={candidate_rmse:.4f}")
        if candidate_rmse < best_rmse:
            best_name = name
            best_rmse = candidate_rmse
            best_params = {
                key.replace("regressor__", "", 1): value
                for key, value in search.best_params_.items()
            }
            best_estimator = clone(search.best_estimator_.named_steps["regressor"])

    if best_estimator is None:
        raise RuntimeError("En iyi imputasyon modeli belirlenemedi.")
    return best_name, best_params, best_rmse, best_estimator


def constrain_imputed_values(values: np.ndarray, column: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if column in NONNEGATIVE_COLUMNS:
        values = np.clip(values, 0.0, None)
    if column == "Humidity":
        values = np.clip(values, 0.0, 100.0)
    if column == "Winddirection":
        values = values % 360.0
    return values


def impute_train_test_leakage_free(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Fit each imputer on pre-test rows and apply it to both time blocks.

    The selected model family/hyperparameters are reused for every variable, as
    requested.  A training median is used only when a variable has too few
    observed pre-test values to fit a reliable regression model.
    """
    train_imputed = train_df.copy()
    test_imputed = test_df.copy()
    audit_records = []

    for target in train_imputed.columns:
        train_missing = train_imputed[target].isna()
        test_missing = test_imputed[target].isna()
        train_before = int(train_missing.sum())
        test_before = int(test_missing.sum())
        if train_before == 0 and test_before == 0:
            audit_records.append(
                {
                    "Variable": target,
                    "Method": "No missing values",
                    "Observed_train": int(train_imputed[target].notna().sum()),
                    "Train_missing_before": 0,
                    "Test_missing_before": 0,
                    "Train_missing_after": 0,
                    "Test_missing_after": 0,
                }
            )
            continue

        observed_train = ~train_missing
        observed_count = int(observed_train.sum())
        features = [
            column
            for column in train_imputed.columns
            if column != target and train_imputed[column].notna().sum() >= 2
        ]

        if observed_count < MIN_IMPUTE_OBS or not features:
            training_median = train_imputed.loc[observed_train, target].median()
            if pd.isna(training_median):
                raise ValueError(
                    f"{target}: egitim doneminde imputasyon icin hic gozlem yok."
                )
            if train_before:
                train_imputed.loc[train_missing, target] = training_median
            if test_before:
                test_imputed.loc[test_missing, target] = training_median
            method = "Training median fallback"
        else:
            target_pipeline = Pipeline(
                [
                    ("feature_imputer", SimpleImputer(strategy="median")),
                    ("regressor", clone(model)),
                ]
            )
            target_pipeline.fit(
                train_imputed.loc[observed_train, features],
                train_imputed.loc[observed_train, target],
            )
            if train_before:
                predictions = target_pipeline.predict(
                    train_imputed.loc[train_missing, features]
                )
                train_imputed.loc[train_missing, target] = constrain_imputed_values(
                    predictions, target
                )
            if test_before:
                predictions = target_pipeline.predict(
                    test_imputed.loc[test_missing, features]
                )
                test_imputed.loc[test_missing, target] = constrain_imputed_values(
                    predictions, target
                )
            method = type(model).__name__

        audit_records.append(
            {
                "Variable": target,
                "Method": method,
                "Observed_train": observed_count,
                "Train_missing_before": train_before,
                "Test_missing_before": test_before,
                "Train_missing_after": int(train_imputed[target].isna().sum()),
                "Test_missing_after": int(test_imputed[target].isna().sum()),
            }
        )

    return train_imputed, test_imputed, audit_records


def prepare_station_with_imputation(
    station: str,
    file_path: Path,
) -> tuple[pd.DataFrame, pd.Timestamp, dict, list[dict]]:
    """Load a station, split it once and impute without final-year leakage."""
    station_df, station_summary = load_and_clean_station(file_path)
    test_start = calculate_test_start(station_df)
    numeric_columns = [TARGET] + [
        column for column in BASE_FEATURE_CANDIDATES if column in station_df.columns
    ]

    # The first test target at ``test_start`` is issued as early as 24 h before
    # that cutoff.  End imputer fitting before this earliest possible test
    # issue time, so no observation later than a test forecast's issue time can
    # influence its missing-input replacement.
    imputation_fit_end = test_start - pd.Timedelta(
        hours=max(FORECAST_HORIZONS_H)
    )
    train_mask = station_df["Tarih"] < imputation_fit_end
    test_mask = ~train_mask
    train_numeric = station_df.loc[train_mask, numeric_columns].reset_index(drop=True)
    test_numeric = station_df.loc[test_mask, numeric_columns].reset_index(drop=True)
    if train_numeric.empty or test_numeric.empty:
        raise ValueError(f"{station}: kronolojik train/test bloklarindan biri bos.")

    print(
        f"\n{station}: imputasyon train={len(train_numeric)}, "
        f"uygulama blogu={len(test_numeric)}, imputer fit sonu={imputation_fit_end}, "
        f"hedef-test baslangici={test_start}"
    )
    best_name, best_params, best_rmse, best_model = find_best_impute_model(
        train_numeric
    )
    print(
        f"{station}: secilen imputasyon modeli={best_name}, "
        f"CV RMSE={best_rmse:.4f}, params={best_params}"
    )
    train_imputed, test_imputed, audit = impute_train_test_leakage_free(
        train_numeric, test_numeric, best_model
    )

    imputed_numeric = pd.concat([train_imputed, test_imputed], ignore_index=True)
    if len(imputed_numeric) != len(station_df):
        raise RuntimeError("Imputasyon sonrasi satir sayisi degisti.")

    prepared = station_df[["Tarih", OBSERVED_TARGET]].copy()
    for column in numeric_columns:
        prepared[column] = imputed_numeric[column].to_numpy()
    prepared = add_cyclic_time_wind_features(prepared)

    remaining = int(prepared[numeric_columns].isna().sum().sum())
    if remaining:
        raise RuntimeError(
            f"{station}: imputasyon sonrasi {remaining} temel degisken hucresi eksik."
        )

    station_summary.update(
        {
            "Station": station,
            "Test_start": test_start,
            "Imputation_fit_end": imputation_fit_end,
            "Imputation_train_rows": int(train_mask.sum()),
            "Imputation_test_rows": int(test_mask.sum()),
            "Imputation_model": best_name,
            "Imputation_CV_RMSE": best_rmse,
            "Imputation_parameters": json.dumps(
                best_params, ensure_ascii=False, default=str
            ),
        }
    )
    for record in audit:
        record.update(
            {
                "Station": station,
                "Test_start": test_start,
                "Imputation_fit_end": imputation_fit_end,
                "Selected_model": best_name,
                "Selected_model_CV_RMSE": best_rmse,
                "Selected_model_parameters": json.dumps(
                    best_params, ensure_ascii=False, default=str
                ),
            }
        )
    return prepared, test_start, station_summary, audit


# =============================================================================
# EXACT-HORIZON DATASETS AND CHRONOLOGICAL OUTER SPLIT
# =============================================================================


def select_features(df: pd.DataFrame) -> list[str]:
    candidates = BASE_FEATURE_CANDIDATES + CYCLIC_FEATURE_CANDIDATES
    if INCLUDE_CURRENT_PM10_AS_PREDICTOR:
        candidates = ["PM10_t"] + candidates
    features = [
        column
        for column in candidates
        if column in df.columns and df[column].notna().any()
    ]
    if not features:
        raise ValueError("Kullanilabilir predictor sutunu bulunamadi.")
    return features


def build_forecast_dataset(
    station_df: pd.DataFrame,
    horizon_h: int,
) -> pd.DataFrame:
    """Create PM10(t+h) by exact timestamp lookup; never by row shift."""
    if horizon_h <= 0:
        raise ValueError("Tahmin ufku pozitif saat olmalidir.")
    out = station_df.copy()
    out["Issue_Time"] = out["Tarih"]
    out["Target_Time"] = out["Issue_Time"] + pd.Timedelta(hours=horizon_h)
    observed_by_time = station_df.set_index("Tarih")[OBSERVED_TARGET]
    out[FORECAST_TARGET] = out["Target_Time"].map(observed_by_time)
    if INCLUDE_CURRENT_PM10_AS_PREDICTOR:
        out["PM10_t"] = out[TARGET]

    # Only genuinely measured future targets enter model fitting/evaluation.
    out = out.dropna(subset=[FORECAST_TARGET]).sort_values("Issue_Time")
    return out.reset_index(drop=True)


def chronological_last_year_split(
    forecast_df: pd.DataFrame,
    test_start: pd.Timestamp,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Split by target time so no final-year outcome can enter training."""
    train_mask = forecast_df["Target_Time"] < test_start
    test_mask = forecast_df["Target_Time"] >= test_start
    train_df = forecast_df.loc[train_mask].copy()
    test_df = forecast_df.loc[test_mask].copy()
    if len(train_df) < MIN_MODEL_ROWS or len(test_df) < 2:
        raise ValueError(
            f"Kronolojik split yetersiz: train={len(train_df)}, test={len(test_df)}"
        )
    if train_df["Target_Time"].max() >= test_df["Target_Time"].min():
        raise RuntimeError("Kronolojik hedef zamanlari birbiriyle ortusuyor.")

    X_train = train_df[feature_columns].reset_index(drop=True)
    X_test = test_df[feature_columns].reset_index(drop=True)
    y_train = train_df[FORECAST_TARGET].reset_index(drop=True)
    y_test = test_df[FORECAST_TARGET].reset_index(drop=True)
    test_times = test_df[["Issue_Time", "Target_Time", TARGET]].reset_index(
        drop=True
    )
    test_times.rename(columns={TARGET: "Current_PM10"}, inplace=True)
    return X_train, X_test, y_train, y_test, test_times


# =============================================================================
# BAYESIAN BASE MODELS AND FORWARD-CHAINING OOF STACKING
# =============================================================================


def make_pipeline(estimator) -> Pipeline:
    """A training-only median safety net handles any unexpected residual NaN."""
    return Pipeline(
        [
            ("feature_imputer", SimpleImputer(strategy="median")),
            ("regressor", estimator),
        ]
    )


def model_definitions() -> OrderedDict:
    definitions = OrderedDict()
    definitions["ExtraTrees"] = (
        ExtraTreesRegressor(
            random_state=RANDOM_STATE, n_jobs=1, criterion="squared_error"
        ),
        {
            "regressor__n_estimators": Integer(250, 800),
            "regressor__max_depth": Integer(6, 40),
            "regressor__min_samples_split": Integer(2, 16),
            "regressor__min_samples_leaf": Integer(1, 8),
            "regressor__max_features": Real(0.45, 1.0, prior="uniform"),
        },
    )
    definitions["XGBoost"] = (
        XGBRegressor(
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=1,
            verbosity=0,
        ),
        {
            "regressor__n_estimators": Integer(200, 800),
            "regressor__max_depth": Integer(2, 10),
            "regressor__learning_rate": Real(0.01, 0.20, prior="log-uniform"),
            "regressor__subsample": Real(0.60, 1.0, prior="uniform"),
            "regressor__colsample_bytree": Real(0.55, 1.0, prior="uniform"),
            "regressor__min_child_weight": Real(1.0, 20.0, prior="log-uniform"),
            "regressor__gamma": Real(0.0, 5.0, prior="uniform"),
            "regressor__reg_alpha": Real(1e-4, 2.0, prior="log-uniform"),
            "regressor__reg_lambda": Real(0.1, 20.0, prior="log-uniform"),
        },
    )
    definitions["LightGBM"] = (
        LGBMRegressor(
            random_state=RANDOM_STATE,
            objective="regression",
            subsample_freq=1,
            n_jobs=1,
            verbosity=-1,
        ),
        {
            "regressor__n_estimators": Integer(200, 800),
            "regressor__max_depth": Integer(3, 16),
            "regressor__num_leaves": Integer(15, 127),
            "regressor__learning_rate": Real(0.01, 0.20, prior="log-uniform"),
            "regressor__min_child_samples": Integer(5, 80),
            "regressor__subsample": Real(0.60, 1.0, prior="uniform"),
            "regressor__colsample_bytree": Real(0.55, 1.0, prior="uniform"),
            "regressor__reg_alpha": Real(1e-4, 2.0, prior="log-uniform"),
            "regressor__reg_lambda": Real(0.1, 20.0, prior="log-uniform"),
        },
    )
    definitions["HistGB"] = (
        HistGradientBoostingRegressor(
            random_state=RANDOM_STATE,
            loss="squared_error",
            early_stopping=True,
        ),
        {
            "regressor__max_iter": Integer(150, 600),
            "regressor__learning_rate": Real(0.01, 0.20, prior="log-uniform"),
            "regressor__max_leaf_nodes": Integer(15, 127),
            "regressor__max_depth": Integer(3, 20),
            "regressor__min_samples_leaf": Integer(5, 60),
            "regressor__l2_regularization": Real(
                1e-5, 10.0, prior="log-uniform"
            ),
        },
    )
    return definitions


def tune_base_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    station: str,
    horizon_h: int,
) -> tuple[OrderedDict, list[dict]]:
    tuned_models = OrderedDict()
    parameter_records = []
    cv = safe_time_series_cv(len(X_train))
    for name, (estimator, search_space) in model_definitions().items():
        print(f"\n[{station} - {horizon_h}h] {name} Bayesian optimizasyonu...")
        search = BayesSearchCV(
            estimator=make_pipeline(estimator),
            search_spaces=search_space,
            n_iter=BAYES_N_ITER,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=N_JOBS_SEARCH,
            random_state=RANDOM_STATE,
            refit=True,
            return_train_score=False,
            error_score="raise",
            verbose=BAYES_VERBOSE,
            pre_dispatch=PRE_DISPATCH,
        )
        search.fit(X_train, y_train)
        tuned_models[name] = search.best_estimator_
        clean_params = {
            key.replace("regressor__", ""): value
            for key, value in search.best_params_.items()
        }
        parameter_records.append(
            {
                "Station": station,
                "Horizon_h": horizon_h,
                "Model": name,
                "Search_Type": "BayesSearchCV + TimeSeriesSplit",
                "CV_RMSE": -float(search.best_score_),
                "Best_Parameters": json.dumps(
                    clean_params, ensure_ascii=False, default=str
                ),
            }
        )
        print(f"[{station} - {horizon_h}h] {name} CV RMSE: {-search.best_score_:.4f}")
    return tuned_models, parameter_records


def build_time_series_oof_predictions(
    tuned_models: OrderedDict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[np.ndarray, pd.Series, np.ndarray]:
    """Create forward-chaining OOF predictions for the Ridge meta-learner."""
    cv = safe_time_series_cv(len(X_train))
    oof = np.full((len(X_train), len(tuned_models)), np.nan, dtype=float)
    for fold_number, (fit_idx, valid_idx) in enumerate(cv.split(X_train), start=1):
        for model_number, (_, tuned_pipeline) in enumerate(tuned_models.items()):
            fold_model = clone(tuned_pipeline)
            fold_model.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
            oof[valid_idx, model_number] = fold_model.predict(
                X_train.iloc[valid_idx]
            )
        print(f"  Forward OOF fold {fold_number}/{cv.get_n_splits()} tamamlandi.")

    valid_rows = ~np.isnan(oof).any(axis=1)
    if valid_rows.sum() < 6:
        raise RuntimeError("Stacking icin yeterli forward OOF satiri uretilmedi.")
    return (
        np.clip(oof[valid_rows], 0.0, None),
        y_train.iloc[valid_rows].reset_index(drop=True),
        valid_rows,
    )


def fit_ridge_stacker(
    oof_predictions: np.ndarray,
    y_oof: pd.Series,
) -> RidgeCV:
    stacker = RidgeCV(
        alphas=RIDGE_ALPHA_GRID,
        cv=safe_time_series_cv(len(y_oof)),
        scoring="neg_root_mean_squared_error",
        fit_intercept=True,
    )
    stacker.fit(oof_predictions, y_oof)
    return stacker


def fit_ridge_benchmark(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    station: str,
    horizon_h: int,
) -> tuple[Pipeline, dict]:
    pipeline = Pipeline(
        [
            ("feature_imputer", SimpleImputer(strategy="median")),
            ("feature_scaler", StandardScaler()),
            ("ridge", Ridge(fit_intercept=True)),
        ]
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid={"ridge__alpha": RIDGE_ALPHA_GRID},
        cv=safe_time_series_cv(len(X_train)),
        scoring="neg_root_mean_squared_error",
        n_jobs=N_JOBS_SEARCH,
        refit=True,
        error_score="raise",
        pre_dispatch=PRE_DISPATCH,
    )
    search.fit(X_train, y_train)
    record = {
        "Station": station,
        "Horizon_h": horizon_h,
        "Model": "Ridge Benchmark (Temporal CV)",
        "Search_Type": "GridSearchCV + TimeSeriesSplit",
        "CV_RMSE": -float(search.best_score_),
        "Best_Parameters": json.dumps(
            {"alpha": float(search.best_params_["ridge__alpha"])}
        ),
    }
    return search.best_estimator_, record


# =============================================================================
# EVALUATION AND FIGURES
# =============================================================================


def calculate_metrics(y_true, y_pred) -> dict:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    r2 = np.nan if len(actual) < 2 or np.isclose(np.var(actual), 0.0) else r2_score(actual, predicted)
    return {
        "R2": r2,
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
    }


def result_record(
    station: str,
    horizon_h: int,
    model: str,
    y_true,
    y_pred,
) -> dict:
    return {
        "Station": station,
        "Horizon_h": horizon_h,
        "Model": model,
        "N_Test": len(y_true),
        **calculate_metrics(y_true, y_pred),
    }


def save_figure(path: Path, fig=None, rect=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fig is None:
        fig = plt.gcf()
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    if SHOW_FIGURES_IN_NOTEBOOK:
        plt.show()
    plt.close(fig)


FORECAST_COMPARISON_MODELS = OrderedDict(
    {
        "Persistence Benchmark": "Persistence",
        "ExtraTrees": "ExtraTrees",
        "XGBoost": "XGBoost",
        "LightGBM": "LightGBM",
        "HistGB": "HistGB",
        "Stacking Ensemble (RidgeCV-OOF)": "Stacking",
    }
)

MODEL_COLORS = {
    "Persistence Benchmark": "#7F7F7F",
    "ExtraTrees": "#1F77B4",
    "XGBoost": "#FF7F0E",
    "LightGBM": "#2CA02C",
    "HistGB": "#9467BD",
    "Stacking Ensemble (RidgeCV-OOF)": "#D62728",
}

MODEL_LINESTYLES = {
    "Persistence Benchmark": "--",
    "ExtraTrees": "-",
    "XGBoost": "-",
    "LightGBM": "-",
    "HistGB": "-",
    "Stacking Ensemble (RidgeCV-OOF)": "-",
}


def forecast_figure_record(figure_type: str, path: Path) -> dict:
    return {
        "Station": "All",
        "Horizon_h": "All",
        "Figure_Type": figure_type,
        "File": str(path),
        "DPI": FIGURE_DPI,
        "Status": "Saved",
    }


def make_residual_records(
    station: str,
    horizon_h: int,
    y_true: pd.Series,
    model_predictions: OrderedDict,
) -> list[dict]:
    actual = np.asarray(y_true, dtype=float)
    records = []
    for model_name, predictions in model_predictions.items():
        residuals = actual - np.asarray(predictions, dtype=float)
        records.extend(
            {
                "Station": station,
                "Horizon_h": horizon_h,
                "Model": model_name,
                "Residual": float(residual),
            }
            for residual in residuals
        )
    return records


def plot_horizon_metric_curves(metrics_records: list[dict]) -> dict:
    """Plot RMSE, MAE and R² against forecast horizon for every station."""
    path = FIGURE_DIR / "Forecast_Horizon_Performance_RMSE_MAE_R2_3x3.png"
    frame = pd.DataFrame(metrics_records).copy()
    if frame.empty:
        raise ValueError("Ufuk performans grafigi icin metrik kaydi yok.")
    frame["Horizon_h"] = pd.to_numeric(frame["Horizon_h"], errors="coerce")
    stations = list(FILES.keys())
    metrics = ["RMSE", "MAE", "R2"]
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    metric_labels = {
        "RMSE": "RMSE (µg/m³)",
        "MAE": "MAE (µg/m³)",
        "R2": "R²",
    }

    fig, axes = plt.subplots(
        len(stations),
        len(metrics),
        figsize=HORIZON_FIGURE_SIZE_INCHES,
        sharex=True,
        squeeze=False,
    )
    legend_handles = OrderedDict()
    for row, station in enumerate(stations):
        for column, metric in enumerate(metrics):
            ax = axes[row, column]
            for model, display_name in FORECAST_COMPARISON_MODELS.items():
                subset = frame.loc[
                    (frame["Station"] == station) & (frame["Model"] == model)
                ].sort_values("Horizon_h")
                if subset.empty:
                    continue
                line = ax.plot(
                    subset["Horizon_h"],
                    subset[metric],
                    marker="o",
                    markersize=4.0,
                    linewidth=1.45 if model != "Stacking Ensemble (RidgeCV-OOF)" else 2.2,
                    linestyle=MODEL_LINESTYLES[model],
                    color=MODEL_COLORS[model],
                    label=display_name,
                    zorder=4 if model == "Stacking Ensemble (RidgeCV-OOF)" else 2,
                )[0]
                legend_handles.setdefault(display_name, line)
            ax.set_title(
                f"{station} — {metric_labels[metric]}",
                fontsize=PANEL_TITLE_FONTSIZE,
            )
            ax.set_xticks(FORECAST_HORIZONS_H)
            ax.tick_params(labelsize=AXIS_TICK_FONTSIZE)
            ax.grid(True, linestyle="--", alpha=0.30)
            if row == len(stations) - 1:
                ax.set_xlabel("Forecast horizon (h)", fontsize=AXIS_LABEL_FONTSIZE)
            if column == 0:
                ax.set_ylabel(metric_labels[metric], fontsize=AXIS_LABEL_FONTSIZE)

    fig.suptitle(
        "Forecast performance across lead times",
        fontsize=MAIN_TITLE_FONTSIZE,
        y=0.995,
    )
    fig.legend(
        legend_handles.values(),
        legend_handles.keys(),
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        bbox_to_anchor=(0.5, 0.005),
    )
    save_figure(path, fig=fig, rect=[0, 0.075, 1, 0.975])
    return forecast_figure_record("Horizon metric curves 3x3", path)


def _prepare_prediction_frame(prediction_records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(prediction_records).copy()
    required = {
        "Station",
        "Horizon_h",
        "Target_Time",
        "Actual_PM10",
        "Pred_Stacking",
        "Pred_Persistence",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Forecast zaman-serisi sutunlari eksik: {sorted(missing)}")
    frame["Target_Time"] = pd.to_datetime(frame["Target_Time"], errors="coerce")
    frame["Horizon_h"] = pd.to_numeric(frame["Horizon_h"], errors="coerce")
    return frame.dropna(subset=["Target_Time"]).sort_values("Target_Time")


def _format_time_axis(ax, concise: bool = True) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    if concise:
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)
    ax.grid(True, linestyle="--", alpha=0.25)


def _add_common_timeseries_legend(fig) -> None:
    handles = [
        plt.Line2D([0], [0], color="black", linewidth=1.2, label="Measured"),
        plt.Line2D([0], [0], color="#D62728", linewidth=1.15, label="Stacking"),
        plt.Line2D(
            [0], [0], color="#7F7F7F", linewidth=1.0, linestyle="--", label="Persistence"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        bbox_to_anchor=(0.5, 0.005),
    )


def plot_full_test_daily_trajectories(prediction_records: list[dict]) -> dict:
    """Plot full final-year daily means without selecting a favourable episode."""
    path = FIGURE_DIR / "Full_Test_Year_Daily_Measured_vs_Forecast_3x4.png"
    frame = _prepare_prediction_frame(prediction_records)
    stations = list(FILES.keys())
    fig, axes = plt.subplots(
        len(stations),
        len(FORECAST_HORIZONS_H),
        figsize=TIMESERIES_FIGURE_SIZE_INCHES,
        squeeze=False,
    )
    for row, station in enumerate(stations):
        for column, horizon_h in enumerate(FORECAST_HORIZONS_H):
            ax = axes[row, column]
            subset = frame.loc[
                (frame["Station"] == station) & (frame["Horizon_h"] == horizon_h),
                ["Target_Time", "Actual_PM10", "Pred_Stacking", "Pred_Persistence"],
            ].drop_duplicates(subset=["Target_Time"])
            if subset.empty:
                ax.set_visible(False)
                continue
            daily = (
                subset.set_index("Target_Time")
                .resample("D")
                .mean(numeric_only=True)
                .dropna(how="all")
            )
            ax.plot(daily.index, daily["Actual_PM10"], color="black", linewidth=1.0)
            ax.plot(daily.index, daily["Pred_Stacking"], color="#D62728", linewidth=1.0)
            ax.plot(
                daily.index,
                daily["Pred_Persistence"],
                color="#7F7F7F",
                linewidth=0.9,
                linestyle="--",
                alpha=0.85,
            )
            ax.set_title(
                f"{station} — {horizon_h} h",
                fontsize=PANEL_TITLE_FONTSIZE,
            )
            _format_time_axis(ax)
            if row == len(stations) - 1:
                ax.set_xlabel("Target date", fontsize=AXIS_LABEL_FONTSIZE)
            if column == 0:
                ax.set_ylabel("Daily PM10 (µg/m³)", fontsize=AXIS_LABEL_FONTSIZE)

    fig.suptitle(
        "Measured and forecast daily PM10 during the full chronological test year",
        fontsize=MAIN_TITLE_FONTSIZE,
        y=0.995,
    )
    _add_common_timeseries_legend(fig)
    save_figure(path, fig=fig, rect=[0, 0.065, 1, 0.975])
    return forecast_figure_record("Full-test daily trajectories 3x4", path)


def plot_peak_event_hourly_trajectories(prediction_records: list[dict]) -> dict:
    """Plot a fixed-length window centred on each panel's largest measured event."""
    path = FIGURE_DIR / "Peak_Event_30Day_Hourly_Measured_vs_Forecast_3x4.png"
    frame = _prepare_prediction_frame(prediction_records)
    stations = list(FILES.keys())
    fig, axes = plt.subplots(
        len(stations),
        len(FORECAST_HORIZONS_H),
        figsize=TIMESERIES_FIGURE_SIZE_INCHES,
        squeeze=False,
    )
    half_window = pd.Timedelta(days=FORECAST_EVENT_WINDOW_DAYS / 2.0)
    for row, station in enumerate(stations):
        for column, horizon_h in enumerate(FORECAST_HORIZONS_H):
            ax = axes[row, column]
            subset = frame.loc[
                (frame["Station"] == station) & (frame["Horizon_h"] == horizon_h)
            ].drop_duplicates(subset=["Target_Time"])
            if subset.empty:
                ax.set_visible(False)
                continue
            peak_time = subset.loc[subset["Actual_PM10"].idxmax(), "Target_Time"]
            window = subset.loc[
                subset["Target_Time"].between(
                    peak_time - half_window, peak_time + half_window
                )
            ]
            ax.plot(
                window["Target_Time"],
                window["Actual_PM10"],
                color="black",
                linewidth=1.0,
            )
            ax.plot(
                window["Target_Time"],
                window["Pred_Stacking"],
                color="#D62728",
                linewidth=1.0,
            )
            ax.plot(
                window["Target_Time"],
                window["Pred_Persistence"],
                color="#7F7F7F",
                linewidth=0.85,
                linestyle="--",
                alpha=0.85,
            )
            ax.axvline(peak_time, color="#555555", linewidth=0.7, alpha=0.5)
            ax.set_title(
                f"{station} — {horizon_h} h",
                fontsize=PANEL_TITLE_FONTSIZE,
            )
            _format_time_axis(ax, concise=False)
            if row == len(stations) - 1:
                ax.set_xlabel("Target time", fontsize=AXIS_LABEL_FONTSIZE)
            if column == 0:
                ax.set_ylabel("Hourly PM10 (µg/m³)", fontsize=AXIS_LABEL_FONTSIZE)

    fig.suptitle(
        f"Hourly forecasts in a {FORECAST_EVENT_WINDOW_DAYS}-day window centred on the largest measured event",
        fontsize=MAIN_TITLE_FONTSIZE,
        y=0.995,
    )
    _add_common_timeseries_legend(fig)
    save_figure(path, fig=fig, rect=[0, 0.065, 1, 0.975])
    return forecast_figure_record("Peak-event hourly trajectories 3x4", path)


def evaluate_station_horizon(
    station_df: pd.DataFrame,
    test_start: pd.Timestamp,
    station: str,
    horizon_h: int,
    imputation_summary: dict,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict], list[dict]]:
    forecast_df = build_forecast_dataset(station_df, horizon_h)
    features = select_features(forecast_df)
    X_train, X_test, y_train, y_test, test_times = chronological_last_year_split(
        forecast_df, test_start, features
    )
    print(
        f"\n{station}-{horizon_h}h: train={len(X_train)}, test={len(X_test)}, "
        f"predictor={len(features)} | hedef-test "
        f"{test_times['Target_Time'].min()} -- {test_times['Target_Time'].max()}"
    )

    tuned_models, parameter_records = tune_base_models(
        X_train, y_train, station, horizon_h
    )
    ridge_model, ridge_record = fit_ridge_benchmark(
        X_train, y_train, station, horizon_h
    )
    parameter_records.append(ridge_record)

    oof, y_oof, _ = build_time_series_oof_predictions(
        tuned_models, X_train, y_train
    )
    stacker = fit_ridge_stacker(oof, y_oof)

    test_matrix = np.column_stack(
        [model.predict(X_test) for model in tuned_models.values()]
    )
    test_matrix = np.clip(test_matrix, 0.0, None)
    metrics_records = []
    prediction_columns = {}
    model_predictions = OrderedDict()
    for index, model_name in enumerate(tuned_models):
        prediction = test_matrix[:, index]
        metrics_records.append(
            result_record(station, horizon_h, model_name, y_test, prediction)
        )
        prediction_columns[f"Pred_{model_name}"] = prediction
        model_predictions[model_name] = prediction

    average_prediction = test_matrix.mean(axis=1)
    stacking_prediction = np.clip(stacker.predict(test_matrix), 0.0, None)
    # Operational no-change reference: the latest available (observed or
    # training-only imputed) PM10 value is carried forward to t+h.
    persistence_prediction = np.asarray(test_times["Current_PM10"], dtype=float)
    mean_prediction = np.repeat(float(y_train.mean()), len(y_test))
    median_prediction = np.repeat(float(y_train.median()), len(y_test))
    ridge_prediction = np.clip(ridge_model.predict(X_test), 0.0, None)
    model_predictions["Ensemble Average"] = average_prediction
    model_predictions["Stacking"] = stacking_prediction
    model_predictions["Persistence"] = persistence_prediction

    metrics_records.extend(
        [
            result_record(station, horizon_h, "Persistence Benchmark", y_test, persistence_prediction),
            result_record(station, horizon_h, "Mean Benchmark", y_test, mean_prediction),
            result_record(station, horizon_h, "Median Benchmark", y_test, median_prediction),
            result_record(station, horizon_h, "Ridge Benchmark (Temporal CV)", y_test, ridge_prediction),
            result_record(station, horizon_h, "Ensemble Average", y_test, average_prediction),
            result_record(station, horizon_h, "Stacking Ensemble (RidgeCV-OOF)", y_test, stacking_prediction),
        ]
    )

    predictions = pd.DataFrame(
        {
            "Station": station,
            "Horizon_h": horizon_h,
            "Issue_Time": test_times["Issue_Time"],
            "Target_Time": test_times["Target_Time"],
            "Actual_PM10": y_test,
            **prediction_columns,
            "Pred_Persistence": persistence_prediction,
            "Pred_Mean_Benchmark": mean_prediction,
            "Pred_Median_Benchmark": median_prediction,
            "Pred_Ridge_Benchmark": ridge_prediction,
            "Pred_Average": average_prediction,
            "Pred_Stacking": stacking_prediction,
            "Stacking_Residual": y_test.to_numpy() - stacking_prediction,
        }
    )

    coefficients = [
        {
            "Station": station,
            "Horizon_h": horizon_h,
            "Base_Model": model_name,
            "Ridge_Coefficient": float(coefficient),
            "Ridge_Alpha": float(stacker.alpha_),
            "Ridge_Intercept": float(stacker.intercept_),
        }
        for model_name, coefficient in zip(tuned_models.keys(), stacker.coef_)
    ]
    residuals = make_residual_records(
        station, horizon_h, y_test, model_predictions
    )

    best_base_record = min(
        (
            record
            for record in parameter_records
            if record["Model"] in tuned_models
        ),
        key=lambda record: record["CV_RMSE"],
    )
    stacking_metrics = calculate_metrics(y_test, stacking_prediction)
    persistence_metrics = calculate_metrics(y_test, persistence_prediction)
    persistence_rmse = float(persistence_metrics["RMSE"])
    persistence_mae = float(persistence_metrics["MAE"])
    summary = {
        "Station": station,
        "Horizon_h": horizon_h,
        "Test_start_target_time": test_start,
        "Train_target_start": forecast_df.loc[forecast_df["Target_Time"] < test_start, "Target_Time"].min(),
        "Train_target_end": forecast_df.loc[forecast_df["Target_Time"] < test_start, "Target_Time"].max(),
        "Test_target_start": test_times["Target_Time"].min(),
        "Test_target_end": test_times["Target_Time"].max(),
        "Train_rows_observed_target": len(X_train),
        "Test_rows_observed_target": len(X_test),
        "Predictor_count": len(features),
        "Predictors": ", ".join(features),
        "Current_PM10_included": INCLUDE_CURRENT_PM10_AS_PREDICTOR,
        "Future_targets_imputed": False,
        "Imputation_model": imputation_summary["Imputation_model"],
        "Imputation_CV_RMSE": imputation_summary["Imputation_CV_RMSE"],
        "Best_base_model_by_temporal_CV": best_base_record["Model"],
        "Best_base_model_CV_RMSE": best_base_record["CV_RMSE"],
        "Persistence_RMSE": persistence_rmse,
        "Persistence_MAE": persistence_mae,
        "Persistence_R2": persistence_metrics["R2"],
        "Stacking_RMSE_skill_vs_persistence": (
            np.nan
            if np.isclose(persistence_rmse, 0.0)
            else 1.0 - float(stacking_metrics["RMSE"]) / persistence_rmse
        ),
        "Stacking_MAE_skill_vs_persistence": (
            np.nan
            if np.isclose(persistence_mae, 0.0)
            else 1.0 - float(stacking_metrics["MAE"]) / persistence_mae
        ),
        "Stacker_alpha": float(stacker.alpha_),
        "Combination_Status": "Completed",
        "Completed_at_UTC": datetime.now(timezone.utc).isoformat(),
    }

    # Forecast figures compare all stations and horizons and are therefore
    # generated once, after every station-horizon evaluation has completed.
    figures = []

    return (
        metrics_records,
        parameter_records,
        predictions.to_dict("records"),
        coefficients,
        [summary],
        residuals,
        figures,
    )


# =============================================================================
# CHECKPOINTING AND EXCEL OUTPUT
# =============================================================================


def file_sha256(file_path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_configuration() -> OrderedDict:
    source_metadata = OrderedDict()
    for station, file_path in FILES.items():
        if file_path.exists():
            source_metadata[station] = {
                "path": str(file_path),
                "size_bytes": int(file_path.stat().st_size),
                "sha256": file_sha256(file_path),
            }
        else:
            source_metadata[station] = {"path": str(file_path), "status": "missing"}
    return OrderedDict(
        {
            "Config_Schema_Version": CONFIG_SCHEMA_VERSION,
            "Problem": "PM10 future forecasting",
            "Forecast_Horizons_h": FORECAST_HORIZONS_H,
            "Outer_Split": "Final calendar year, chronological by target time",
            "Test_Years": TEST_YEARS,
            "Current_PM10_Predictor": INCLUDE_CURRENT_PM10_AS_PREDICTOR,
            "Future_Target_Imputed": False,
            "Persistence_Benchmark": "PM10(t+h) = PM10(t)",
            "Imputation_Selection": "RandomizedSearchCV + TimeSeriesSplit on pre-test PM10",
            "Imputation_Application": (
                "Target-specific models fit before the earliest 24-h test "
                "issue time; then applied forward"
            ),
            "Imputation_Search_Iterations": IMPUTE_SEARCH_ITER,
            "Model_CV": "TimeSeriesSplit",
            "CV_Splits": CV_SPLITS,
            "Bayes_Iterations": BAYES_N_ITER,
            "Base_Models": list(model_definitions().keys()),
            "Raw_Features": BASE_FEATURE_CANDIDATES,
            "Cyclic_Features": CYCLIC_FEATURE_CANDIDATES,
            "Random_State": RANDOM_STATE,
            "Figure_DPI": FIGURE_DPI,
            "Forecast_Figures": [
                "RMSE/MAE/R2 curves across horizons",
                "Full-test-year daily measured vs stacking vs persistence",
                "Peak-event 30-day hourly measured vs stacking vs persistence",
            ],
            "Peak_Event_Window_Days": FORECAST_EVENT_WINDOW_DAYS,
            "Source_Files": source_metadata,
        }
    )


def configuration_fingerprint(configuration: OrderedDict) -> str:
    payload = json.dumps(
        configuration,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def configuration_dataframe(configuration: OrderedDict) -> pd.DataFrame:
    rows = [{"Key": "Config_Fingerprint", "Value": configuration_fingerprint(configuration)}]
    rows.extend(
        {
            "Key": key,
            "Value": json.dumps(value, ensure_ascii=False, default=str)
            if isinstance(value, (dict, list, tuple, OrderedDict))
            else value,
        }
        for key, value in configuration.items()
    )
    return pd.DataFrame(rows)


CHECKPOINT_SHEET_MAP = OrderedDict(
    {
        "metrics": "All_Metrics",
        "parameters": "Best_Parameters",
        "predictions": "Test_Predictions",
        "coefficients": "Stacking_Coefficients",
        "summaries": "Horizon_Summary",
        "stations": "Station_Imputation",
        "imputation_audit": "Imputation_Audit",
        "residuals": "Residuals",
        "figures": "Figure_Manifest",
    }
)


def empty_state() -> dict[str, list[dict]]:
    return {key: [] for key in CHECKPOINT_SHEET_MAP}


def completed_combinations(state: dict[str, list[dict]]) -> set[tuple[str, int]]:
    summary_pairs = {
        (str(row.get("Station")), int(row.get("Horizon_h")))
        for row in state["summaries"]
        if row.get("Combination_Status") == "Completed" and pd.notna(row.get("Horizon_h"))
    }
    metric_pairs = {
        (str(row.get("Station")), int(row.get("Horizon_h")))
        for row in state["metrics"]
        if row.get("Model") == "Stacking Ensemble (RidgeCV-OOF)" and pd.notna(row.get("Horizon_h"))
    }
    prediction_pairs = {
        (str(row.get("Station")), int(row.get("Horizon_h")))
        for row in state["predictions"]
        if pd.notna(row.get("Horizon_h"))
    }
    return summary_pairs & metric_pairs & prediction_pairs


def load_checkpoint(configuration: OrderedDict) -> dict[str, list[dict]]:
    state = empty_state()
    if not RESUME_FROM_CHECKPOINT or not OUTPUT_FILE.exists():
        return state
    with pd.ExcelFile(OUTPUT_FILE, engine="openpyxl") as workbook:
        if "Run_Configuration" not in workbook.sheet_names:
            raise RuntimeError("Checkpoint Run_Configuration sayfasini icermiyor.")
        saved = pd.read_excel(workbook, sheet_name="Run_Configuration")
        fingerprint_rows = saved.loc[saved["Key"] == "Config_Fingerprint", "Value"]
        if fingerprint_rows.empty or str(fingerprint_rows.iloc[0]) != configuration_fingerprint(configuration):
            raise RuntimeError(
                "Mevcut sonuc dosyasi bu kod/veri konfigurasyonuyla uyumlu degil. "
                "OUTPUT_FILE adini degistirin veya eski dosyayi ayri yere tasiyin."
            )
        for state_key, sheet_name in CHECKPOINT_SHEET_MAP.items():
            if sheet_name in workbook.sheet_names:
                frame = pd.read_excel(workbook, sheet_name=sheet_name)
                state[state_key] = frame.to_dict("records")
    print(f"Uyumlu checkpoint yuklendi: {OUTPUT_FILE}")
    return state


def autosize_excel_columns(writer: pd.ExcelWriter, sheet_names: list[str]) -> None:
    for sheet_name in sheet_names:
        worksheet = writer.sheets[sheet_name]
        for column_cells in worksheet.columns:
            values = [str(cell.value) if cell.value is not None else "" for cell in column_cells]
            max_length = min(max(len(value) for value in values) + 2, 70)
            worksheet.column_dimensions[column_cells[0].column_letter].width = max_length
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions


def save_results(
    state: dict[str, list[dict]],
    configuration: OrderedDict,
    checkpoint: bool,
) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(state["metrics"])
    stacking_only = (
        metrics.loc[metrics["Model"] == "Stacking Ensemble (RidgeCV-OOF)"].copy()
        if not metrics.empty
        else pd.DataFrame()
    )
    sheet_frames = OrderedDict(
        {
            "Run_Configuration": configuration_dataframe(configuration),
            "Stacking_Only": stacking_only,
            "All_Metrics": metrics,
            "Best_Parameters": pd.DataFrame(state["parameters"]),
            "Test_Predictions": pd.DataFrame(state["predictions"]),
            "Stacking_Coefficients": pd.DataFrame(state["coefficients"]),
            "Horizon_Summary": pd.DataFrame(state["summaries"]),
            "Station_Imputation": pd.DataFrame(state["stations"]),
            "Imputation_Audit": pd.DataFrame(state["imputation_audit"]),
            "Residuals": pd.DataFrame(state["residuals"]),
            "Figure_Manifest": pd.DataFrame(state["figures"]),
        }
    )
    temporary = OUTPUT_FILE.with_name(f"{OUTPUT_FILE.stem}.writing{OUTPUT_FILE.suffix}")
    with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
        for sheet_name, frame in sheet_frames.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
        autosize_excel_columns(writer, list(sheet_frames))
    os.replace(temporary, OUTPUT_FILE)
    print(("Checkpoint" if checkpoint else "Final sonuc") + f" kaydedildi: {OUTPUT_FILE}")


def replace_station_records(records: list[dict], station: str, new_records: list[dict]) -> list[dict]:
    return [row for row in records if str(row.get("Station")) != station] + new_records


FORECAST_FIGURE_TYPES = {
    "Horizon metric curves 3x3",
    "Full-test daily trajectories 3x4",
    "Peak-event hourly trajectories 3x4",
}


def generate_forecast_figures(state: dict[str, list[dict]]) -> None:
    """Regenerate the three manuscript-oriented forecast diagnostics once."""
    state["figures"] = [
        record
        for record in state["figures"]
        if str(record.get("Figure_Type")) not in FORECAST_FIGURE_TYPES
    ]
    figure_jobs = [
        (plot_horizon_metric_curves, state["metrics"], "Horizon metric curves 3x3"),
        (
            plot_full_test_daily_trajectories,
            state["predictions"],
            "Full-test daily trajectories 3x4",
        ),
        (
            plot_peak_event_hourly_trajectories,
            state["predictions"],
            "Peak-event hourly trajectories 3x4",
        ),
    ]
    for figure_function, records, figure_type in figure_jobs:
        try:
            state["figures"].append(figure_function(records))
        except Exception as exc:
            plt.close("all")
            warnings.warn(f"{figure_type} cizilemedi: {exc}", RuntimeWarning)
            state["figures"].append(
                {
                    "Station": "All",
                    "Horizon_h": "All",
                    "Figure_Type": figure_type,
                    "File": "",
                    "DPI": FIGURE_DPI,
                    "Status": f"Failed: {type(exc).__name__}: {exc}",
                }
            )


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    set_reproducibility()
    start = datetime.now()
    print(f"Calisma klasoru: {BASE_DIR}")
    print(f"Sonuc dosyasi: {OUTPUT_FILE}")
    print(f"Ufuklar: {FORECAST_HORIZONS_H} saat | test: son {TEST_YEARS} yil")

    configuration = build_run_configuration()
    state = load_checkpoint(configuration)
    completed = completed_combinations(state)

    for station, file_path in FILES.items():
        missing_horizons = [
            horizon for horizon in FORECAST_HORIZONS_H if (station, horizon) not in completed
        ]
        if not missing_horizons:
            print(f"{station}: dort ufuk da checkpoint'te tamam; atlandi.")
            continue

        prepared, test_start, station_summary, audit = prepare_station_with_imputation(
            station, file_path
        )
        state["stations"] = replace_station_records(
            state["stations"], station, [station_summary]
        )
        state["imputation_audit"] = replace_station_records(
            state["imputation_audit"], station, audit
        )

        for horizon_h in missing_horizons:
            (
                metrics,
                parameters,
                predictions,
                coefficients,
                summaries,
                residuals,
                figures,
            ) = evaluate_station_horizon(
                prepared, test_start, station, horizon_h, station_summary
            )
            state["metrics"].extend(metrics)
            state["parameters"].extend(parameters)
            state["predictions"].extend(predictions)
            state["coefficients"].extend(coefficients)
            state["summaries"].extend(summaries)
            state["residuals"].extend(residuals)
            state["figures"].extend(figures)
            completed.add((station, horizon_h))
            if SAVE_CHECKPOINT_AFTER_EACH_HORIZON:
                save_results(state, configuration, checkpoint=True)

    if RUN_FIGURES:
        generate_forecast_figures(state)

    save_results(state, configuration, checkpoint=False)
    end = datetime.now()
    print("Baslama zamani:", start.strftime("%Y-%m-%d %H:%M:%S"))
    print("Bitis zamani:", end.strftime("%Y-%m-%d %H:%M:%S"))
    print("Toplam calisma suresi:", end - start)


if __name__ == "__main__":
    main()
