# -*- coding: utf-8 -*-
"""
Leakage-controlled direct-horizon SARIMAX evaluation for hourly PM10.

Stations
--------
Aliağa, Bornova and Menemen

Forecast horizons
-----------------
1, 3, 6 and 24 hours ahead

Purpose
-------
This script is designed as a directly comparable SARIMAX extension of
``Spyder_PM10_Future_1_3_6_24h_LeakageFree_Imputation_Stacking.py``.

1. The station-specific imputation model family and hyperparameters are read
   from the ``Station_Imputation`` sheet of
   ``PM10_Future_1_3_6_24h_Forecast_Diagnostics.xlsx``. Randomized search is
   NOT repeated.
2. The selected imputer is fitted only before the same 24-h safety boundary
   used by the machine-learning workflow and is then applied forward.
3. Future PM10 labels are never imputed for training/evaluation. The original
   observed PM10 copy is retained as the endogenous target; missing target
   hours are handled by the SARIMAX state-space filter and excluded from test
   metrics.
4. The outer split is exactly the same target-time split: the final calendar
   year is the chronological test block and targets before that boundary form
   the training block.
5. A separate direct-horizon SARIMAX is fitted for every station-horizon pair.
   The response at issue time t is observed PM10(t+h), while exogenous inputs
   are a nonredundant linear representation of the issue-time information used
   by the ML workflow. Thus, no future pollutant or meteorological covariate is
   supplied.
6. Redundant linear exogenous representations are removed a priori; remaining
   constant or nearly duplicate columns are screened using training data only.
7. A compact training-only candidate search retains only converged fits. The
   complete training-period fit is retried with multiple optimizers, and no
   metric is published for a station-horizon pair that does not converge.
8. The output workbook contains metrics, timestamp-level predictions,
   imputation/feature audits, optimizer attempts, residual diagnostics,
   model/order diagnostics and common-issue-time metrics for later comparison
   with the ML results.

Required packages in the Spyder/conda environment
-------------------------------------------------
python -m pip install pandas numpy openpyxl scikit-learn statsmodels xgboost

Runtime note
------------
SARIMAX maximum-likelihood estimation is computationally intensive. The
default training-only BIC search uses six modest candidate specifications on
the last 90 training days. The selected specification is then refitted on the
complete chronological training block. Alternative optimizers are tried only
when an earlier optimizer is not admissible. Setting RUN_ORDER_SEARCH=False
uses the fixed order but does not bypass the mandatory convergence check.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import warnings
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.stats.diagnostic import acorr_ljungbox
except ImportError as exc:
    raise ImportError(
        "statsmodels eksik. PM10 conda ortaminda su komutu calistirin: "
        "python -m pip install statsmodels"
    ) from exc

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "xgboost eksik. PM10 conda ortaminda su komutu calistirin: "
        "python -m pip install xgboost"
    ) from exc


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42
FORECAST_HORIZONS_H = (1, 3, 6, 24)
TEST_YEARS = 1
MAX_HORIZON_H = max(FORECAST_HORIZONS_H)
INCLUDE_CURRENT_PM10_AS_EXOG = True

# Training-only SARIMAX order selection. Only converged candidates are
# eligible. BIC is used because all candidates for a station-horizon pair use
# the same response/exogenous sample.
RUN_ORDER_SEARCH = True
ORDER_SELECTION_CRITERION = "BIC"
ORDER_SEARCH_TAIL_DAYS = 90
ORDER_SEARCH_MAXITER = 200
FINAL_FIT_MAXITER = 500
ORDER_SEARCH_OPTIMIZERS = ("lbfgs", "powell")
FINAL_FIT_OPTIMIZERS = ("lbfgs", "powell", "bfgs")
REQUIRE_CONVERGENCE = True
CONTINUE_ON_COMBINATION_FAILURE = True

FIXED_ORDER = (1, 0, 1)
FIXED_SEASONAL_ORDER = (1, 0, 0, 24)

# A compact search is intentional: 12 full station-horizon analyses are run.
SARIMAX_CANDIDATES = (
    ((1, 0, 0), (0, 0, 0, 0)),
    ((2, 0, 0), (0, 0, 0, 0)),
    ((1, 0, 1), (0, 0, 0, 0)),
    ((1, 0, 0), (1, 0, 0, 24)),
    ((2, 0, 0), (1, 0, 0, 24)),
    ((1, 0, 1), (1, 0, 0, 24)),
)

SARIMAX_TREND = "c"
ENFORCE_STATIONARITY = True
ENFORCE_INVERTIBILITY = True
CONCENTRATE_SCALE = True
CLIP_NEGATIVE_FORECASTS = True
SAVE_PREDICTION_INTERVALS = True
PREDICTION_INTERVAL_ALPHA = 0.05

# A linear SARIMAX regression should not receive redundant encodings of the
# same physical quantity. The parsimonious specification preserves the same
# issue-time information while using NO and NO2 instead of their aggregate
# NOX, and wind speed plus sine/cosine direction instead of simultaneously
# supplying raw direction and u/v products.
EXOGENOUS_SPECIFICATION = "parsimonious"
REDUNDANT_EXOGENOUS_FEATURES = (
    "NOX",
    "Winddirection",
    "Wind_u",
    "Wind_v",
)
DROP_HIGHLY_CORRELATED_EXOGENOUS = True
MAX_ABS_EXOGENOUS_CORRELATION = 0.995
MIN_EXOGENOUS_STANDARD_DEVIATION = 1.0e-8

LJUNG_BOX_LAGS = (24, 48)
CONDITION_NUMBER_WARNING_THRESHOLD = 1.0e8

MIN_MODEL_ROWS = 100
MIN_IMPUTE_OBS = 20
RESUME_FROM_CHECKPOINT = True
SAVE_AFTER_EACH_HORIZON = True
CONFIG_SCHEMA_VERSION = "2026-08-31-v2-sarimax-convergence-controlled"

SCRIPT_DIR = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd().resolve()
)
# Optional override is useful when the script is stored separately from the
# data files. With no override, the original "all files in one folder"
# workflow is preserved.
BASE_DIR = Path(os.environ.get("PM10_SARIMAX_BASE_DIR", SCRIPT_DIR)).resolve()


def resolve_input_file(expected_name: str) -> Path:
    """Resolve an exact filename, then a single prefixed uploaded copy."""
    exact = BASE_DIR / expected_name
    if exact.exists():
        return exact
    matches = sorted(BASE_DIR.glob(f"*{expected_name}"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileExistsError(
            f"{expected_name} icin birden fazla aday bulundu: "
            + ", ".join(path.name for path in matches)
        )
    raise FileNotFoundError(f"Dosya bulunamadi: {exact}")


FILES = OrderedDict(
    {
        "Aliaga": resolve_input_file("Aliaga_Meteor_Son.xlsx"),
        "Bornova": resolve_input_file("Bornova_Meteor_Son.xlsx"),
        "Menemen": resolve_input_file("Menemen_Meteor_Son.xlsx"),
    }
)
ML_DIAGNOSTICS_FILE = resolve_input_file(
    "PM10_Future_1_3_6_24h_Forecast_Diagnostics.xlsx"
)
OUTPUT_FILE = BASE_DIR / (
    "PM10_Future_1_3_6_24h_SARIMAX_ConvergenceControlled_Diagnostics.xlsx"
)

TARGET = "PM10"
OBSERVED_TARGET = "PM10_observed"
FORECAST_TARGET = "Target_PM10_observed"

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
# REPRODUCIBILITY, INPUT VALIDATION AND CLEANING
# =============================================================================


def set_reproducibility(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def coerce_numeric(series: pd.Series) -> pd.Series:
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
    duplicate_rows = int(duplicate_mask.sum())
    if duplicate_rows:
        examples = (
            raw.loc[duplicate_mask, "Tarih"]
            .drop_duplicates()
            .sort_values()
            .head(5)
            .dt.strftime("%Y-%m-%d %H:%M:%S")
            .tolist()
        )
        raise ValueError(
            f"{file_path.name}: {duplicate_rows} yinelenen saat bulundu. "
            f"Ornekler: {examples}"
        )

    cleaned = raw.sort_values("Tarih").reset_index(drop=True)
    if len(cleaned) < MIN_MODEL_ROWS:
        raise ValueError(f"{file_path.name}: yalnizca {len(cleaned)} satir var.")

    # Exact hourly regularity matters for SARIMAX lag interpretation.
    full_index = pd.date_range(
        cleaned["Tarih"].min(), cleaned["Tarih"].max(), freq="h"
    )
    if len(full_index) != len(cleaned) or not np.array_equal(
        cleaned["Tarih"].to_numpy(), full_index.to_numpy()
    ):
        missing_hours = full_index.difference(pd.DatetimeIndex(cleaned["Tarih"]))
        raise ValueError(
            f"{file_path.name}: SARIMAX icin saat serisi kesintisiz olmali. "
            f"Eksik saat sayisi={len(missing_hours)}."
        )

    cleaned[OBSERVED_TARGET] = cleaned[TARGET].copy()
    summary = {
        "Raw_rows": raw_rows,
        "Removed_invalid_date": invalid_date_rows,
        "Duplicate_hour_rows": duplicate_rows,
        "Rows_after_date_filter": len(cleaned),
        "Observed_PM10_rows": int(cleaned[OBSERVED_TARGET].notna().sum()),
        "Missing_PM10_rows": int(cleaned[OBSERVED_TARGET].isna().sum()),
        "First_timestamp": cleaned["Tarih"].min(),
        "Last_timestamp": cleaned["Tarih"].max(),
    }
    return cleaned, summary


def add_cyclic_time_wind_features(df: pd.DataFrame) -> pd.DataFrame:
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
    last_timestamp = pd.Timestamp(station_df["Tarih"].max())
    test_start = last_timestamp - pd.DateOffset(years=TEST_YEARS)
    if test_start <= station_df["Tarih"].min():
        raise ValueError("Son bir yil ayrildiginda egitim donemi bos kaliyor.")
    return pd.Timestamp(test_start)


# =============================================================================
# SAVED IMPUTER REGISTRY AND LEAKAGE-CONTROLLED IMPUTATION
# =============================================================================


def parse_json_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise TypeError(f"Imputation_parameters JSON nesnesi degil: {value}")
    return parsed


def load_saved_imputation_registry() -> dict[str, dict]:
    with pd.ExcelFile(ML_DIAGNOSTICS_FILE, engine="openpyxl") as workbook:
        required = {"Station_Imputation", "Horizon_Summary"}
        missing = required.difference(workbook.sheet_names)
        if missing:
            raise KeyError(
                f"{ML_DIAGNOSTICS_FILE.name} eksik sayfalar: {sorted(missing)}"
            )
        station_frame = pd.read_excel(workbook, sheet_name="Station_Imputation")
        horizon_frame = pd.read_excel(workbook, sheet_name="Horizon_Summary")

    required_columns = {
        "Station",
        "Test_start",
        "Imputation_fit_end",
        "Imputation_model",
        "Imputation_CV_RMSE",
        "Imputation_parameters",
    }
    missing_columns = required_columns.difference(station_frame.columns)
    if missing_columns:
        raise KeyError(
            "Station_Imputation eksik sutunlar: " + str(sorted(missing_columns))
        )

    registry = {}
    for station in FILES:
        rows = station_frame.loc[
            station_frame["Station"].astype(str).str.strip() == station
        ]
        if len(rows) != 1:
            raise ValueError(
                f"Station_Imputation icinde {station} icin tek satir bekleniyordu; "
                f"bulunan={len(rows)}"
            )
        row = rows.iloc[0].to_dict()
        row["Test_start"] = pd.Timestamp(row["Test_start"])
        row["Imputation_fit_end"] = pd.Timestamp(row["Imputation_fit_end"])
        row["Imputation_parameters_parsed"] = parse_json_dict(
            row["Imputation_parameters"]
        )

        expected = horizon_frame.loc[
            horizon_frame["Station"].astype(str).str.strip() == station,
            [
                "Horizon_h",
                "Train_rows_observed_target",
                "Test_rows_observed_target",
                "Test_start_target_time",
            ],
        ].copy()
        expected["Horizon_h"] = expected["Horizon_h"].astype(int)
        if set(expected["Horizon_h"]) != set(FORECAST_HORIZONS_H):
            raise ValueError(
                f"Horizon_Summary icinde {station} ufuklari eksik veya fazla."
            )
        row["Expected_horizon_rows"] = expected.set_index("Horizon_h").to_dict(
            orient="index"
        )
        registry[station] = row
    return registry


def instantiate_saved_imputer(model_name: str, parameters: dict):
    params = dict(parameters)
    if model_name == "XGBoost":
        return XGBRegressor(
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=1,
            verbosity=0,
            **params,
        )
    if model_name == "ExtraTrees":
        return ExtraTreesRegressor(
            random_state=RANDOM_STATE,
            n_jobs=1,
            **params,
        )
    if model_name == "RandomForest":
        return RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=1,
            **params,
        )
    if model_name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(
            random_state=RANDOM_STATE,
            **params,
        )
    raise ValueError(f"Desteklenmeyen kayitli imputasyon modeli: {model_name}")


def constrain_imputed_values(values: np.ndarray, column: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if column in NONNEGATIVE_COLUMNS:
        values = np.clip(values, 0.0, None)
    if column == "Humidity":
        values = np.clip(values, 0.0, 100.0)
    if column == "Winddirection":
        values = values % 360.0
    return values


def impute_train_test_with_saved_model(
    train_df: pd.DataFrame,
    application_df: pd.DataFrame,
    model,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Reuse the original variable-by-variable imputation procedure."""
    train_imputed = train_df.copy()
    application_imputed = application_df.copy()
    audit_records = []

    for target in train_imputed.columns:
        train_missing = train_imputed[target].isna()
        application_missing = application_imputed[target].isna()
        train_before = int(train_missing.sum())
        application_before = int(application_missing.sum())

        if train_before == 0 and application_before == 0:
            audit_records.append(
                {
                    "Variable": target,
                    "Method": "No missing values",
                    "Observed_train": int(train_imputed[target].notna().sum()),
                    "Train_missing_before": 0,
                    "Application_missing_before": 0,
                    "Train_missing_after": 0,
                    "Application_missing_after": 0,
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
                    f"{target}: egitim doneminde imputasyon icin gozlem yok."
                )
            if train_before:
                train_imputed.loc[train_missing, target] = training_median
            if application_before:
                application_imputed.loc[application_missing, target] = training_median
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
            if application_before:
                predictions = target_pipeline.predict(
                    application_imputed.loc[application_missing, features]
                )
                application_imputed.loc[
                    application_missing, target
                ] = constrain_imputed_values(predictions, target)
            method = type(model).__name__

        audit_records.append(
            {
                "Variable": target,
                "Method": method,
                "Observed_train": observed_count,
                "Train_missing_before": train_before,
                "Application_missing_before": application_before,
                "Train_missing_after": int(train_imputed[target].isna().sum()),
                "Application_missing_after": int(
                    application_imputed[target].isna().sum()
                ),
            }
        )

    return train_imputed, application_imputed, audit_records


def prepare_station_with_saved_imputation(
    station: str,
    file_path: Path,
    registry_row: dict,
) -> tuple[pd.DataFrame, pd.Timestamp, dict, list[dict]]:
    station_df, station_summary = load_and_clean_station(file_path)
    calculated_test_start = calculate_test_start(station_df)
    saved_test_start = pd.Timestamp(registry_row["Test_start"])
    if calculated_test_start != saved_test_start:
        raise RuntimeError(
            f"{station}: hesaplanan test baslangici {calculated_test_start}, "
            f"diagnostics kaydi {saved_test_start}."
        )
    test_start = saved_test_start

    calculated_fit_end = test_start - pd.Timedelta(hours=MAX_HORIZON_H)
    saved_fit_end = pd.Timestamp(registry_row["Imputation_fit_end"])
    if calculated_fit_end != saved_fit_end:
        raise RuntimeError(
            f"{station}: hesaplanan imputer fit sonu {calculated_fit_end}, "
            f"diagnostics kaydi {saved_fit_end}."
        )
    imputation_fit_end = saved_fit_end

    numeric_columns = [TARGET] + [
        column for column in BASE_FEATURE_CANDIDATES if column in station_df.columns
    ]
    train_mask = station_df["Tarih"] < imputation_fit_end
    application_mask = ~train_mask
    train_numeric = station_df.loc[train_mask, numeric_columns].reset_index(drop=True)
    application_numeric = station_df.loc[
        application_mask, numeric_columns
    ].reset_index(drop=True)
    if train_numeric.empty or application_numeric.empty:
        raise ValueError(f"{station}: imputasyon bloklarindan biri bos.")

    model_name = str(registry_row["Imputation_model"])
    model_params = dict(registry_row["Imputation_parameters_parsed"])
    saved_model = instantiate_saved_imputer(model_name, model_params)
    print(
        f"\n{station}: kayitli imputer={model_name}, params={model_params}, "
        f"fit satiri={len(train_numeric)}, ileri uygulama={len(application_numeric)}"
    )

    train_imputed, application_imputed, audit = impute_train_test_with_saved_model(
        train_numeric, application_numeric, saved_model
    )
    imputed_numeric = pd.concat(
        [train_imputed, application_imputed], ignore_index=True
    )
    if len(imputed_numeric) != len(station_df):
        raise RuntimeError(f"{station}: imputasyon sonrasi satir sayisi degisti.")

    prepared = station_df[["Tarih", OBSERVED_TARGET]].copy()
    for column in numeric_columns:
        prepared[column] = imputed_numeric[column].to_numpy()
    prepared = add_cyclic_time_wind_features(prepared)

    remaining = int(prepared[numeric_columns].isna().sum().sum())
    if remaining:
        raise RuntimeError(
            f"{station}: imputasyon sonrasi {remaining} temel hucre eksik."
        )

    station_summary.update(
        {
            "Station": station,
            "Test_start": test_start,
            "Imputation_fit_end": imputation_fit_end,
            "Imputation_train_rows": int(train_mask.sum()),
            "Imputation_application_rows": int(application_mask.sum()),
            "Imputation_model": model_name,
            "Imputation_CV_RMSE_from_ML_diagnostics": float(
                registry_row["Imputation_CV_RMSE"]
            ),
            "Imputation_parameters": json.dumps(
                model_params, ensure_ascii=False, default=str
            ),
            "Imputation_parameter_source": (
                f"{ML_DIAGNOSTICS_FILE.name} / Station_Imputation"
            ),
            "Imputation_search_repeated": False,
        }
    )
    for record in audit:
        record.update(
            {
                "Station": station,
                "Test_start": test_start,
                "Imputation_fit_end": imputation_fit_end,
                "Selected_model": model_name,
                "Selected_model_CV_RMSE": float(
                    registry_row["Imputation_CV_RMSE"]
                ),
                "Selected_model_parameters": json.dumps(
                    model_params, ensure_ascii=False, default=str
                ),
                "Parameter_source": (
                    f"{ML_DIAGNOSTICS_FILE.name} / Station_Imputation"
                ),
            }
        )
    return prepared, test_start, station_summary, audit


# =============================================================================
# DIRECT-HORIZON SARIMAX DATASETS
# =============================================================================


def select_exogenous_features(
    df: pd.DataFrame,
) -> tuple[list[str], list[dict]]:
    """Build the available exogenous list and document explicit exclusions."""
    candidates = BASE_FEATURE_CANDIDATES + CYCLIC_FEATURE_CANDIDATES
    if INCLUDE_CURRENT_PM10_AS_EXOG:
        candidates = ["PM10_t"] + candidates

    features = []
    audit = []
    for column in candidates:
        if column not in df.columns or not df[column].notna().any():
            continue
        if (
            EXOGENOUS_SPECIFICATION == "parsimonious"
            and column in REDUNDANT_EXOGENOUS_FEATURES
        ):
            audit.append(
                {
                    "Feature": column,
                    "Status": "Dropped",
                    "Reason": "A priori redundant representation",
                    "Correlated_with": "",
                    "Absolute_correlation": np.nan,
                    "Train_standard_deviation": np.nan,
                }
            )
            continue
        features.append(column)
        audit.append(
            {
                "Feature": column,
                "Status": "Candidate",
                "Reason": "Available issue-time feature",
                "Correlated_with": "",
                "Absolute_correlation": np.nan,
                "Train_standard_deviation": np.nan,
            }
        )

    if EXOGENOUS_SPECIFICATION not in {"parsimonious", "full"}:
        raise ValueError(
            "EXOGENOUS_SPECIFICATION 'parsimonious' veya 'full' olmalidir."
        )
    if not features:
        raise ValueError("Kullanilabilir SARIMAX exogenous feature bulunamadi.")
    return features, audit


def prune_exogenous_features_on_training(
    X_train: pd.DataFrame,
    candidate_features: list[str],
    initial_audit: list[dict],
) -> tuple[list[str], list[dict]]:
    """Remove constants and near-duplicates using training data only."""
    audit_by_feature = {
        str(record["Feature"]): dict(record) for record in initial_audit
    }
    retained = []

    for feature in candidate_features:
        values = X_train[feature].to_numpy(dtype=float)
        std = float(np.nanstd(values, ddof=0))
        record = audit_by_feature[feature]
        record["Train_standard_deviation"] = std
        if not np.isfinite(std) or std <= MIN_EXOGENOUS_STANDARD_DEVIATION:
            record.update(
                {
                    "Status": "Dropped",
                    "Reason": "Constant or near-constant in training data",
                }
            )
            continue

        correlated_with = ""
        absolute_correlation = np.nan
        if DROP_HIGHLY_CORRELATED_EXOGENOUS and retained:
            for kept_feature in retained:
                correlation = X_train[[feature, kept_feature]].corr().iloc[0, 1]
                if (
                    np.isfinite(correlation)
                    and abs(float(correlation))
                    >= MAX_ABS_EXOGENOUS_CORRELATION
                ):
                    correlated_with = kept_feature
                    absolute_correlation = abs(float(correlation))
                    break

        if correlated_with:
            record.update(
                {
                    "Status": "Dropped",
                    "Reason": "Training-only high-correlation filter",
                    "Correlated_with": correlated_with,
                    "Absolute_correlation": absolute_correlation,
                }
            )
        else:
            retained.append(feature)
            record.update(
                {
                    "Status": "Retained",
                    "Reason": "Retained after training-only screening",
                }
            )

    if not retained:
        raise RuntimeError(
            "Egitim-donemi taramasindan sonra exogenous feature kalmadi."
        )
    ordered_audit = [audit_by_feature[str(row["Feature"])] for row in initial_audit]
    return retained, ordered_audit


def build_direct_hourly_frame(
    prepared: pd.DataFrame,
    horizon_h: int,
) -> pd.DataFrame:
    """Keep the hourly grid; missing observed future PM10 remains NaN."""
    if horizon_h <= 0:
        raise ValueError("Tahmin ufku pozitif olmalidir.")
    out = prepared.copy()
    out["Issue_Time"] = out["Tarih"]
    out["Target_Time"] = out["Issue_Time"] + pd.Timedelta(hours=horizon_h)
    observed_by_time = prepared.set_index("Tarih")[OBSERVED_TARGET]
    out[FORECAST_TARGET] = out["Target_Time"].map(observed_by_time)
    out["PM10_t"] = out[TARGET]

    # Remove only rows whose target timestamp lies beyond the source series.
    out = out.loc[out["Target_Time"] <= prepared["Tarih"].max()].copy()
    return out.sort_values("Issue_Time").reset_index(drop=True)


def chronological_target_time_split(
    frame: pd.DataFrame,
    test_start: pd.Timestamp,
    features: list[str],
) -> dict:
    train_mask = frame["Target_Time"] < test_start
    test_mask = frame["Target_Time"] >= test_start
    train = frame.loc[train_mask].copy()
    test = frame.loc[test_mask].copy()
    if len(train) < MIN_MODEL_ROWS or len(test) < 2:
        raise ValueError(
            f"Kronolojik split yetersiz: train={len(train)}, test={len(test)}"
        )
    if train["Target_Time"].max() >= test["Target_Time"].min():
        raise RuntimeError("Train ve test hedef zamanlari ortusuyor.")
    if train[features].isna().any().any() or test[features].isna().any().any():
        raise RuntimeError("SARIMAX exogenous matriste imputasyon sonrasi NaN var.")

    return {
        "train": train,
        "test": test,
        "X_train": train[features].reset_index(drop=True),
        "X_test": test[features].reset_index(drop=True),
        # NaN endogenous values are deliberately retained; Kalman filtering
        # handles them as missing observations on the regular hourly grid.
        "y_train": train[FORECAST_TARGET].reset_index(drop=True),
        "y_test": test[FORECAST_TARGET].reset_index(drop=True),
    }


def verify_alignment_with_ml_diagnostics(
    station: str,
    horizon_h: int,
    split: dict,
    registry_row: dict,
) -> None:
    expected = registry_row["Expected_horizon_rows"][horizon_h]
    train_observed = int(split["y_train"].notna().sum())
    test_observed = int(split["y_test"].notna().sum())
    expected_train = int(expected["Train_rows_observed_target"])
    expected_test = int(expected["Test_rows_observed_target"])
    expected_start = pd.Timestamp(expected["Test_start_target_time"])

    if train_observed != expected_train or test_observed != expected_test:
        raise RuntimeError(
            f"{station}-{horizon_h}h: ML ile satir eslesmesi bozuldu. "
            f"train observed={train_observed}/{expected_train}, "
            f"test observed={test_observed}/{expected_test}."
        )
    if pd.Timestamp(split["test"]["Target_Time"].min()) != expected_start:
        raise RuntimeError(
            f"{station}-{horizon_h}h: test hedef baslangici diagnostics ile ayni degil."
        )


# =============================================================================
# SARIMAX ORDER SELECTION, FITTING AND FORECASTING
# =============================================================================


def order_text(order: tuple[int, int, int]) -> str:
    return str(tuple(int(value) for value in order))


def seasonal_order_text(order: tuple[int, int, int, int]) -> str:
    return str(tuple(int(value) for value in order))


def scale_exogenous_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    train_values = scaler.fit_transform(X_train)
    test_values = scaler.transform(X_test)
    train_scaled = pd.DataFrame(train_values, columns=X_train.columns)
    test_scaled = pd.DataFrame(test_values, columns=X_test.columns)
    if not np.isfinite(train_scaled.to_numpy()).all():
        raise RuntimeError("Olceklenmis train exogenous matriste sonlu olmayan deger var.")
    if not np.isfinite(test_scaled.to_numpy()).all():
        raise RuntimeError("Olceklenmis test exogenous matriste sonlu olmayan deger var.")
    return train_scaled, test_scaled, scaler


class SARIMAXConvergenceError(RuntimeError):
    """Raised when no admissible converged SARIMAX fit is available."""

    def __init__(
        self,
        message: str,
        optimizer_attempts: list[dict] | None = None,
        order_records: list[dict] | None = None,
        feature_audit: list[dict] | None = None,
    ) -> None:
        super().__init__(message)
        self.optimizer_attempts = optimizer_attempts or []
        self.order_records = order_records or []
        self.feature_audit = feature_audit or []


def _safe_float(value) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return np.nan
    return converted if np.isfinite(converted) else np.nan


def _iteration_count(result) -> float:
    retvals = getattr(result, "mle_retvals", {}) or {}
    for key in ("iterations", "nit", "fcalls"):
        if key in retvals and retvals[key] is not None:
            return _safe_float(retvals[key])
    return np.nan


def fit_one_sarimax(
    y: pd.Series,
    X: pd.DataFrame,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    method: str,
    maxiter: int,
    start_params: np.ndarray | None = None,
):
    model = SARIMAX(
        endog=y.astype(float),
        exog=X.astype(float),
        order=order,
        seasonal_order=seasonal_order,
        trend=SARIMAX_TREND,
        enforce_stationarity=ENFORCE_STATIONARITY,
        enforce_invertibility=ENFORCE_INVERTIBILITY,
        concentrate_scale=CONCENTRATE_SCALE,
        simple_differencing=False,
        missing="none",
    )
    fit_kwargs = {
        "method": method,
        "maxiter": maxiter,
        "disp": False,
        "cov_type": "none",
    }
    if start_params is not None:
        fit_kwargs["start_params"] = np.asarray(start_params, dtype=float)
    return model.fit(**fit_kwargs)


def fit_sarimax_with_optimizer_fallback(
    y: pd.Series,
    X: pd.DataFrame,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    optimizer_methods: tuple[str, ...],
    maxiter: int,
    station: str,
    horizon_h: int,
    stage: str,
):
    """Try optimizers sequentially and return only an admissible fit."""
    attempts = []
    warm_start_params = None

    for method in optimizer_methods:
        started = time.perf_counter()
        warning_text = ""
        start_source = "default" if warm_start_params is None else "previous optimizer"
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = fit_one_sarimax(
                    y=y,
                    X=X,
                    order=order,
                    seasonal_order=seasonal_order,
                    method=method,
                    maxiter=maxiter,
                    start_params=warm_start_params,
                )
            warning_text = " | ".join(
                sorted({str(item.message) for item in caught})
            )
            retvals = getattr(result, "mle_retvals", {}) or {}
            converged = bool(retvals.get("converged", False))
            parameter_values = np.asarray(result.params, dtype=float)
            finite_parameters = bool(np.isfinite(parameter_values).all())
            finite_objective = bool(
                np.isfinite(_safe_float(result.llf))
                and np.isfinite(_safe_float(result.bic))
            )
            admissible = finite_parameters and finite_objective and (
                converged or not REQUIRE_CONVERGENCE
            )
            record = {
                "Station": station,
                "Horizon_h": horizon_h,
                "Stage": stage,
                "Order": order_text(order),
                "Seasonal_Order": seasonal_order_text(seasonal_order),
                "Optimizer": method,
                "Max_iterations": maxiter,
                "Start_source": start_source,
                "Converged": converged,
                "Admissible": admissible,
                "Iterations": _iteration_count(result),
                "AIC": _safe_float(result.aic),
                "BIC": _safe_float(result.bic),
                "Log_Likelihood": _safe_float(result.llf),
                "Finite_parameters": finite_parameters,
                "Finite_objective": finite_objective,
                "Elapsed_seconds": time.perf_counter() - started,
                "Warnings": warning_text,
                "Error": "",
                "Selected_Attempt": False,
            }
            attempts.append(record)
            if admissible:
                record["Selected_Attempt"] = True
                return result, method, warning_text, attempts

            if finite_parameters:
                warm_start_params = parameter_values.copy()
        except Exception as exc:
            attempts.append(
                {
                    "Station": station,
                    "Horizon_h": horizon_h,
                    "Stage": stage,
                    "Order": order_text(order),
                    "Seasonal_Order": seasonal_order_text(seasonal_order),
                    "Optimizer": method,
                    "Max_iterations": maxiter,
                    "Start_source": start_source,
                    "Converged": False,
                    "Admissible": False,
                    "Iterations": np.nan,
                    "AIC": np.nan,
                    "BIC": np.nan,
                    "Log_Likelihood": np.nan,
                    "Finite_parameters": False,
                    "Finite_objective": False,
                    "Elapsed_seconds": time.perf_counter() - started,
                    "Warnings": warning_text,
                    "Error": f"{type(exc).__name__}: {exc}",
                    "Selected_Attempt": False,
                }
            )

    attempted = ", ".join(optimizer_methods)
    raise SARIMAXConvergenceError(
        f"{station}-{horizon_h}h {stage}: {attempted} ile yakinayan "
        "ve sonlu bir SARIMAX uyumu elde edilemedi.",
        optimizer_attempts=attempts,
    )


def select_sarimax_order(
    y_train: pd.Series,
    X_train: pd.DataFrame,
    station: str,
    horizon_h: int,
) -> tuple[
    tuple[int, int, int],
    tuple[int, int, int, int],
    list[dict],
    list[dict],
]:
    criterion = ORDER_SELECTION_CRITERION.upper()
    if criterion not in {"AIC", "BIC", "HQIC"}:
        raise ValueError("ORDER_SELECTION_CRITERION AIC, BIC veya HQIC olmalidir.")

    if not RUN_ORDER_SEARCH:
        return FIXED_ORDER, FIXED_SEASONAL_ORDER, [
            {
                "Station": station,
                "Horizon_h": horizon_h,
                "Order": order_text(FIXED_ORDER),
                "Seasonal_Order": seasonal_order_text(FIXED_SEASONAL_ORDER),
                "Status": "Fixed configuration; final convergence still required",
                "Selected": True,
            }
        ], []

    tail_rows = min(len(y_train), int(ORDER_SEARCH_TAIL_DAYS * 24))
    y_tail = y_train.iloc[-tail_rows:].reset_index(drop=True)
    X_tail = X_train.iloc[-tail_rows:].reset_index(drop=True)
    if int(y_tail.notna().sum()) < MIN_MODEL_ROWS:
        raise ValueError(
            f"{station}-{horizon_h}h: order search icin yalnizca "
            f"{int(y_tail.notna().sum())} gozlenen hedef var."
        )

    records = []
    optimizer_attempts = []
    eligible_candidates = []
    print(
        f"  {station}-{horizon_h}h: SARIMAX order search, "
        f"tail={tail_rows} saat, aday={len(SARIMAX_CANDIDATES)}"
    )
    for order, seasonal_order in SARIMAX_CANDIDATES:
        started = time.perf_counter()
        try:
            result, optimizer, warning_text, attempts = (
                fit_sarimax_with_optimizer_fallback(
                    y=y_tail,
                    X=X_tail,
                    order=order,
                    seasonal_order=seasonal_order,
                    optimizer_methods=ORDER_SEARCH_OPTIMIZERS,
                    maxiter=ORDER_SEARCH_MAXITER,
                    station=station,
                    horizon_h=horizon_h,
                    stage="Order_search",
                )
            )
            optimizer_attempts.extend(attempts)
            record = {
                "Station": station,
                "Horizon_h": horizon_h,
                "Order": order_text(order),
                "Seasonal_Order": seasonal_order_text(seasonal_order),
                "Selection_rows": tail_rows,
                "Selection_observed_targets": int(y_tail.notna().sum()),
                "AIC": _safe_float(result.aic),
                "BIC": _safe_float(result.bic),
                "HQIC": _safe_float(result.hqic),
                "Log_Likelihood": _safe_float(result.llf),
                "Converged": True,
                "Optimizer": optimizer,
                "Iterations": _iteration_count(result),
                "Elapsed_seconds": time.perf_counter() - started,
                "Warnings": warning_text,
                "Status": "Converged and eligible",
                "Selected": False,
            }
            records.append(record)
            if np.isfinite(record[criterion]):
                eligible_candidates.append((record, order, seasonal_order))
            print(
                f"    order={order}, seasonal={seasonal_order}, "
                f"{criterion}={record[criterion]:.2f}, optimizer={optimizer}, "
                "converged=True"
            )
        except SARIMAXConvergenceError as exc:
            optimizer_attempts.extend(exc.optimizer_attempts)
            records.append(
                {
                    "Station": station,
                    "Horizon_h": horizon_h,
                    "Order": order_text(order),
                    "Seasonal_Order": seasonal_order_text(seasonal_order),
                    "Selection_rows": tail_rows,
                    "Selection_observed_targets": int(y_tail.notna().sum()),
                    "AIC": np.nan,
                    "BIC": np.nan,
                    "HQIC": np.nan,
                    "Log_Likelihood": np.nan,
                    "Converged": False,
                    "Iterations": np.nan,
                    "Elapsed_seconds": time.perf_counter() - started,
                    "Warnings": "",
                    "Status": f"No converged optimizer: {exc}",
                    "Selected": False,
                }
            )
            print(f"    order={order}, seasonal={seasonal_order} FAILED: {exc}")

    if not eligible_candidates:
        raise SARIMAXConvergenceError(
            f"{station}-{horizon_h}h: egitim-donemi aramasinda yakinayan "
            "SARIMAX adayi bulunamadi. Gecersiz sabit modele geri donulmedi.",
            optimizer_attempts=optimizer_attempts,
            order_records=records,
        )

    selected_record, selected_order, selected_seasonal = min(
        eligible_candidates,
        key=lambda item: float(item[0][criterion]),
    )
    selected_record["Selected"] = True
    return selected_order, selected_seasonal, records, optimizer_attempts


def calculate_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 2:
        return {"N_Test": int(valid.sum()), "R2": np.nan, "MAE": np.nan, "RMSE": np.nan}
    actual = y_true[valid]
    forecast = y_pred[valid]
    return {
        "N_Test": int(valid.sum()),
        "R2": float(r2_score(actual, forecast)),
        "MAE": float(mean_absolute_error(actual, forecast)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, forecast))),
    }


def calculate_residual_diagnostics(
    result,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> dict:
    """Return compact training-residual and root diagnostics."""
    residuals = pd.Series(np.asarray(result.resid, dtype=float))
    burn = max(
        int(getattr(result, "loglikelihood_burn", 0) or 0),
        int(getattr(result, "nobs_diffuse", 0) or 0),
    )
    residuals = residuals.iloc[burn:]
    residuals = residuals[np.isfinite(residuals)].reset_index(drop=True)

    diagnostics = {
        "Residual_burn_rows": burn,
        "Residual_rows": int(len(residuals)),
        "Residual_mean": _safe_float(residuals.mean()),
        "Residual_standard_deviation": _safe_float(residuals.std(ddof=1)),
        "Residual_skewness": _safe_float(residuals.skew()),
        "Residual_excess_kurtosis": _safe_float(residuals.kurt()),
    }

    model_df = int(order[0] + order[2] + seasonal_order[0] + seasonal_order[2])
    valid_lags = [
        int(lag)
        for lag in LJUNG_BOX_LAGS
        if int(lag) > model_df and int(lag) < len(residuals)
    ]
    ljung_box = None
    if valid_lags:
        try:
            ljung_box = acorr_ljungbox(
                residuals,
                lags=valid_lags,
                model_df=model_df,
                return_df=True,
            )
        except Exception:
            ljung_box = None
    for lag in LJUNG_BOX_LAGS:
        lag = int(lag)
        q_value = np.nan
        p_value = np.nan
        if ljung_box is not None and lag in ljung_box.index:
            q_value = _safe_float(ljung_box.loc[lag, "lb_stat"])
            p_value = _safe_float(ljung_box.loc[lag, "lb_pvalue"])
        diagnostics[f"Ljung_Box_Q_{lag}"] = q_value
        diagnostics[f"Ljung_Box_p_{lag}"] = p_value

    ar_roots = np.asarray(getattr(result, "arroots", []), dtype=complex)
    ma_roots = np.asarray(getattr(result, "maroots", []), dtype=complex)
    diagnostics["Minimum_AR_root_modulus"] = (
        float(np.min(np.abs(ar_roots))) if ar_roots.size else np.nan
    )
    diagnostics["Minimum_MA_root_modulus"] = (
        float(np.min(np.abs(ma_roots))) if ma_roots.size else np.nan
    )
    return diagnostics


def evaluate_sarimax_horizon(
    prepared: pd.DataFrame,
    test_start: pd.Timestamp,
    station: str,
    horizon_h: int,
    registry_row: dict,
) -> dict[str, list[dict]]:
    frame = build_direct_hourly_frame(prepared, horizon_h)
    candidate_features, feature_audit = select_exogenous_features(frame)
    split = chronological_target_time_split(
        frame, test_start, candidate_features
    )
    verify_alignment_with_ml_diagnostics(
        station, horizon_h, split, registry_row
    )

    features, feature_audit = prune_exogenous_features_on_training(
        split["X_train"], candidate_features, feature_audit
    )
    for record in feature_audit:
        record.update({"Station": station, "Horizon_h": horizon_h})
    split["X_train"] = split["X_train"][features].copy()
    split["X_test"] = split["X_test"][features].copy()

    X_train_scaled, X_test_scaled, scaler = scale_exogenous_train_test(
        split["X_train"], split["X_test"]
    )
    condition_number = _safe_float(
        np.linalg.cond(X_train_scaled.to_numpy(dtype=float))
    )
    if (
        np.isfinite(condition_number)
        and condition_number > CONDITION_NUMBER_WARNING_THRESHOLD
    ):
        warnings.warn(
            f"{station}-{horizon_h}h: olceklenmis exogenous matris kosul "
            f"sayisi yuksek ({condition_number:.3e}).",
            RuntimeWarning,
        )

    try:
        (
            selected_order,
            selected_seasonal,
            order_records,
            optimizer_attempts,
        ) = select_sarimax_order(
            split["y_train"], X_train_scaled, station, horizon_h
        )
    except SARIMAXConvergenceError as exc:
        exc.feature_audit = feature_audit
        raise

    print(
        f"  {station}-{horizon_h}h: final fit order={selected_order}, "
        f"seasonal={selected_seasonal}, train grid={len(split['train'])}, "
        f"test grid={len(split['test'])}"
    )
    try:
        result, final_optimizer, fit_warnings, final_attempts = (
            fit_sarimax_with_optimizer_fallback(
                y=split["y_train"],
                X=X_train_scaled,
                order=selected_order,
                seasonal_order=selected_seasonal,
                optimizer_methods=FINAL_FIT_OPTIMIZERS,
                maxiter=FINAL_FIT_MAXITER,
                station=station,
                horizon_h=horizon_h,
                stage="Final_fit",
            )
        )
    except SARIMAXConvergenceError as exc:
        exc.optimizer_attempts = optimizer_attempts + exc.optimizer_attempts
        exc.order_records = order_records
        exc.feature_audit = feature_audit
        raise
    optimizer_attempts.extend(final_attempts)
    fit_seconds = float(
        sum(
            float(row.get("Elapsed_seconds", 0.0) or 0.0)
            for row in final_attempts
        )
    )

    forecast_result = result.get_forecast(
        steps=len(X_test_scaled), exog=X_test_scaled
    )
    raw_prediction = np.asarray(forecast_result.predicted_mean, dtype=float)
    if not np.isfinite(raw_prediction).all():
        raise RuntimeError(
            f"{station}-{horizon_h}h: nihai tahminlerde sonlu olmayan deger var."
        )
    if CLIP_NEGATIVE_FORECASTS:
        prediction = np.clip(raw_prediction, 0.0, None)
    else:
        prediction = raw_prediction.copy()

    lower = np.full(len(prediction), np.nan)
    upper = np.full(len(prediction), np.nan)
    if SAVE_PREDICTION_INTERVALS:
        interval = np.asarray(
            forecast_result.conf_int(alpha=PREDICTION_INTERVAL_ALPHA), dtype=float
        )
        lower = interval[:, 0]
        upper = interval[:, 1]
        if CLIP_NEGATIVE_FORECASTS:
            lower = np.clip(lower, 0.0, None)

    test = split["test"].reset_index(drop=True)
    actual = split["y_test"].to_numpy(dtype=float)
    evaluable = np.isfinite(actual) & np.isfinite(prediction)
    metrics = calculate_metrics(actual[evaluable], prediction[evaluable])
    metric_record = {
        "Station": station,
        "Horizon_h": horizon_h,
        "Model": "SARIMAX",
        "Order": order_text(selected_order),
        "Seasonal_Order": seasonal_order_text(selected_seasonal),
        "Optimizer": final_optimizer,
        "Converged": True,
        **metrics,
    }

    coverage = np.nan
    mean_width = np.nan
    if SAVE_PREDICTION_INTERVALS and evaluable.any():
        coverage = float(
            np.mean(
                (actual[evaluable] >= lower[evaluable])
                & (actual[evaluable] <= upper[evaluable])
            )
        )
        mean_width = float(np.mean(upper[evaluable] - lower[evaluable]))

    predictions = pd.DataFrame(
        {
            "Station": station,
            "Horizon_h": horizon_h,
            "Issue_Time": test["Issue_Time"],
            "Target_Time": test["Target_Time"],
            "Actual_PM10": actual,
            "Current_PM10": test[TARGET].to_numpy(dtype=float),
            "Pred_SARIMAX_Raw": raw_prediction,
            "Pred_SARIMAX": prediction,
            "PI95_Lower": lower,
            "PI95_Upper": upper,
            "SARIMAX_Residual": actual - prediction,
            "Observed_Target_Available": np.isfinite(actual),
        }
    )
    # ML result sheets contain only genuinely observed future targets.
    predictions = predictions.loc[predictions["Observed_Target_Available"]].copy()

    converged = bool(result.mle_retvals.get("converged", False))
    if REQUIRE_CONVERGENCE and not converged:
        raise RuntimeError(
            f"{station}-{horizon_h}h: optimizer yardimcisi yakinmayan sonucu "
            "yanlislikla kabul etti."
        )
    residual_diagnostics = calculate_residual_diagnostics(
        result, selected_order, selected_seasonal
    )
    dropped_features = [
        str(row["Feature"])
        for row in feature_audit
        if row.get("Status") == "Dropped"
    ]
    diagnostics = {
        "Station": station,
        "Horizon_h": horizon_h,
        "Order": order_text(selected_order),
        "Seasonal_Order": seasonal_order_text(selected_seasonal),
        "Trend": SARIMAX_TREND,
        "Selection_criterion": ORDER_SELECTION_CRITERION,
        "Order_search_enabled": RUN_ORDER_SEARCH,
        "Order_search_tail_days": ORDER_SEARCH_TAIL_DAYS,
        "Train_grid_rows": len(split["train"]),
        "Train_observed_targets": int(split["y_train"].notna().sum()),
        "Train_missing_targets_state_space": int(split["y_train"].isna().sum()),
        "Test_grid_rows": len(split["test"]),
        "Test_observed_targets": int(evaluable.sum()),
        "Exogenous_feature_count": len(features),
        "Exogenous_features": ", ".join(features),
        "Dropped_exogenous_feature_count": len(dropped_features),
        "Dropped_exogenous_features": ", ".join(dropped_features),
        "Exogenous_design_condition_number": condition_number,
        "Condition_number_warning_threshold": (
            CONDITION_NUMBER_WARNING_THRESHOLD
        ),
        "Scaler_means_JSON": json.dumps(
            dict(zip(features, scaler.mean_)), default=float
        ),
        "Scaler_scales_JSON": json.dumps(
            dict(zip(features, scaler.scale_)), default=float
        ),
        "AIC": _safe_float(result.aic),
        "BIC": _safe_float(result.bic),
        "HQIC": _safe_float(result.hqic),
        "Log_Likelihood": _safe_float(result.llf),
        "Converged": converged,
        "Optimizer": final_optimizer,
        "Iterations": _iteration_count(result),
        "Fit_elapsed_seconds": fit_seconds,
        "Fit_warnings": fit_warnings,
        "Raw_negative_forecasts": int((raw_prediction < 0).sum()),
        "Forecasts_clipped_at_zero": CLIP_NEGATIVE_FORECASTS,
        "PI95_Coverage": coverage,
        "PI95_Mean_Width": mean_width,
        **residual_diagnostics,
        "Completed_at_UTC": datetime.now(timezone.utc).isoformat(),
    }

    coefficients = []
    parameter_names = list(result.param_names)
    parameter_values = np.asarray(result.params, dtype=float)
    for name, value in zip(parameter_names, parameter_values):
        coefficients.append(
            {
                "Station": station,
                "Horizon_h": horizon_h,
                "Parameter": name,
                "Estimate": float(value),
                "Order": order_text(selected_order),
                "Seasonal_Order": seasonal_order_text(selected_seasonal),
            }
        )

    summary = {
        "Station": station,
        "Horizon_h": horizon_h,
        "Test_start_target_time": test_start,
        "Train_issue_start": split["train"]["Issue_Time"].min(),
        "Train_issue_end": split["train"]["Issue_Time"].max(),
        "Train_target_start": split["train"]["Target_Time"].min(),
        "Train_target_end": split["train"]["Target_Time"].max(),
        "Test_issue_start": split["test"]["Issue_Time"].min(),
        "Test_issue_end": split["test"]["Issue_Time"].max(),
        "Test_target_start": split["test"]["Target_Time"].min(),
        "Test_target_end": split["test"]["Target_Time"].max(),
        "Train_rows_observed_target": int(split["y_train"].notna().sum()),
        "Test_rows_observed_target": int(evaluable.sum()),
        "Exogenous_feature_count": len(features),
        "Exogenous_features": ", ".join(features),
        "Dropped_exogenous_features": ", ".join(dropped_features),
        "Current_PM10_included": INCLUDE_CURRENT_PM10_AS_EXOG,
        "Future_targets_imputed": False,
        "Imputation_model": registry_row["Imputation_model"],
        "Imputation_parameters": json.dumps(
            registry_row["Imputation_parameters_parsed"],
            ensure_ascii=False,
            default=str,
        ),
        "SARIMAX_Order": order_text(selected_order),
        "SARIMAX_Seasonal_Order": seasonal_order_text(selected_seasonal),
        "SARIMAX_Optimizer": final_optimizer,
        "SARIMAX_Converged": converged,
        "Exogenous_scaler": type(scaler).__name__,
        "Evaluation_mode": "Fixed final-year holdout; direct horizon",
        "Combination_Status": "Completed",
    }

    print(
        f"    RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, "
        f"R2={metrics['R2']:.4f}, n={metrics['N_Test']}, "
        f"converged={converged}, fit={fit_seconds/60:.1f} min"
    )
    return {
        "metrics": [metric_record],
        "predictions": predictions.to_dict("records"),
        "order_search": order_records,
        "optimizer_attempts": optimizer_attempts,
        "diagnostics": [diagnostics],
        "coefficients": coefficients,
        "summaries": [summary],
        "feature_audit": feature_audit,
    }


# =============================================================================
# COMMON ISSUE-TIME REEVALUATION
# =============================================================================


def calculate_common_issue_time_outputs(
    prediction_records: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    if not prediction_records:
        return [], [], []
    frame = pd.DataFrame(prediction_records)
    frame["Issue_Time"] = pd.to_datetime(frame["Issue_Time"], errors="coerce")
    frame["Target_Time"] = pd.to_datetime(frame["Target_Time"], errors="coerce")
    frame["In_Common_Issue_Time"] = False

    common_metrics = []
    windows = []
    for station, station_frame in frame.groupby("Station", sort=False):
        horizon_groups = {
            int(horizon): group
            for horizon, group in station_frame.groupby("Horizon_h")
        }
        if set(horizon_groups) != set(FORECAST_HORIZONS_H):
            continue

        start = max(
            max(group["Issue_Time"].min() for group in horizon_groups.values()),
            station_frame["Target_Time"].min(),
        )
        end = min(
            group["Issue_Time"].max() for group in horizon_groups.values()
        )
        if start > end:
            raise RuntimeError(f"{station}: ortak issue-time araligi bos.")

        station_mask = (
            (frame["Station"] == station)
            & frame["Issue_Time"].between(start, end, inclusive="both")
        )
        frame.loc[station_mask, "In_Common_Issue_Time"] = True
        windows.append(
            {
                "Station": station,
                "Common_Issue_Start": start,
                "Common_Issue_End": end,
                "Identical_Issue_Times_Required": False,
                "Definition": (
                    "Common calendar interval; exact stamps may differ when "
                    "observed future PM10 is missing"
                ),
            }
        )

        for horizon_h in FORECAST_HORIZONS_H:
            subset = frame.loc[
                station_mask & (frame["Horizon_h"].astype(int) == horizon_h)
            ]
            metrics = calculate_metrics(
                subset["Actual_PM10"], subset["Pred_SARIMAX"]
            )
            common_metrics.append(
                {
                    "Station": station,
                    "Horizon_h": horizon_h,
                    "Model": "SARIMAX",
                    **metrics,
                    "Common_Issue_Start": start,
                    "Common_Issue_End": end,
                }
            )
    return common_metrics, windows, frame.to_dict("records")


# =============================================================================
# CHECKPOINTING AND EXCEL OUTPUT
# =============================================================================


STATE_SHEETS = OrderedDict(
    {
        "metrics": "SARIMAX_Metrics",
        "predictions": "Test_Predictions",
        "order_search": "Order_Search",
        "optimizer_attempts": "Optimizer_Attempts",
        "diagnostics": "Model_Diagnostics",
        "coefficients": "Model_Coefficients",
        "feature_audit": "Feature_Audit",
        "summaries": "Horizon_Summary",
        "stations": "Station_Imputation",
        "imputation_audit": "Imputation_Audit",
    }
)


def empty_state() -> dict[str, list[dict]]:
    return {key: [] for key in STATE_SHEETS}


def replace_combination_records(
    records: list[dict],
    station: str,
    horizon_h: int,
    new_records: list[dict],
) -> list[dict]:
    return [
        row
        for row in records
        if not (
            str(row.get("Station")) == station
            and pd.notna(row.get("Horizon_h"))
            and int(row.get("Horizon_h")) == int(horizon_h)
        )
    ] + new_records


def replace_station_records(
    records: list[dict], station: str, new_records: list[dict]
) -> list[dict]:
    return [row for row in records if str(row.get("Station")) != station] + new_records


def completed_combinations(state: dict[str, list[dict]]) -> set[tuple[str, int]]:
    metric_pairs = {
        (str(row.get("Station")), int(row.get("Horizon_h")))
        for row in state["metrics"]
        if str(row.get("Model")) == "SARIMAX" and pd.notna(row.get("Horizon_h"))
    }
    prediction_pairs = {
        (str(row.get("Station")), int(row.get("Horizon_h")))
        for row in state["predictions"]
        if pd.notna(row.get("Horizon_h"))
    }
    summary_pairs = {
        (str(row.get("Station")), int(row.get("Horizon_h")))
        for row in state["summaries"]
        if row.get("Combination_Status") == "Completed"
        and pd.notna(row.get("Horizon_h"))
    }
    return metric_pairs & prediction_pairs & summary_pairs


def file_sha256(file_path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_configuration() -> OrderedDict:
    source_files = OrderedDict()
    for station, path in FILES.items():
        source_files[station] = {
            "name": path.name,
            "size_bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
    source_files["ML_Diagnostics"] = {
        "name": ML_DIAGNOSTICS_FILE.name,
        "size_bytes": int(ML_DIAGNOSTICS_FILE.stat().st_size),
        "sha256": file_sha256(ML_DIAGNOSTICS_FILE),
    }
    return OrderedDict(
        {
            "Config_Schema_Version": CONFIG_SCHEMA_VERSION,
            "Problem": "Direct-horizon PM10 SARIMAX forecasting",
            "Forecast_Horizons_h": FORECAST_HORIZONS_H,
            "Outer_Split": "Final calendar year, chronological by target time",
            "Test_Years": TEST_YEARS,
            "Future_Target_Imputed": False,
            "Endogenous_Missing_Handling": (
                "Observed PM10 retained; missing endogenous hours handled by "
                "state-space filtering and excluded from metrics"
            ),
            "Imputation_Selection": (
                "Read selected model and parameters from ML diagnostics; "
                "RandomizedSearchCV not repeated"
            ),
            "Imputation_Application": (
                "Variable-specific models fitted before the earliest 24-h "
                "test issue-time boundary and applied forward"
            ),
            "Current_PM10_Exogenous": INCLUDE_CURRENT_PM10_AS_EXOG,
            "Raw_Features": BASE_FEATURE_CANDIDATES,
            "Cyclic_Features": CYCLIC_FEATURE_CANDIDATES,
            "Exogenous_Specification": EXOGENOUS_SPECIFICATION,
            "A_Priori_Redundant_Features_Dropped": (
                REDUNDANT_EXOGENOUS_FEATURES
            ),
            "Training_Only_Correlation_Filter": (
                DROP_HIGHLY_CORRELATED_EXOGENOUS
            ),
            "Maximum_Absolute_Exogenous_Correlation": (
                MAX_ABS_EXOGENOUS_CORRELATION
            ),
            "Minimum_Exogenous_Standard_Deviation": (
                MIN_EXOGENOUS_STANDARD_DEVIATION
            ),
            "Exogenous_Scaling": "StandardScaler fitted on each training block only",
            "SARIMAX_Formulation": (
                "Target PM10(t+h) with issue-time t exogenous features; "
                "separate model per station and horizon"
            ),
            "Evaluation_Mode": "Fixed final-year holdout; direct horizon",
            "Run_Order_Search": RUN_ORDER_SEARCH,
            "Order_Selection_Criterion": ORDER_SELECTION_CRITERION,
            "Order_Search_Tail_Days": ORDER_SEARCH_TAIL_DAYS,
            "Order_Search_Maxiter": ORDER_SEARCH_MAXITER,
            "Order_Search_Optimizers": ORDER_SEARCH_OPTIMIZERS,
            "Final_Fit_Maxiter": FINAL_FIT_MAXITER,
            "Final_Fit_Optimizers": FINAL_FIT_OPTIMIZERS,
            "Convergence_Required": REQUIRE_CONVERGENCE,
            "Invalid_Fixed_Fallback_Allowed": False,
            "SARIMAX_Candidates": SARIMAX_CANDIDATES,
            "Fixed_Order": FIXED_ORDER,
            "Fixed_Seasonal_Order": FIXED_SEASONAL_ORDER,
            "Trend": SARIMAX_TREND,
            "Enforce_Stationarity": ENFORCE_STATIONARITY,
            "Enforce_Invertibility": ENFORCE_INVERTIBILITY,
            "Concentrate_Scale": CONCENTRATE_SCALE,
            "Ljung_Box_Lags": LJUNG_BOX_LAGS,
            "Condition_Number_Warning_Threshold": (
                CONDITION_NUMBER_WARNING_THRESHOLD
            ),
            "Clip_Negative_Forecasts": CLIP_NEGATIVE_FORECASTS,
            "Prediction_Intervals": SAVE_PREDICTION_INTERVALS,
            "Prediction_Interval_Alpha": PREDICTION_INTERVAL_ALPHA,
            "Random_State": RANDOM_STATE,
            "Source_Files": source_files,
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
    rows = [
        {
            "Key": "Config_Fingerprint",
            "Value": configuration_fingerprint(configuration),
        }
    ]
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


def load_checkpoint(configuration: OrderedDict) -> dict[str, list[dict]]:
    state = empty_state()
    if not RESUME_FROM_CHECKPOINT or not OUTPUT_FILE.exists():
        return state
    with pd.ExcelFile(OUTPUT_FILE, engine="openpyxl") as workbook:
        if "Run_Configuration" not in workbook.sheet_names:
            raise RuntimeError("Checkpoint Run_Configuration sayfasini icermiyor.")
        saved = pd.read_excel(workbook, sheet_name="Run_Configuration")
        fingerprint_rows = saved.loc[
            saved["Key"] == "Config_Fingerprint", "Value"
        ]
        expected = configuration_fingerprint(configuration)
        if fingerprint_rows.empty or str(fingerprint_rows.iloc[0]) != expected:
            raise RuntimeError(
                "Mevcut SARIMAX sonuc dosyasi bu kod/veri konfigurasyonuyla "
                "uyumlu degil. OUTPUT_FILE adini degistirin veya eski dosyayi "
                "ayri yere tasiyin."
            )
        for key, sheet_name in STATE_SHEETS.items():
            if sheet_name in workbook.sheet_names:
                frame = pd.read_excel(workbook, sheet_name=sheet_name)
                state[key] = frame.to_dict("records")
    print(f"Uyumlu SARIMAX checkpoint yuklendi: {OUTPUT_FILE}")
    return state


def style_excel_writer(writer: pd.ExcelWriter, sheet_names: list[str]) -> None:
    header_fill = PatternFill("solid", fgColor="D9E5F1")
    for sheet_name in sheet_names:
        worksheet = writer.sheets[sheet_name]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cell in worksheet[1]:
            cell.font = Font(bold=True)
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")
        for column_cells in worksheet.columns:
            values = [
                str(cell.value) if cell.value is not None else ""
                for cell in column_cells
            ]
            max_length = min(max(len(value) for value in values) + 2, 70)
            worksheet.column_dimensions[column_cells[0].column_letter].width = max_length


def save_results(
    state: dict[str, list[dict]],
    configuration: OrderedDict,
    checkpoint: bool,
) -> None:
    common_metrics, common_windows, flagged_predictions = (
        calculate_common_issue_time_outputs(state["predictions"])
    )
    sheet_frames = OrderedDict(
        {
            "Run_Configuration": configuration_dataframe(configuration),
            "SARIMAX_Metrics": pd.DataFrame(state["metrics"]),
            "CommonIssueTime_Metrics": pd.DataFrame(common_metrics),
            "Common_Windows": pd.DataFrame(common_windows),
            "Test_Predictions": pd.DataFrame(flagged_predictions),
            "Horizon_Summary": pd.DataFrame(state["summaries"]),
            "Model_Diagnostics": pd.DataFrame(state["diagnostics"]),
            "Order_Search": pd.DataFrame(state["order_search"]),
            "Optimizer_Attempts": pd.DataFrame(state["optimizer_attempts"]),
            "Model_Coefficients": pd.DataFrame(state["coefficients"]),
            "Feature_Audit": pd.DataFrame(state["feature_audit"]),
            "Station_Imputation": pd.DataFrame(state["stations"]),
            "Imputation_Audit": pd.DataFrame(state["imputation_audit"]),
        }
    )
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_FILE.with_name(
        f"{OUTPUT_FILE.stem}.writing{OUTPUT_FILE.suffix}"
    )
    with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
        for sheet_name, frame in sheet_frames.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
        style_excel_writer(writer, list(sheet_frames))
    os.replace(temporary, OUTPUT_FILE)
    label = "Checkpoint" if checkpoint else "Final sonuc"
    print(f"{label} kaydedildi: {OUTPUT_FILE}")


def record_failed_combination(
    state: dict[str, list[dict]],
    station: str,
    horizon_h: int,
    exc: Exception,
) -> None:
    """Persist a failed combination without publishing invalid metrics."""
    error_text = f"{type(exc).__name__}: {exc}"
    order_records = getattr(exc, "order_records", [])
    optimizer_attempts = getattr(exc, "optimizer_attempts", [])
    feature_audit = getattr(exc, "feature_audit", [])

    state["metrics"] = replace_combination_records(
        state["metrics"], station, horizon_h, []
    )
    state["predictions"] = replace_combination_records(
        state["predictions"], station, horizon_h, []
    )
    state["coefficients"] = replace_combination_records(
        state["coefficients"], station, horizon_h, []
    )
    state["order_search"] = replace_combination_records(
        state["order_search"], station, horizon_h, order_records
    )
    state["optimizer_attempts"] = replace_combination_records(
        state["optimizer_attempts"], station, horizon_h, optimizer_attempts
    )
    state["feature_audit"] = replace_combination_records(
        state["feature_audit"], station, horizon_h, feature_audit
    )
    state["diagnostics"] = replace_combination_records(
        state["diagnostics"],
        station,
        horizon_h,
        [
            {
                "Station": station,
                "Horizon_h": horizon_h,
                "Converged": False,
                "Metrics_published": False,
                "Failure": error_text,
                "Completed_at_UTC": datetime.now(timezone.utc).isoformat(),
            }
        ],
    )
    state["summaries"] = replace_combination_records(
        state["summaries"],
        station,
        horizon_h,
        [
            {
                "Station": station,
                "Horizon_h": horizon_h,
                "Combination_Status": "Failed",
                "Failure": error_text,
                "Metrics_published": False,
            }
        ],
    )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    set_reproducibility()
    started = datetime.now()
    print(f"Calisma klasoru: {BASE_DIR}")
    print(f"ML diagnostics: {ML_DIAGNOSTICS_FILE}")
    print(f"SARIMAX sonucu: {OUTPUT_FILE}")
    print(f"Ufuklar: {FORECAST_HORIZONS_H} saat | test: son {TEST_YEARS} yil")

    registry = load_saved_imputation_registry()
    configuration = build_run_configuration()
    state = load_checkpoint(configuration)
    completed = completed_combinations(state)

    for station, file_path in FILES.items():
        missing_horizons = [
            horizon_h
            for horizon_h in FORECAST_HORIZONS_H
            if (station, horizon_h) not in completed
        ]
        if not missing_horizons:
            print(f"{station}: dort ufuk da checkpoint'te tamam; atlandi.")
            continue

        prepared, test_start, station_summary, audit = (
            prepare_station_with_saved_imputation(
                station, file_path, registry[station]
            )
        )
        state["stations"] = replace_station_records(
            state["stations"], station, [station_summary]
        )
        state["imputation_audit"] = replace_station_records(
            state["imputation_audit"], station, audit
        )

        for horizon_h in missing_horizons:
            print(f"\n=== {station} | {horizon_h} saat SARIMAX ===")
            try:
                output = evaluate_sarimax_horizon(
                    prepared,
                    test_start,
                    station,
                    horizon_h,
                    registry[station],
                )
            except Exception as exc:
                record_failed_combination(
                    state, station, horizon_h, exc
                )
                print(
                    f"  HATA: {station}-{horizon_h}h sonucu yayinlanmadi: "
                    f"{type(exc).__name__}: {exc}"
                )
                save_results(state, configuration, checkpoint=True)
                if CONTINUE_ON_COMBINATION_FAILURE:
                    continue
                raise
            for key in [
                "metrics",
                "predictions",
                "order_search",
                "optimizer_attempts",
                "diagnostics",
                "coefficients",
                "feature_audit",
                "summaries",
            ]:
                state[key] = replace_combination_records(
                    state[key], station, horizon_h, output[key]
                )
            completed.add((station, horizon_h))
            if SAVE_AFTER_EACH_HORIZON:
                save_results(state, configuration, checkpoint=True)

    save_results(state, configuration, checkpoint=False)
    elapsed = datetime.now() - started
    completed = completed_combinations(state)
    expected = {
        (station, horizon_h)
        for station in FILES
        for horizon_h in FORECAST_HORIZONS_H
    }
    failed = sorted(expected.difference(completed))
    print(f"\nTamamlandi. Toplam sure: {elapsed}")
    print(f"Gecerli ve yakinmis kombinasyon: {len(completed)}/{len(expected)}")
    if failed:
        print(
            "Gecerli metrik uretilmeyen kombinasyonlar: "
            + ", ".join(f"{station}-{horizon}h" for station, horizon in failed)
        )


if __name__ == "__main__":
    main()
