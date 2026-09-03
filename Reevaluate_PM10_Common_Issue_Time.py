"""
Mevcut PM10 gelecek-tahminlerini ortak Issue_Time araliginda yeniden degerlendirir.

Bu betik model egitmez ve imputasyon yapmaz. Daha once uretilen
``PM10_Future_1_3_6_24h_Forecast_Diagnostics.xlsx`` dosyasindaki
``Test_Predictions`` sayfasini okuyarak:

1. Her istasyonda 1, 3, 6 ve 24 saat ufuklarinin ortak Issue_Time araligini bulur.
2. Issue_Time'in ilk test Target_Time'inden once olmasina izin vermez.
3. Mevcut tahminleri bu ortak araliga filtreler.
4. RMSE, MAE ve R2 degerlerini yeniden hesaplar.
5. Filtreleme denetimini, eski-yeni metrik karsilastirmasini ve 600 dpi
   sekilleri ayri bir Excel dosyasina kaydeder.

Varsayilan kapsam PER_STATION'dir: her istasyon kendi ortak zaman araliginda
degerlendirilir. Tum istasyonlarda ayni takvim araligi istenirse
COMMON_WINDOW_SCOPE = "GLOBAL" yapilabilir.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# USER SETTINGS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "PM10_Future_1_3_6_24h_Forecast_Diagnostics.xlsx"
OUTPUT_FILE = BASE_DIR / "PM10_Future_1_3_6_24h_CommonIssueTime_Reevaluated.xlsx"
FIGURE_DIR = BASE_DIR / "PM10_Future_1_3_6_24h_CommonIssueTime_Figures"

PREDICTIONS_SHEET = "Test_Predictions"
ORIGINAL_METRICS_SHEET = "All_Metrics"
FORECAST_HORIZONS_H = (1, 3, 6, 24)

# "PER_STATION" onerilir. "GLOBAL", uc istasyona da tek bir ortak aralik uygular.
COMMON_WINDOW_SCOPE = "PER_STATION"

# False: ayni takvim araligi, fakat eksik hedefler nedeniyle N ufuklar arasinda
# farkli olabilir. True: yalnizca dort ufukta da bulunan ayni Issue_Time damgalari
# tutulur; ufuklar arasi tam eslestirilmis karsilastirma saglar fakat orneklemi
# daha fazla azaltabilir.
REQUIRE_IDENTICAL_ISSUE_TIMES = False

# Otomatik alt sinir, ilk test Target_Time'inden onceki issue zamanlarini eler.
# Gerekirse ISO biciminde acik bir sinir verilebilir: "2023-01-01 00:00:00".
FORCED_COMMON_START: str | None = None
FORCED_COMMON_END: str | None = None

GENERATE_FIGURES = True
FIGURE_DPI = 600


MODEL_NAME_MAP = OrderedDict(
    {
        "Pred_ExtraTrees": "ExtraTrees",
        "Pred_XGBoost": "XGBoost",
        "Pred_LightGBM": "LightGBM",
        "Pred_HistGB": "HistGB",
        "Pred_Persistence": "Persistence Benchmark",
        "Pred_Mean_Benchmark": "Mean Benchmark",
        "Pred_Median_Benchmark": "Median Benchmark",
        "Pred_Ridge_Benchmark": "Ridge Benchmark (Temporal CV)",
        "Pred_Average": "Ensemble Average",
        "Pred_Stacking": "Stacking Ensemble (RidgeCV-OOF)",
    }
)


# =============================================================================
# VALIDATION AND METRICS
# =============================================================================


def _timestamp_or_none(value: str | None) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = pd.to_datetime(value, errors="raise")
    if isinstance(parsed, pd.DatetimeIndex):
        raise ValueError("Tek bir tarih-zaman degeri girilmelidir.")
    return pd.Timestamp(parsed)


def load_predictions(path: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if not path.exists():
        raise FileNotFoundError(
            f"Girdi dosyasi bulunamadi: {path}\n"
            "INPUT_FILE yolunu eski sonuc dosyanizin konumuna gore duzeltin."
        )

    with pd.ExcelFile(path, engine="openpyxl") as workbook:
        if PREDICTIONS_SHEET not in workbook.sheet_names:
            raise KeyError(
                f"'{PREDICTIONS_SHEET}' sayfasi bulunamadi. Mevcut sayfalar: "
                f"{workbook.sheet_names}"
            )
        predictions = pd.read_excel(workbook, sheet_name=PREDICTIONS_SHEET)
        original_metrics = (
            pd.read_excel(workbook, sheet_name=ORIGINAL_METRICS_SHEET)
            if ORIGINAL_METRICS_SHEET in workbook.sheet_names
            else None
        )

    required = {
        "Station",
        "Horizon_h",
        "Issue_Time",
        "Target_Time",
        "Actual_PM10",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"Test_Predictions sutunlari eksik: {sorted(missing)}")

    predictions = predictions.copy()
    predictions["Station"] = predictions["Station"].astype(str).str.strip()
    predictions["Horizon_h"] = pd.to_numeric(
        predictions["Horizon_h"], errors="coerce"
    )
    predictions["Issue_Time"] = pd.to_datetime(
        predictions["Issue_Time"], errors="coerce"
    )
    predictions["Target_Time"] = pd.to_datetime(
        predictions["Target_Time"], errors="coerce"
    )
    predictions["Actual_PM10"] = pd.to_numeric(
        predictions["Actual_PM10"], errors="coerce"
    )

    invalid = predictions[
        ["Horizon_h", "Issue_Time", "Target_Time", "Actual_PM10"]
    ].isna().any(axis=1)
    if invalid.any():
        raise ValueError(
            f"Test_Predictions icinde {int(invalid.sum())} satirda gecersiz "
            "Horizon_h, Issue_Time, Target_Time veya Actual_PM10 var."
        )

    predictions["Horizon_h"] = predictions["Horizon_h"].astype(int)
    predictions = predictions.loc[
        predictions["Horizon_h"].isin(FORECAST_HORIZONS_H)
    ].copy()

    duplicated = predictions.duplicated(
        subset=["Station", "Horizon_h", "Issue_Time"], keep=False
    )
    if duplicated.any():
        examples = predictions.loc[
            duplicated, ["Station", "Horizon_h", "Issue_Time"]
        ].head(10)
        raise ValueError(
            "Ayni Station-Horizon_h-Issue_Time icin yinelenen tahminler var:\n"
            + examples.to_string(index=False)
        )

    expected_delta = pd.to_timedelta(predictions["Horizon_h"], unit="h")
    mismatch = predictions["Target_Time"] != predictions["Issue_Time"] + expected_delta
    if mismatch.any():
        raise ValueError(
            f"{int(mismatch.sum())} satirda Target_Time != Issue_Time + Horizon_h. "
            "Tahmin zamani tanimi tutarsiz."
        )

    prediction_columns = [
        column
        for column in predictions.columns
        if column.startswith("Pred_")
        and pd.to_numeric(predictions[column], errors="coerce").notna().any()
    ]
    if not prediction_columns:
        raise KeyError("Test_Predictions icinde Pred_ ile baslayan tahmin sutunu yok.")
    for column in prediction_columns:
        predictions[column] = pd.to_numeric(predictions[column], errors="coerce")

    return predictions.sort_values(
        ["Station", "Horizon_h", "Issue_Time"]
    ).reset_index(drop=True), original_metrics


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    paired = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if paired.empty:
        return {"N_Test": 0, "R2": np.nan, "MAE": np.nan, "RMSE": np.nan}

    y_true = paired["actual"].to_numpy(dtype=float)
    y_pred = paired["predicted"].to_numpy(dtype=float)
    r2 = (
        np.nan
        if len(y_true) < 2 or np.isclose(np.var(y_true), 0.0)
        else r2_score(y_true, y_pred)
    )
    return {
        "N_Test": len(y_true),
        "R2": r2,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def model_label(prediction_column: str) -> str:
    return MODEL_NAME_MAP.get(
        prediction_column,
        prediction_column.removeprefix("Pred_").replace("_", " "),
    )


# =============================================================================
# COMMON ISSUE-TIME WINDOW
# =============================================================================


def station_candidate_window(
    station_frame: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    present = set(station_frame["Horizon_h"].unique())
    missing = set(FORECAST_HORIZONS_H).difference(present)
    if missing:
        station = station_frame["Station"].iloc[0]
        raise ValueError(f"{station} icin eksik tahmin ufuklari: {sorted(missing)}")

    issue_minima = station_frame.groupby("Horizon_h")["Issue_Time"].min()
    issue_maxima = station_frame.groupby("Horizon_h")["Issue_Time"].max()

    # Ortak takvim araliginin baslangici, hem tum ufuklarin mevcut oldugu ilk
    # issue zamanini hem de ilk dis-test hedef zamanini gecmelidir. Bu ikinci
    # kosul, hedef-zamani tabanli eski bolmede test yilindan once kalan issue
    # zamanlarini disarida birakir.
    inferred_test_start = station_frame["Target_Time"].min()
    start = max(issue_minima.max(), inferred_test_start)
    end = issue_maxima.min()

    forced_start = _timestamp_or_none(FORCED_COMMON_START)
    forced_end = _timestamp_or_none(FORCED_COMMON_END)
    if forced_start is not None:
        start = max(start, forced_start)
    if forced_end is not None:
        end = min(end, forced_end)
    if start > end:
        raise ValueError(
            f"{station_frame['Station'].iloc[0]} icin ortak Issue_Time araligi bos: "
            f"{start} > {end}"
        )
    return start, end, inferred_test_start


def determine_windows(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for station, station_frame in predictions.groupby("Station", sort=True):
        start, end, inferred_test_start = station_candidate_window(station_frame)
        records.append(
            {
                "Station": station,
                "Candidate_Common_Issue_Start": start,
                "Candidate_Common_Issue_End": end,
                "Inferred_First_Test_Target_Time": inferred_test_start,
            }
        )
    windows = pd.DataFrame(records)

    scope = COMMON_WINDOW_SCOPE.strip().upper()
    if scope not in {"PER_STATION", "GLOBAL"}:
        raise ValueError("COMMON_WINDOW_SCOPE 'PER_STATION' veya 'GLOBAL' olmali.")
    if scope == "GLOBAL":
        global_start = windows["Candidate_Common_Issue_Start"].max()
        global_end = windows["Candidate_Common_Issue_End"].min()
        if global_start > global_end:
            raise ValueError(
                f"Tum istasyonlar icin ortak aralik bos: {global_start} > {global_end}"
            )
        windows["Common_Issue_Start"] = global_start
        windows["Common_Issue_End"] = global_end
    else:
        windows["Common_Issue_Start"] = windows[
            "Candidate_Common_Issue_Start"
        ]
        windows["Common_Issue_End"] = windows["Candidate_Common_Issue_End"]

    windows["Window_Scope"] = scope
    windows["Identical_Issue_Times_Required"] = REQUIRE_IDENTICAL_ISSUE_TIMES
    return windows


def filter_to_common_windows(
    predictions: pd.DataFrame, windows: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered_parts = []
    audit_records = []

    for _, window in windows.iterrows():
        station = window["Station"]
        start = pd.Timestamp(window["Common_Issue_Start"])
        end = pd.Timestamp(window["Common_Issue_End"])
        station_frame = predictions.loc[predictions["Station"] == station].copy()
        in_interval = station_frame["Issue_Time"].between(start, end, inclusive="both")
        interval_frame = station_frame.loc[in_interval].copy()

        shared_times: set[pd.Timestamp] | None = None
        if REQUIRE_IDENTICAL_ISSUE_TIMES:
            time_sets = [
                set(
                    interval_frame.loc[
                        interval_frame["Horizon_h"] == horizon_h, "Issue_Time"
                    ]
                )
                for horizon_h in FORECAST_HORIZONS_H
            ]
            shared_times = set.intersection(*time_sets)
            if not shared_times:
                raise ValueError(
                    f"{station} icin dort ufukta ortak tek bir Issue_Time bulunamadi."
                )
            interval_frame = interval_frame.loc[
                interval_frame["Issue_Time"].isin(shared_times)
            ].copy()

        for horizon_h in FORECAST_HORIZONS_H:
            original_h = station_frame.loc[station_frame["Horizon_h"] == horizon_h]
            kept_h = interval_frame.loc[interval_frame["Horizon_h"] == horizon_h]
            if kept_h.empty:
                raise ValueError(
                    f"{station}, {horizon_h} h icin ortak aralikta tahmin kalmadi."
                )
            audit_records.append(
                {
                    "Station": station,
                    "Horizon_h": horizon_h,
                    "Original_N": len(original_h),
                    "Kept_N": len(kept_h),
                    "Removed_N": len(original_h) - len(kept_h),
                    "Original_Issue_Start": original_h["Issue_Time"].min(),
                    "Original_Issue_End": original_h["Issue_Time"].max(),
                    "Common_Issue_Start": start,
                    "Common_Issue_End": end,
                    "Kept_Issue_Start": kept_h["Issue_Time"].min(),
                    "Kept_Issue_End": kept_h["Issue_Time"].max(),
                    "Kept_Target_Start": kept_h["Target_Time"].min(),
                    "Kept_Target_End": kept_h["Target_Time"].max(),
                    "Identical_Issue_Times": REQUIRE_IDENTICAL_ISSUE_TIMES,
                }
            )

        filtered_parts.append(interval_frame)

    filtered = pd.concat(filtered_parts, ignore_index=True).sort_values(
        ["Station", "Horizon_h", "Issue_Time"]
    )
    if "Pred_Stacking" in filtered.columns:
        filtered["Stacking_Residual"] = (
            filtered["Actual_PM10"] - filtered["Pred_Stacking"]
        )
    return filtered.reset_index(drop=True), pd.DataFrame(audit_records)


def recompute_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    prediction_columns = [
        column for column in predictions.columns if column.startswith("Pred_")
    ]
    records = []
    for (station, horizon_h), group in predictions.groupby(
        ["Station", "Horizon_h"], sort=True
    ):
        for column in prediction_columns:
            if group[column].notna().sum() == 0:
                continue
            records.append(
                {
                    "Station": station,
                    "Horizon_h": int(horizon_h),
                    "Model": model_label(column),
                    "Prediction_Column": column,
                    **calculate_metrics(group["Actual_PM10"], group[column]),
                    "Issue_Time_Start": group["Issue_Time"].min(),
                    "Issue_Time_End": group["Issue_Time"].max(),
                    "Target_Time_Start": group["Target_Time"].min(),
                    "Target_Time_End": group["Target_Time"].max(),
                }
            )
    return pd.DataFrame(records).sort_values(
        ["Station", "Horizon_h", "Model"]
    ).reset_index(drop=True)


def compare_with_original_metrics(
    common_metrics: pd.DataFrame, original_metrics: pd.DataFrame | None
) -> pd.DataFrame:
    if original_metrics is None:
        return pd.DataFrame()
    required = {"Station", "Horizon_h", "Model", "N_Test", "R2", "MAE", "RMSE"}
    if not required.issubset(original_metrics.columns):
        return pd.DataFrame()

    old = original_metrics[list(required)].copy()
    old["Horizon_h"] = pd.to_numeric(old["Horizon_h"], errors="coerce")
    old = old.dropna(subset=["Horizon_h"])
    old["Horizon_h"] = old["Horizon_h"].astype(int)
    old = old.rename(
        columns={
            "N_Test": "Original_N_Test",
            "R2": "Original_R2",
            "MAE": "Original_MAE",
            "RMSE": "Original_RMSE",
        }
    )
    new = common_metrics[
        ["Station", "Horizon_h", "Model", "N_Test", "R2", "MAE", "RMSE"]
    ].rename(
        columns={
            "N_Test": "Common_N_Test",
            "R2": "Common_R2",
            "MAE": "Common_MAE",
            "RMSE": "Common_RMSE",
        }
    )
    comparison = old.merge(new, on=["Station", "Horizon_h", "Model"], how="inner")
    comparison["Delta_R2_Common_minus_Original"] = (
        comparison["Common_R2"] - comparison["Original_R2"]
    )
    comparison["Delta_MAE_Common_minus_Original"] = (
        comparison["Common_MAE"] - comparison["Original_MAE"]
    )
    comparison["Delta_RMSE_Common_minus_Original"] = (
        comparison["Common_RMSE"] - comparison["Original_RMSE"]
    )
    return comparison.sort_values(["Station", "Horizon_h", "Model"])


# =============================================================================
# FIGURES
# =============================================================================


def save_figure(fig: plt.Figure, path: Path, rect=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_horizon_performance(metrics: pd.DataFrame) -> Path | None:
    selected_models = [
        "Stacking Ensemble (RidgeCV-OOF)",
        "Persistence Benchmark",
        "Ensemble Average",
    ]
    available = [model for model in selected_models if model in set(metrics["Model"])]
    if not available:
        return None

    colors = {
        "Stacking Ensemble (RidgeCV-OOF)": "#D62728",
        "Persistence Benchmark": "#6F6F6F",
        "Ensemble Average": "#1F77B4",
    }
    stations = sorted(metrics["Station"].unique())
    metric_specs = [("RMSE", "RMSE (µg/m³)"), ("MAE", "MAE (µg/m³)"), ("R2", "R²")]
    fig, axes = plt.subplots(len(stations), 3, figsize=(11.2, 8.2), squeeze=False)

    for row, station in enumerate(stations):
        for column, (metric, ylabel) in enumerate(metric_specs):
            ax = axes[row, column]
            for model in available:
                subset = metrics.loc[
                    (metrics["Station"] == station) & (metrics["Model"] == model)
                ].sort_values("Horizon_h")
                ax.plot(
                    subset["Horizon_h"],
                    subset[metric],
                    marker="o",
                    linewidth=1.4,
                    markersize=4,
                    color=colors[model],
                    linestyle="--" if model == "Persistence Benchmark" else "-",
                    label=model,
                )
            ax.set_xticks(FORECAST_HORIZONS_H)
            ax.set_xlabel("Forecast horizon (h)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{station} — {metric}")
            ax.grid(True, linestyle="--", alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(available),
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle("Performance on the common issue-time test interval", y=0.995)
    path = FIGURE_DIR / "CommonIssueTime_Horizon_Performance_3x3.png"
    save_figure(fig, path, rect=[0, 0.065, 1, 0.975])
    return path


def plot_daily_forecasts(predictions: pd.DataFrame) -> Path | None:
    if not {"Pred_Stacking", "Pred_Persistence"}.issubset(predictions.columns):
        return None
    stations = sorted(predictions["Station"].unique())
    fig, axes = plt.subplots(
        len(stations), len(FORECAST_HORIZONS_H), figsize=(12.2, 8.4), squeeze=False
    )
    for row, station in enumerate(stations):
        for column, horizon_h in enumerate(FORECAST_HORIZONS_H):
            ax = axes[row, column]
            subset = predictions.loc[
                (predictions["Station"] == station)
                & (predictions["Horizon_h"] == horizon_h),
                ["Target_Time", "Actual_PM10", "Pred_Stacking", "Pred_Persistence"],
            ].drop_duplicates(subset=["Target_Time"])
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
                color="#6F6F6F",
                linewidth=0.9,
                linestyle="--",
            )
            locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax.set_title(f"{station} — {horizon_h} h")
            ax.grid(True, linestyle="--", alpha=0.25)
            if row == len(stations) - 1:
                ax.set_xlabel("Target date")
            if column == 0:
                ax.set_ylabel("Daily PM10 (µg/m³)")

    handles = [
        plt.Line2D([0], [0], color="black", linewidth=1.2, label="Measured"),
        plt.Line2D([0], [0], color="#D62728", linewidth=1.2, label="Stacking"),
        plt.Line2D(
            [0], [0], color="#6F6F6F", linewidth=1.0, linestyle="--", label="Persistence"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        "Measured and forecast daily PM10 on the common issue-time interval",
        y=0.995,
    )
    path = FIGURE_DIR / "CommonIssueTime_Daily_Measured_vs_Forecast_3x4.png"
    save_figure(fig, path, rect=[0, 0.06, 1, 0.975])
    return path


# =============================================================================
# EXCEL OUTPUT
# =============================================================================


def autosize_excel_columns(writer: pd.ExcelWriter) -> None:
    for worksheet in writer.sheets.values():
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for column_cells in worksheet.columns:
            values = [str(cell.value) if cell.value is not None else "" for cell in column_cells]
            width = min(max(len(value) for value in values) + 2, 55)
            worksheet.column_dimensions[column_cells[0].column_letter].width = width


def configuration_frame() -> pd.DataFrame:
    configuration = OrderedDict(
        {
            "Input_File": str(INPUT_FILE),
            "Input_Predictions_Sheet": PREDICTIONS_SHEET,
            "Output_File": str(OUTPUT_FILE),
            "Forecast_Horizons_h": list(FORECAST_HORIZONS_H),
            "Common_Window_Scope": COMMON_WINDOW_SCOPE,
            "Require_Identical_Issue_Times": REQUIRE_IDENTICAL_ISSUE_TIMES,
            "Forced_Common_Start": FORCED_COMMON_START,
            "Forced_Common_End": FORCED_COMMON_END,
            "Retraining_Performed": False,
            "Imputation_Performed": False,
            "Existing_Predictions_Modified": False,
            "Metrics_Recomputed": True,
            "Figure_DPI": FIGURE_DPI,
        }
    )
    return pd.DataFrame(
        {
            "Key": configuration.keys(),
            "Value": [
                json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict, tuple))
                else value
                for value in configuration.values()
            ],
        }
    )


def save_outputs(
    windows: pd.DataFrame,
    audit: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    figure_paths: list[Path],
) -> None:
    stacking_only = metrics.loc[
        metrics["Model"] == "Stacking Ensemble (RidgeCV-OOF)"
    ].copy()
    figure_manifest = pd.DataFrame(
        {
            "Figure_Type": [path.stem for path in figure_paths],
            "File": [str(path) for path in figure_paths],
            "DPI": FIGURE_DPI,
        }
    )
    sheets = OrderedDict(
        {
            "Reeval_Configuration": configuration_frame(),
            "Common_Windows": windows,
            "Filter_Audit": audit,
            "Stacking_Only_Common": stacking_only,
            "All_Metrics_Common": metrics,
            "Old_vs_Common_Metrics": comparison,
            "Test_Predictions_Common": predictions,
            "Figure_Manifest": figure_manifest,
        }
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_FILE.with_name(f"{OUTPUT_FILE.stem}.writing{OUTPUT_FILE.suffix}")
    with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
        autosize_excel_columns(writer)
    temporary.replace(OUTPUT_FILE)


def main() -> None:
    predictions, original_metrics = load_predictions(INPUT_FILE)
    windows = determine_windows(predictions)
    filtered, audit = filter_to_common_windows(predictions, windows)
    metrics = recompute_metrics(filtered)
    comparison = compare_with_original_metrics(metrics, original_metrics)

    figure_paths: list[Path] = []
    if GENERATE_FIGURES:
        for path in (
            plot_horizon_performance(metrics),
            plot_daily_forecasts(filtered),
        ):
            if path is not None:
                figure_paths.append(path)

    save_outputs(windows, audit, filtered, metrics, comparison, figure_paths)

    print("\nOrtak Issue_Time araliklari:")
    print(
        windows[
            ["Station", "Common_Issue_Start", "Common_Issue_End", "Window_Scope"]
        ].to_string(index=False)
    )
    print("\nTutulan satir sayilari:")
    print(audit[["Station", "Horizon_h", "Original_N", "Kept_N", "Removed_N"]].to_string(index=False))
    print(f"\nYeniden degerlendirme tamamlandi: {OUTPUT_FILE}")
    if figure_paths:
        print(f"Sekil klasoru: {FIGURE_DIR}")
    print(
        "Not: Bu islem dis-test ufuklarini ortak Issue_Time araliginda yeniden "
        "degerlendirir; eski egitim/icapraz-dogrulama akisinin ic sinirlamalarini "
        "geriye donuk olarak degistirmez."
    )


if __name__ == "__main__":
    main()
