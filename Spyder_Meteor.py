# -*- coding: utf-8 -*-
"""İzmir istasyonları için saatlik tarihsel meteoroloji verilerini indirir.

Open-Meteo zaman damgaları ``Europe/Istanbul`` yerel saatinde istenir. Her
istasyonun verisi tek bir Excel çalışma kitabında ayrı bir sekmeye yazılır.
"""

from pathlib import Path

import pandas as pd
import requests


API_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEZONE = "Europe/Istanbul"
OUTPUT_FILE = Path("./Izmir_Meteor_Eslenik_Saatler.xlsx")

# Başlangıç tarihleri, istasyonların mevcut hava-kirliliği veri dönemleriyle
# eşleşecek biçimde ayrı tutulmuştur. Gerekirse yalnızca bu sözlüğü güncelleyin.
STATIONS = {
    "Menemen": {
        "latitude": 38.610,
        "longitude": 27.070,
        "start_date": "2022-01-24",
        "end_date": "2026-06-16",
    },
    "Aliaga": {
        "latitude": 38.799,
        "longitude": 26.972,
        "start_date": "2022-01-24",
        "end_date": "2026-06-16",
    },
    "Bornova": {
        "latitude": 38.470,
        "longitude": 27.220,
        "start_date": "2022-01-24",
        "end_date": "2026-06-16",
    },
}

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "shortwave_radiation",
]


def download_station(station_name, station):
    """Bir istasyonun saatlik verisini yerel saatle indirip doğrular."""
    params = {
        "latitude": station["latitude"],
        "longitude": station["longitude"],
        "start_date": station["start_date"],
        "end_date": station["end_date"],
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": TIMEZONE,
        "timeformat": "iso8601",
    }

    response = requests.get(API_URL, params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()

    if payload.get("error"):
        raise RuntimeError(
            f"{station_name} için Open-Meteo hatası: "
            f"{payload.get('reason', 'Bilinmeyen hata')}"
        )
    if "hourly" not in payload or "time" not in payload["hourly"]:
        raise RuntimeError(f"{station_name} için saatlik veri alınamadı.")

    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"], errors="raise")
    df = df.rename(columns={"time": "Tarih"})
    df = df.sort_values("Tarih", kind="mergesort")

    if df["Tarih"].duplicated().any():
        duplicate_count = int(df["Tarih"].duplicated().sum())
        raise ValueError(
            f"{station_name} verisinde {duplicate_count} yinelenen saat bulundu."
        )

    # API'den eksik bir saat dönerse satırı NaN değerlerle açıkça gösterir.
    # Böylece hava-kirliliği tablosuyla Tarih üzerinden yapılan eşleştirmede
    # saatler kaymaz ve eksikler sonradan güvenli biçimde ele alınabilir.
    expected_hours = pd.date_range(
        start=f"{station['start_date']} 00:00:00",
        end=f"{station['end_date']} 23:00:00",
        freq="h",
    )
    df = (
        df.set_index("Tarih")
        .reindex(expected_hours)
        .rename_axis("Tarih")
        .reset_index()
    )

    # Excel saat dilimi bilgisi taşıyamadığı için zamanlar yerel saat olarak
    # timezone-naive yazılır; kullanılan bölge ayrıca sütunda belirtilir.
    df.insert(1, "Timezone", TIMEZONE)
    df.insert(2, "Station", station_name)
    return df


def main():
    station_frames = {}

    for station_name, station in STATIONS.items():
        print(f"{station_name} verisi indiriliyor...")
        station_frames[station_name] = download_station(station_name, station)
        print(f"  {len(station_frames[station_name]):,} saat hazırlandı.")

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        for station_name in ("Menemen", "Aliaga", "Bornova"):
            station_frames[station_name].to_excel(
                writer,
                sheet_name=station_name,
                index=False,
            )

    print(f"Tamamlandı: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()