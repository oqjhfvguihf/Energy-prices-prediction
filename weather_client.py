import pandas as pd
import requests

# Prognozy sprzed doby (Open-Meteo "Previous Model Runs API"):
# zmienna *_previous_day1 = prognoza wydana ok. 24 h przed godziną docelową,
# czyli to, co model faktycznie mógłby znać dzień wcześniej.
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"  # faktyczna pogoda (tylko do porównań)

FORECAST_VARS = ["temperature_2m", "wind_speed_10m", "cloud_cover"]
ARCHIVE_VARS = ["temperature_2m", "wind_speed_10m", "cloud_cover", "shortwave_radiation"]


def fetch_weather_range(latitude, longitude, start_date, end_date, source="forecast"):
    """
    Pobiera godzinową pogodę dla całego zakresu dat jednym zapytaniem.
    Czas w strefie Europe/Warsaw (tak jak PSE).

    source="forecast": prognoza z poprzedniej doby (domyślnie, bez przecieku informacji
                       z przyszłości); kolumny mają te same nazwy co w archiwum.
    source="archive":  faktyczna pogoda (idealne warunki - zawyża wyniki na teście).

    Zwraca DataFrame z kolumną 'dtime' lub None przy błędzie.
    """
    if source == "forecast":
        url = PREVIOUS_RUNS_URL
        hourly = ",".join(f"{v}_previous_day1" for v in FORECAST_VARS)
    else:
        url = ARCHIVE_URL
        hourly = ",".join(ARCHIVE_VARS)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly,
        "timezone": "Europe/Warsaw",
    }
    try:
        r = requests.get(url, params=params, timeout=120)
        r.raise_for_status()
        data = r.json().get("hourly")
    except requests.exceptions.RequestException as e:
        detail = getattr(getattr(e, "response", None), "text", "")
        print(f"Błąd pobierania pogody: {e} {detail[:300]}")
        return None

    if not data:
        print("Brak danych pogodowych w odpowiedzi.")
        return None

    df = pd.DataFrame(data)
    df["dtime"] = pd.to_datetime(df.pop("time"))
    df.columns = [c.replace("_previous_day1", "") for c in df.columns]
    return df.drop_duplicates("dtime").reset_index(drop=True)


if __name__ == "__main__":
    w = fetch_weather_range(52.06, 19.25, "2026-08-01", "2026-08-03")
    if w is not None:
        print(w.head())
        print("Braki:", w.isna().sum().to_dict())