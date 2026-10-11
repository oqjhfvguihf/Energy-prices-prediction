import os
import time
import pandas as pd
 
from api_client import fetch_pse_data
from data_processor import process_pse_data
from weather_client import fetch_weather_range
 
PSE_FILE = "pse_hourly.csv"
WEATHER_FILE = "weather_hourly.csv"
LAT, LON = 52.06, 19.25
 
 
def _done_dates():
    """Doby, które już mamy w pliku (>=23 godziny) - można wznawiać po przerwie."""
    if not os.path.exists(PSE_FILE):
        return set()
    df = pd.read_csv(PSE_FILE, parse_dates=["dtime"])
    counts = df.groupby(df["dtime"].dt.date).size()
    return set(counts[counts >= 23].index)
 
 
def build_pse_dataset(start_date, end_date):
    done = _done_dates()
    failed = []
 
    for day in pd.date_range(start_date, end_date):
        if day.date() in done:
            continue
        day_str = day.strftime("%Y-%m-%d")
 
        try:
            raw_physics = fetch_pse_data("his-wlk-cal", day_str)
            raw_price = fetch_pse_data("csdac-pln", day_str)
            df = process_pse_data(raw_physics, raw_price)
        except Exception as e:  # jedna zła doba nie może zatrzymać całego pobierania
            print(f"BŁĄD przy {day_str}: {e}. Przechodzę dalej.")
            failed.append(day_str)
            continue
 
        if df is None or df.empty:
            failed.append(day_str)
            continue
 
        df.to_csv(PSE_FILE, mode="a", header=not os.path.exists(PSE_FILE), index=False)
        print(f"{day_str}: {len(df)} godzin")
        time.sleep(0.5)
 
    if failed:
        print(f"\nNie udało się pobrać {len(failed)} dób (uruchom skrypt ponownie, "
              f"wznowi tylko brakujące):\n{failed}")
 
 
def build_weather_dataset(start_date, end_date):
    w = fetch_weather_range(LAT, LON, start_date, end_date)
    if w is None:
        print("Pogoda nie została pobrana - uruchom ponownie.")
        return
    w.to_csv(WEATHER_FILE, index=False)
    print(f"Zapisano pogodę: {len(w)} godzin -> {WEATHER_FILE}")
 
 
if __name__ == "__main__":
    start = "2025-01-01"
    end = "2026-08-31"
    print(f"Pobieranie PSE od {start} do {end}...")
    build_pse_dataset(start, end)
    print("Pobieranie pogody...")
    build_weather_dataset(start, end)
    print("Gotowe.")