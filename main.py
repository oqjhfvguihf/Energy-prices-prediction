import datetime

from api_client import fetch_pse_data
from data_processor import process_pse_data
from weather_client import fetch_weather_range


def main():
    lat, lon = 52.06, 19.25
    report_date = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Pobieranie danych dla {report_date}...")

    raw_physics = fetch_pse_data("his-wlk-cal", report_date)
    raw_price = fetch_pse_data("csdac-pln", report_date)
    df = process_pse_data(raw_physics, raw_price)
    if df is None or df.empty:
        print("Proces przerwany: brak danych PSE.")
        return

    # Archiwum pogodowe ma kilkudniowe opóźnienie - dla wczoraj pogody może jeszcze nie być.
    weather = fetch_weather_range(lat, lon, report_date, report_date)
    if weather is not None:
        df = df.merge(weather, on="dtime", how="left")

    file_name = f"pse_report_{report_date}.csv"
    df.to_csv(file_name, index=False)
    print(df.head(3))


if __name__ == "__main__":
    main()
