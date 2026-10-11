import time
import requests
from urllib.parse import urljoin

BASE_URL = "https://api.raporty.pse.pl/api/"


def fetch_pse_data(endpoint, report_date, retries=3):
    """
    Pobiera dane z API PSE dla endpointu i doby handlowej (business_date).
    Ponawia próbę przy błędach sieci / 429 / 5xx. Obsługuje paginację OData.
    Zwraca {'value': [...]} lub None, gdy się nie udało.
    """
    url = urljoin(BASE_URL, endpoint)
    params = {"$filter": f"business_date eq '{report_date}'"}

    for attempt in range(1, retries + 1):
        try:
            records = []
            next_url, next_params = url, params
            while next_url:
                r = requests.get(next_url, params=next_params, timeout=30)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.exceptions.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                data = r.json()
                records.extend(data.get("value", []))
                next_url = data.get("@odata.nextLink")  # pełny URL, bez dodatkowych params
                next_params = None
            return {"value": records}
        except requests.exceptions.RequestException as e:
            print(f"[{endpoint} {report_date}] próba {attempt}/{retries} nieudana: {e}")
            time.sleep(2 * attempt)
    return None


if __name__ == "__main__":
    test = fetch_pse_data("his-wlk-cal", "2025-01-01")
    if test and test["value"]:
        print(len(test["value"]), "rekordów; klucze:", list(test["value"][0].keys()))
