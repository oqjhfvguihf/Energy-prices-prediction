import pandas as pd
 
PHYSICS_COLS = ["demand", "wi", "pv", "swm_p"]
PRICE_COLS = ["csdac_pln"]
 
 
def _to_hourly(records, cols):
    """
    Zamienia rekordy z API na średnie godzinowe.
 
    PSE podaje 'dtime' jako KONIEC okresu (godzina lub kwadrans). Krok
    wykrywamy z danych (mediana różnic), odejmujemy go, żeby dostać początek
    okresu, i uśredniamy do pełnej godziny. Dzięki temu nie gubimy 3 z 4
    kwadransów i cena po przejściu na 15 minut jest średnią godzinową.
    Brakujące kolumny zostają jako NaN (nie zera!).
    """
    df = pd.DataFrame(records)
    if df.empty or "dtime" not in df.columns:
        return None
    for c in cols:
        if c not in df.columns:
            df[c] = float("nan")
 
    # Doba jesiennej zmiany czasu: PSE oznacza powtórzoną godzinę jako "02a:15:00".
    # Zdejmujemy literę, więc obie godziny 02:xx trafiają do jednego kosza i są
    # uśredniane (siatka czasu lokalnego nie ma jak pomieścić dwóch godzin 02:00).
    df["dtime"] = pd.to_datetime(
        df["dtime"].astype(str).str.replace(r"(?<=\d{2})[a-zA-Z](?=:)", "", regex=True)
    )
    df = df[["dtime"] + cols].sort_values("dtime")
 
    step = df["dtime"].drop_duplicates().diff().median()
    if pd.isna(step):
        step = pd.Timedelta(hours=1)
 
    df["hour_start"] = (df["dtime"] - step).dt.floor("h")
    out = df.groupby("hour_start")[cols].mean()
    return out.rename_axis("dtime").reset_index()
 
 
def process_pse_data(raw_physics, raw_price):
    """
    Zwraca DataFrame godzinowy: dtime, demand, wi, pv, swm_p, csdac_pln.
    Pogoda jest pobierana osobno (weather_client) i łączona w prepare_model_data.py.
    """
    if not raw_physics or not raw_price or "value" not in raw_physics or "value" not in raw_price:
        return None
 
    phys = _to_hourly(raw_physics["value"], PHYSICS_COLS)
    price = _to_hourly(raw_price["value"], PRICE_COLS)
    if phys is None or price is None:
        return None
 
    df = pd.merge(phys, price, on="dtime", how="inner")
    # Bez fillna(0): brak ceny / zapotrzebowania to brak danych, nie wartość 0.
    return df.dropna(subset=["csdac_pln", "demand"]).reset_index(drop=True)