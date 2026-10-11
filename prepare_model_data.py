import os
import pandas as pd
import holidays

PSE_FILE = "pse_hourly.csv"
WEATHER_FILE = "weather_hourly.csv"
OUT_FILE = "model_ready_dataset.csv"
# Opcjonalne serie dzienne (z fetch_fuels.py): plik -> nazwa cechy
DAILY_SERIES = {"ttf.csv": "ttf_prev_close", "co2.csv": "co2_prev_close"}

# --- 1. Wczytanie i pełna siatka godzinowa ---
df = pd.read_csv(PSE_FILE, parse_dates=["dtime"])
df = df.drop_duplicates("dtime", keep="last").set_index("dtime").sort_index()

# Siatka bez dziur: brakujące godziny stają się NaN, więc shift(24) zawsze
# oznacza dokładnie "24 godziny wcześniej" (a wiersze po dziurze wypadną w dropna).
full_index = pd.date_range(df.index.min(), df.index.max(), freq="h", name="dtime")
df = df.reindex(full_index)

if os.path.exists(WEATHER_FILE):
    weather = pd.read_csv(WEATHER_FILE, parse_dates=["dtime"]).drop_duplicates("dtime").set_index("dtime")
    df = df.join(weather, how="left")
    weather_cols = list(weather.columns)
else:
    print("Uwaga: brak pliku pogody.")
    weather_cols = []

# --- 2. Cechy kalendarzowe ---
years = sorted(set(df.index.year))
pl_holidays = holidays.Poland(years=years)
df["hour"] = df.index.hour
df["day_of_week"] = df.index.dayofweek
df["month"] = df.index.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["is_holiday"] = pd.Series(df.index.date, index=df.index).map(lambda d: d in pl_holidays).astype(int)
df["is_non_working_day"] = ((df["is_weekend"] == 1) | (df["is_holiday"] == 1)).astype(int)

# --- 3. Cechy opóźnione (tylko to, co znamy przed aukcją na dobę D) ---
df["residual_demand"] = df["demand"] - df["wi"] - df["pv"]
for col in ["demand", "wi", "pv", "swm_p", "residual_demand"]:
    df[f"{col}_lag_24h"] = df[col].shift(24)

df["price_lag_24h"] = df["csdac_pln"].shift(24)
df["price_lag_48h"] = df["csdac_pln"].shift(48)
df["price_lag_week"] = df["csdac_pln"].shift(168)

# Statystyki całej poprzedniej doby (ceny doby D-1 są znane przed aukcją D)
day = df.index.normalize()
daily = df["csdac_pln"].groupby(day).agg(["mean", "min", "max", "std"])
prev7 = daily["mean"].rolling(7, min_periods=7).mean().shift(1)
daily = daily.shift(1)
for stat in ["mean", "min", "max", "std"]:
    df[f"prev_day_price_{stat}"] = daily[stat].reindex(day).values
df["prev7d_price_mean"] = prev7.reindex(day).values

# --- 4. Opcjonalnie: gaz / CO2 (znamy tylko wczorajsze zamknięcie) ---
for path, name in DAILY_SERIES.items():
    if not os.path.exists(path):
        continue
    s = pd.read_csv(path, parse_dates=["Data"]).drop_duplicates("Data").set_index("Data")["Zamkniecie"]
    days = pd.date_range(day.min() - pd.Timedelta(days=10), day.max(), freq="D")
    s = s.reindex(days).ffill().shift(1)
    df[name] = s.reindex(day).values
    print(f"Dodano cechę {name}")

# --- 5. Usunięcie surowych kolumn z "przyszłości" i pustych wierszy ---
df = df.drop(columns=["demand", "wi", "pv", "swm_p", "residual_demand"])
before = len(df)
df = df.dropna()
print(f"Usunięto {before - len(df)} wierszy z brakami (dziury w danych / początek zbioru).")
print(f"Zbiór: {df.shape}, zakres {df.index.min()} -> {df.index.max()}")
print(f"Cechy pogodowe użyte w zbiorze: {weather_cols}")

df.reset_index().to_csv(OUT_FILE, index=False)
