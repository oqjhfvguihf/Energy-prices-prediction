"""
Pobiera dzienne ceny gazu (TTF) i proxy dla CO2 z Yahoo Finance.
Zastępuje stary fetch_co2.py (KE=F to pszenica).

UWAGA: 'KRBN' to ETF na uprawnienia do emisji - tylko przybliżenie ceny EUA.
Jeśli znajdziesz lepszy ticker/źródło EUA, podmień go poniżej.
Wynik: ttf.csv i co2.csv w formacie 'Data,Zamkniecie' (czyta je prepare_model_data.py).
"""
import pandas as pd
import yfinance as yf

START, END = "2024-12-01", "2026-09-05"
TICKERS = {"ttf.csv": "TTF=F", "co2.csv": "KRBN"}

for out_file, ticker in TICKERS.items():
    df = yf.download(ticker, start=START, end=END, progress=False)
    if df.empty:
        print(f"Brak danych dla {ticker}")
        continue
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()[["Date", "Close"]]
    df.columns = ["Data", "Zamkniecie"]
    df.to_csv(out_file, index=False)
    print(f"{ticker}: {len(df)} wierszy -> {out_file}")
