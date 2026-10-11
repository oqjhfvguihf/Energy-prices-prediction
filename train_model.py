import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
 
warnings.filterwarnings("ignore")
 
PEAK_HOURS = [18, 19, 20, 21]
TARGET = "csdac_pln"
 
df = pd.read_csv("model_ready_dataset.csv", parse_dates=["dtime"]).sort_values("dtime").reset_index(drop=True)
feature_columns = [c for c in df.columns if c not in ("dtime", TARGET)]
X, y = df[feature_columns], df[TARGET]
print(f"Cechy ({len(feature_columns)}): {feature_columns}")
 
# Podział chronologiczny 80/20 (bez tasowania)
cut = int(len(df) * 0.8)
X_train, X_test, y_train, y_test = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
print(f"Trening: {len(X_train)} wierszy ({df.dtime.iloc[0]:%Y-%m-%d} -> {df.dtime.iloc[cut-1]:%Y-%m-%d})")
print(f"Test:    {len(X_test)} wierszy ({df.dtime.iloc[cut]:%Y-%m-%d} -> {df.dtime.iloc[-1]:%Y-%m-%d})")
 
 
def metrics(y_true, y_pred, hours):
    err = np.asarray(y_true) - np.asarray(y_pred)
    peak = np.asarray(hours.isin(PEAK_HOURS))
    return {
        "MAE": np.abs(err).mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "MAE_peak": np.abs(err[peak]).mean(),
        "MAE_offpeak": np.abs(err[~peak]).mean(),
    }
 
 
# --- Baseline'y (naiwne prognozy) ---
naive = {
    "naive lag 24h": X_test["price_lag_24h"],
    "naive lag tydzień": X_test["price_lag_week"],
    "naive średnia(24h, tydzień)": (X_test["price_lag_24h"] + X_test["price_lag_week"]) / 2,
}
naive_results = {n: metrics(y_test, p, X_test["hour"]) for n, p in naive.items()}
best_naive_name = min(naive_results, key=lambda n: naive_results[n]["MAE"])
naive_mae = naive_results[best_naive_name]["MAE"]
 
# --- Modele: wszędzie strata L1 (MAE), poza Ridge, który jej nie wspiera ---
models_and_params = {
    "Ridge": (
        make_pipeline(StandardScaler(), Ridge(random_state=42)),
        {"ridge__alpha": [0.1, 1.0, 10.0, 100.0]},
    ),
    "HistGB": (
        HistGradientBoostingRegressor(loss="absolute_error", random_state=42),
        {"max_depth": [4, 6], "learning_rate": [0.03, 0.08], "max_iter": [300, 500]},
    ),
}
try:
    from lightgbm import LGBMRegressor
    models_and_params["LightGBM"] = (
        LGBMRegressor(objective="l1", random_state=42, verbose=-1),
        {"max_depth": [5, 7], "num_leaves": [31, 63], "learning_rate": [0.01, 0.05], "n_estimators": [300, 500]},
    )
except ImportError:
    print("(pomijam LightGBM - niezainstalowany)")
try:
    from xgboost import XGBRegressor
    models_and_params["XGBoost"] = (
        XGBRegressor(random_state=42, objective="reg:absoluteerror"),
        {"max_depth": [4, 6], "learning_rate": [0.01, 0.05], "n_estimators": [300, 500], "subsample": [0.8, 1.0]},
    )
except ImportError:
    print("(pomijam XGBoost - niezainstalowany)")
try:
    from catboost import CatBoostRegressor
    models_and_params["CatBoost"] = (
        CatBoostRegressor(random_state=42, silent=True, loss_function="MAE"),
        {"depth": [4, 6], "learning_rate": [0.01, 0.05], "iterations": [300, 500], "l2_leaf_reg": [3, 5]},
    )
except ImportError:
    print("(pomijam CatBoost - niezainstalowany)")
 
tscv = TimeSeriesSplit(n_splits=3)
results, fitted, test_preds = {}, {}, {}
for name, (model, params) in models_and_params.items():
    grid = GridSearchCV(model, params, scoring="neg_mean_absolute_error", cv=tscv, n_jobs=-1)
    grid.fit(X_train, y_train)
    test_preds[name] = grid.best_estimator_.predict(X_test)
    results[name] = metrics(y_test, test_preds[name], X_test["hour"])
    results[name]["CV_MAE"] = -grid.best_score_
    fitted[name] = grid
    print(f"Zakończono: {name}  (najlepsze parametry: {grid.best_params_})")
 
# --- Uśrednienie modeli (bez Ridge, który nie wnosi nic ponad baseline) ---
ENSEMBLE_NAME = "Ensemble (średnia)"
members = [n for n in fitted if n != "Ridge"]
if len(members) >= 2:
    test_preds[ENSEMBLE_NAME] = np.mean([test_preds[n] for n in members], axis=0)
    results[ENSEMBLE_NAME] = metrics(y_test, test_preds[ENSEMBLE_NAME], X_test["hour"])
    results[ENSEMBLE_NAME]["CV_MAE"] = float("nan")
    print(f"\nEnsemble = średnia z: {members}")
else:
    ENSEMBLE_NAME = None
    print("\nZa mało modeli boostingowych do uśredniania (zainstaluj LightGBM/XGBoost/CatBoost).")
 
# --- Raport na zbiorze testowym ---
print("\n=== WYNIKI NA ZBIORZE TESTOWYM (PLN/MWh) ===")
print(f"{'':30s}{'MAE':>8s}{'RMSE':>8s}{'MAE/naive':>11s}{'peak 18-21':>12s}{'poza szczytem':>15s}{'CV MAE':>9s}")
for name, m in naive_results.items():
    print(f"{name:30s}{m['MAE']:8.1f}{m['RMSE']:8.1f}{m['MAE']/naive_mae:11.2f}{m['MAE_peak']:12.1f}{m['MAE_offpeak']:15.1f}{'':>9s}")
print("-" * 93)
for name, m in sorted(results.items(), key=lambda kv: kv[1]["MAE"]):
    print(f"{name:30s}{m['MAE']:8.1f}{m['RMSE']:8.1f}{m['MAE']/naive_mae:11.2f}{m['MAE_peak']:12.1f}{m['MAE_offpeak']:15.1f}{m['CV_MAE']:9.1f}")
print(f"(MAE/naive liczone względem najlepszego baseline'u: {best_naive_name}; <1 = model coś dodaje)")
 
# --- Walk-forward dla WSZYSTKICH modeli i ensemble (4 ostatnie bloki po ok. 45 dni) ---
# Uwaga: hiperparametry były dobrane na pierwszych 80% danych, więc bloki 1-3 częściowo
# nakładają się na ten okres - to lekko zawyża wyniki, ale jednakowo dla wszystkich modeli.
print("\n=== WALK-FORWARD: MAE w kolejnych blokach (rozszerzające się okno) ===")
wf = TimeSeriesSplit(n_splits=4, test_size=24 * 45)
wf_mae, periods = {}, []
for tr, te in wf.split(X):
    periods.append(f"{df.dtime.iloc[te[0]]:%m-%d}..{df.dtime.iloc[te[-1]]:%m-%d}")
    fold_preds = {}
    for name in fitted:
        mdl = clone(fitted[name].best_estimator_).fit(X.iloc[tr], y.iloc[tr])
        fold_preds[name] = mdl.predict(X.iloc[te])
    if ENSEMBLE_NAME:
        fold_preds[ENSEMBLE_NAME] = np.mean([fold_preds[n] for n in members], axis=0)
    fold_preds["naive lag 24h"] = X.iloc[te]["price_lag_24h"].values
    fold_preds["naive średnia(24h, tydzień)"] = ((X.iloc[te]["price_lag_24h"] + X.iloc[te]["price_lag_week"]) / 2).values
    for name, p in fold_preds.items():
        wf_mae.setdefault(name, []).append(np.abs(y.iloc[te].values - p).mean())
 
print(f"{'':30s}" + "".join(f"{'blok ' + str(i + 1):>14s}" for i in range(len(periods))) + f"{'średnia':>10s}")
print(f"{'':30s}" + "".join(f"{p:>14s}" for p in periods))
for name, vals in sorted(wf_mae.items(), key=lambda kv: np.mean(kv[1])):
    print(f"{name:30s}" + "".join(f"{v:14.1f}" for v in vals) + f"{np.mean(vals):10.1f}")
 
model_only = {n: v for n, v in wf_mae.items() if not n.startswith("naive")}
best_wf = min(model_only, key=lambda n: np.mean(model_only[n]))
print(f"\nNajlepszy po średniej z walk-forward: {best_wf} ({np.mean(model_only[best_wf]):.1f} PLN/MWh)")

 # --- Ważność cech ---
name = "CatBoost" if "CatBoost" in fitted else next(
    n for n in fitted if hasattr(fitted[n].best_estimator_, "feature_importances_"))
imp = pd.Series(fitted[name].best_estimator_.feature_importances_, index=feature_columns)
print(f"\nWażność cech ({name}, % sumy):")
print((imp / imp.sum() * 100).sort_values(ascending=False).round(1).to_string())