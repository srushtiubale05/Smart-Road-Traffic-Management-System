# train_all_traffic_models.py
import os
import json
import warnings
import joblib
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Regressors
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# Try to import XGBoost (optional)
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
DATA_PATH = "data/traffic.csv"
OUT_DIR = "trained_traffic_models"
METRICS_OUT = os.path.join(OUT_DIR, "metrics.csv")
REPORTS_JSON = os.path.join(OUT_DIR, "reports.json")
ENC_OUT = os.path.join(OUT_DIR, "junction_encoder.pkl")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")

TEST_SIZE = 0.20     # fraction for test split (keeps order = time-series)
RANDOM_STATE = 42
N_LAGS = 3           # number of lag features (prev hour, prev2, prev3)
ROLL_WINDOWS = [3, 6]  # rolling mean windows (in samples/hours)
USE_TIMESERIES_SPLIT = False  # if True, uses TimeSeriesSplit for CV (not used for final training)
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------- 1) Load data ----------------
df = pd.read_csv(DATA_PATH)
if df.empty:
    raise SystemExit("traffic.csv is empty or missing.")

# Ensure DateTime parsed
df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df.sort_values('DateTime').reset_index(drop=True)

print(f"Loaded traffic dataset: {df.shape[0]} rows")

# ---------------- 2) Feature engineering ----------------
df['hour'] = df['DateTime'].dt.hour
df['day'] = df['DateTime'].dt.day
df['month'] = df['DateTime'].dt.month
df['weekday'] = df['DateTime'].dt.weekday

# create lag features per junction
df = df.groupby('Junction', group_keys=False).apply(lambda g: g.sort_values('DateTime')).reset_index(drop=True)

for lag in range(1, N_LAGS + 1):
    df[f'lag_{lag}'] = df.groupby('Junction')['Vehicles'].shift(lag)

# rolling means
for w in ROLL_WINDOWS:
    df[f'roll_mean_{w}'] = df.groupby('Junction')['Vehicles'].shift(1).rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)

# fill missing lags/rolls with a sensible default: previous value or 0
df[['lag_1', 'lag_2', 'lag_3']] = df[['lag_1', 'lag_2', 'lag_3']].fillna(0)
for w in ROLL_WINDOWS:
    df[f'roll_mean_{w}'] = df[f'roll_mean_{w}'].fillna(df['Vehicles'].mean())

# encode junction
junction_encoder = LabelEncoder()
df['junction_encoded'] = junction_encoder.fit_transform(df['Junction'])
print(f"Found junctions: {list(junction_encoder.classes_)}")

# ---------------- 3) Features & target ----------------
feature_cols = [
    'junction_encoded',
    'hour',
    'day',
    'month',
    'weekday'
] + [f'lag_{i}' for i in range(1, N_LAGS + 1)] + [f'roll_mean_{w}' for w in ROLL_WINDOWS]

# ensure features exist
for c in feature_cols:
    if c not in df.columns:
        raise SystemExit(f"Feature missing: {c}")

X = df[feature_cols].copy()
y = df['Vehicles'].copy()

# drop rows where target or core features are NaN (after lagging beginning rows may be NaN)
mask = X.notnull().all(axis=1) & y.notnull()
X = X[mask]
y = y[mask]
df_filtered = df.loc[mask].reset_index(drop=True)
print(f"After lag/rolling cleanup: {len(X)} samples")

# ---------------- 4) Train/test split (preserve time order) ----------------
n_test = int(np.ceil(TEST_SIZE * len(X)))
if n_test < 1:
    raise SystemExit("Test size too small for your dataset.")

X_train = X.iloc[:-n_test].copy()
X_test = X.iloc[-n_test:].copy()
y_train = y.iloc[:-n_test].copy()
y_test = y.iloc[-n_test:].copy()

print(f"Training samples: {len(X_train)}  Testing samples: {len(X_test)}")

# ---------------- 5) Define candidate models ----------------
models = OrderedDict({
    "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=150, max_depth=8, random_state=RANDOM_STATE),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=150, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1),
    "DecisionTree": DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
    "LinearRegression": LinearRegression(),
    "KNN": Pipeline([('scale', StandardScaler()), ('knn', KNeighborsRegressor(n_neighbors=6))]),
    "SVR": Pipeline([('scale', StandardScaler()), ('svr', SVR(kernel='rbf'))]),
})

if HAS_XGB:
    models["XGBoost"] = XGBRegressor(n_estimators=200, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)

# ---------------- 6) Train, evaluate, save ----------------
results = []
feature_importances = {}

for name, model in models.items():
    print(f"\nTraining {name} ...")
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Training failed for {name}: {e}")
        continue

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"{name} MAE: {mae:.3f}  RMSE: {rmse:.3f}  R2: {r2:.3f}")

    # save model
    model_path = os.path.join(OUT_DIR, f"{name.lower()}_traffic.pkl")
    try:
        joblib.dump(model, model_path)
    except Exception:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    # collect feature importance if available
    fi = None
    try:
        # if pipeline, get final estimator
        est = model
        if hasattr(model, 'steps'):
            est = model.steps[-1][1]
        if hasattr(est, 'feature_importances_'):
            fi = est.feature_importances_
        elif hasattr(est, 'coef_'):
            # linear coef (may be multi-dim)
            coef = np.ravel(est.coef_)
            # map to absolute importance
            fi = np.abs(coef)
    except Exception:
        fi = None

    if fi is not None:
        feature_importances[name] = dict(zip(feature_cols, (fi / np.sum(fi)).tolist()))

    results.append({
        "model": name,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "model_path": model_path
    })

# ---------------- 7) Save encoder and metrics ----------------
joblib.dump(junction_encoder, ENC_OUT)
print(f"\nSaved junction encoder -> {ENC_OUT}")

metrics_df = pd.DataFrame(results).sort_values("rmse")
metrics_df.to_csv(METRICS_OUT, index=False)
print(f"Saved metrics summary -> {METRICS_OUT}")

with open(REPORTS_JSON, "w") as f:
    json.dump({"results": results, "feature_importances": feature_importances}, f, indent=2)
print(f"Saved detailed reports -> {REPORTS_JSON}")

# ---------------- 8) Plots: Actual vs Predicted (first N) for top models ----------------
top_k = min(4, len(results))
top_models = metrics_df['model'].iloc[:top_k].tolist()

plt.figure(figsize=(12, 6 * top_k))
for i, model_name in enumerate(top_models):
    model_path = [r["model_path"] for r in results if r["model"] == model_name][0]
    # load model
    try:
        m = joblib.load(model_path)
    except Exception:
        with open(model_path, "rb") as f:
            m = pickle.load(f)
    y_pred = m.predict(X_test)
    ax = plt.subplot(top_k, 1, i + 1)
    ax.plot(y_test.values[:200], label="Actual", marker='o', linewidth=1)
    ax.plot(y_pred[:200], label=f"Predicted ({model_name})", marker='x', linewidth=1)
    ax.set_title(f"{model_name} - Actual vs Predicted (first 200 samples)")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(PLOTS_DIR, "actual_vs_predicted_top_models.png")
plt.savefig(plot_path, dpi=200, bbox_inches='tight')
print(f"Saved comparison plot -> {plot_path}")
plt.close()

# ---------------- 9) Feature importance plots ----------------
for name, fi_map in feature_importances.items():
    labels = list(fi_map.keys())
    vals = list(fi_map.values())
    plt.figure(figsize=(8, 4))
    y_pos = np.arange(len(labels))
    plt.barh(y_pos, vals, align='center')
    plt.yticks(y_pos, labels)
    plt.xlabel("Relative importance")
    plt.title(f"Feature importances - {name}")
    plt.gca().invert_yaxis()
    p = os.path.join(PLOTS_DIR, f"fi_{name}.png")
    plt.tight_layout()
    plt.savefig(p, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance -> {p}")

# ---------------- 10) Print final summary ----------------
print("\nFinal model ranking (by RMSE):")
print(metrics_df[['model', 'mae', 'rmse', 'r2']].to_string(index=False))

print("\nAll artifacts saved under:", OUT_DIR)
