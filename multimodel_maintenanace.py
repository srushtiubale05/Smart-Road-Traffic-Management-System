# multimodel_maintenanace.py (fixed OneHotEncoder param compatibility)
import os
import json
import warnings
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
import joblib

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

HAS_XGB = False
HAS_LGBM = False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

warnings.filterwarnings("ignore")

CANDIDATE_PATHS = [
    Path(__file__).parent / "data" / "vehicle_maintenance_data.csv",
    Path.cwd() / "data" / "vehicle_maintenance_data.csv",
    Path("data") / "vehicle_maintenance_data.csv",
]
OUTDIR = Path("maintenance_models")
OUTDIR.mkdir(exist_ok=True)
MODELS_DIR = OUTDIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

TEST_SIZE = 0.20
RANDOM_STATE = 42
BALANCE_CLASSES = True
USE_CLASS_WEIGHT = True

def find_dataset(candidates):
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError(f"No dataset found. Tried: {candidates}")

DATA_PATH = find_dataset(CANDIDATE_PATHS)
print("Using dataset:", DATA_PATH)

expected_dates = ["Last_Service_Date", "Warranty_Expiry_Date"]
csv_cols = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
parse_dates = [c for c in expected_dates if c in csv_cols]
if parse_dates:
    df = pd.read_csv(DATA_PATH, parse_dates=parse_dates)
else:
    df = pd.read_csv(DATA_PATH)

print("Loaded rows:", len(df))
print("Columns:", list(df.columns))

target_col = "Need_Maintenance"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in CSV.")

optional_numeric = ["Reported_Issues", "Service_History", "Accident_History", "Fuel_Efficiency"]
for c in optional_numeric:
    if c in df.columns:
        df[c] = df[c].fillna(0)

today = pd.to_datetime("today")
if "Last_Service_Date" in df.columns:
    df["days_since_last_service"] = (today - pd.to_datetime(df["Last_Service_Date"])).dt.days
else:
    df["days_since_last_service"] = 0

if "Warranty_Expiry_Date" in df.columns:
    df["warranty_remaining_days"] = (pd.to_datetime(df["Warranty_Expiry_Date"]) - today).dt.days
else:
    df["warranty_remaining_days"] = 0

df["warranty_active"] = (df["warranty_remaining_days"] > 0).astype(int)

ord_map = {"Worn Out": 0, "Good": 1, "New": 2}
for col in ["Maintenance_History", "Tire_Condition", "Brake_Condition", "Battery_Status"]:
    if col in df.columns:
        df[col + "_ord"] = df[col].map(ord_map)

numeric_features = []
candidate_numeric = [
    "Mileage",
    "Engine_Size",
    "Vehicle_Age",
    "Reported_Issues",
    "Service_History",
    "Accident_History",
    "Odometer_Reading",
    "Fuel_Efficiency",
    "days_since_last_service",
    "warranty_remaining_days",
    "warranty_active",
    "Maintenance_History_ord",
    "Tire_Condition_ord",
    "Brake_Condition_ord",
    "Battery_Status_ord",
]
for c in candidate_numeric:
    if c in df.columns:
        numeric_features.append(c)

categorical_features = []
candidate_categorical = ["Vehicle_Model", "Fuel_Type", "Transmission_Type", "Owner_Type"]
for c in candidate_categorical:
    if c in df.columns:
        categorical_features.append(c)

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

X = df[numeric_features + categorical_features].copy()
y = df[target_col].copy()

if y.dtype == object or y.dtype.name == "category":
    y_unique = sorted(y.dropna().unique())
    if set(y_unique) <= {"No", "Yes"} or set(y_unique) <= {"0", "1"}:
        y = y.replace({"No": 0, "Yes": 1}).astype(int)
    else:
        mapping = {v: i for i, v in enumerate(y_unique)}
        y = y.map(mapping).astype(int)
        print("Mapped target classes:", mapping)

mask = ~y.isnull()
X = X.loc[mask].reset_index(drop=True)
y = y.loc[mask].reset_index(drop=True)

if BALANCE_CLASSES:
    df_xy = pd.concat([X, y.rename("target")], axis=1)
    counts = df_xy["target"].value_counts()
    if len(counts) > 1:
        max_n = counts.max()
        df_list = [df_xy[df_xy["target"] == cls] for cls in counts.index]
        upsamples = []
        for grp in df_list:
            if len(grp) < max_n:
                upsamples.append(
                    resample(grp, replace=True, n_samples=max_n, random_state=RANDOM_STATE)
                )
            else:
                upsamples.append(grp)
        df_bal = pd.concat(upsamples).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        X = df_bal.drop(columns=["target"])
        y = df_bal["target"]
        print("Balanced class distribution:", y.value_counts().to_dict())
    else:
        print("Single-class target after filtering; skipping balancing.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print("Train/test sizes:", X_train.shape, X_test.shape)

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

# choose OneHotEncoder parameter compatible with sklearn version
import sklearn
skv = sklearn.__version__
major_minor = tuple(int(x) for x in skv.split(".")[:2])
ohe_kwargs = {"handle_unknown": "ignore"}
# sklearn>=1.2 uses sparse_output, older versions use sparse
if major_minor >= (1, 2):
    ohe_kwargs["sparse_output"] = False
else:
    ohe_kwargs["sparse"] = False

categorical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(**ohe_kwargs))]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)

models = OrderedDict()

models["RandomForest"] = Pipeline(
    steps=[
        ("pre", preprocessor),
        (
            "clf",
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced" if USE_CLASS_WEIGHT else None, n_jobs=-1),
        ),
    ]
)

models["ExtraTrees"] = Pipeline(
    steps=[
        ("pre", preprocessor),
        ("clf", ExtraTreesClassifier(n_estimators=150, random_state=RANDOM_STATE, class_weight="balanced" if USE_CLASS_WEIGHT else None, n_jobs=-1)),
    ]
)

models["GradientBoosting"] = Pipeline(
    steps=[("pre", preprocessor), ("clf", GradientBoostingClassifier(n_estimators=150, random_state=RANDOM_STATE))],
)

models["DecisionTree"] = Pipeline(
    steps=[("pre", preprocessor), ("clf", DecisionTreeClassifier(max_depth=12, random_state=RANDOM_STATE, class_weight="balanced" if USE_CLASS_WEIGHT else None))],
)

models["LogisticRegression"] = Pipeline(
    steps=[
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced" if USE_CLASS_WEIGHT else None, n_jobs=-1)),
    ]
)

models["KNN"] = Pipeline(steps=[("pre", preprocessor), ("clf", KNeighborsClassifier(n_neighbors=7))])

models["SVM"] = Pipeline(steps=[("pre", preprocessor), ("clf", SVC(kernel="rbf", probability=True))])

models["NaiveBayes"] = Pipeline(steps=[("pre", preprocessor), ("clf", GaussianNB())])

models["MLP"] = Pipeline(
    steps=[
        ("pre", preprocessor),
        ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=RANDOM_STATE)),
    ]
)

if HAS_XGB:
    models["XGBoost"] = Pipeline(
        steps=[("pre", preprocessor), ("clf", XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1))]
    )

if HAS_LGBM:
    models["LightGBM"] = Pipeline(
        steps=[("pre", preprocessor), ("clf", LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))]
    )

metrics = []
reports = {}

for name, pipeline in models.items():
    print(f"\nTraining {name} ...")
    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        print(f"Training failed for {name}: {e}")
        continue

    y_pred = pipeline.predict(X_test)
    average_mode = "binary" if len(np.unique(y_test)) == 2 else "macro"

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=average_mode, zero_division=0)
    rec = recall_score(y_test, y_pred, average=average_mode, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=average_mode, zero_division=0)

    roc = None
    if len(np.unique(y_test)) == 2:
        try:
            proba = pipeline.predict_proba(X_test)[:, 1]
            roc = float(roc_auc_score(y_test, proba))
        except Exception:
            roc = None

    try:
        rep = classification_report(y_test, y_pred, output_dict=True)
    except Exception:
        rep = None

    model_path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(pipeline, model_path)

    metrics.append(
        {
            "model": name,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(roc) if roc is not None else None,
            "model_path": str(model_path),
        }
    )
    reports[name] = rep
    print(f"{name} -> acc: {acc:.4f}, prec: {prec:.4f}, rec: {rec:.4f}, f1: {f1:.4f}, roc_auc: {roc}")

metrics_df = pd.DataFrame(metrics).sort_values(by=["f1", "recall", "accuracy"], ascending=[False, False, False])
metrics_df.to_csv(OUTDIR / "metrics_summary.csv", index=False)
with open(OUTDIR / "detailed_reports.json", "w") as f:
    json.dump(reports, f, indent=2)
print("\nSaved metrics to:", OUTDIR / "metrics_summary.csv")
print("Saved reports to:", OUTDIR / "detailed_reports.json")

print("\nModel comparison (sorted):")
print(metrics_df.to_string(index=False))

if not metrics_df.empty:
    best = metrics_df.iloc[0]
    best_model_name = best["model"]
    best_model_path = MODELS_DIR / f"{best_model_name}.joblib"
    print(f"\nBest model by ranking: {best_model_name} ({best_model_path})")
    joblib.dump(joblib.load(best_model_path), OUTDIR / "best_maintenance_model.joblib")
    print("Saved best model to:", OUTDIR / "best_maintenance_model.joblib")

print("\nAll done. Models & metrics are in the folder:", OUTDIR.resolve())
