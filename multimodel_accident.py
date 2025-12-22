# train_all_models.py
import os
import pickle
import json
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Naive Bayes: try categorical NB, fallback to GaussianNB/MultinomialNB
try:
    from sklearn.naive_bayes import CategoricalNB as NB_CLS
except Exception:
    try:
        from sklearn.naive_bayes import GaussianNB as NB_CLS
    except Exception:
        from sklearn.naive_bayes import MultinomialNB as NB_CLS

# ---------- CONFIG ----------
DATA_PATH = "data\RTA Dataset.csv"
OUT_DIR = "trained_models"
METRICS_OUT = os.path.join(OUT_DIR, "metrics.csv")
ENC_OUT = os.path.join(OUT_DIR, "encoders.pkl")
MODEL_FORMAT = "pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.20
BALANCE_CLASSES = True
# ----------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# 1) Load dataset
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape}")

# Feature list (same as your script)
features = [
    "Day_of_week",
    "Time",
    "Age_band_of_driver",
    "Sex_of_driver",
    "Driving_experience",
    "Type_of_vehicle",
    "Service_year_of_vehicle",
    "Defect_of_vehicle",
    "Area_accident_occured",
    "Road_surface_conditions",
    "Light_conditions",
    "Weather_conditions",
]

# 2) Ensure target exists and canonicalize severity -> risk labels
if "Accident_severity" not in df.columns:
    raise ValueError("Dataset must contain 'Accident_severity' column to derive risk labels.")

def normalize_severity(x):
    x = str(x).strip().lower()
    if "fatal" in x:
        return "Fatal"
    if "serious" in x:
        return "Serious"
    if "slight" in x or "minor" in x:
        return "Slight"
    return x.title()

df["Accident_severity_clean"] = df["Accident_severity"].astype(str).apply(normalize_severity)
severity_to_risk = {"Slight": "Low", "Serious": "Medium", "Fatal": "High"}
df["Accident_risk_level"] = df["Accident_severity_clean"].map(severity_to_risk)
df = df.dropna(subset=["Accident_risk_level"]).reset_index(drop=True)

# 3) Time column fallback
if "Time" not in df.columns:
    if "Time_of_day" in df.columns:
        df["Time"] = df["Time_of_day"]
    else:
        df["Time"] = "Unknown"

# 4) Ensure all features exist; fill with 'Unknown' if missing
for col in features:
    if col not in df.columns:
        print(f"Warning: {col} not in CSV. Filling with 'Unknown'.")
        df[col] = "Unknown"

# Restrict dataframe to used columns
df = df[features + ["Accident_risk_level"]].copy()

# 5) Fill missing and ensure string dtype for categorical encoding
for col in features:
    df[col] = df[col].fillna("Unknown").astype(str)
df["Accident_risk_level"] = df["Accident_risk_level"].astype(str)

# 6) Encode categorical variables with LabelEncoder (keeps mapping simple)
encoders = {}
df_encoded = df.copy()
for col in features + ["Accident_risk_level"]:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    encoders[col] = le

# 7) Optional: balance classes via upsampling minority classes
if BALANCE_CLASSES:
    major_class = df_encoded["Accident_risk_level"].value_counts().idxmax()
    dfs = [df_encoded[df_encoded["Accident_risk_level"] == major_class]]
    for cls in df_encoded["Accident_risk_level"].unique():
        if cls != major_class:
            dfg = df_encoded[df_encoded["Accident_risk_level"] == cls]
            dfg_up = resample(
                dfg,
                replace=True,
                n_samples=len(df_encoded[df_encoded["Accident_risk_level"] == major_class]),
                random_state=RANDOM_STATE,
            )
            dfs.append(dfg_up)
    df_bal = pd.concat(dfs).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
else:
    df_bal = df_encoded.copy()

print("Class distribution (encoded) after balancing:")
print(df_bal["Accident_risk_level"].value_counts())

# 8) Train-test split
X = df_bal[features]
y = df_bal["Accident_risk_level"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# 9) Define models (use pipelines for models that need scaling)
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=15, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "LogisticRegression": Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    ),
    "KNN": Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
    "DecisionTree": DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=RANDOM_STATE),
    "NeuralNet": Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE)),
        ]
    ),
    "NaiveBayes": NB_CLS(),
}

# 10) Train, evaluate, save
results = []
for name, model in models.items():
    print(f"\nTraining {name} ...")
    # fit
    try:
        model.fit(X_train, y_train)
    except Exception as ex:
        print(f"Error training {name}: {ex}")
        continue

    # predict
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # classification report (use readable class names)
    target_names = list(encoders["Accident_risk_level"].classes_)
    try:
        report_dict = classification_report(y_test, y_pred, output_dict=True, target_names=target_names)
    except Exception:
        # fallback: no target_names support or mismatch - convert numeric labels back
        report = classification_report(y_test, y_pred, output_dict=True)
        report_dict = report

    # save model
    model_path = os.path.join(OUT_DIR, f"{name.lower()}.{MODEL_FORMAT}")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # append metrics summary
    per_class_recall = {}
    for cls_name in target_names:
        recall_val = report_dict.get(cls_name, {}).get("recall", None)
        per_class_recall[cls_name] = recall_val

    results.append(
        {
            "model": name,
            "accuracy": acc,
            "report": report_dict,
            "per_class_recall": per_class_recall,
            "model_path": model_path,
        }
    )

    print(f"{name} accuracy: {acc:.4f}")

# 11) Save encoders
with open(ENC_OUT, "wb") as f:
    pickle.dump(encoders, f)
print(f"Encoders saved to {ENC_OUT}")

# 12) Write a metrics CSV summary (flattened)
rows = []
for r in results:
    row = {
        "model": r["model"],
        "accuracy": r["accuracy"],
        "model_path": r["model_path"],
    }
    # add per-class recall fields
    for cls_name, rec in r["per_class_recall"].items():
        row[f"recall_{cls_name}"] = rec
    rows.append(row)

df_metrics = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
df_metrics.to_csv(METRICS_OUT, index=False)
print(f"Metrics summary saved to {METRICS_OUT}")

# 13) Save full reports (json) for detailed analysis
reports_out = os.path.join(OUT_DIR, "reports.json")
all_reports = {r["model"]: r["report"] for r in results}
with open(reports_out, "w") as f:
    json.dump(all_reports, f, indent=2)
print(f"Detailed classification reports saved to {reports_out}")

print("\nTraining complete. Models saved in:", OUT_DIR)
print("Best models by accuracy:")
print(df_metrics[["model", "accuracy"]].head())
