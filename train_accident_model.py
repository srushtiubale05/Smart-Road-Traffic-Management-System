# train_risk_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample

# ---------- CONFIG ----------
DATA_PATH = "data/RTA Dataset.csv"   # update if needed
MODEL_OUT = "risk_model.pkl"
ENC_OUT = "risk_encoders.pkl"
RANDOM_STATE = 42
# ----------------------------

# 1) Load dataset
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape}")

# 2) Define features that are realistic pre-accident predictors
features = [
    "Day_of_week",
    "Time",  # if your dataset doesn't have 'Time', change to 'Time_of_day' or similar
    "Age_band_of_driver",
    "Sex_of_driver",
    "Driving_experience",
    "Type_of_vehicle",
    "Service_year_of_vehicle",
    "Defect_of_vehicle",
    "Area_accident_occured",
    "Road_surface_conditions",
    "Light_conditions",
    "Weather_conditions"
]

# If your dataset uses different column names for time or service year, adapt the list above.
# 3) If Accident_severity exists — map it to Risk (Low/Medium/High).
if "Accident_severity" not in df.columns:
    raise ValueError("Dataset must contain 'Accident_severity' column to derive risk labels.")

# Normalize some common severity label variants to a canonical form first:
def normalize_severity(x):
    x = str(x).strip().lower()
    if "fatal" in x:
        return "Fatal"
    if "serious" in x:
        return "Serious"
    if "slight" in x or "minor" in x:
        return "Slight"
    # fallback
    return x.title()

df["Accident_severity_clean"] = df["Accident_severity"].astype(str).apply(normalize_severity)

# Map severity -> risk
severity_to_risk = {
    "Slight": "Low",
    "Serious": "Medium",
    "Fatal": "High"
}
df["Accident_risk_level"] = df["Accident_severity_clean"].map(severity_to_risk)

# Remove rows without a mapped risk level (if any)
df = df.dropna(subset=["Accident_risk_level"])

# If 'Time' isn't present, create a simple 'Time' column from existing fields if possible,
# otherwise create a placeholder 'Unknown'. (Adjust this if your CSV has 'Time_of_day' etc.)
if "Time" not in df.columns:
    # attempt to use Time_of_day or create placeholder
    if "Time_of_day" in df.columns:
        df["Time"] = df["Time_of_day"]
    else:
        df["Time"] = "Unknown"

# Keep only desired columns if they exist. For missing columns create 'Unknown' placeholders.
for col in features:
    if col not in df.columns:
        print(f"Warning: {col} not in CSV. Filling with 'Unknown'.")
        df[col] = "Unknown"

df = df[features + ["Accident_risk_level"]]

# 4) Handle missing values for features: fill with "Unknown"
for col in features:
    df[col] = df[col].fillna("Unknown").astype(str)

# 5) Encode categorical variables and the target
encoders = {}
df_encoded = df.copy()
for col in features + ["Accident_risk_level"]:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    encoders[col] = le

# 6) Optional: balance classes by upsampling minority risk levels
major_class = df_encoded["Accident_risk_level"].value_counts().idxmax()
dfs = [df_encoded[df_encoded["Accident_risk_level"] == major_class]]

for cls in df_encoded["Accident_risk_level"].unique():
    if cls != major_class:
        dfg = df_encoded[df_encoded["Accident_risk_level"] == cls]
        dfg_up = resample(dfg,
                          replace=True,
                          n_samples=len(df_encoded[df_encoded["Accident_risk_level"] == major_class]),
                          random_state=RANDOM_STATE)
        dfs.append(dfg_up)

df_bal = pd.concat(dfs).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
print("Class distribution after balancing:")
print(df_bal["Accident_risk_level"].value_counts())

# 7) Train-test split
X = df_bal[features]
y = df_bal["Accident_risk_level"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# 8) Train model
model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, max_depth=15, class_weight="balanced")
model.fit(X_train, y_train)

# 9) Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {acc:.4f}")
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=encoders["Accident_risk_level"].classes_))

# 10) Save model & encoders
with open(MODEL_OUT, "wb") as f:
    pickle.dump(model, f)
with open(ENC_OUT, "wb") as f:
    pickle.dump(encoders, f)

print(f"Saved model to {MODEL_OUT} and encoders to {ENC_OUT}")
