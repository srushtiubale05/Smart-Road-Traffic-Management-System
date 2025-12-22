import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
import os

# ==============================================
# CONFIG
# ==============================================
DATA_PATH = "data/vehicle_maintenance_data.csv"
MODEL_OUT = "maintenance_model.pkl"
COMPONENT_MODEL_OUT = "component_model.pkl"
ENC_OUT = "maintenance_encoders.pkl"
SCALER_OUT = "maintenance_scaler.pkl"
RANDOM_STATE = 42

# ==============================================
# LOAD DATA
# ==============================================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded dataset with shape {df.shape}")

# ==============================================
# FEATURE ENGINEERING
# ==============================================
target = "Need_Maintenance"

features = [
    "Vehicle_Model", "Mileage", "Maintenance_History", "Reported_Issues",
    "Vehicle_Age", "Fuel_Type", "Transmission_Type", "Engine_Size",
    "Odometer_Reading", "Owner_Type", "Insurance_Premium", "Service_History",
    "Accident_History", "Fuel_Efficiency", "Tire_Condition",
    "Brake_Condition", "Battery_Status"
]

# Drop unnecessary date columns
for col in ["Last_Service_Date", "Warranty_Expiry_Date"]:
    if col in df.columns:
        df = df.drop(columns=[col])

# Handle missing values
for col in features:
    if col not in df.columns:
        df[col] = "Unknown"

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna("Unknown")
    else:
        df[col] = df[col].fillna(df[col].median())

# ==============================================
# ENCODING
# ==============================================
encoders = {}
df_encoded = df.copy()

for col in df_encoded.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    encoders[col] = le

print("🔤 Encoding complete.")

# ==============================================
# SCALING
# ==============================================
scaler = StandardScaler()
X = df_encoded[features]
y = df_encoded[target]
X_scaled = scaler.fit_transform(X)
print("📏 Scaling complete.")

# ==============================================
# SUPERVISED MODEL 1: Logistic Regression (SGD)
# ==============================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

log_model = SGDClassifier(
    loss="log_loss", learning_rate="optimal", max_iter=1000, random_state=RANDOM_STATE
)
log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)
print("\n📊 Logistic Regression (SGD) Results:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred))

# ==============================================
# SUPERVISED MODEL 2: Decision Tree (Component Health)
# ==============================================
# Derived component labels based on feature thresholds
def component_issue(row):
    issues = []
    if row["Tire_Condition"] == "Worn Out":
        issues.append("Tyres")
    if row["Brake_Condition"] == "Worn Out":
        issues.append("Brakes")
    if row["Battery_Status"] == "Weak":
        issues.append("Battery")
    if row["Fuel_Efficiency"] < 10:
        issues.append("Engine")
    if row["Maintenance_History"] == "Poor":
        issues.append("Oil & Filters")
    return ", ".join(issues) if issues else "None"

df["Component_Issue"] = df.apply(component_issue, axis=1)
df_encoded["Component_Issue"] = LabelEncoder().fit_transform(df["Component_Issue"])

X_comp = df_encoded[features]
y_comp = df_encoded["Component_Issue"]

comp_tree = DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)
comp_tree.fit(X_comp, y_comp)
print("🌳 Component-level Decision Tree trained successfully!")

# ==============================================
# UNSUPERVISED MODEL: K-Means Clustering
# ==============================================
kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE)
clusters = kmeans.fit_predict(X_scaled)
df["Cluster"] = clusters

print("\nK-Means clustering results:")
print(df.groupby("Cluster")[["Mileage", "Fuel_Efficiency", "Reported_Issues"]].mean())

# ==============================================
# SAVE MODELS
# ==============================================
with open(MODEL_OUT, "wb") as f:
    pickle.dump(log_model, f)

with open(COMPONENT_MODEL_OUT, "wb") as f:
    pickle.dump(comp_tree, f)

with open(ENC_OUT, "wb") as f:
    pickle.dump(encoders, f)

with open(SCALER_OUT, "wb") as f:
    pickle.dump(scaler, f)

print("\nFiles saved successfully:")
print(f"  - Main Maintenance Model: {MODEL_OUT}")
print(f"  - Component Model: {COMPONENT_MODEL_OUT}")
print(f"  - Encoders: {ENC_OUT}")
print(f"  - Scaler: {SCALER_OUT}")

print("\nTraining complete — ready for Flask integration!")
