# ===============================================
# train_full_model.py
# Vehicle Maintenance Prediction (VS Code Version)
# ===============================================

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import joblib

# ------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------
DATA_PATH = os.path.join("data", "vehicle_maintenance_data.csv")

print(f"Loading dataset from: {DATA_PATH}")

df = pd.read_csv(
    DATA_PATH,
    parse_dates=['Last_Service_Date', 'Warranty_Expiry_Date']
)

# Fill optional NaNs with 0
optional_cols = ['Reported_Issues', 'Service_History', 'Accident_History', 'Fuel_Efficiency']
for col in optional_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# ------------------------------------------------
# 2. Feature Engineering
# ------------------------------------------------
today = pd.to_datetime('today')

df['days_since_last_service'] = (today - df['Last_Service_Date']).dt.days
df['warranty_remaining_days'] = (df['Warranty_Expiry_Date'] - today).dt.days
df['warranty_active'] = (df['warranty_remaining_days'] > 0).astype(int)

ord_map = {'Worn Out': 0, 'Good': 1, 'New': 2}
for col in ['Maintenance_History', 'Tire_Condition', 'Brake_Condition', 'Battery_Status']:
    if col in df.columns:
        df[col + '_ord'] = df[col].map(ord_map)

# ------------------------------------------------
# 3. Define Features & Target
# ------------------------------------------------
numeric_features = [
    'Mileage', 'Engine_Size', 'Vehicle_Age', 'Reported_Issues', 'Service_History',
    'Accident_History', 'Odometer_Reading', 'Fuel_Efficiency',
    'days_since_last_service', 'warranty_remaining_days', 'warranty_active'
]

for col in ['Maintenance_History_ord', 'Tire_Condition_ord', 'Brake_Condition_ord', 'Battery_Status_ord']:
    if col in df.columns:
        numeric_features.append(col)

categorical_features = ['Vehicle_Model', 'Fuel_Type', 'Transmission_Type', 'Owner_Type']
categorical_features = [c for c in categorical_features if c in df.columns]

target = 'Need_Maintenance'
if target not in df.columns:
    raise ValueError("Target column 'Need_Maintenance' not found in dataset.")

X = df[numeric_features + categorical_features]
y = df[target]

# ------------------------------------------------
# 4. Train-Test Split
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ------------------------------------------------
# 5. Preprocessing Pipelines
# ------------------------------------------------
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, numeric_features),
    ('cat', cat_transformer, categorical_features)
])

# ------------------------------------------------
# 6. Model Pipeline
# ------------------------------------------------
clf = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight='balanced'
    ))
])

# ------------------------------------------------
# 7. Train Model
# ------------------------------------------------
print("\nTraining model...")
clf.fit(X_train, y_train)
print("Model training complete.")

# ------------------------------------------------
# 8. Evaluate Model
# ------------------------------------------------
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("\nModel Evaluation:")
print("-" * 40)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# ------------------------------------------------
# 9. Calibrate Probabilities
# ------------------------------------------------
print("\nCalibrating model probabilities...")
clf_calibrated = CalibratedClassifierCV(clf, method='isotonic', cv=5)
clf_calibrated.fit(X_train, y_train)
print("Calibration complete.")

# ------------------------------------------------
# 10. Save Models (Inside VS Code Project Folder)
# ------------------------------------------------
MODEL_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(clf, os.path.join(MODEL_DIR, "vehicle_maintenance_model.joblib"))
joblib.dump(clf, os.path.join(MODEL_DIR, "vehicle_maintenance_model.pkl"))
joblib.dump(clf_calibrated, os.path.join(MODEL_DIR, "vehicle_maintenance_model_calibrated.pkl"))

print(f"\nAll models saved inside:\n   {MODEL_DIR}")
print("\nTraining complete! You can now load and use the model in your Flask app or another script.")
