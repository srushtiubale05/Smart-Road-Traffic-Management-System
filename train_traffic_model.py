# train_traffic_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# ---------------------------
# 1. Load traffic dataset
# ---------------------------
df = pd.read_csv("data/traffic.csv")  # CSV should have: DateTime, Junction, Vehicles
df['DateTime'] = pd.to_datetime(df['DateTime'])

print("Dataset loaded successfully!")
print(f"Total records: {len(df)}")

# ---------------------------
# 2. Extract datetime features
# ---------------------------
df['hour'] = df['DateTime'].dt.hour
df['day'] = df['DateTime'].dt.day
df['month'] = df['DateTime'].dt.month
df['weekday'] = df['DateTime'].dt.weekday

# Previous hour vehicles
df['prev_hour'] = df.groupby('Junction')['Vehicles'].shift(1)
df['prev_hour'].fillna(0, inplace=True)

# ---------------------------
# 3. Encode Junctions
# ---------------------------
junction_encoder = LabelEncoder()
df['junction_encoded'] = junction_encoder.fit_transform(df['Junction'])

print(f"\nJunctions found: {list(junction_encoder.classes_)}")

# ---------------------------
# 4. Features and target
# ---------------------------
X = df[['junction_encoded', 'hour', 'day', 'month', 'weekday', 'prev_hour']]
y = df['Vehicles']

# ---------------------------
# 5. Train-Test Split (80-20)
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ---------------------------
# 6. Train Random Forest Regressor
# ---------------------------
print("\nTraining Random Forest model...")
traffic_model = RandomForestRegressor(
    n_estimators=100, 
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    n_jobs=-1
)
traffic_model.fit(X_train, y_train)
print("Model training complete!")

# ---------------------------
# 7. Make predictions on test set
# ---------------------------
y_pred = traffic_model.predict(X_test)

# ---------------------------
# 8. Calculate evaluation metrics
# ---------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("Model Performance (Random Forest):")
print("="*50)
print(f"MAE  (Mean Absolute Error):     {mae:.3f}")
print(f"MSE  (Mean Squared Error):      {mse:.3f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.3f}")
print(f"R² Score:                       {r2:.3f}")
print("="*50)

# ---------------------------
# 9. Print comparison table (first 20 values)
# ---------------------------
comparison_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
comparison_df['Difference'] = comparison_df['Actual'] - comparison_df['Predicted']
comparison_df['Error %'] = (abs(comparison_df['Difference']) / comparison_df['Actual'] * 100).round(2)

print("\nSample Actual vs Predicted (First 20):")
print(comparison_df.head(20).to_string(index=False))

# ---------------------------
# 10. Feature Importance
# ---------------------------
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': traffic_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance.to_string(index=False))

# ---------------------------
# 11. Plot results
# ---------------------------
plt.figure(figsize=(14, 8))

# Plot 1: Actual vs Predicted
plt.subplot(2, 1, 1)
plt.plot(y_test.values[:100], label="Actual Traffic", marker='o', linewidth=2)
plt.plot(y_pred[:100], label="Predicted Traffic", marker='x', linewidth=2)
plt.title("Traffic Prediction using Random Forest (ML Model) - First 100 Samples", fontsize=14, fontweight='bold')
plt.xlabel("Sample Index")
plt.ylabel("Vehicles")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Residuals (Prediction Error)
plt.subplot(2, 1, 2)
residuals = y_test.values - y_pred
plt.scatter(y_pred, residuals, alpha=0.5, color='red')
plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
plt.title("Residual Plot (Prediction Error)", fontsize=14, fontweight='bold')
plt.xlabel("Predicted Vehicles")
plt.ylabel("Residuals (Actual - Predicted)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('traffic_model_evaluation.png', dpi=300, bbox_inches='tight')
print("\nPlots saved as 'traffic_model_evaluation.png'")
plt.show()

# ---------------------------
# 12. Additional visualization: Feature Importance Bar Chart
# ---------------------------
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
plt.xlabel('Importance Score', fontsize=12)
plt.title('Feature Importance in Traffic Prediction Model', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("Feature importance plot saved as 'feature_importance.png'")
plt.show()

# ---------------------------
# 13. Save model and encoder
# ---------------------------
joblib.dump(traffic_model, "traffic_rf_model.pkl")
joblib.dump(junction_encoder, "junction_encoder.pkl")

print("\n Traffic model and junction encoder saved successfully!")
print("   - traffic_rf_model.pkl")
print("   - junction_encoder.pkl")

# ---------------------------
# 14. Summary Statistics
# ---------------------------
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
print(f"Total training time: Random Forest with {traffic_model.n_estimators} trees")
print(f"Average vehicles in dataset: {df['Vehicles'].mean():.2f}")
print(f"Max vehicles recorded: {df['Vehicles'].max()}")
print(f"Min vehicles recorded: {df['Vehicles'].min()}")
print(f"Prediction accuracy (R² Score): {r2*100:.2f}%")
print("="*50)