# 🚦 SmartRaahi

**AI-Based Helmet Detection and Traffic Risk Prediction System**

SmartRaahi is a **unified intelligent transportation platform** that integrates **computer vision** and **machine learning** to enable real-time traffic monitoring, violation detection, accident-risk prediction, traffic forecasting, and vehicle maintenance analysis.
The system is designed for **smart-city traffic enforcement, safety analytics, and preventive decision-making**.

---

## 🔍 Project Objectives

* Predict **road-accident likelihood** using historical traffic, weather, and road-condition data
* Forecast **traffic congestion levels** across junctions
* Predict **vehicle maintenance requirements** for timely servicing
* Detect **helmet / no-helmet violations** from images and videos
* Crop **number plates** for enforcement workflows
* Detect **red-line boundary violations**
* Provide a **single integrated platform** for analytics, detection, and enforcement

---

## 🪖 Module 1: Helmet / No-Helmet Detection

### 1.1 Dataset Details

* **Total Images:** 124 traffic-scene images
* **Annotated Classes:**

  * rider
  * helmet
  * no_helmet
  * number_plate
* **Dataset Split:**

  * Train: 104 images
  * Validation: 20 images

---

### 1.2 Model Used

* **YOLOv8s (Ultralytics)**
* **Training Configuration:**

  * Epochs: 50
  * Batch Size: 8
  * Image Size: 640

**Detection Pipeline**

1. Frame preprocessing
2. YOLO object detection
3. If *no-helmet detected* → crop number plate


### 1.4 Conclusion

YOLOv8 demonstrates **high reliability and accuracy** for helmet-violation detection even with a **small dataset**, making it suitable for real-world traffic enforcement scenarios.

---

### 1.5 Future Scope

* Advanced YOLO models (YOLOv8m/l/x)
* Seat-belt and triple-riding detection
* End-to-end OCR-based challan generation
* Multi-city and multi-camera datasets

---

## 🚥 Module 2: Traffic Flow Prediction

### 2.1 Dataset

**File:** `traffic.csv`
**Columns:**

* DateTime
* Junction
* Vehicles

---

### 2.2 Feature Engineering

* **Temporal Features:** hour, day, weekday, month
* **Lag Features:** lag_1, lag_2, lag_3
* **Rolling Averages:** roll_mean_3, roll_mean_6
* **Encoded Junction IDs**

---

### 2.3 Model

* **Random Forest Regressor**
* **Why Random Forest?**

  * Handles non-linear traffic patterns
  * Robust to noisy real-world data
* **Parameters:**

  * n_estimators = 100
  * max_depth = 10

---

### 2.4 Capabilities

* 6-hour and 24-hour traffic forecasting
* Congestion-level prediction
* Real-time junction traffic status

---

### 2.5 Conclusion

Random Forest provides **accurate short-term traffic forecasts** and performs well in urban traffic environments.

---

### 2.6 Future Scope

* LSTM / GRU time-series models
* Weather and event-aware forecasting
* Graph Neural Networks (GNNs) for multi-junction learning
* Reinforcement Learning for adaptive signal control

---

## 🚗 Module 3: Vehicle Maintenance Prediction

### 3.1 Dataset

**File:** `vehicle_maintenance.csv`
**Columns:**

* Mileage
* EngineTemp
* OilQuality
* BatteryVoltage
* LastServiceDate
* BreakdownHistory
* MaintenanceNeeded

---

### 3.2 Feature Engineering

* Mileage-based degradation indicators
* Oil-quality deterioration trends
* Time since last service
* Breakdown frequency analysis

---

### 3.3 Model

* **Random Forest Classifier**
* **Parameters:**

  * n_estimators = 200
  * max_depth = 12

---

### 3.4 Results

* **Accuracy:** ~95%
* **Precision:** 0.93
* **Recall:** 0.91
* **F1-Score:** 0.92

---

### 3.5 Capabilities

* Predict maintenance requirement
* Breakdown probability estimation
* Vehicle health dashboard

---

### 3.6 Conclusion

The model delivers **highly accurate and reliable predictions**, enabling preventive vehicle maintenance.

---

### 3.7 Future Scope

* IoT sensor integration
* Component-level fault prediction
* Fleet-wide optimization analytics

---

## ⚠️ Module 4: Accident Risk Prediction

### 4.1 Dataset

**File:** `accident_data.csv`
**Fields:**

* Location
* Time
* Date
* Weather
* RoadType
* TrafficDensity
* Severity
* AccidentRisk

---

### 4.2 Feature Analysis

* Time-based accident patterns
* Weather impact
* Road geometry influence
* Traffic density correlations

---

### 4.3 Model

* **Random Forest Classifier**
* **Parameters:**

  * n_estimators = 150
  * max_depth = 15

---

### 4.4 Results

* **Accuracy:** ~92%
* **Recall (High-Risk Class):** 0.89
* **F1-Score:** 0.90
* **ROC-AUC:** 0.94

---

### 4.5 Capabilities

* Accident probability scoring
* High-risk hotspot heatmaps
* Time-based accident alerts

---

### 4.6 Conclusion

The model effectively identifies **high-risk locations and time windows**, supporting proactive safety measures.

---

### 4.7 Future Scope

* Real-time accident prediction
* LSTM for temporal risk trends
* Integration with navigation and smart-map platforms
* GNN-based spatial modeling

* Convert this into a **one-page hackathon README**
* Add **installation & execution steps**
* Write a **problem statement + innovation section**
* Create **architecture diagrams text for PPT or Eraser.io**
