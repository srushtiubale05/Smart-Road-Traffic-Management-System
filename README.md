# Smart-Road-Traffic-Management-System
A multi-module AI system for automated helmet detection, traffic prediction, accident risk assessment, and vehicle maintenance forecasting.

SmartRaahi: AI-Based Helmet Detection and Traffic Risk Prediction System
A unified AI platform integrating computer vision + machine learning for traffic monitoring, violation detection, accident prediction, and vehicle maintenance analysis.

Project Objectives
Predict road-accident likelihood using historical traffic, weather, and road-condition data.
Forecast traffic levels across junctions to reduce congestion.
Predict vehicle maintenance requirements for timely servicing.
Detect riders without helmets in images/videos and crop number plates.
Detect bikers crossing red-line boundaries.
Provide an integrated platform combining analytics, detection, and enforcement.

1. Helmet / No-Helmet Detection Module
1.1 Dataset Details
Total: 124 traffic-scene images.
Annotated classes: rider, helmet, no_helmet, number_plate

Dataset split:
Train: 104 images
Val: 20 images

1.2 Model Used
YOLOv8s (Ultralytics)
Training:
Epochs: 50
Batch size: 8
Image size: 640
Pipeline:
- Preprocess frame
- YOLO detection
- If no-helmet -> crop number plate

1.3 Results
mAP@50: 0.94368
mAP@50-95: 0.789
F1-score: 0.92
Precision: 1.00
Recall: 0.97

1.4 Conclusion
YOLOv8 provides reliable helmet violation detection even with a small dataset.

1.5 Future Scope
Advanced YOLO models
Seat belt / triple riding detection
End-to-end OCR challan system
Multi-location datasets

2. Traffic Flow Prediction Module
2.1 Dataset
traffic.csv  
Columns: DateTime, Junction, Vehicles

2.2 Features
Temporal: hour, day, weekday, month
Lag: lag_1, lag_2, lag_3
Rolling avg: roll_mean_3, roll_mean_6
Junction ID encoded

2.3 Model: Random Forest Regressor
Why RF?
Handles non-linearity, works well with noisy data.
Params: n_estimators=100, max_depth=10

2.4 Capabilities
6-hour and 24-hour forecasting
Congestion levels
Real-time junction status

2.5 Conclusion
RF provides accurate short-term forecasts.

2.6 Future Scope
LSTM/GRU
Weather + event-based prediction
GNNs for multi-junction learning
RL-based adaptive signals

3. Vehicle Maintenance Prediction Module
3.1 Dataset
vehicle_maintenance.csv  
Columns: Mileage, EngineTemp, OilQuality, BatteryVoltage, LastServiceDate, BreakdownHistory, MaintenanceNeeded

3.2 Features
Mileage indicators
Oil-quality deterioration
Time since last service
Breakdown frequency

3.3 Model: Random Forest Classifier
Params: n_estimators=200, max_depth=12

3.4 Results
Accuracy: ~95%
Precision: 0.93
Recall: 0.91
F1: 0.92

3.5 Capabilities
Maintenance prediction
Breakdown probability
Health dashboard

3.6 Conclusion
Accurate and reliable model for vehicle monitoring.

3.7 Future Scope
IoT integration
Component-level prediction
Fleet optimization

4. Accident Risk Prediction Module
4.1 Dataset
accident_data.csv  
Fields: Location, Time, Date, Weather, RoadType, TrafficDensity, Severity, AccidentRisk

4.2 Features
Time patterns
Weather influence
Road geometry
Density patterns

4.3 Model: Random Forest Classifier
Params: n_estimators=150, max_depth=15

4.4 Results
Accuracy: ~92%
Recall (high-risk): 0.89
F1: 0.90
ROC-AUC: 0.94

4.5 Capabilities
Accident probability scoring
Hotspot heatmaps
Time-based alerts

4.6 Conclusion
RF successfully identifies high-risk zones and times.

4.7 Future Scope
Real-time prediction
LSTM for time-series trends
Integration with navigation apps
GNNs for spatial modeling

5. System Architecture
smartraahi/
│── app.py
│── data/
│── models/
│── helmet_detection/
│── traffic_prediction/
│── accident_risk/
│── vehicle_maintenance/
│── static/
│── templates/


Final Summary
SmartRaahi integrates YOLOv8 for helmet violation detection, Random Forest for traffic forecasting, vehicle maintenance prediction, and accident-risk analysis. It forms a complete intelligent transportation system ready for smart-city applications.
