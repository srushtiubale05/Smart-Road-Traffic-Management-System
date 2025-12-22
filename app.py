import os
import random
import pickle
import joblib
import numpy as np
import pandas as pd
import time
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

try:
    from ultralytics import YOLO
    import cv2
except Exception:
    YOLO = None
    cv2 = None

app = Flask(__name__)
CORS(app)
app.secret_key = 'traffic_prediction_secret_key_2024'

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "outputs")
STOPLINE_PLATES_DIR = os.path.join(OUTPUT_DIR, "stopline_plates")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STOPLINE_PLATES_DIR, exist_ok=True)

ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg"}
ALLOWED_VIDEO_EXT = {"mp4", "mov", "avi", "mkv", "webm"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXT

def allowed_video_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXT

STOPLINE_CLASS_NAMES = ["helmet", "no helmet", "rider", "number plate"]
PLATE_CLASS_IDX = 3
BIKE_RIDER_CLASSES = {0, 1, 2}
FRAME_SKIP = 3

USERS = {
    'admin': {
        'password': 'admin123',
        'name': 'Administrator',
        'email': 'admin@traffic.com',
        'role': 'admin'
    },
    'user': {
        'password': 'user123',
        'name': 'John Doe',
        'email': 'user@traffic.com',
        'role': 'user'
    }
}

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('root_login'))
        return f(*args, **kwargs)
    return decorated

try:
    risk_model = pickle.load(open("risk_model.pkl", "rb"))
    risk_encoders = pickle.load(open("risk_encoders.pkl", "rb"))
    print("Accident Risk Model loaded")
except Exception as e:
    risk_model, risk_encoders = None, None
    print("Accident risk model not loaded:", e)

traffic_model, junction_encoder = None, None
try:
    traffic_model = joblib.load("traffic_rf_model.pkl")
    junction_encoder = joblib.load("junction_encoder.pkl")
    print("Traffic Model loaded successfully!")
except Exception as e:
    traffic_model, junction_encoder = None, None
    print(f"Error loading traffic model: {e}")

try:
    df_traffic = pd.read_csv("data/traffic.csv")
    df_traffic["DateTime"] = pd.to_datetime(df_traffic["DateTime"])
except Exception as e:
    print(f"Error loading traffic data: {e}")
    df_traffic = pd.DataFrame()

try:
    MAINTENANCE_MODEL_PATH = os.path.join("models", "vehicle_maintenance_model.joblib")
    maintenance_model = joblib.load(MAINTENANCE_MODEL_PATH)
    print(f"Vehicle Maintenance Model loaded from {MAINTENANCE_MODEL_PATH}")
except Exception as e:
    maintenance_model = None
    print(f"Could not load Vehicle Maintenance Model: {e}")

helmet_model = None
helmet_class_names = ["with helmet", "without helmet", "rider", "number plate"]
try:
    if YOLO is not None:
        helmet_weights_path = os.path.join("models", "helmet_detection.pt")
        if os.path.exists(helmet_weights_path):
            helmet_model = YOLO(helmet_weights_path)
            print(f"YOLO helmet model loaded from {helmet_weights_path}")
        else:
            print(f"Helmet weights not found at {helmet_weights_path}")
    else:
        print("ultralytics YOLO not available (not installed)")
except Exception as e:
    helmet_model = None
    print("Error loading helmet model:", e)

stopline_model = None
try:
    if YOLO is not None:
        stopline_weights_path = os.path.join("data", "best.pt")
        if os.path.exists(stopline_weights_path):
            stopline_model = YOLO(stopline_weights_path)
            print(f"YOLO stop-line model loaded from {stopline_weights_path}")
        else:
            if helmet_model is not None:
                stopline_model = helmet_model
                print(f"Stop-line model not found, using helmet model as fallback")
            else:
                print(f"Stop-line weights not found at {stopline_weights_path}")
    else:
        print("ultralytics YOLO not available for stop-line detection")
except Exception as e:
    stopline_model = None
    print("Error loading stop-line model:", e)

@app.route('/', methods=['GET'])
def root_login():
    return render_template('login.html')

@app.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json() or {}
    username = (data.get('username') or "").strip()
    password = data.get('password') or ""
    if username in USERS and USERS[username]['password'] == password:
        user_obj = {
            'username': username,
            'name': USERS[username]['name'],
            'email': USERS[username]['email'],
            'role': USERS[username]['role']
        }
        session['user'] = user_obj
        return jsonify({'success': True, 'message': 'Login successful', 'user': user_obj})
    else:
        return jsonify({'success': False, 'message': 'Invalid username or password'}), 401

@app.route('/logout', methods=['GET'])
def logout():
    session.pop('user', None)
    return redirect(url_for('root_login'))

@app.route('/dashboard', methods=['GET'])
@login_required
def home():
    user = session.get('user')
    if user['role'] == 'admin':
        return render_template('index.html', user=user)
    else:
        return render_template('user_dashboard.html', user=user)

@app.route('/index', methods=['GET'])
@login_required
def index_page():
    user = session.get('user')
    if user['role'] == 'admin':
        return render_template('index.html', user=user)
    else:
        return render_template('user_dashboard.html', user=user)

@app.route('/traffic', methods=['GET'])
@login_required
def traffic():
    return render_template('traffic.html', user=session.get('user'))

@app.route('/accident', methods=['GET'])
@login_required
def accident():
    return render_template('accident.html', user=session.get('user'))

@app.route('/maintenance', methods=['GET'])
@login_required
def maintenance():
    return render_template('car_maintenance.html', user=session.get('user'))

@app.route('/numberplate', methods=['GET'])
@login_required
def number_plate():
    return render_template('number_plate.html', user=session.get('user'))

@app.route('/helmet', methods=['GET'])
@login_required
def helmet():
    return render_template('helmet.html', user=session.get('user'))

@app.route('/helmet_video', methods=['GET'])
@login_required
def helmet_video_page():
    return render_template('helmet_video.html', user=session.get('user'))

@app.route('/stopline', methods=['GET'])
@login_required
def stopline_page():
    return render_template('stopline.html', user=session.get('user'))


@app.route('/api/predict', methods=['POST'])
@login_required
def predict_accident():
    if risk_model is None or risk_encoders is None:
        return jsonify({'error': 'Accident model not loaded'}), 500

    try:
        data = request.json or {}

        # --------------------------------------
        # AUTO-FILL SYSTEM FIELDS
        # --------------------------------------
        auto_fields = {
            "Day_of_week": datetime.now().strftime("%A"),
            "Time": datetime.now().strftime("%H:%M"),
            "Light_conditions": (
                "Daylight" if 6 <= datetime.now().hour <= 18 else "Dark"
            ),
            "Road_surface_conditions": (
                "Wet" if data.get("Weather_conditions") in ["Rainy", "Storm"] else "Dry"
            ),
            "Area_accident_occured": "Unknown",
            "Service_year_of_vehicle": "Unknown",
        }

        # --------------------------------------
        # USER PROVIDES ONLY THESE FIELDS:
        # Age_band_of_driver, Sex_of_driver, Driving_experience,
        # Type_of_vehicle, Defect_of_vehicle, Weather_conditions
        # --------------------------------------

        # Merge user inputs with auto-generated fields
        full_data = {**auto_fields, **data}

        # --------------------------------------
        # FINAL FEATURE ORDER (MUST MATCH TRAINING)
        # --------------------------------------
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

        # --------------------------------------
        # ENCODING INPUT USING SAVED LABEL ENCODERS
        # --------------------------------------
        encoded_values = []

        for feature in features:
            value = full_data.get(feature, "Unknown")

            try:
                encoded_value = risk_encoders[feature].transform([str(value)])[0]
            except Exception:
                # fallback to the first class
                encoded_value = risk_encoders[feature].transform(
                    [risk_encoders[feature].classes_[0]]
                )[0]

            encoded_values.append(encoded_value)

        # --------------------------------------
        # FIX: Convert to DataFrame with column names
        # --------------------------------------
        input_df = pd.DataFrame([encoded_values], columns=features)

        # --------------------------------------
        # PREDICTION
        # --------------------------------------
        prediction = risk_model.predict(input_df)[0]
        prediction_proba = risk_model.predict_proba(input_df)[0]

        # Decode prediction
        risk_level = risk_encoders["Accident_risk_level"].inverse_transform(
            [prediction]
        )[0]

        # Prepare probability output
        risk_classes = risk_encoders["Accident_risk_level"].classes_
        probabilities = {
            risk_classes[i]: float(prediction_proba[i]) * 100
            for i in range(len(risk_classes))
        }

        return jsonify({
            'risk_level': risk_level,
            'probabilities': probabilities,
            'success': True
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 400


@app.route('/predict_traffic', methods=['POST'])
@login_required
def predict_traffic():
    if traffic_model is None or junction_encoder is None:
        return jsonify({"success": False, "error": "Traffic model not loaded"}), 500
    try:
        junction = request.form.get("junction")
        datetime_str = request.form.get("datetime")
        dt = datetime.fromisoformat(datetime_str)
        if junction in junction_encoder.classes_:
            junction_encoded = junction_encoder.transform([junction])[0]
        else:
            junction_encoded = junction_encoder.transform([junction_encoder.classes_[0]])[0]
        prev_rows = df_traffic[df_traffic['Junction'] == junction]
        max_vehicles = prev_rows['Vehicles'].max() if not prev_rows.empty else 100
        predictions, labels, timestamps = [], [], []
        for i in range(6):
            future_dt = dt + pd.Timedelta(hours=i)
            hour = future_dt.hour
            day = future_dt.day
            month = future_dt.month
            weekday = future_dt.weekday()
            prev_hour_time = future_dt - pd.Timedelta(hours=1)
            prev_hour_row = df_traffic[
                (df_traffic['Junction'] == junction) &
                (df_traffic['DateTime'] == prev_hour_time)
            ]
            if not prev_hour_row.empty:
                prev_hour_vehicles = int(prev_hour_row['Vehicles'].values[0])
            elif not prev_rows.empty:
                prev_hour_vehicles = int(prev_rows['Vehicles'].mean())
            else:
                prev_hour_vehicles = 0
            X_input = pd.DataFrame([{
                "junction_encoded": junction_encoded,
                "hour": hour,
                "day": day,
                "month": month,
                "weekday": weekday,
                "prev_hour": prev_hour_vehicles
            }])
            y_pred = traffic_model.predict(X_input)[0]
            predictions.append(float(round(y_pred, 2)))
            traffic_pct = (y_pred / max_vehicles) * 100 if max_vehicles > 0 else 0
            if traffic_pct < 33:
                label = "Low Traffic"
            elif traffic_pct < 66:
                label = "Moderate Traffic"
            else:
                label = "High Traffic"
            labels.append(label)
            timestamps.append(future_dt.strftime("%H:%M"))
        min_traffic_idx = predictions.index(min(predictions))
        best_time = timestamps[min_traffic_idx]
        advice = f"Best time to travel: {best_time}" if min_traffic_idx != 0 else \
                 f"Current time ({timestamps[0]}) has the lowest traffic. Travel now!"
        return jsonify({
            "success": True,
            "current_prediction": predictions[0],
            "current_label": labels[0],
            "current_time": timestamps[0],
            "predictions": predictions,
            "labels": labels,
            "timestamps": timestamps,
            "advice": advice
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

JUNCTIONS = df_traffic['Junction'].unique().tolist() if not df_traffic.empty else ["Junction A", "Junction B", "Junction C"]

@app.route("/api/realtime")
@login_required
def api_realtime():
    data = []
    now = datetime.now()
    for junction in JUNCTIONS:
        df_j = df_traffic[df_traffic['Junction'] == junction]
        latest_row = df_j[df_j['DateTime'] <= now]
        vehicles = int(latest_row['Vehicles'].iloc[-1]) if not latest_row.empty else random.randint(20, 100)
        traffic_pct = min(vehicles, 100)
        status = "Low" if traffic_pct < 33 else "Medium" if traffic_pct < 66 else "High"
        data.append({"junction": junction, "traffic_level": traffic_pct, "status": status})
    return jsonify(data)

@app.route("/api/historical/<junction>")
@login_required
def api_historical(junction):
    if df_traffic.empty or junction not in JUNCTIONS:
        today = datetime.today()
        data = [{"date": (today - pd.Timedelta(days=i)).strftime("%Y-%m-%d"), "traffic_level": random.randint(20, 100)} for i in range(30)]
    else:
        df_j = df_traffic[df_traffic['Junction'] == junction].sort_values("DateTime")
        last_30_days = df_j[df_j['DateTime'] >= (datetime.now() - pd.Timedelta(days=30))]
        data = [{"date": row['DateTime'].strftime("%Y-%m-%d"), "traffic_level": int(row['Vehicles'])} for idx, row in last_30_days.iterrows()]
    return jsonify(data[::-1])

@app.route("/api/predictive/<junction>")
@login_required
def api_predictive(junction):
    hours = [f"{h}:00" for h in range(24)]
    if traffic_model is None or junction_encoder is None:
        predictions = [random.randint(20, 100) for _ in range(24)]
    else:
        if junction in junction_encoder.classes_:
            junction_encoded = junction_encoder.transform([junction])[0]
        else:
            junction_encoded = junction_encoder.transform([junction_encoder.classes_[0]])[0]
        predictions = []
        now = datetime.now()
        for i in range(24):
            future_dt = now + pd.Timedelta(hours=i)
            hour = future_dt.hour
            day = future_dt.day
            month = future_dt.month
            weekday = future_dt.weekday()
            prev_hour_row = df_traffic[(df_traffic['Junction'] == junction) & (df_traffic['DateTime'] == (future_dt - pd.Timedelta(hours=1)))]
            prev_hour = int(prev_hour_row['Vehicles'].iloc[0]) if not prev_hour_row.empty else 50
            X_input = pd.DataFrame([{
                "junction_encoded": junction_encoded,
                "hour": hour,
                "day": day,
                "month": month,
                "weekday": weekday,
                "prev_hour": prev_hour
            }])
            y_pred = float(traffic_model.predict(X_input)[0]) if traffic_model else random.randint(20, 100)
            predictions.append(y_pred)
    alerts = [{"hour": hours[i], "alert": "High Traffic"} for i, val in enumerate(predictions) if val > 80]
    return jsonify({"hours": hours, "predictions": predictions, "alerts": alerts})


# assume maintenance_model is your trained pipeline loaded earlier (joblib.load)
# if you named it differently, adjust the name

@app.route("/api/predict_maintenance", methods=["POST"])
@login_required
def predict_maintenance():
    try:
        data = request.json or {}

        # 1) Minimal user-provided fields (from simplified UI)
        user_vehicle_model = data.get("Vehicle_Model", "Unknown")
        user_mileage = data.get("Mileage", None)               # numeric or None
        user_vehicle_age = data.get("Vehicle_Age", None)       # numeric or None
        user_fuel_type = data.get("Fuel_Type", "Unknown")
        user_last_service = data.get("Last_Service_Date", None) # ISO date string or None
        user_symptoms = data.get("Symptoms", [])               # list of symptom codes

        # 2) Training features expected by your pipeline (must match training script)
        expected_features = [
            "Mileage", "Engine_Size", "Vehicle_Age", "Reported_Issues", "Service_History",
            "Accident_History", "Odometer_Reading", "Fuel_Efficiency",
            "days_since_last_service", "warranty_remaining_days", "warranty_active",
            "Maintenance_History_ord", "Tire_Condition_ord", "Brake_Condition_ord", "Battery_Status_ord",
            "Vehicle_Model", "Fuel_Type", "Transmission_Type", "Owner_Type"
        ]

        # 3) Derive/compute sensible defaults where user didn't supply a field

        # parse last service to compute days_since_last_service
        today = pd.to_datetime(datetime.now().date())
        if user_last_service:
            try:
                last_service_dt = pd.to_datetime(user_last_service)
                days_since_last_service = (today - last_service_dt.date()).days
            except Exception:
                days_since_last_service = 180  # fallback: 6 months
        else:
            days_since_last_service = 180

        # warranty defaults (assume 5-year warranty if model < 5 years)
        try:
            vehicle_age_val = float(user_vehicle_age) if user_vehicle_age is not None else 0.0
        except Exception:
            vehicle_age_val = 0.0
        warranty_days_est = max(0, 365*5 - int(vehicle_age_val * 365))
        warranty_active = int(warranty_days_est > 0)

        # reported issues derived from symptom list length
        reported_issues_val = int(len(user_symptoms)) if isinstance(user_symptoms, (list, tuple)) else 0

        # Odometer_Reading: prefer explicit field else use Mileage
        odometer_val = None
        if "Odometer_Reading" in data and data.get("Odometer_Reading") not in [None, ""]:
            try:
                odometer_val = float(data.get("Odometer_Reading"))
            except Exception:
                odometer_val = None
        if odometer_val is None:
            # fallback to user mileage if provided
            try:
                odometer_val = float(user_mileage) if user_mileage not in [None, ""] else 0.0
            except Exception:
                odometer_val = 0.0

        # Mileage fallback
        try:
            mileage_val = float(user_mileage) if user_mileage not in [None, ""] else odometer_val
        except Exception:
            mileage_val = odometer_val if odometer_val is not None else 0.0

        # Engine size: optional; use average if not given
        engine_size_val = float(data.get("Engine_Size", 1200))

        # Service history / other counts (use defaults if not supplied)
        service_history_val = int(data.get("Service_History", 1))
        accident_history_val = int(data.get("Accident_History", 0))
        fuel_efficiency_val = float(data.get("Fuel_Efficiency", 15.0))

        # ordinal maintenance/parts conditions: map text -> ordinal if given, else default=1 (Good)
        ord_map = {"Worn Out": 0, "Good": 1, "New": 2}
        def ord_from_field(field_name, default=1):
            val = data.get(field_name, None)
            if val is None:
                return default
            return ord_map.get(str(val), default)

        maintenance_history_ord = ord_from_field("Maintenance_History", 1)
        tire_condition_ord = ord_from_field("Tire_Condition", 1)  # default Good
        brake_condition_ord = ord_from_field("Brake_Condition", 1)
        battery_status_ord = ord_from_field("Battery_Status", 1)

        # categorical defaults
        vehicle_model_val = user_vehicle_model or "Unknown"
        fuel_type_val = user_fuel_type or "Petrol"
        transmission_val = data.get("Transmission_Type", "Manual")
        owner_type_val = data.get("Owner_Type", "First")

        # 4) Build final dict in EXACT order / names expected
        input_dict = {
            "Mileage": mileage_val,
            "Engine_Size": engine_size_val,
            "Vehicle_Age": vehicle_age_val,
            "Reported_Issues": reported_issues_val,
            "Service_History": service_history_val,
            "Accident_History": accident_history_val,
            "Odometer_Reading": odometer_val,
            "Fuel_Efficiency": fuel_efficiency_val,
            "days_since_last_service": days_since_last_service,
            "warranty_remaining_days": warranty_days_est,
            "warranty_active": warranty_active,
            "Maintenance_History_ord": maintenance_history_ord,
            "Tire_Condition_ord": tire_condition_ord,
            "Brake_Condition_ord": brake_condition_ord,
            "Battery_Status_ord": battery_status_ord,
            "Vehicle_Model": vehicle_model_val,
            "Fuel_Type": fuel_type_val,
            "Transmission_Type": transmission_val,
            "Owner_Type": owner_type_val
        }

        # Ensure all expected features are present (defensive)
        for feat in expected_features:
            if feat not in input_dict:
                # set a numerical 0/empty default or 0.0
                input_dict[feat] = 0 if feat.endswith("_ord") or feat.endswith("_History") else 0.0

        # 5) Convert to DataFrame (names preserved)
        input_df = pd.DataFrame([input_dict], columns=expected_features)

        # 6) Predict using your pipeline
        pred_label = maintenance_model.predict(input_df)[0]
        pred_proba = None
        try:
            # if classifier supports predict_proba
            pred_proba = float(maintenance_model.predict_proba(input_df)[0][1]) * 100
        except Exception:
            pred_proba = None

        # Build readable response
        need_maintenance = "YES" if pred_label in [1, "1", True] else "NO"
        response = {
            "success": True,
            "need_maintenance": need_maintenance,
            "probability": round(pred_proba, 2) if pred_proba is not None else None,
            "input_used": input_dict  # (optional) useful for debugging & report
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400



@app.route("/api/helmet_detect", methods=["POST"]) 
@login_required
def api_helmet_detect():
    if helmet_model is None:
        return jsonify({"success": False, "error": "Helmet model not loaded"}), 500

    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file"}), 400

    filename = secure_filename(file.filename)
    upload_path = os.path.join(UPLOAD_DIR, filename)
    base, ext = os.path.splitext(filename)
    counter = 0
    while os.path.exists(upload_path):
        counter += 1
        filename = f"{base}_{counter}{ext}"
        upload_path = os.path.join(UPLOAD_DIR, filename)

    try:
        file.save(upload_path)
        results = helmet_model.predict(source=upload_path, conf=0.45, verbose=False)

        detections = []
        plate_cropped_path = None
        annotated_url = None

        if len(results) > 0:
            r = results[0]
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = helmet_class_names[cls] if cls < len(helmet_class_names) else f"class_{cls}"
                detections.append({
                    "label": label,
                    "confidence": round(conf * 100, 2),
                    "bbox": [x1, y1, x2, y2]
                })

            try:
                plotted = r.plot()
                if isinstance(plotted, (np.ndarray,)):
                    out_filename = f"annot_{os.path.splitext(os.path.basename(upload_path))[0]}.jpg"
                    out_path = os.path.join(OUTPUT_DIR, out_filename)
                    cv2.imwrite(out_path, plotted)
                    annotated_url = f"/static/outputs/{out_filename}"
            except Exception:
                annotated_url = None

            if annotated_url is None:
                try:
                    results.save()
                    annotated_url = f"/static/uploads/{os.path.basename(upload_path)}"
                except Exception:
                    annotated_url = f"/static/uploads/{os.path.basename(upload_path)}"

            plate_det = next((d for d in detections if 'number plate' in d['label'].lower()), None)
            if plate_det:
                x1, y1, x2, y2 = plate_det['bbox']
                img = cv2.imread(upload_path)
                if img is not None and y2 > y1 and x2 > x1:
                    crop = img[y1:y2, x1:x2]
                    plate_filename = f"plate_{os.path.splitext(os.path.basename(upload_path))[0]}.jpg"
                    plate_path = os.path.join(OUTPUT_DIR, plate_filename)
                    cv2.imwrite(plate_path, crop)
                    plate_cropped_path = f"/static/outputs/{plate_filename}"

        return jsonify({
            "success": True,
            "detections": detections,
            "annotated_image": annotated_url,
            "plate_image": plate_cropped_path
        })

    except Exception as e:
        print("Helmet detection error:", e)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/helmet_video_upload", methods=["POST"]) 
@login_required
def helmet_video_upload():
    if helmet_model is None:
        return jsonify({"success": False, "error": "Helmet model not loaded"}), 500

    if 'video' not in request.files:
        return jsonify({"success": False, "error": "No video uploaded"}), 400

    video_file = request.files['video']
    if video_file.filename == "" or not allowed_video_file(video_file.filename):
        return jsonify({"success": False, "error": "Invalid video file"}), 400

    filename = secure_filename(video_file.filename)
    upload_path = os.path.join(UPLOAD_DIR, filename)
    base, ext = os.path.splitext(filename)
    counter = 0
    while os.path.exists(upload_path):
        counter += 1
        filename = f"{base}_{counter}{ext}"
        upload_path = os.path.join(UPLOAD_DIR, filename)

    video_file.save(upload_path)
    print(f"Video uploaded: {upload_path}")
    
    try:
        out_filename = f"annotated_{base}.mp4"
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cap = cv2.VideoCapture(upload_path)
        
        if not cap.isOpened():
            raise Exception("Could not open uploaded video.")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        
        plate_paths, plate_id, frame_count = [], 0, 0
        print(f"Starting video inference ({w}x{h} @ {fps}fps)...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            
            results = helmet_model(frame, stream=True, conf=0.45)
            no_helmet_detected = False
            plate_boxes = []
            
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls == 1:
                        no_helmet_detected = True
                        color, label = (0, 0, 255), f"NO HELMET {conf:.2f}"
                    elif cls == 0:
                        color, label = (0, 255, 0), f"HELMET {conf:.2f}"
                    elif cls == 3:
                        plate_boxes.append((x1, y1, x2, y2))
                        color, label = (0, 255, 255), "Plate"
                    else:
                        color, label = (255, 255, 255), "Rider"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if no_helmet_detected and plate_boxes:
                for (px1, py1, px2, py2) in plate_boxes:
                    crop = frame[py1:py2, px1:px2]
                    if crop.size > 0:
                        plate_file = f"plate_{base}_{plate_id}.jpg"
                        plate_path = os.path.join(OUTPUT_DIR, plate_file)
                        cv2.imwrite(plate_path, crop)
                        plate_paths.append(f"/static/outputs/{plate_file}")
                        plate_id += 1
                
                cv2.putText(frame, "VIOLATION DETECTED", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            out.write(frame)
        
        cap.release()
        out.release()
        
        print(f"Processing complete for {filename}")
        return jsonify({
            "success": True,
            "annotated_video": f"/static/outputs/{out_filename}",
            "plates": plate_paths,
            "message": f"Processed {frame_count} frames successfully"
        })

    except Exception as e:
        print("Helmet video detection error:", e)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/stopline_upload', methods=['POST'])
@login_required
def api_stopline_upload():
    if stopline_model is None:
        return jsonify({"success": False, "error": "Stop-line detection model not loaded"}), 500
    
    if 'video' not in request.files:
        return jsonify({"success": False, "error": "No video uploaded"}), 400

    video_file = request.files['video']
    if video_file.filename == "" or not allowed_video_file(video_file.filename):
        return jsonify({"success": False, "error": "Invalid video file. Use MP4/MOV/AVI/MKV"}), 400

    filename = secure_filename(video_file.filename)
    base, ext = os.path.splitext(filename)
    timestamp = int(time.time())
    saved_name = f"{base}_{timestamp}{ext}"
    upload_path = os.path.join(UPLOAD_DIR, saved_name)
    video_file.save(upload_path)
    
    print(f"Stop-line video uploaded: {upload_path}")

    out_video_name = f"stopline_annot_{base}_{timestamp}.mp4"
    out_video_path = os.path.join(OUTPUT_DIR, out_video_name)

    try:
        cap = cv2.VideoCapture(upload_path)
        if not cap.isOpened():
            return jsonify({"success": False, "error": "Cannot open video file"}), 500

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

        plate_urls = []
        plate_count = 0
        frame_idx = 0
        violation_count = 0

        stop_line_y = int(h * 0.88)
        
        print(f"Processing video: {w}x{h} @ {fps}fps, stop-line at y={stop_line_y}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % FRAME_SKIP != 0:
                cv2.line(frame, (0, stop_line_y), (w, stop_line_y), (0, 0, 255), 3)
                out.write(frame)
                continue

            results = stopline_model(frame, stream=True, conf=0.4)
            stopline_violation = False
            plate_boxes = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls in BIKE_RIDER_CLASSES:
                        bike_bottom = y2
                        if bike_bottom > stop_line_y:
                            stopline_violation = True

                    if cls == PLATE_CLASS_IDX:
                        plate_boxes.append((x1, y1, x2, y2))

                    if cls == PLATE_CLASS_IDX:
                        color = (0, 255, 255)
                        label = "Plate"
                    else:
                        color = (255, 255, 255)
                        label = STOPLINE_CLASS_NAMES[cls] if cls < len(STOPLINE_CLASS_NAMES) else f"class_{cls}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.line(frame, (0, stop_line_y), (w, stop_line_y), (0, 0, 255), 3)
            if stopline_violation and plate_boxes:
                violation_count += 1
                for (px1, py1, px2, py2) in plate_boxes:
                    px1c = max(0, px1)
                    py1c = max(0, py1)
                    px2c = min(w - 1, px2)
                    py2c = min(h - 1, py2)
                    crop = frame[py1c:py2c, px1c:px2c].copy()
                    if crop.size == 0:
                        continue
                    plate_fname = f"STOPLINE_{timestamp}_{plate_count}.jpg"
                    plate_path = os.path.join(STOPLINE_PLATES_DIR, plate_fname)
                    cv2.imwrite(plate_path, crop)
                    plate_url = f"/static/outputs/stopline_plates/{plate_fname}"
                    plate_urls.append(plate_url)
                    plate_count += 1
                    print(f"Plate cropped: {plate_fname}")

                cv2.putText(frame, "STOP-LINE VIOLATION", (30, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            out.write(frame)
        cap.release()
        out.release()
        annotated_url = f"/static/outputs/{out_video_name}"
        print(f"Stop-line processing complete:")
        print(f"   - Frames processed: {frame_idx}")
        print(f"   - Violations detected: {violation_count}")
        print(f"   - Plates cropped: {len(plate_urls)}")
        return jsonify({
            "success": True,
            "annotated_video": annotated_url,
            "plates": plate_urls,
            "message": f"Processing complete - {len(plate_urls)} violation(s) detected"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Stop-line detection error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/static/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/static/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route('/static/outputs/stopline_plates/<path:filename>')
def serve_stopline_plates(filename):
    return send_from_directory(STOPLINE_PLATES_DIR, filename)


@app.route('/heatmap')
def heatmap_page():
    return render_template('heatmap.html')


@app.route("/api/heatmap", methods=["POST"])
def api_heatmap():
    import numpy as np
    import cv2
    from scipy.stats import gaussian_kde
    import base64
    import os
    from datetime import datetime

    if "image" not in request.files:
        return jsonify(success=False, error="No image uploaded"), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify(success=False, error="Empty filename"), 400

    # ----- Save file -----
    filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    upload_path = os.path.join("static/uploads", filename)
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/outputs", exist_ok=True)
    file.save(upload_path)

    # ----- Read image -----
    img = cv2.imdecode(np.fromfile(upload_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(success=False, error="Image read error"), 500

    # ----- Preprocess -----
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    edges_canny = cv2.Canny(gray_smooth, 70, 150)
    sobel = cv2.Sobel(gray_smooth, cv2.CV_64F, 1, 1, ksize=5)
    sobel_abs = cv2.convertScaleAbs(sobel)
    edges = cv2.addWeighted(edges_canny, 0.6, sobel_abs, 0.4, 0)

    kernel = np.ones((5,5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    ys, xs = np.where(edges > 50)
    if len(xs) < 50:
        heatmap = img
    else:
        coords = np.vstack([xs, ys])
        kde = gaussian_kde(coords, bw_method=0.25)

        h, w = img.shape[:2]
        xgrid = np.linspace(0, w, w)
        ygrid = np.linspace(0, h, h)
        xx, yy = np.meshgrid(xgrid, ygrid)
        grid_coords = np.vstack([xx.ravel(), yy.ravel()])
        density = kde(grid_coords).reshape(h, w)

        density_norm = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        density_smooth = cv2.GaussianBlur(density_norm, (35, 35), 0)

        cmap = cv2.applyColorMap(density_smooth, cv2.COLORMAP_TURBO)
        heatmap = cmap

    # ----- Save output -----
    out_name = f"heatmap_{filename}.jpg"
    out_path = os.path.join("static/outputs", out_name)
    cv2.imwrite(out_path, heatmap)

    # ----- Convert to Base64 for UI -----
    success, encoded = cv2.imencode(".jpg", heatmap)
    b64 = base64.b64encode(encoded).decode("utf-8")
    data_url = f"data:image/jpg;base64,{b64}"

    return jsonify(success=True, heatmap_image=data_url)







@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'risk_model_loaded': risk_model is not None,
        'risk_encoders_loaded': risk_encoders is not None,
        'traffic_model_loaded': traffic_model is not None,
        'junction_encoder_loaded': junction_encoder is not None,
        'maintenance_model_loaded': maintenance_model is not None,
        'helmet_model_loaded': helmet_model is not None,
        'stopline_model_loaded': stopline_model is not None,
        'logged_in': 'user' in session
    })

if __name__ == '__main__':
    print("SmartRaahi Traffic Management System Starting...")
    print(f"Upload Directory: {UPLOAD_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f" Stop-line Plates: {STOPLINE_PLATES_DIR}")
    print("Default Login Credentials:")
    print("   Admin: admin / admin123")
    print("   User:  user / user123")
    app.run(debug=True, port=5000, threaded=True)
