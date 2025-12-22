"""
config.py
---------
Configuration settings for traffic management system
"""

import os

# ============================================================
# DATABASE CONFIGURATION
# ============================================================
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Ssu@2005',  # Change this
    'database': 'traffic_management',
    'charset': 'utf8mb4',
    'raise_on_warnings': True
}

# ============================================================
# TESSERACT OCR CONFIGURATION
# ============================================================
# Windows
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Linux
# TESSERACT_PATH = '/usr/bin/tesseract'

# ============================================================
# FILE PATHS
# ============================================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ============================================================
# YOLO MODEL CONFIGURATION
# ============================================================
HELMET_MODEL_PATH = os.path.join(MODELS_DIR, "helmet_detection.pt")
HELMET_CONFIDENCE_THRESHOLD = 0.45

# Class names for helmet detection model
HELMET_CLASS_NAMES = [
    "with helmet",
    "without helmet", 
    "rider",
    "number plate"
]

# ============================================================
# OCR CONFIGURATION
# ============================================================
OCR_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for OCR results
USE_EASYOCR = True  # Use EasyOCR if available
USE_TESSERACT = True  # Use Tesseract if available

# ============================================================
# VIOLATION CONFIGURATION
# ============================================================
VIOLATION_FINES = {
    'No Helmet': 500.00,
    'Wrong Side': 1000.00,
    'Signal Jump': 1000.00,
    'Speed Limit': 2000.00,
    'Triple Riding': 500.00
}

# Default fine if violation type not found
DEFAULT_FINE = 500.00

# ============================================================
# VIDEO PROCESSING CONFIGURATION
# ============================================================
PROCESS_EVERY_N_FRAMES = 30  # Process every Nth frame to speed up
VIDEO_OUTPUT_CODEC = 'mp4v'  # Video codec for output

# ============================================================
# ALLOWED FILE EXTENSIONS
# ============================================================
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# ============================================================
# FLASK CONFIGURATION
# ============================================================
SECRET_KEY = 'traffic_prediction_secret_key_2024'  # Change in production
DEBUG_MODE = True  # Set to False in production
PORT = 5000

# ============================================================
# INDIAN NUMBER PLATE PATTERNS
# ============================================================
PLATE_PATTERNS = [
    r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$',  # MH12AB1234
    r'^[A-Z]{2}\d{2}[A-Z]\d{4}$',        # MH12A1234
    r'^[A-Z]{3}\d{3,4}$',                # DL1234
]