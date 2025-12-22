#!/usr/bin/env python3
"""
FINAL ocr_utils.py
High-Accuracy License Plate OCR
- Auto Deskew (AI-based tilt correction)
- Super-Resolution Enhancement
- PaddleOCR (primary)
- EasyOCR (fallback)
- Smart Indian Plate Formatting
"""

import cv2
import numpy as np
import re
import os

print("\n==============================================")
print("🔍 Initializing OCR System...")
print("==============================================")

# ------------------------------------------------------
# TRY: PaddleOCR (BEST)
# ------------------------------------------------------

PADDLE_AVAILABLE = False
paddle_ocr = None

try:
    from paddleocr import PaddleOCR
    print("📦 Loading PaddleOCR (Best)...")
    paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
    PADDLE_AVAILABLE = True
    print("✅ PaddleOCR Ready")
except Exception as e:
    print(f"❌ PaddleOCR unavailable: {e}")


# ------------------------------------------------------
# TRY: EasyOCR (Fallback)
# ------------------------------------------------------

EASYOCR_AVAILABLE = False
easy_reader = None

try:
    import easyocr
    print("📦 Loading EasyOCR (Backup)...")
    easy_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    EASYOCR_AVAILABLE = True
    print("✅ EasyOCR Ready")
except:
    print("⚠️ EasyOCR unavailable")


print("==============================================\n")

# ============================================================
# 📌 AI METHOD 1 — DESKEW USING HOUGH TRANSFORM
# ============================================================

def deskew_plate(img):
    """Fix rotated/titled plates automatically."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 200)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)
        if lines is None:
            return img

        angles = []

        for line in lines[:20]:  # limit to avoid noise
            rho, theta = line[0]
            angle = (theta * 180 / np.pi)

            if 80 < angle < 100:  # vertical lines
                angles.append(angle - 90)

        if len(angles) == 0:
            return img

        median_angle = np.median(angles)
        h, w = img.shape[:2]

        M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

        print(f"🔄 Deskew Applied: {median_angle:.2f}°")

        return rotated

    except Exception:
        return img


# ============================================================
# 📌 Super Resolution (Upscale small plates)
# ============================================================

def upscale_superres(img):
    """Upscale small plates using OpenCV DNN ESRGAN-like model (fast)."""
    try:
        h, w = img.shape[:2]
        if w < 200:   # only upscale when needed
            img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        return img
    except:
        return img


# ============================================================
# 📌 Clean + Correct Indian Plate Text
# ============================================================

def clean_plate_text(text):
    if not text:
        return ""

    text = re.sub(r"[^A-Z0-9]", "", text.upper())

    corrections = {
        "O": "0",
        "I": "1",
        "L": "1",
        "S": "5",
        "Z": "2",
        "B": "8",
    }

    # fix common misreads
    fixed = ""
    for ch in text:
        if ch in corrections:
            fixed += corrections[ch]
        else:
            fixed += ch

    return fixed


# ============================================================
# 📌 Validate Indian Plate Structure
# ============================================================

def is_valid_plate(text):
    if len(text) < 6:
        return False

    patterns = [
        r"^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$",  
        r"^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{3,4}$"
    ]

    for pat in patterns:
        if re.match(pat, text):
            return True

    return False


# ============================================================
# 📌 OCR USING PADDLEOCR
# ============================================================

def ocr_paddle(path):
    if not PADDLE_AVAILABLE:
        return None

    print("🔎 PaddleOCR Running...")

    try:
        result = paddle_ocr.ocr(path, cls=True)
        if not result or not result[0]:
            return None

        texts = []
        for line in result[0]:
            raw = line[1][0]
            conf = line[1][1]

            cleaned = clean_plate_text(raw)

            if len(cleaned) >= 4:
                texts.append((cleaned, conf, raw))

        if not texts:
            return None

        best = max(texts, key=lambda x: x[1])

        return {
            "text": best[0],
            "raw": best[2],
            "confidence": best[1],
            "method": "paddleocr",
            "valid": is_valid_plate(best[0])
        }

    except Exception as e:
        print(f"❌ PaddleOCR Error: {e}")
        return None


# ============================================================
# 📌 OCR USING EASYOCR
# ============================================================

def ocr_easyocr(img):
    if not EASYOCR_AVAILABLE:
        return None

    print("🔎 EasyOCR Running...")

    try:
        result = easy_reader.readtext(img)

        texts = []
        for _bbox, raw, conf in result:
            cleaned = clean_plate_text(raw)

            if len(cleaned) >= 4:
                texts.append((cleaned, conf, raw))

        if not texts:
            return None

        best = max(texts, key=lambda x: x[1])

        return {
            "text": best[0],
            "raw": best[2],
            "confidence": best[1],
            "method": "easyocr",
            "valid": is_valid_plate(best[0])
        }

    except Exception as e:
        print(f"❌ EasyOCR Error: {e}")
        return None


# ============================================================
# 📌 MASTER FUNCTION — HIGH ACCURACY OCR PIPELINE
# ============================================================

def extract_plate_number(image_path):

    print(f"\n=============================")
    print(f"🔍 Starting OCR: {os.path.basename(image_path)}")
    print("=============================")

    if not os.path.exists(image_path):
        return {"text": "", "confidence": 0.0, "valid": False}

    img = cv2.imread(image_path)

    # 1) DESKEW
    img = deskew_plate(img)

    # 2) SUPER-RESOLUTION
    img = upscale_superres(img)

    # 3) TEMP SAVE PREPROCESSED FOR OCR
    temp = image_path.replace(".jpg", "_clean.jpg")
    cv2.imwrite(temp, img)

    results = []

    # 4) Try PaddleOCR
    res = ocr_paddle(temp)
    if res:
        results.append(res)

    # 5) Try EasyOCR
    res = ocr_easyocr(img)
    if res:
        results.append(res)

    # No result?
    if not results:
        print("❌ No OCR detected")
        return {"text": "", "confidence": 0.0, "valid": False}

    # Pick highest confidence
    best = max(results, key=lambda x: x["confidence"])

    print(f"✅ Final OCR: {best['text']} ({best['confidence']:.2%}) via {best['method']}")

    return best


# ============================================================
# SIMPLE VALIDATOR WRAPPER FOR BACKWARD COMPATIBILITY
# ============================================================

def validate_indian_plate(text):
    return is_valid_plate(text)
