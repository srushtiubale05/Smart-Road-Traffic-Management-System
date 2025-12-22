#!/usr/bin/env python3
"""
detect_helmet_video.py
----------------------
Detect riders with / without helmets in a video and crop number plates
whenever a "no helmet" violation occurs.


📁 Place your YOLO weights (.pt) inside the 'models' folder, or change the path below.
🎞️ Place your test video in the same folder or give full path.
💾 Output:
   - Annotated video: output_video2.mp4
   - Cropped plates:  cropped_plates/
"""


import os
import cv2
from ultralytics import YOLO


# ============================
# CONFIGURATION
# ============================
MODEL_PATH = r"data\best.pt"
VIDEO_PATH = r"D:\14571074_3840_2160_60fps.mp4"     # change this to your input video file
OUTPUT_VIDEO = "output_video2.mp4"
CROPPED_DIR = "cropped_plates"


# ============================
# SETUP
# ============================
os.makedirs(CROPPED_DIR, exist_ok=True)


print("📦 Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("✅ Model loaded successfully!\n")


# Initialize video reader
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"❌ Could not open video: {VIDEO_PATH}")


fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None
plate_id = 0
frame_count = 0


# ============================
# MAIN LOOP
# ============================
while True:
    ret, frame = cap.read()
    if not ret:
        break


    frame_count += 1
    results = model(frame, stream=True)


    no_helmet_detected = False
    plate_boxes = []


    # Loop through all detections in this frame
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])


            # Define class behavior
            if cls == 1:  # Without helmet
                no_helmet_detected = True
                color = (0, 0, 255)
                label = f"NO HELMET {conf:.2f}"
            elif cls == 0:  # With helmet
                color = (0, 255, 0)
                label = f"HELMET {conf:.2f}"
            elif cls == 3:  # Number plate
                plate_boxes.append((x1, y1, x2, y2))
                color = (0, 255, 255)
                label = "Plate"
            else:  # Rider or other
                color = (255, 255, 255)
                label = "Rider"


            # Draw detection on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


    # Save plates if violation
    if no_helmet_detected and plate_boxes:
        for (px1, py1, px2, py2) in plate_boxes:
            plate_crop = frame[py1:py2, px1:px2]
            if plate_crop.size > 0:
                filename = os.path.join(CROPPED_DIR, f"plate_{plate_id}.jpg")
                cv2.imwrite(filename, plate_crop)
                plate_id += 1


        cv2.putText(frame, "⚠️ VIOLATION DETECTED", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)


    # Initialize output writer once (after first valid frame)
    if out is None:
        h, w = frame.shape[:2]
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))
        print(f"🎬 Output video initialized: {OUTPUT_VIDEO} ({w}x{h}, {fps} FPS)\n")


    out.write(frame)


    # Optional: show live preview (press Q to stop early)
    cv2.imshow("Helmet Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("⏹️ Stopped early by user.")
        break


# ============================
# CLEANUP
# ============================
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()


print("\n✅ Processing Complete!")
print(f"🎥 Output video saved as: {OUTPUT_VIDEO}")
print(f"📸 Cropped plates saved in: {CROPPED_DIR}\\")
print(f"🧮 Total frames processed: {frame_count}")
print(f"🪖 Plates cropped: {plate_id}")



