
import cv2
import os
from ultralytics import YOLO

# Load YOLO model
model = YOLO(r"data\best.pt")

# Folder for cropped plates
os.makedirs("cropped_plates2", exist_ok=True)

# Input video
video_path = r"testing 3 video.mp4"
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None

plate_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Draw stop-line near bottom right
    stop_line_y = int(h * 0.88)
    cv2.line(frame, (0, stop_line_y), (w, stop_line_y), (0, 0, 255), 3)

    results = model(frame, stream=True)

    stopline_violation = False
    plate_boxes = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])  # class id
            conf = float(box.conf[0])

            bike_bottom = y2  # bottom of detection box

            # ---- STOP-LINE VIOLATION DETECTION ----
            # cls values:
            # 0 = helmet, 1 = no helmet, 2 = rider, 3 = number plate
            if cls in [0, 1, 2]:  # bike + rider classes
                if bike_bottom > stop_line_y:
                    stopline_violation = True

            # ---- NUMBER PLATE DETECTION ----
            if cls == 3:  # plate class
                plate_boxes.append((x1, y1, x2, y2))
                color = (0, 255, 255)
                label = "Plate"
            else:
                color = (255, 255, 255)
                label = str(cls)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # ---- HANDLE STOP-LINE VIOLATION ----
    if stopline_violation and len(plate_boxes) > 0:
        for (px1, py1, px2, py2) in plate_boxes:
            crop = frame[py1:py2, px1:px2]
            cv2.imwrite(f"cropped_plates2/STOPLINE_{plate_id}.jpg", crop)
            plate_id += 1

        cv2.putText(frame, "⚠️ STOP-LINE VIOLATION", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    # Initialize output writer
    if out is None:
        out = cv2.VideoWriter("output_stopline.mp4", fourcc, 20, (w, h))

    out.write(frame)

cap.release()
out.release()

print("\n✅ Stop-line Processing Complete!")
print("🎥 Output saved as: output_stopline.mp4")
print("📸 Cropped plates saved in: cropped_plates2/\n")
