#!/usr/bin/env python3
"""
detect_helmet.py
Runs YOLO helmet detection on a default or specified image,
saves annotated result + cropped plate if found.

Usage (optional):
    python detect_helmet.py --image data/new22.jpg --weights data/best.pt
"""

import os
import argparse
import cv2
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="data/new22.jpg",
                   help="Path to input image (default: data/new22.jpg)")
    p.add_argument("--weights", default="data/best.pt",
                   help="Path to YOLO .pt weights (default: data/best.pt)")
    p.add_argument("--out", default="static/outputs",
                   help="Output directory (default: static/outputs)")
    p.add_argument("--conf", type=float, default=0.45,
                   help="Confidence threshold")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    # must match your training order
    class_names = ["with helmet", "without helmet", "rider", "number plate"]

    print(f"Loading image: {args.image}")
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Image not found: {args.image}")

    print("Running detection...")
    results = model(img, conf=args.conf, verbose=False)

    no_helmet_detected = False
    plate_cropped = None
    detections_output = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = class_names[cls] if cls < len(class_names) else f"cls_{cls}"

            # Draw bounding boxes
            color = (0, 200, 0) if label != "without helmet" else (0, 80, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            detections_output.append({
                "label": label,
                "confidence": round(conf * 100, 2),
                "bbox": [x1, y1, x2, y2]
            })

            if label == "without helmet":
                no_helmet_detected = True
            if label == "number plate":
                plate_cropped = img[y1:y2, x1:x2].copy()

    # Save annotated image
    base = os.path.splitext(os.path.basename(args.image))[0]
    out_annot = os.path.join(args.out, f"{base}_annotated.jpg")
    cv2.imwrite(out_annot, img)
    print(f"Saved annotated image: {out_annot}")

    # Save cropped plate
    plate_path = None
    if plate_cropped is not None:
        plate_path = os.path.join(args.out, f"{base}_plate.jpg")
        cv2.imwrite(plate_path, plate_cropped)
        print(f"Saved cropped plate: {plate_path}")

    if no_helmet_detected:
        print("Rider(s) without helmet detected!")
    else:
        print("All riders wearing helmets (no violation found).")

    print("\nDetections:")
    for d in detections_output:
        print(f" - {d['label']} ({d['confidence']}%)")

if __name__ == "__main__":
    main()
