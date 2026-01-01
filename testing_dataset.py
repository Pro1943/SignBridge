import cv2
import mediapipe as mp
import csv
import os
import time
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MediaPipe Setup
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

mp_image = mp.Image
mp_image_format = mp.ImageFormat

base_options = BaseOptions(model_asset_path="./hand_landmarker.task")
options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# TESTING dataset
os.makedirs("dataset/testing", exist_ok=True)
TEST_CSV = "dataset/testing/signbridge_landmarks_test.csv"

# Header
header = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)] + ["label"]

try:
    with open(TEST_CSV, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
except FileExistsError:
    pass

print("🎥 TESTING Dataset Collection")
print("Keys: A='A' | H='H' | G='GOOD' | B='BAD' | Q=Quit")
print(f"Saving: {TEST_CSV}")

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp_image(image_format=mp_image_format.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_img, timestamp_ms)

        hand_detected = len(result.hand_landmarks) > 0 if result.hand_landmarks else False
        
        if hand_detected:
            h = result.hand_landmarks[0]
            for i, lm in enumerate(h):
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5 if i % 5 == 0 else 3, (0, 255, 0), -1)

        cv2.putText(frame, "HAND DETECTED ✓" if hand_detected else "SHOW HAND", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255 if hand_detected else 0, 0), 2)
        cv2.putText(frame, "A/H/G/B=Save | Q=Quit", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("TESTING Dataset Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break

        if key in [ord("a"), ord("h"), ord("g"), ord("b")] and hand_detected:
            label_map = {ord("a"):"A", ord("h"):"H", ord("g"):"GOOD", ord("b"):"BAD"}
            label = label_map[key]
            
            row = []
            for lm in result.hand_landmarks[0]:
                row.extend([lm.x, lm.y, lm.z])
            row.append(label)

            with open(TEST_CSV, "a", newline="") as f:
                csv.writer(f).writerow(row)

            print(f"✅ TEST '{label}' saved!")
            time.sleep(0.3)

    cap.release()
    cv2.destroyAllWindows()

print(f"📁 Test data ready: {TEST_CSV}")
