import joblib
import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load model
try:
    model = joblib.load('signbridge_model.pkl')
    print("✅ Model loaded (63 landmarks)")
except FileNotFoundError:
    print("❌ signbridge_model.pkl not found! Run model_trainer.py first.")
    exit()

# MediaPipe
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

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp_image(image_format=mp_image_format.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_img, timestamp_ms)

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            h = result.hand_landmarks[0]
            
            # EXTRACT 63 FEATURES ✓
            features = []
            for lm in h:
                features.extend([lm.x, lm.y, lm.z])
            features = np.array([features])
            
            # PREDICT
            prediction = model.predict(features)[0]
            confidence = (model.predict_proba(features).max() * 100)
            
            # DRAW LANDMARKS
            for i, lm in enumerate(h):
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                color = (0, 255, 0) if i % 4 == 0 else (255, 255, 255)
                cv2.circle(frame, (x, y), 6 if i % 4 == 0 else 3, color, -1)
            
            # DISPLAY RESULT
            color = (0, 255, 0) if confidence > 75 else (0, 165, 255)
            cv2.rectangle(frame, (40, 40), (580, 120), color, 3)
            cv2.putText(frame, f"{prediction}", (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 5)
            cv2.putText(frame, f"Conf: {confidence:.0f}%", (50, 115), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

        # FPS
        fps = int(1 / (time.time() - fps_start))
        fps_start = time.time()
        cv2.putText(frame, f"FPS: {fps}", (20, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 50, 50), 2)
        cv2.putText(frame, "Q=Quit", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("SignBridge Model Tester", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
print("✅ Model testing complete.")