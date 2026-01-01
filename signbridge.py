import cv2
import mediapipe as mp
import numpy as np
import time
import joblib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("🚀 SignBridge - AI Sign Language Translator")

# LOAD MODEL
try:
    model = joblib.load('signbridge_model.pkl')
    print("✅ Model loaded")
except:
    print("❌ Model missing!")
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
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# Wave
wave_positions = []
WAVE_FRAMES = 12
WAVE_THRESHOLD = 0.2

last_gesture = None
gesture_cooldown = 0

# MAIN LOOP
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps_start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp_image(image_format=mp_image_format.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_img, timestamp_ms)

        gesture = None

        # ML PREDICTION
        if result.hand_landmarks:
            for h in result.hand_landmarks[:1]:
                # Landmarks
                for i, lm in enumerate(h):
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    color = (0, 255, 255) if i % 5 == 0 else (0, 255, 0)
                    cv2.circle(frame, (x, y), 6 if i % 5 == 0 else 3, color, -1)
                
                # PREDICT
                features = np.array([[lm.x, lm.y, lm.z] for lm in h]).flatten()
                prediction = model.predict([features])[0]
                confidence = model.predict_proba([features]).max() * 100
                
                if confidence > 75:
                    gesture = f"{prediction}"

        # WAVE
        if result.hand_landmarks:
            wrist_x = result.hand_landmarks[0][0].x
            wave_positions.append(wrist_x)
            if len(wave_positions) > WAVE_FRAMES: wave_positions.pop(0)
            if len(wave_positions) == WAVE_FRAMES:
                if max(wave_positions) - min(wave_positions) > WAVE_THRESHOLD:
                    gesture = "HELLO"
                    wave_positions.clear()

        # COOLDOWN
        if gesture and gesture != last_gesture and gesture_cooldown == 0:
            last_gesture = gesture
            gesture_cooldown = 15
            print(f"🎯 {gesture}")

        if gesture_cooldown > 0: gesture_cooldown -= 1

        # SMALL CORNER UI
        cv2.rectangle(frame, (20, 30), (380, 90), (30, 30, 30), -1)
        pred_text = last_gesture or "No Gesture"
        cv2.putText(frame, f"Prediction: {pred_text}", 
                   (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # FPS + INFO
        fps = int(1 / (time.time() - fps_start))
        fps_start = time.time()
        cv2.putText(frame, f"FPS: {fps}", (20, frame.shape[0]-60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        cv2.putText(frame, "Q=Quit", (20, frame.shape[0]-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("SignBridge", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()