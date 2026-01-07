from pathlib import Path
import time
import joblib
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# Paths (match your structure)
# =========================
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "signbridge_model.pkl"
TASK_PATH = (ROOT / "hand_landmarker.task").resolve()

# Load model
if not MODEL_PATH.exists():
    print(f"❌ Model not found: {MODEL_PATH}")
    print("Run: python models/model_trainer.py")
    raise SystemExit(1)

model = joblib.load(MODEL_PATH)
print("✅ Model loaded (63 landmarks)")

# MediaPipe setup
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

mp_image = mp.Image
mp_image_format = mp.ImageFormat

base_options = BaseOptions(model_asset_path=str(TASK_PATH))
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

    last_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp_image(image_format=mp_image_format.SRGB, data=rgb)

        result = landmarker.detect_for_video(mp_img, int(time.time() * 1000))

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            h = result.hand_landmarks[0]

            # 63 features
            features = np.array([[lm.x, lm.y, lm.z] for lm in h], dtype=np.float32).flatten()
            features = features.reshape(1, -1)

            prediction = model.predict(features)[0]
            confidence = float(model.predict_proba(features).max() * 100.0)

            # draw landmarks
            for i, lm in enumerate(h):
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                color = (0, 255, 0) if i % 4 == 0 else (255, 255, 255)
                cv2.circle(frame, (x, y), 6 if i % 4 == 0 else 3, color, -1)

            # UI
            box_color = (0, 255, 0) if confidence > 75 else (0, 165, 255)
            cv2.rectangle(frame, (20, 20), (420, 95), box_color, -1)
            cv2.putText(frame, f"Pred: {prediction}", (30, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame, f"Conf: {confidence:.0f}%", (30, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # FPS
        now = time.time()
        fps = int(1.0 / max(1e-6, now - last_time))
        last_time = now
        cv2.putText(frame, f"FPS: {fps}", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)

        cv2.imshow("SignBridge Model Tester (Static)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

print("✅ Model testing complete.")
