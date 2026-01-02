import cv2
import mediapipe as mp
import numpy as np
import time
import joblib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# LOAD MODELS
# =========================
static_model = joblib.load("signbridge_model.pkl")              # 63 features
motion_model = joblib.load("signbridge_motion_model.pkl")       # 16*63 features

# =========================
# MediaPipe Setup
# =========================
BaseOptions = python.BaseOptions
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

# =========================
# MOTION BUFFER
# =========================
SEQ_LEN = 16
motion_buffer = []     # list of (63,) vectors
wrist_x_buf = []
wrist_y_buf = []

MOTION_GATE = 0.20         # increase if motion triggers too easily
STATIC_MIN_CONF = 75.0
MOTION_MIN_CONF = 60.0

last_text = "---"
cooldown = 0

with vision.HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps_prev = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp_image(image_format=mp_image_format.SRGB, data=rgb)

        result = landmarker.detect_for_video(mp_img, int(time.time() * 1000))

        chosen = None
        chosen_conf = 0.0

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            h = result.hand_landmarks[0]

            # Draw landmarks
            for i, lm in enumerate(h):
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # Current 63
            feats63 = np.array([[lm.x, lm.y, lm.z] for lm in h], dtype=np.float32).flatten()

            # Update motion buffers
            motion_buffer.append(feats63)
            wrist_x_buf.append(h[0].x)
            wrist_y_buf.append(h[0].y)

            if len(motion_buffer) > SEQ_LEN:
                motion_buffer.pop(0)
                wrist_x_buf.pop(0)
                wrist_y_buf.pop(0)

            # --- STATIC PRED ---
            static_pred = static_model.predict([feats63])[0]
            static_conf = static_model.predict_proba([feats63]).max() * 100.0

            # --- MOTION SCORE ---
            motion_score = 0.0
            if len(wrist_x_buf) >= SEQ_LEN:
                motion_score = (max(wrist_x_buf) - min(wrist_x_buf)) + (max(wrist_y_buf) - min(wrist_y_buf))

            # --- MOTION PRED (only if buffer full) ---
            motion_pred = None
            motion_conf = 0.0
            if len(motion_buffer) == SEQ_LEN:
                seq_feats = np.concatenate(motion_buffer, axis=0)  # (16*63,)
                motion_pred = motion_model.predict([seq_feats])[0]
                motion_conf = motion_model.predict_proba([seq_feats]).max() * 100.0

            # --- CHOOSE WINNER ---
            # If strong motion, trust motion model; otherwise trust static model
            if motion_pred and motion_score >= MOTION_GATE and motion_conf >= MOTION_MIN_CONF:
                chosen = motion_pred
                chosen_conf = motion_conf
            elif static_conf >= STATIC_MIN_CONF:
                chosen = static_pred
                chosen_conf = static_conf

        # cooldown to avoid flicker
        if chosen and cooldown == 0:
            last_text = chosen
            cooldown = 10
        if cooldown > 0:
            cooldown -= 1

        # UI corner
        cv2.rectangle(frame, (20, 20), (420, 80), (25, 25, 25), -1)
        cv2.putText(frame, f"Pred: {last_text}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # FPS
        now = time.time()
        fps = int(1.0 / max(1e-6, (now - fps_prev)))
        fps_prev = now
        cv2.putText(frame, f"FPS: {fps}", (20, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 255, 120), 2)

        cv2.imshow("SignBridge", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
