from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import time
import joblib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

CAM_W, CAM_H = 640, 480         # lower capture res = faster
PROCESS_EVERY = 2               # run MediaPipe every N frames (2 = ~half compute)
TS_STEP_MS = 33 * PROCESS_EVERY # monotonic timestamps for VIDEO mode
DRAW_LANDMARKS = True

MOTION_GATE = 0.15               # min motion score to run motion model
STATIC_MIN_CONF = 60.0
MOTION_MIN_CONF = 52.0
SEQ_LEN = 16
COOLDOWN_FRAMES = 5

# =========================
# PATHS
# =========================
ROOT = Path(__file__).resolve().parent
TASK_PATH = ROOT / "hand_landmarker.task"
STATIC_MODEL_PATH = ROOT / "models" / "signbridge_model.pkl"
MOTION_MODEL_PATH = ROOT / "models" / "signbridge_motion_model.pkl"

# =========================
# LOAD MODELS
# =========================
if not STATIC_MODEL_PATH.exists():
    print(f"❌ Static model not found: {STATIC_MODEL_PATH}")
    raise SystemExit(1)
if not MOTION_MODEL_PATH.exists():
    print(f"❌ Motion model not found: {MOTION_MODEL_PATH}")
    raise SystemExit(1)

static_model = joblib.load(STATIC_MODEL_PATH)
motion_model = joblib.load(MOTION_MODEL_PATH)
print("✅ Models loaded")

# =========================
# MediaPipe Setup
# =========================
BaseOptions = python.BaseOptions
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

mp_image = mp.Image
mp_image_format = mp.ImageFormat

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(TASK_PATH)),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# =========================
# State
# =========================
motion_buffer = []
wrist_x_buf = []
wrist_y_buf = []

last_text = "---"
cooldown = 0

last_hand = None          # cache last detected landmarks
last_feats63 = None       # cache last 63 feature vector

frame_idx = 0
ts = 0

fps_prev = time.time()
fps_smooth = 0.0

# =========================
# MAIN LOOP
# =========================
with vision.HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except:
        pass

    if not cap.isOpened():
        print("❌ Camera not opened")
        raise SystemExit(1)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        frame_idx += 1

        # Run MediaPipe only every N frames (major FPS win)
        run_mp = (frame_idx % PROCESS_EVERY == 0)

        if run_mp:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp_image(image_format=mp_image_format.SRGB, data=rgb)

            ts += TS_STEP_MS  # monotonic timestamp required for VIDEO mode
            result = landmarker.detect_for_video(mp_img, ts)

            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                last_hand = result.hand_landmarks[0]
                last_feats63 = np.array([[lm.x, lm.y, lm.z] for lm in last_hand], dtype=np.float32).flatten()

                # Update motion buffers only when we have a detection
                motion_buffer.append(last_feats63)
                wrist_x_buf.append(last_hand[0].x)
                wrist_y_buf.append(last_hand[0].y)

                if len(motion_buffer) > SEQ_LEN:
                    motion_buffer.pop(0)
                    wrist_x_buf.pop(0)
                    wrist_y_buf.pop(0)

        chosen = None

        # Only classify if we have cached landmarks
        if last_hand is not None and last_feats63 is not None:
            # Draw landmarks (cheap)
            if DRAW_LANDMARKS:
                for i, lm in enumerate(last_hand):
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    color = (0, 255, 255) if i % 5 == 0 else (0, 255, 0)
                    cv2.circle(frame, (x, y), 5 if i % 5 == 0 else 3, color, -1)

            # Static prediction (cheap vs MediaPipe)
            static_pred = static_model.predict([last_feats63])[0]
            static_conf = float(static_model.predict_proba([last_feats63]).max() * 100.0)

            # Motion score
            motion_score = 0.0
            if len(wrist_x_buf) >= SEQ_LEN:
                motion_score = (max(wrist_x_buf) - min(wrist_x_buf)) + (max(wrist_y_buf) - min(wrist_y_buf))

            # Only run motion model when it’s plausible (saves time)
            motion_pred = None
            motion_conf = 0.0
            if len(motion_buffer) == SEQ_LEN and motion_score >= MOTION_GATE:
                seq_feats = np.concatenate(motion_buffer, axis=0)
                motion_pred = motion_model.predict([seq_feats])[0]
                motion_conf = float(motion_model.predict_proba([seq_feats]).max() * 100.0)

            # Choose
            if motion_pred and motion_conf >= MOTION_MIN_CONF:
                chosen = motion_pred
            elif static_conf >= STATIC_MIN_CONF:
                chosen = static_pred

        # Cooldown (reduce flicker)
        if chosen and cooldown == 0:
            last_text = chosen
            cooldown = COOLDOWN_FRAMES
        if cooldown > 0:
            cooldown -= 1

        # FPS (smoothed)
        now = time.time()
        inst_fps = 1.0 / max(1e-6, (now - fps_prev))
        fps_prev = now
        fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps if fps_smooth else inst_fps

        # UI
        cv2.rectangle(frame, (15, 15), (340, 80), (25, 25, 25), -1)
        cv2.putText(frame, f"Pred: {last_text}", (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {int(fps_smooth)}", (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 255, 160), 2)

        cv2.imshow("SignBridge", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
# =========================