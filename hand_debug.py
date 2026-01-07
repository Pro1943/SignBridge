from pathlib import Path
import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent
TASK_PATH = ROOT / "hand_landmarker.task"

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
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

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows: more reliable for some webcams
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Camera not opened")
        raise SystemExit(1)

    ts = 0  # manual monotonic timestamp
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_img = mp_image(image_format=mp_image_format.SRGB, data=rgb)

        ts += 33  # ~30 FPS monotonic timestamps (IMPORTANT)
        result = landmarker.detect_for_video(mp_img, ts)

        detected = (result.hand_landmarks is not None) and (len(result.hand_landmarks) > 0)
        cv2.putText(frame, f"Hand: {'YES' if detected else 'NO'}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if detected else (0,0,255), 2)

        if detected:
            h = result.hand_landmarks[0]
            for lm in h:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        cv2.imshow("Debug HandLandmarker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
# =========================