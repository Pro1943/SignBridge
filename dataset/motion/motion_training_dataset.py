import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# SETTINGS
# =========================
SEQ_LEN = 16          # frames per sequence
FRAME_DELAY = 0.03    # ~30 FPS
OUT_DIR = "dataset/motion/training"
CSV_PATH = os.path.join(OUT_DIR, "motion_sequences_train.csv")

LABEL_MAP = {
    ord("h"): "HELLO",
    ord("j"): "J",
    ord("z"): "Z",
}

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
# CSV Setup
# =========================
os.makedirs(OUT_DIR, exist_ok=True)

header = []
for t in range(SEQ_LEN):
    for i in range(21):
        header += [f"x{i}_t{t}", f"y{i}_t{t}", f"z{i}_t{t}"]
header.append("label")

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(header)

print("🎥 Motion TRAIN collector")
print("Keys: H=HELLO | J=J | Z=Z | Q=Quit")
print(f"Saves to: {CSV_PATH}")

def extract_63(hand_landmarks):
    feats = []
    for lm in hand_landmarks:
        feats.extend([lm.x, lm.y, lm.z])
    return np.array(feats, dtype=np.float32)

with vision.HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        cv2.putText(frame, "MOTION TRAIN: H=HELLO J=J Z=Z", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Press key then perform gesture | Q=Quit", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("SignBridge Motion TRAIN", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key in LABEL_MAP:
            label = LABEL_MAP[key]
            print(f"\n⏺ Recording {label} sequence ({SEQ_LEN} frames)")

            seq = []
            ok = True

            for t in range(SEQ_LEN):
                ret2, f2 = cap.read()
                if not ret2:
                    ok = False
                    break

                f2 = cv2.flip(f2, 1)
                rgb = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
                mp_img = mp_image(image_format=mp_image_format.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_img, int(time.time() * 1000))

                if not (result.hand_landmarks and len(result.hand_landmarks) > 0):
                    ok = False
                    cv2.putText(f2, "NO HAND - RETRY", (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.imshow("SignBridge Motion TRAIN", f2)
                    cv2.waitKey(500)
                    break

                h = result.hand_landmarks[0]
                feats63 = extract_63(h)
                seq.append(feats63)

                # Draw + progress
                for lm in h:
                    x, y = int(lm.x * f2.shape[1]), int(lm.y * f2.shape[0])
                    cv2.circle(f2, (x, y), 4, (0, 255, 0), -1)

                cv2.putText(f2, f"{label}: {t+1}/{SEQ_LEN}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.imshow("SignBridge Motion TRAIN", f2)
                cv2.waitKey(1)
                time.sleep(FRAME_DELAY)

            if ok and len(seq) == SEQ_LEN:
                row = np.concatenate(seq, axis=0).tolist()
                row.append(label)

                with open(CSV_PATH, "a", newline="") as f:
                    csv.writer(f).writerow(row)

                print(f"✅ Saved motion sequence: {label}")
            else:
                print("❌ Sequence discarded (no hand / not enough frames)")

    cap.release()
    cv2.destroyAllWindows()

print(f"\n📁 Final motion training dataset: {CSV_PATH}")
