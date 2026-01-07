from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import csv
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# SETTINGS
# =========================
SEQ_LEN = 16
FRAME_DELAY = 0.01          # keep small; motion needs more attempts
MAX_ATTEMPTS = 90           # max frames to try to collect 16 valid ones (~3 sec)

ROOT = Path(__file__).resolve().parents[2]  # .../SignBridge
TASK_PATH = ROOT / "hand_landmarker.task"
OUT_DIR = Path(__file__).resolve().parent / "training"
CSV_PATH = OUT_DIR / "motion_sequences_train.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

base_options = BaseOptions(model_asset_path=str(TASK_PATH))
options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# =========================
# CSV HEADER
# =========================
header = []
for t in range(SEQ_LEN):
    for i in range(21):
        header += [f"x{i}_t{t}", f"y{i}_t{t}", f"z{i}_t{t}"]
header.append("label")

need_header = True
if CSV_PATH.exists():
    try:
        with open(CSV_PATH, "r", newline="") as f:
            first_line = f.readline().strip()
        # If empty file OR doesn't contain the word label -> treat as broken
        need_header = (first_line == "") or ("label" not in first_line)
    except:
        need_header = True

if need_header:
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(header)
    print("🛠️ CSV header written/repaired!")

print("🎥 Motion TRAIN collector")
print("Keys: H=HELLO | J=J | Z=Z | Q=Quit")
print(f"Saves to: {CSV_PATH}")

def extract_63(hand_landmarks):
    feats = []
    for lm in hand_landmarks:
        feats.extend([lm.x, lm.y, lm.z])
    return np.array(feats, dtype=np.float32)

with vision.HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows stable
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Camera not opened")
        raise SystemExit(1)

    ts = 0  # monotonic timestamp counter for VIDEO mode

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)

        cv2.putText(frame, "MOTION TRAIN: H=HELLO  J=J  Z=Z", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Tip: keep hand centered + move slower", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, "Press key to record | Q=Quit", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("SignBridge Motion TRAIN", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key in LABEL_MAP:
            label = LABEL_MAP[key]
            print(f"\n⏺ Recording {label} (need {SEQ_LEN} valid frames)")

            # Countdown so you can start moving AFTER pressing key
            for s in [3, 2, 1]:
                retc, fc = cap.read()
                if not retc or fc is None:
                    continue
                fc = cv2.flip(fc, 1)
                cv2.putText(fc, f"GET READY: {s}", (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.imshow("SignBridge Motion TRAIN", fc)
                cv2.waitKey(300)

            seq = []
            attempts = 0

            while len(seq) < SEQ_LEN and attempts < MAX_ATTEMPTS:
                attempts += 1

                ret2, f2 = cap.read()
                if not ret2 or f2 is None:
                    continue

                f2 = cv2.flip(f2, 1)
                rgb = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
                mp_img = mp_image(image_format=mp_image_format.SRGB, data=rgb)

                ts += 33  # monotonic timestamps for VIDEO mode
                result = landmarker.detect_for_video(mp_img, ts)

                detected = result.hand_landmarks and len(result.hand_landmarks) > 0

                if detected:
                    h = result.hand_landmarks[0]
                    seq.append(extract_63(h))

                    for lm in h:
                        x, y = int(lm.x * f2.shape[1]), int(lm.y * f2.shape[0])
                        cv2.circle(f2, (x, y), 4, (0, 255, 0), -1)

                # UI progress even when not detected
                cv2.putText(f2, f"{label} frames: {len(seq)}/{SEQ_LEN}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0) if detected else (0, 0, 255), 3)
                cv2.putText(f2, f"Attempts: {attempts}/{MAX_ATTEMPTS}", (20, 155),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("SignBridge Motion TRAIN", f2)
                cv2.waitKey(1)
                time.sleep(FRAME_DELAY)

            if len(seq) == SEQ_LEN:
                row = np.concatenate(seq, axis=0).tolist()
                row.append(label)

                with open(CSV_PATH, "a", newline="") as f:
                    csv.writer(f).writerow(row)

                print(f"✅ Saved motion sequence: {label}")
            else:
                print("❌ Failed to collect enough valid frames. Try slower + better light.")

    cap.release()
    cv2.destroyAllWindows()

print(f"\n📁 Final motion training dataset: {CSV_PATH}")
print("✅ Collection complete.")
# =========================