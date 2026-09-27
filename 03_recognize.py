import json
from collections import deque
from pathlib import Path

import cv2

from face_utils import (
    CONFIDENCE_THRESHOLD,
    create_recognizer,
    detect_faces,
    largest_face,
    preprocess_face,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "trainer" / "trainer.yml"
NAMES_PATH = BASE_DIR / "trainer" / "names.json"
STABLE_FRAMES = 4

if not MODEL_PATH.exists():
    raise SystemExit("No trained model found. Run 02_train.py first.")

names = {}
if NAMES_PATH.exists():
    names = {int(k): v for k, v in json.loads(NAMES_PATH.read_text(encoding="utf-8-sig")).items()}

recognizer = create_recognizer()
recognizer.read(str(MODEL_PATH))

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cam.isOpened():
    raise SystemExit("Could not open webcam.")

print("Recognizing faces. Press Q to quit.")
history = deque(maxlen=STABLE_FRAMES)

while True:
    ok, frame = cam.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    box = largest_face(faces)

    if box is not None:
        x, y, w, h = [int(v) for v in box]
        face = preprocess_face(gray, box)
        if face is not None:
            user_id, confidence = recognizer.predict(face)
            is_match = confidence < CONFIDENCE_THRESHOLD
            label = names.get(user_id, f"ID {user_id}") if is_match else "Unknown"
            history.append((label, confidence, is_match))

            if len(history) == STABLE_FRAMES and all(item[0] == history[0][0] for item in history):
                shown_label, shown_conf, shown_match = history[-1]
            elif is_match:
                shown_label, shown_conf, shown_match = "Checking...", confidence, False
            else:
                shown_label, shown_conf, shown_match = "Unknown", confidence, False

            color = (0, 255, 0) if shown_match else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{shown_label} ({round(float(shown_conf), 1)})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

    cv2.imshow("Recognize Face", frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break

cam.release()
cv2.destroyAllWindows()
