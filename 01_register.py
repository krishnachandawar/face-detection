import argparse
import json
from pathlib import Path

import cv2

from face_utils import detect_faces, is_sharp, largest_face, preprocess_face

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
NAMES_PATH = BASE_DIR / "trainer" / "names.json"
SAMPLES = 120

parser = argparse.ArgumentParser()
parser.add_argument("name", nargs="?", help="Person name to register")
parser.add_argument("--headless", action="store_true", help="Capture without a preview window")
args = parser.parse_args()

DATASET_DIR.mkdir(exist_ok=True)
NAMES_PATH.parent.mkdir(exist_ok=True)

if NAMES_PATH.exists():
    names = {int(k): v for k, v in json.loads(NAMES_PATH.read_text(encoding="utf-8-sig")).items()}
else:
    names = {}

person_name = (args.name or input("Enter the person's name: ")).strip()
if not person_name:
    raise SystemExit("Name cannot be empty.")

user_id = (max(names.keys()) + 1) if names else 1
names[user_id] = person_name
NAMES_PATH.write_text(json.dumps(names, indent=2), encoding="utf-8")

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cam.isOpened():
    raise SystemExit("Could not open webcam.")

print(f"Registering '{person_name}' as ID {user_id}.")
print("Move slowly: front, left, right, up, down. Keep one face in frame.")
print("Press Q to stop early.")

count = 0
skipped = 0
empty_frames = 0
max_empty = 400
while count < SAMPLES:
    ok, frame = cam.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    box = largest_face(faces)
    if box is None:
        empty_frames += 1
        if args.headless and empty_frames >= max_empty:
            print("No face found after many frames. Sit in front of the camera and retry.")
            break
    else:
        empty_frames = 0

    if box is not None:
        x, y, w, h = [int(v) for v in box]
        face = preprocess_face(gray, box)
        if face is not None and is_sharp(face):
            count += 1
            cv2.imwrite(str(DATASET_DIR / f"User.{user_id}.{count}.jpg"), face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{person_name} {count}/{SAMPLES}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            if count % 10 == 0:
                print(f"Saved {count}/{SAMPLES}")
            cv2.waitKey(40)
        else:
            skipped += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(frame, "Hold still / more light", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    if not args.headless:
        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break
    elif count == 0 and skipped and skipped % 30 == 0:
        print(f"Waiting for a clear face... skipped {skipped}")

cam.release()
cv2.destroyAllWindows()
print(f"Saved {count} sharp photos in {DATASET_DIR} (skipped {skipped} blurry/poor frames).")
print("Next: python 02_train.py")
